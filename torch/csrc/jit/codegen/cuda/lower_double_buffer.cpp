#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

unsigned int getDoubleBufferAxisPosition(const TensorView* tv) {
  // Double-buffering prefetches the next subregion of the tensor by
  // doubling the allocation. The subregion is defined by the axes
  // at the CA position till the inner-most position. There must be
  // at least one axis that is outside (left) of the CA position,
  // which defines the loop where prefetching is applied. Therefore,
  // the CA position must be larger than 0.

  TORCH_INTERNAL_ASSERT(tv->getComputeAtPosition() > 0, tv->toString());

  // Unroll must not exist outside of double-buffer axis
  auto first_unroll_it = std::find_if(
      tv->domain()->domain().begin(),
      tv->domain()->domain().end(),
      [](const auto axis) {
        return axis->getParallelType() == ParallelType::Unroll;
      });

  const int first_unroll_pos =
      std::distance(tv->domain()->domain().begin(), first_unroll_it);

  const int unroll_or_ca_pos =
      std::min((int)tv->getComputeAtPosition(), first_unroll_pos);

  TORCH_INTERNAL_ASSERT(
      unroll_or_ca_pos > 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found due to Unroll. ",
      tv->toString());

  int valid_pos = -1;
  // Skip parallelized or broadcast axes
  for (int i = unroll_or_ca_pos - 1; i >= 0; --i) {
    auto pt = tv->axis(i)->getParallelType();
    if (!isParallelTypeThread(pt) && !tv->axis(i)->isBroadcast()) {
      valid_pos = i;
      break;
    }
  }

  TORCH_INTERNAL_ASSERT(
      valid_pos >= 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found. ",
      tv->toString());

  return valid_pos;
}

IterDomain* getDoubleBufferAxis(const TensorView* tv) {
  return tv->axis((int)getDoubleBufferAxisPosition(tv));
}

void validateDoubleBufferedTensor(const TensorView* tv) {
  auto double_buffer_pos = getDoubleBufferAxisPosition(tv);

  // Like vectorization, only UnaryOp::Set with another TensorView is
  // considered.
  auto def = tv->definition();
  TORCH_INTERNAL_ASSERT(
      (def->isA<UnaryOp>() &&
       def->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Set) ||
          // Load store op should generally support double buffering.
          def->isA<LoadStoreOp>(),
      "Invalid tensor to double-buffer. Only tensor defined by UnaryOp::Set is supported: ",
      def->toString());

  TORCH_INTERNAL_ASSERT(
      def->input(0)->isA<TensorView>(),
      "Invalid tensor to double-buffer. Only tensor defined by UnaryOp::Set with TensorView is supported: ",
      def->toString());

  TORCH_INTERNAL_ASSERT(
      !tv->hasComputeWith(),
      "computeWith is not supported with double buffering: ",
      tv->toString());

  // Require the producer tensor to have been computed entirely for
  // the double-buffering loop. Otherwise, the producer itself would
  // also need to be double-bufferred.
  auto producer = def->input(0)->as<TensorView>();
  TORCH_INTERNAL_ASSERT(
      producer->getComputePosition(tv) <= double_buffer_pos,
      "Invalid tensor to double-buffer. The computeAt position of the producer tensor must be moved left: ",
      producer->toString());

  // Not strictly necessary, but only gmem -> smem or local and smem -> local
  // are allowed.
  const auto p_mem_type = producer->getMemoryType();
  const auto c_mem_type = tv->getMemoryType();
  TORCH_INTERNAL_ASSERT(
      (p_mem_type == MemoryType::Global &&
       (c_mem_type == MemoryType::Shared || c_mem_type == MemoryType::Local)) ||
          (c_mem_type == MemoryType::Local),
      "Invalid tensor to double-buffer: ",
      tv->toString(),
      ". Producer memory type: ",
      p_mem_type,
      ". Consumer memory type: ",
      c_mem_type);

  return;
}

namespace {

// Initial inspection of a fusion to find and validate double buffered tensors
class DoubleBufferFusionInspector : private IterVisitor {
 public:
  DoubleBufferFusionInspector(Fusion* fusion, DoubleBufferInfo& db_info)
      : db_info_(db_info) {
    traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void handle(TensorView* tv) final {
    if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        tv->definition(), "Fusion input shouldn't be double buffered.", tv);

    validateDoubleBufferedTensor(tv);

    auto db_axis = getDoubleBufferAxis(tv);

    db_info_.setDoubleBufferAxis(tv, db_axis);
  }

 private:
  DoubleBufferInfo& db_info_;
};

// The epilogue loop is only created when the producer of a double
// buffer tensor is on smem, in which case it would otherwise require
// an additional predicate to guard buffer overruns. When it's on
// gmem, that isn't the case, so it does not need to create an
// epilogue loop.
bool requireEpilogue(const std::vector<Expr*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const Expr* expr) {
    return expr->input(0)->as<TensorView>()->getMemoryType() ==
        MemoryType::Shared;
  });
}

bool isGmemIncrement(Expr* expr) {
  if (auto loop = dynamic_cast<kir::ForLoop*>(expr)) {
    if (loop->body().exprs().size() != 1) {
      return false;
    }
    return isGmemIncrement(loop->body().exprs()[0]);
  } else if (auto address_compute = dynamic_cast<kir::AddressCompute*>(expr)) {
    return address_compute->opType() ==
        kir::AddressCompute::AddressComputeOpType::GMEM_INCREMENT;
  }
  return false;
}

//! Hoists the gmem increment ops to the beginning of the loop
//!  within the scope of the given loop.
//! Note: [Gmem Increment Hoisting]
//!
//! This optimization is very useful when inplace increment
//!  is used on the global memory pointers.
//! Before this optimization, the code would look like:
//!
//!  for i in ... // main loop
//!    load.global ... [ptr]
//!    // Here we actually have an anti-dependency (WAR) on
//!    //  the register holding ptr and could result in
//!    //  non-ideal performance when we do not have enough
//!    //  instructions to put between the load and the increment.
//!    // depending on how many other instructions we have
//!    //   within this loop.
//!    ptr += increment_value
//!
//! After this transformation, the code looks like:
//!  ptr -=increment_value // a naive way to compensate
//!                        //  for the first iter.
//!  for i in ... // main loop
//!    ptr += increment_value
//!    // This is actually ok as integer instructions
//!    //   are usually much faster than memory.
//!    load.global ... [ptr]
//!
//! This function hoists the pointer increments, in the given
//!  loop, assuming that the decrements have been inserted
//!  on the CircularInitProlog stage.
kir::ForLoop* hoistGmemIncrement(kir::ForLoop* fl) {
  auto hoisted_loop = IrBuilder::create<kir::ForLoop>(fl);

  // insert all gmem increment exprs
  for (auto expr : fl->body().exprs()) {
    if (isGmemIncrement(expr)) {
      hoisted_loop->body().push_back(expr);
    }
  }

  // insert all non gmem increment exprs
  for (auto expr : fl->body().exprs()) {
    if (!isGmemIncrement(expr)) {
      hoisted_loop->body().push_back(expr);
    }
  }

  return hoisted_loop;
}

// Replicates double buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of double
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything.
class DoubleBufferLoopCloner : public kir::IrVisitor {
 public:
  static kir::ForLoop* clone(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type) {
    DoubleBufferLoopCloner cloner(
        double_buffer_loop, double_buffer_load_exprs, loop_type);
    cloner.clone();
    return cloner.cloned_top_level_loop_;
  }

 private:
  DoubleBufferLoopCloner(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs),
        loop_type_(loop_type) {}

  using kir::IrVisitor::handle;

  void clone() {
    const auto gpu_lower = GpuLower::current();

    // Cloning the double buffer loop as follows:
    //
    // Prologue: 0 to 1
    // Main: 0 to (extent-1)
    // Epilogue: (extent-1) to extent

    auto index = GpuLower::current()->caMap()->getIndexVariable(
        double_buffer_loop_->iter_domain(), loop_type_);
    auto start = double_buffer_loop_->start();
    auto stop = double_buffer_loop_->stop();
    auto stage_depth = gpu_lower->doubleBufferInfo().getStageDepthFor(
        double_buffer_loop_->iter_domain());

    if (loop_type_ == DoubleBufferLoopStage::Prolog) {
      TORCH_INTERNAL_ASSERT(start->isZeroInt());
      stop = SimplifyingIrBuilder::create<Int>(stage_depth - 1);
    } else if (
        loop_type_ == DoubleBufferLoopStage::Main &&
        requireEpilogue(double_buffer_load_exprs_)) {
      stop = IrBuilder::subExpr(
          double_buffer_loop_->stop(), gpu_lower->kernel()->oneVal());
    } else if (loop_type_ == DoubleBufferLoopStage::Epilog) {
      TORCH_INTERNAL_ASSERT(requireEpilogue(double_buffer_load_exprs_));
      start = IrBuilder::subExpr(
          double_buffer_loop_->stop(),
          SimplifyingIrBuilder::create<Int>(stage_depth - 1));
    } else if (loop_type_ == DoubleBufferLoopStage::CircularInitProlog) {
      // See [Predicate Peeling Interaction with Circular Buffering]
      TORCH_INTERNAL_ASSERT(start->isZeroInt());
      start = SimplifyingIrBuilder::create<Int>(stage_depth - 1);
      stop = SimplifyingIrBuilder::create<Int>(stage_depth);
    }

    cloned_top_level_loop_ = IrBuilder::create<kir::ForLoop>(
        double_buffer_loop_->iter_domain(),
        index,
        start,
        stop,
        gpu_lower->kernel()->oneVal(),
        false,
        nullptr,
        double_buffer_loop_->isUnrollRequired(),
        double_buffer_loop_->loopTransformInfo().doubleBufferStage(loop_type_));

    handle(double_buffer_loop_);

    // insert double buffer switching for the read offset:
    if (loop_type_ == DoubleBufferLoopStage::Main) {
      auto& db_info = GpuLower::current()->doubleBufferInfo();

      for (auto load : double_buffer_load_exprs_) {
        if (auto tv_out = ir_utils::getTvOutput(load)) {
          // calculate the switching size
          auto switch_size = db_info.getOriginalAllocSize(tv_out);
          auto switch_size_in_byte = SimplifyingIrBuilder::mulExpr(
              switch_size,
              SimplifyingIrBuilder::create<Int>(dataTypeSize(tv_out->dtype())));

          // insert db switch expressions:
          // Note:[Uniform Double Buffer Offset]
          // This modification is to encourage usage of uniform registers on
          // sm75+ when
          //  accessing shared memory double buffered tensors.
          // The code before transformation:
          //   for i in ... // double buffer loop
          //     ... = ld.shared [... + (i%5) * double_buffer_size]
          // The above code doesn't explictly specify that the double buffer
          // switch
          //  component is uniform. The following transformed code makes it
          //  explicit:
          //   for i in ... // double buffer loop
          //     ... = ld.shared [... + switch_index]
          //     doubleBufferSwitch(switch_index);
          //  So that the double buffer indices are all placed in uniform reg.

          auto maybe_read_index = db_info.getReadSwitchIndex(tv_out);
          if (maybe_read_index.has_value()) {
            // Instantiate and insert the update operator.
            auto address_compute =
                SimplifyingIrBuilder::create<kir::AddressCompute>(
                    tv_out,
                    maybe_read_index.value(),
                    switch_size_in_byte,
                    0, // assume this path only supports read
                       // so offset is 0
                    db_info.getStageDepthFor(
                        double_buffer_loop_->iter_domain()));

            cloned_top_level_loop_->body().push_back(address_compute);
          }
        }
      }
    }

    // Need to insert commits for multi-stage circular buffering
    //  on the prologs, but do not need to wait for them until
    //  the main loop.
    if (stage_depth > 2 && loop_type_ == DoubleBufferLoopStage::Prolog) {
      cloned_top_level_loop_->body().push_back(
          IrBuilder::create<kir::CpAsyncCommit>());
    }

    // Hoist the address increment in the double buffer main
    // loop, see also [Gmem Increment Hoisting]
    if (loop_type_ == DoubleBufferLoopStage::Main &&
        std::any_of(
            double_buffer_loop_->body().exprs().begin(),
            double_buffer_loop_->body().exprs().end(),
            isGmemIncrement) &&
        // FIXME:
        // Below is current condition that is required for gmem increment
        //  hoisting because the gmem decrement is currently placed in
        //  CircularInitProlog which requires predicate peeling to
        //  be generated.
        // To fix this should probably dedicate another double buffer
        //  loop stage, maybe GmemPointerDecrement, that is reserved
        //  for placing the gmem decrement before the main loop stage.
        GpuLower::current()->predicatePeelingInfo().shouldPeelLoop(
            double_buffer_loop_)) {
      cloned_top_level_loop_ = hoistGmemIncrement(cloned_top_level_loop_);
    }
  }

  void handle(kir::ForLoop* fl) final {
    kir::ForLoop* cloned_loop = fl == double_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<kir::ForLoop>(fl);

    cloned_scopes_.push_back(&cloned_loop->body());

    kir::IrVisitor::handle(fl);

    cloned_scopes_.pop_back();

    // Add the cloned loop into the parent loop body only when the
    // cloned loop contains expressions.
    if (!cloned_loop->body().empty() && !cloned_scopes_.empty()) {
      cloned_scopes_.back()->push_back(cloned_loop);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(false, "No IfThenElse should exist yet");
  }

  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::handle(expr);
      return;
    }

    TORCH_INTERNAL_ASSERT(!cloned_scopes_.empty());

    if (loop_type_ == DoubleBufferLoopStage::Main) {
      if (!canOmitInitInMainLoop(expr, double_buffer_loop_)) {
        cloned_scopes_.back()->push_back(expr);
      }
      return;
    }

    // In Prologue and Epilogue, either load expressions or anything
    // else are copied. Note that there can be multiple exprs defining
    // double buffered TVs (e.g., buffer initialization).

    auto out_tv = ir_utils::getTvOutput(expr);
    const auto is_double_buffer_load_expr = std::any_of(
        double_buffer_load_exprs_.begin(),
        double_buffer_load_exprs_.end(),
        [out_tv](const auto load_expr) {
          auto double_buffer_tv = ir_utils::getTvOutput(load_expr);
          TORCH_INTERNAL_ASSERT(double_buffer_tv != nullptr);
          return out_tv == double_buffer_tv;
        });

    if ((loop_type_ == DoubleBufferLoopStage::Prolog &&
         is_double_buffer_load_expr) ||
        (loop_type_ == DoubleBufferLoopStage::Epilog &&
         !is_double_buffer_load_expr)) {
      if (lower_utils::supportInlinePredicate(expr) &&
          expr->isA<LoadStoreOp>()) {
        auto ldst = expr->as<LoadStoreOp>();
        cloned_scopes_.back()->push_back(IrBuilder::create<LoadStoreOp>(
            ldst->opType(), ldst->out(), ldst->in()));
      } else {
        cloned_scopes_.back()->push_back(expr);
      }
    } else if (
        loop_type_ == DoubleBufferLoopStage::CircularInitProlog &&
        is_double_buffer_load_expr) {
      // Only need the init expressions in circular init prolog stage
      if (ir_utils::isTensorScalarFillOp(expr)) {
        cloned_scopes_.back()->push_back(expr);
      }
    }

    if (loop_type_ == DoubleBufferLoopStage::CircularInitProlog) {
      // Convert the address compute ops to decrement in the circular
      //  buffer init prolog, see [Gmem Increment Hoisting].
      if (auto address_compute = dynamic_cast<kir::AddressCompute*>(expr)) {
        if (address_compute->opType() ==
            kir::AddressCompute::AddressComputeOpType::GMEM_INCREMENT) {
          cloned_scopes_.back()->push_back(
              IrBuilder::create<kir::AddressCompute>(
                  address_compute->addressTv(),
                  address_compute->dataTv(),
                  address_compute->incrementValue(),
                  true /* is_decrement */));
        }
      }
    }

    // Include the double buffer update expressions in prologs too as
    //  prolog does write into the double buffered space.
    if (loop_type_ == DoubleBufferLoopStage::Prolog) {
      if (auto address_compute = dynamic_cast<kir::AddressCompute*>(expr)) {
        if (address_compute->opType() ==
            kir::AddressCompute::AddressComputeOpType::DOUBLE_BUFFER_UPDATE) {
          if (std::any_of(
                  double_buffer_load_exprs_.begin(),
                  double_buffer_load_exprs_.end(),
                  [address_compute](Expr* expr) {
                    return ir_utils::getTvOutput(expr)->sameAs(
                        address_compute->dataTv());
                  })) {
            cloned_scopes_.back()->push_back(expr);
          }
        }
      }
    }

    if (loop_type_ != DoubleBufferLoopStage::CircularInitProlog) {
      if (auto address_compute = dynamic_cast<kir::AddressCompute*>(expr)) {
        if (address_compute->opType() ==
            kir::AddressCompute::AddressComputeOpType::GMEM_INCREMENT) {
          cloned_scopes_.back()->push_back(expr);
        }
      }
    }
  }

  //! Returns true if the expression is an initialization expr that
  //!  can be omitted in main loop.
  //! See [Predicate Peeling Interaction with Circular Buffering]
  bool canOmitInitInMainLoop(Expr* expr, kir::ForLoop* double_buffer_loop) {
    // Check that this is an initialization for cp.async.
    if (!ir_utils::isCpAsyncInit(expr) ||
        !GpuLower::current()->predicatePeelingInfo().shouldPeelLoop(
            double_buffer_loop)) {
      return false;
    }

    auto out_tv = ir_utils::getTvOutput(expr);

    // Check that the double buffer loop is the main stage of
    //  the loop defining out_tv as there might be multiple
    //  loops that realize double buffers.
    bool db_loop_found = false;
    const auto& ca_map = GpuLower::current()->caMap();

    if (!(out_tv->isDoubleBuffered() || out_tv->isCircularBuffered()) ||
        !ca_map->areMapped(
            GpuLower::current()->doubleBufferInfo().getDoubleBufferAxis(out_tv),
            double_buffer_loop->iter_domain(),
            IdMappingMode::LOOP)) {
      return false;
    }

    // This optimization only applies when all the loops on the
    //  inner side of the double buffer main loop are either
    //  constant unrolled or parallel.
    // TODO:
    //  Buffer alias and broadcast resolution might still
    // break this. These are not showing in matmul kernels but
    // would need to build out support for general safty usage.
    for (auto id : out_tv->domain()->domain()) {
      if (db_loop_found) {
        auto loop_concrete_id =
            ca_map->getConcreteMappedID(id, IdMappingMode::LOOP);

        if (!loop_concrete_id->isParallelized() &&
            !loop_concrete_id->extent()->isConstInt()) {
          return false;
        }
      }

      db_loop_found = db_loop_found ||
          ca_map->areMapped(
              id, double_buffer_loop->iter_domain(), IdMappingMode::LOOP);
    }

    // Only when double buffer loop was found on out_tv could useful
    //  information have been inferred by this function.
    return db_loop_found;
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;
  const DoubleBufferLoopStage loop_type_;

  kir::ForLoop* cloned_top_level_loop_ = nullptr;
  std::deque<kir::Scope*> cloned_scopes_;
};

using InsertionInfo = std::unordered_map<kir::ForLoop*, std::vector<Expr*>>;

// Traverse lowered loop-nests and find all double buffer loops and
// associated load expressions.
class DoubleBufferLoopNestInspector : private kir::IrVisitor {
 public:
  static InsertionInfo run(const std::vector<Expr*>& exprs) {
    DoubleBufferLoopNestInspector inspector(exprs);
    return inspector.insertion_info_;
  }

 private:
  DoubleBufferLoopNestInspector(const std::vector<Expr*>& exprs) {
    handle(exprs);
  }

  using kir::IrVisitor::handle;

  // Collect double buffer related information on a expr
  //  that is a memory load, i.e. a LoadStore or a Set.
  void handlePossibleLoadExpr(Expr* expr) {
    const auto gpu_lower = GpuLower::current();

    auto out_tv = ir_utils::getTvOutput(expr);

    if (out_tv == nullptr) {
      return;
    }

    // Ignore init loop
    if (!(out_tv->isDoubleBuffered() || out_tv->isCircularBuffered()) ||
        !expr->input(0)->isA<TensorView>()) {
      return;
    }

    auto double_buffer_loop =
        gpu_lower->doubleBufferInfo().getDoubleBufferLoop(out_tv, for_loops_);

    TORCH_INTERNAL_ASSERT(
        double_buffer_loop != nullptr,
        "No double buffer loop found for a double buffered tensor: ",
        out_tv->toString());

    validateDoubleBufferLoop(double_buffer_loop);

    insertion_info_[double_buffer_loop].push_back(expr);
  }

  void handle(UnaryOp* uop) final {
    handlePossibleLoadExpr(uop);
  }

  void handle(LoadStoreOp* ldst) final {
    handlePossibleLoadExpr(ldst);
  }

  static void validateDoubleBufferLoop(kir::ForLoop* loop) {
    TORCH_INTERNAL_ASSERT(
        loop->start()->isZeroInt(), "Unsupported loop: ", loop->toString());
    TORCH_INTERNAL_ASSERT(
        loop->step()->isOneInt(), "Unsupported loop: ", loop->toString());
    TORCH_INTERNAL_ASSERT(
        !loop->vectorize(),
        "Vectorized loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
    TORCH_INTERNAL_ASSERT(
        !loop->vectorize_shift(),
        "Vectorize shift loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
  }

  InsertionInfo insertion_info_;
};

// Apply double buffering transformations
class DoubleBufferInserter : private kir::ExprMutator {
 public:
  // When there exist multiple double buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      InsertionInfo insertion_info) {
    auto inserted_exprs = exprs;
    while (!insertion_info.empty()) {
      DoubleBufferInserter inserter(inserted_exprs, insertion_info);
      inserted_exprs = inserter.exprs_;
    }
    return inserted_exprs;
  }

 private:
  DoubleBufferInserter(
      const std::vector<Expr*>& exprs,
      InsertionInfo& insertion_info)
      : insertion_info_(insertion_info) {
    auto num_double_buffer_loops = insertion_info.size();
    traverseAndInsert(exprs);
    TORCH_INTERNAL_ASSERT(processed_loop_ != nullptr);
    TORCH_INTERNAL_ASSERT(insertion_info.size() == num_double_buffer_loops - 1);
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* loop) final {
    kir::ExprMutator::handle(loop);

    // If another loop is already taken care of, no more loop should
    // be done in the same pass
    if (processed_loop_ != nullptr) {
      return;
    }

    auto it = insertion_info_.find(loop);
    if (it == insertion_info_.end()) {
      return;
    }

    insert(loop, it->second);
    processed_loop_ = loop;
    insertion_info_.erase(loop);
  }

  void insert(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& loads) {
    // Allocate read switching index if they need to be updated
    //  independently. see [Uniform Double Buffer Offset]
    for (auto load : loads) {
      if (auto load_output = dynamic_cast<TensorView*>(load->output(0))) {
        auto uses = load_output->fusion()->unordered_uses(load_output);
        if (load_output->getMemoryType() == MemoryType::Shared &&
            (load_output->isDoubleBuffered() ||
             load_output->isCircularBuffered()) &&
            load_output->shouldLiftReadAddress() &&
            // TODO: read switch index is only enabled for ldmatrix
            //  at the moment.
            // Would need to extend the ld.shared usage to directly
            //  take pointers to use this in other cases.
            std::all_of(uses.begin(), uses.end(), ir_utils::isLdMatrixOp)) {
          auto switch_val = IrBuilder::create<Int>();
          switch_val->to32b();

          // Record the read switch indexing variable so it can be
          //  used in the indexing pass.
          // TODO: maybe want to do this in id graph instead
          GpuLower::current()->doubleBufferInfo().setReadSwitchIndex(
              load_output, switch_val);

          // Place allocation for the switching variable before the
          //  double buffer loop.
          auto index_alloc = IrBuilder::create<kir::Allocate>(
              switch_val,
              MemoryType::Local,
              GpuLower::current()->kernel()->oneVal(),
              true);
          registerInsertBefore(double_buffer_loop, index_alloc);
        }
      }
    }

    auto prologue_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Prolog);
    registerInsertBefore(double_buffer_loop, prologue_loop);

    auto write_to_smem =
        std::any_of(loads.begin(), loads.end(), [](const Expr* expr) {
          return expr->output(0)->as<TensorView>()->getMemoryType() ==
              MemoryType::Shared;
        });

    // If the double buffer loop is to be peeled. Will need to insert
    //  a circular buffer init stage to initialize the final stage of
    //  circular buffer space.
    if (GpuLower::current()->predicatePeelingInfo().shouldPeelLoop(
            double_buffer_loop) &&
        write_to_smem) {
      auto circular_init_loop = DoubleBufferLoopCloner::clone(
          double_buffer_loop, loads, DoubleBufferLoopStage::CircularInitProlog);
      registerInsertBefore(double_buffer_loop, circular_init_loop);
    }

    // RAW sync is not inserted for double buffered tensors. The only
    // exception is the prologue load.
    bool insert_cpasync_wait = false;
    if (write_to_smem) {
      // Here the initial sync before entering double buffer loop is
      //  inserted.

      // If any of the double buffered tensor in this double buffer
      //  loop is async copy. We want to wait for the gmem loads to
      //  finish before synchronizing the block.
      if (std::any_of(loads.begin(), loads.end(), ir_utils::isCpAsyncOp)) {
        auto stage_depth =
            GpuLower::current()->doubleBufferInfo().getStageDepthFor(
                double_buffer_loop->iter_domain());
        auto cp_async_wait =
            IrBuilder::create<kir::CpAsyncWait>(stage_depth - 2);
        registerInsertBefore(double_buffer_loop, cp_async_wait);
        insert_cpasync_wait = true;
      }

      // Insert the initial block sync before entering main loop.
      if (std::any_of(loads.begin(), loads.end(), [](Expr* expr) {
            return GpuLower::current()
                ->syncMap()
                ->needsRawSync(ir_utils::getTvOutput(expr))
                .hasTID();
          })) {
        // If any of the double buffered loads require sync, as indicated
        //  by sync info map, insert the sync before entering the double buffer
        //  loop.
        // TODO:
        //  Currently not supporting double buffer in gmem, but short to mid
        //  term not yet a priority to go for this case.
        auto sync = IrBuilder::create<kir::BlockSync>(false);
        registerInsertBefore(double_buffer_loop, sync);
      }
    }

    auto main_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Main);

    registerReplace(double_buffer_loop, main_loop);

    // Insert the wait instruction in this pass instead
    //  of relying on WAR sync pass to do it.
    // The WAR sync pass today would insert the wait function
    //  exactly where we need it but the purpose of this wait
    //  insertion isn't exactly WAR protection.
    //
    // TODO: [Double Buffer Sync]
    //  We might eventually want to move the block sync inserted
    //   by WAR pass here as well since this sync insertion is kind
    //   of both WAR and RAW (or neither RAW nor WAR, depends
    //   on how we look at it).
    // Eg. in the case when a intermediate
    //   tensor is double buffered.
    //
    //  __block_sync();    // This is the initial sync
    //  For i in ...       // Double buffer loop
    //     A[i%2] = ...;
    //     ...  = A[1-i%2];
    //     __block_sync();  // sync within loop
    //     ...
    //  The "sync within loop" can be placed anywhere in the
    //   double buffer loop while in the case of RAW and WAR
    //   there'd be extra insertion point restrictions.
    //  We are currently not actively exploring opportunities
    //   with this property of "double buffer sync" so this
    //   is more conceptual at the moment, aka low priority.
    if (insert_cpasync_wait) {
      insertCpAsyncWaitInMainLoop(main_loop);
    }

    if (requireEpilogue(loads)) {
      auto epilogue_loop = DoubleBufferLoopCloner::clone(
          double_buffer_loop, loads, DoubleBufferLoopStage::Epilog);
      registerInsertAfter(double_buffer_loop, epilogue_loop);
    }
  }

  // Simple conservative rule for inserting async copy wait
  //  primitive in the double buffer loop:
  void insertCpAsyncWaitInMainLoop(kir::ForLoop* main_loop) {
    TORCH_INTERNAL_ASSERT(
        !main_loop->body().empty(),
        "Double buffer sync insertion: empty main loop.");
    // Note: This pass explicitly assumes that WAR sync has been
    //  inserted so would need to be updated if we re-order the
    //  passes. Cleanups suggested in [Double Buffer Sync]
    //  would resolve this dependency on pass ordering.
    auto end_of_loop_expr = main_loop->body().exprs().back();
    auto stage_depth = GpuLower::current()->doubleBufferInfo().getStageDepthFor(
        main_loop->iter_domain());
    auto cp_async_wait = IrBuilder::create<kir::CpAsyncWait>(stage_depth - 2);

    // Make sure the commit is inserted right before the
    //  cp.async.wait in circular buffering.
    bool need_insert_commit = stage_depth > 2;

    // Check if a sync has been inserted by WAR sync pass.
    auto block_sync_it = std::find_if(
        main_loop->body().exprs().rbegin(),
        main_loop->body().exprs().rend(),
        [](const Expr* expr) { return expr->isA<kir::BlockSync>(); });
    if (block_sync_it == main_loop->body().exprs().rend()) {
      // If there's no sync, i.e. no tensor needs cross
      //  thread communication. We still need a wait but
      //  it can just be anywhere in the loop. Chose to
      //  place at the end arbitrarily.
      main_loop->body().insert_after(end_of_loop_expr, cp_async_wait);
      if (need_insert_commit) {
        main_loop->body().insert_after(
            end_of_loop_expr, IrBuilder::create<kir::CpAsyncCommit>());
      }
    } else {
      // If a sync has been inserted, wait needs to be placed
      //  before the sync.
      main_loop->body().insert_before(*block_sync_it, cp_async_wait);
      if (need_insert_commit) {
        main_loop->body().insert_before(
            *block_sync_it, IrBuilder::create<kir::CpAsyncCommit>());
      }
    }
  }

 private:
  InsertionInfo& insertion_info_;
  kir::ForLoop* processed_loop_ = nullptr;
};

// Apply a loop transformation related to double buffering
//  that is particularly useful in matmul kernels.
// Note: [Skew Double Buffer Loop Transformation]
//
// This optimization is used particularly in a situation
//  where a producer-consumer pair are both double buffered.
// in e.g.
//   producer[Id0, Id1] (double buffer loop at Id0) = ...
//   consumer[Id0, Id1] (double buffer loop at Id1) = producer [Id0, Id1]
//
// * Note that the current double buffering check will only allow consumer
//   to have double buffer loop at strictly right of Id0.
//
// The generated code would look like:
//  ```
//  for i in 0..Id0.stage_depth-1: // Id0 prolog
//    for j in 0..Id1.size:
//      producer [i,...] = ...;
//
//  for i in 0..Id0.size: // Id0 main
//    for j in 0..Id1.size:
//      producer [i+1 % stage_depth,...] = ...;
//
//    // consumer could not have been circular buffered
//    //  as it's a consumer so it's not a cp.async output,
//    //  which is the only case we have so far (sm80) that
//    //  can benefit from circular buffering.
//
//    for j in 0..1: // Id1 prolog
//      consumer[j] = producer[i, j]
//
//    for j in 0..Id1.size-1: //Id1 main
//      consumer[j] = producer[i, j]
//      ... = consumer[j]
//
//    ... = consumer[Id1.size-1] // Id1 epilog
//  ```
//  The transformed code looks like:
//  ```
//  for i in 0..Id0.stage_depth-1: // Id0 prolog
//    for j in 0..Id1.size:
//      producer [i,...] = ...;
//
//  for i in 0..1: // first iteration of Id0 main
//    for j in 0..1: // Id1 **Upper Prolog**
//      consumer[j] = producer[i, j]
//
//  for i in 0..Id0.size: // Id0 main
//    for j in 0..Id1.size:
//      producer [i+1 % stage_depth,...] = ...;
//
//    // consumer could not have been circular buffered
//    //  as it's a consumer so it's not a cp.async output,
//    //  which is the only case we have so far (sm80) that
//    //  can benefit from circular buffering.
//
//    for j in 0..Id1.size-1: //Id1 main
//      consumer[j] = producer[i, j]
//      ... = consumer[j]
//
//    ... = consumer[Id1.size-1] // Id1 epilog
//
//    for j in 0..1: // Id1 **Lower Prolog**
//      consumer[j] = producer[i+1, j]
//  ```
// Essentially the prolog of Id1 is skewed ahead by 1 iteration of Id0.
//
// This allows the loop body of Id1 main to execute at the beginning
//  of Id0 main and thus enables optimal instruction interleaving. by
//  the cuda compiler.
class SkewDoubleBufferLoop : private kir::ExprMutator {
 public:
  // When there exist multiple double buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(const std::vector<Expr*>& exprs) {
    auto skewed_exprs = exprs;
    auto& double_buffer_info = GpuLower::current()->doubleBufferInfo();

    // keep track of the lifted loops.
    std::unordered_set<IterDomain*> lifted;

    // Each entry in `nestLiftingMap` corresponds to a pair of Id0,Id1
    //  described above, use a new instance of SkewDoubleBufferLoop to
    //  lift each one.
    for (auto& loop_nest_entry : double_buffer_info.nestLiftingMap()) {
      if (lifted.insert(loop_nest_entry.first).second) {
        SkewDoubleBufferLoop skew_loop(
            skewed_exprs, loop_nest_entry.first, loop_nest_entry.second);
        skewed_exprs = skew_loop.exprs_;
      }
    }
    return skewed_exprs;
  }

 private:
  SkewDoubleBufferLoop(
      const std::vector<Expr*>& exprs,
      IterDomain* concrete_double_buffer_loop_id,
      IterDomain* concrete_outer_main_loop_id)
      : concrete_double_buffer_loop_id_(concrete_double_buffer_loop_id),
        concrete_outer_main_loop_id_(concrete_outer_main_loop_id) {
    traverseAndInsert(exprs);
  }

  using kir::ExprMutator::handle;

  // Create the upper prolog and lower prolog of the given
  //  prolog loop, and insert them to the intended position
  //  as described above.
  void splitProlog(kir::ForLoop* loop) {
    // Create upper prolog
    auto upper_prolog = makeWrapedUpperProlog(loop);

    // Upper prolog needs to be lifted out of
    //  the outer main loop.
    registerInsertBefore(
        outer_main_loop_, upper_prolog, outer_main_loop_scope_);

    // Create lower prolog
    auto lower_prolog = makeLowerProlog(loop);

    // Lower prolog goes to the end of outer main loop
    TORCH_INTERNAL_ASSERT(!outer_main_loop_->body().empty());
    registerInsertAfter(
        outer_main_loop_->body().exprs().back(),
        lower_prolog,
        &outer_main_loop_->body());

    // Remove the original prolog
    registerRemove(loop);
  }

  // Clones the expressions and outer loop nest levels
  //  to ensure valid insertion of upper and lower prologs.
  // Given the original prolog loop to clone and
  //  an **empty** cloned_prolog kir::ForLoop with the meta
  //  data modified.
  // In particular, this function clones:
  //  1. The loop nest between Id0 and Id1 mentioned above.
  //  2. The expressions inside original prolog, possibled
  // with further loopnests within.
  kir::ForLoop* getClonedPrologLoopNest(
      kir::ForLoop* original_prolog,
      kir::ForLoop* cloned_prolog) {
    // Perform step 1:
    //  clone the loop nest all the way to the original prolog
    //  need to identify the loop nest between outer_main_loop (Id0)
    //  and original prolog (Id1).
    std::vector<kir::ForLoop*> loop_nest_to_clone;
    bool outer_main_loop_found = false;
    for (auto loop : for_loops_) {
      if (loop == original_prolog) {
        // Don't need to make copy beyond
        //  the prolog nest level.
        break;
      }
      if (outer_main_loop_found) {
        loop_nest_to_clone.push_back(loop);
      }
      outer_main_loop_found = outer_main_loop_found || loop == outer_main_loop_;
    }

    TORCH_INTERNAL_ASSERT(
        outer_main_loop_found, "cannot find outer main loop on the loop nest");

    kir::ForLoop *outer_loop = cloned_prolog, *inner_loop = cloned_prolog;

    // Clone the loopnest between outer_main_loop and original_prolog
    //  (Step1 above).
    if (!loop_nest_to_clone.empty()) {
      std::tie(outer_loop, inner_loop) = makeLoopNest(loop_nest_to_clone);
      inner_loop->body().push_back(cloned_prolog);
    }

    // Perform step 2: copy all the expressions from original prolog.
    // Put actual expressions inside original prolog
    //  into the upper prolog.
    for (auto expr : original_prolog->body().exprs()) {
      cloned_prolog->body().push_back(cloneMaybeLoopNest(expr));
    }

    return outer_loop;
  }

  // Makes the upper prolog loop nest to be inserted before the
  //  outer (Id0) main loop.
  kir::ForLoop* makeWrapedUpperProlog(kir::ForLoop* original_prolog) {
    // Peel iteration 0 of outer main loop.
    // So the upper prolog can be inserted at the same loop nest
    //  level as the outer main loop (Id0 main loop).
    auto cloned_main_loop = IrBuilder::create<kir::ForLoop>(
        outer_main_loop_->iter_domain(),
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->zeroVal(),
        GpuLower::current()->kernel()->oneVal(),
        GpuLower::current()->kernel()->oneVal(),
        false,
        nullptr,
        outer_main_loop_->isUnrollRequired(),
        kir::LoopTransformInfo());

    // Make the upper prolog loop object.
    auto upper_prolog_loop = IrBuilder::create<kir::ForLoop>(
        original_prolog->iter_domain(),
        original_prolog->index(),
        original_prolog->start(),
        original_prolog->stop(),
        original_prolog->step(),
        false,
        nullptr,
        original_prolog->isUnrollRequired(),
        original_prolog->loopTransformInfo().doubleBufferStage(
            DoubleBufferLoopStage::UpperProlog));

    // Complete the loop nest.
    auto outer_loop =
        getClonedPrologLoopNest(original_prolog, upper_prolog_loop);

    // Put the cloned loop nest into the cloned main loop
    //  and insert the main loop.
    cloned_main_loop->body().push_back(outer_loop);

    return cloned_main_loop;
  }

  // Makes the upper prolog loop nest to be inserted at the end of
  //  the outer_main_loop (Id0 loop) body.
  kir::ForLoop* makeLowerProlog(kir::ForLoop* original_prolog) {
    auto lower_prolog_loop = IrBuilder::create<kir::ForLoop>(
        original_prolog->iter_domain(),
        original_prolog->index(),
        original_prolog->start(),
        original_prolog->stop(),
        original_prolog->step(),
        false,
        nullptr,
        original_prolog->isUnrollRequired(),
        original_prolog->loopTransformInfo().doubleBufferStage(
            DoubleBufferLoopStage::LowerProlog));

    return getClonedPrologLoopNest(original_prolog, lower_prolog_loop);
  }

  void handle(kir::ForLoop* loop) final {
    // Check if this loop is a prolog we need to transform.
    bool is_lifted_prolog = GpuLower::current()->caMap()->areMapped(
                                loop->iter_domain(),
                                concrete_double_buffer_loop_id_,
                                IdMappingMode::LOOP) &&
        loop->doubleBufferLoopStage() == DoubleBufferLoopStage::Prolog &&
        within_outer_main_loop_;

    // Check if this loop is a main loop or not a prolog loop.
    bool is_main_loop =
        loop->doubleBufferLoopStage() == DoubleBufferLoopStage::NotApplicable ||
        loop->doubleBufferLoopStage() == DoubleBufferLoopStage::Main;

    // Check if the current loop is the outer main loop.
    bool is_outer_main_loop = is_main_loop &&
        GpuLower::current()->caMap()->areMapped(
            loop->iter_domain(),
            concrete_outer_main_loop_id_,
            IdMappingMode::LOOP);

    // Do the skew transform if this is an applicable case.
    if (is_lifted_prolog) {
      splitProlog(loop);
      return;
    }

    // Keep track of outer main loop info if it is detected.
    if (is_outer_main_loop) {
      within_outer_main_loop_ = true;
      outer_main_loop_ = loop;
      outer_main_loop_scope_ = scope_.empty() ? nullptr : scope_.back();
    }

    kir::ExprMutator::handle(loop);

    // Invalidate the within outer main loop flag once
    //  all the loop nest level within has been processed.
    if (is_outer_main_loop) {
      within_outer_main_loop_ = false;
    }
  }

  // Helper function to deep clone an expr if it is a loop nest.
  Expr* cloneMaybeLoopNest(Expr* expr) {
    auto loop = dynamic_cast<kir::ForLoop*>(expr);
    if (loop == nullptr) {
      return expr;
    }

    auto cloned_loop = IrBuilder::create<kir::ForLoop>(loop);
    for (auto expr : loop->body().exprs()) {
      cloned_loop->body().push_back(cloneMaybeLoopNest(expr));
    }
    return cloned_loop;
  }

  // Makes the given vector of for loops into a loop nest,
  // Returns <Outermost, Innermost> level as a pair.
  std::pair<kir::ForLoop*, kir::ForLoop*> makeLoopNest(
      std::vector<kir::ForLoop*> original_loop_nest) {
    TORCH_INTERNAL_ASSERT(
        !original_loop_nest.empty(), "cannot copy empty loop nest");
    kir::ForLoop *outermost = nullptr, *innermost = nullptr;
    for (auto loop : original_loop_nest) {
      auto cloned_loop = IrBuilder::create<kir::ForLoop>(loop);
      if (outermost == nullptr) {
        outermost = cloned_loop;
      }
      if (innermost != nullptr) {
        innermost->body().push_back(cloned_loop);
      }
      innermost = cloned_loop;
    }
    return std::make_pair(outermost, innermost);
  }

 private:
  // Running State:
  // Keeps track of the actual loop object representing the
  //  outer main loop.
  kir::ForLoop* outer_main_loop_ = nullptr;

  // Keeps track of the scope level of outer main loop.
  kir::Scope* outer_main_loop_scope_ = nullptr;

  // Keeps track of whether the pass is processing within
  //  the outer main loop level.
  bool within_outer_main_loop_ = false;

  // Interface parameters:
  // The loop concrete id of the prolog loop that this instance
  //  is skewing.
  IterDomain* concrete_double_buffer_loop_id_;

  // The loop concrete id of the outer main loop where the prolog
  //  loop to skew is assumed within.
  IterDomain* concrete_outer_main_loop_id_;
};

} // namespace

void DoubleBufferInfo::build(Fusion* fusion) {
  DoubleBufferFusionInspector inspector(fusion, *this);

  // Build double buffered loop id's
  for (auto& info : map_) {
    auto double_buffer_axis = info.second.double_buffer_axis;
    // Keeps track of which loop disjoint set has been
    //  double buffered. In index allocation, one index
    //  variable would need to be allocated in each
    //  double buffer stage.
    concrete_double_buffered_loop_id_.insert(
        GpuLower::current()->caMap()->getConcreteMappedID(
            double_buffer_axis, IdMappingMode::LOOP));
  }

  // Add a second pass to keep track of lifted
  //  double buffer loop nest see also [Skew Double Buffer Loop Transformation].
  for (auto& info : map_) {
    buildSkewInfo(info.first, info.second);
  }
}

void DoubleBufferInfo::buildSkewInfo(
    const TensorView* tv,
    const TvInfo& tv_info) {
  if (tv->shouldSkewDoubleBuffer()) {
    // Detect the outer main loop
    IterDomain* outer_loop_id = nullptr;
    bool double_buffer_axis_found = false;
    for (auto id_it = tv->domain()->domain().rbegin();
         id_it != tv->domain()->domain().rend();
         id_it++) {
      // The outer loop to lift prolog out of would
      //  be the first serial loop on the left of
      //  the double buffer loop
      if (double_buffer_axis_found &&
          (*id_it)->getParallelType() == ParallelType::Serial) {
        outer_loop_id = *id_it;
        break;
      }

      // Mark double buffer axis found
      if (GpuLower::current()->caMap()->areMapped(
              *id_it, tv_info.double_buffer_axis, IdMappingMode::LOOP)) {
        double_buffer_axis_found = true;
      }
    }

    TORCH_INTERNAL_ASSERT(
        outer_loop_id != nullptr,
        "cannot lift double buffered tensor ",
        tv->toString(),
        "double buffer loop ",
        tv_info.double_buffer_axis->toString());

    // Record the loop concrete id of both the outer main loop
    //  and the prolog loop to be skewed.
    auto concrete_outer_loop_id =
        GpuLower::current()->caMap()->getConcreteMappedID(
            outer_loop_id, IdMappingMode::LOOP);
    auto concrete_double_buffer_axis =
        GpuLower::current()->caMap()->getConcreteMappedID(
            tv_info.double_buffer_axis, IdMappingMode::LOOP);

    concrete_skewed_double_buffer_loop_map_.insert(
        std::make_pair(concrete_double_buffer_axis, concrete_outer_loop_id));
  }
}

bool DoubleBufferInfo::isLowerPrologWithin(
    IterDomain* double_buffer_id,
    IterDomain* outer_id) {
  auto concrete_double_buffer_id =
      GpuLower::current()->caMap()->getConcreteMappedID(
          double_buffer_id, IdMappingMode::LOOP);
  auto lift_id_it =
      concrete_skewed_double_buffer_loop_map_.find(concrete_double_buffer_id);
  if (lift_id_it == concrete_skewed_double_buffer_loop_map_.end()) {
    return false;
  }
  return GpuLower::current()->caMap()->areMapped(
      lift_id_it->second, outer_id, IdMappingMode::LOOP);
}

bool DoubleBufferInfo::isDoubleBufferedIterDomain(IterDomain* id) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return concrete_double_buffered_loop_id_.count(concrete_loop_id);
}

DoubleBufferInfo::TvInfo& DoubleBufferInfo::getTvInfo(const TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      tv->isDoubleBuffered() || tv->isCircularBuffered(),
      "Not a double-buffered tensor: ",
      tv->toString());
  return map_[tv];
}

void DoubleBufferInfo::setDoubleBufferAxis(
    const TensorView* tv,
    IterDomain* axis) {
  getTvInfo(tv).double_buffer_axis = axis;

  // Also validate the stage consistency with CA map.
  unsigned int stage_depth = 0;
  if (tv->isCircularBuffered()) {
    stage_depth = tv->circularBufferDepth();
  } else {
    // Double buffer is essentially
    //  circular buffer with depth 2.
    stage_depth = 2;
  }

  // Set and validate the new stage depth.
  setStageDepth(axis, stage_depth);
}

void DoubleBufferInfo::setStageDepth(IterDomain* id, unsigned int stage_depth) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);

  auto maybe_exisiting_depth_it = stage_depth_.find(concrete_loop_id);
  if (maybe_exisiting_depth_it == stage_depth_.end()) {
    stage_depth_[concrete_loop_id] = stage_depth;
  } else {
    TORCH_INTERNAL_ASSERT(
        stage_depth == maybe_exisiting_depth_it->second,
        "Unsupported multiple depth pipelining, was set to ",
        maybe_exisiting_depth_it->second,
        " by ",
        maybe_exisiting_depth_it->first->toString(),
        " and then set to ",
        stage_depth,
        " by ",
        concrete_loop_id->toString());
  }
}

IterDomain* DoubleBufferInfo::getDoubleBufferAxis(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).double_buffer_axis;
}

unsigned int DoubleBufferInfo::getStageDepthFor(
    IterDomain* double_buffer_axis) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      double_buffer_axis, IdMappingMode::LOOP);

  auto maybe_depth_it = stage_depth_.find(concrete_id);

  TORCH_INTERNAL_ASSERT(
      maybe_depth_it != stage_depth_.end(), "Stage depth not found");

  return maybe_depth_it->second;
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    IterDomain* axis,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return GpuLower::current()->caMap()->areMapped(
               loop->iter_domain(), axis, IdMappingMode::EXACT) &&
        (!ignore_prologue || !isProlog(loop->doubleBufferLoopStage()));
  });

  if (loop_it != loops.end()) {
    return *loop_it;
  } else {
    return nullptr;
  }
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto axis = getDoubleBufferAxis(tv);

  if (axis == nullptr) {
    return nullptr;
  }

  return getDoubleBufferLoop(axis, loops, ignore_prologue);
}

void DoubleBufferInfo::setOriginalAllocSize(
    const TensorView* tv,
    Val* original_alloc_size) {
  getTvInfo(tv).original_alloc_size = original_alloc_size;
}

Val* DoubleBufferInfo::getOriginalAllocSize(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).original_alloc_size;
}

std::vector<Expr*> DoubleBufferPass::run(const std::vector<Expr*>& exprs) {
  auto insertion_info = DoubleBufferLoopNestInspector::run(exprs);
  const auto double_buffer_inserted =
      DoubleBufferInserter::run(exprs, insertion_info);
  return SkewDoubleBufferLoop::run(double_buffer_inserted);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
