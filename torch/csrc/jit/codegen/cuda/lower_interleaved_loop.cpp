#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_interleaved_loop.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
};

//! Returns the next level unrolled loop that is within the given
//!  main loop on a tensorview. Returns a c10::nullopt if the unrolled
//!  loop cannot be found.
c10::optional<IterDomain*> getMaybeSubloop(
    TensorView* tv,
    IterDomain* main_loop) {
  bool main_loop_found = false;
  auto& ca_map = GpuLower::current()->caMap();

  for (auto leaf_id : tv->domain()->domain()) {
    if (main_loop_found && !leaf_id->isParallelized()) {
      return ca_map->getConcreteMappedID(leaf_id, IdMappingMode::LOOP);
    }
    main_loop_found = main_loop_found ||
        ca_map->areMapped(leaf_id, main_loop, IdMappingMode::LOOP);
  }

  return c10::nullopt;
}

} // namespace

void InterleaveLoopInfo::build(Fusion* fusion) {
  fusion_ = fusion;
  auto used_math_vals = fusion->usedMathVals();
  auto filtered_used_math_vals =
      ir_utils::filterByType<TensorView>(used_math_vals);

  // Cache used tvs for multiple visit.
  used_tvs_ = {filtered_used_math_vals.begin(), filtered_used_math_vals.end()};

  // Collect loop information from fusion
  collectInterleaveMainLoops();
  collectInterleavedSubLoops();

  // Validate interleaved expressions for data consistency
  validate();
}

void InterleaveLoopInfo::collectInterleaveMainLoops() {
  for (auto tv : used_tvs_) {
    auto maybe_main_axis = tv->getMaybeInterleavedAxis();
    if (maybe_main_axis.has_value()) {
      auto concrete_main_loop_id =
          GpuLower::current()->caMap()->getConcreteMappedID(
              tv->axis(maybe_main_axis.value()), IdMappingMode::LOOP);

      // Create new record for this loop id if not found
      if (!concrete_main_loop_to_interleaved_tv_.count(concrete_main_loop_id)) {
        concrete_main_loop_to_subloop_map_.insert(
            std::make_pair(concrete_main_loop_id, ConcreteIdVector()));
        concrete_main_loop_to_interleaved_tv_.insert(
            std::make_pair(concrete_main_loop_id, TensorViewVector()));
      }
    }
  }
}

bool InterleaveLoopInfo::isMainLoop(IterDomain* id) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return concrete_main_loop_to_interleaved_tv_.count(concrete_id);
}

bool InterleaveLoopInfo::isSubLoopof(
    IterDomain* id,
    IterDomain* concrete_main_id) {
  auto it = concrete_main_loop_to_subloop_map_.find(concrete_main_id);
  TORCH_INTERNAL_ASSERT(
      it != concrete_main_loop_to_subloop_map_.end(), "Invalid main loop");
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return it->second.has(concrete_id);
}

void InterleaveLoopInfo::insertEntry(
    TensorView* tv,
    IterDomain* main_loop,
    IterDomain* sub_loop) {
  auto concrete_main_loop = GpuLower::current()->caMap()->getConcreteMappedID(
      main_loop, IdMappingMode::LOOP);
  auto concrete_sub_loop = GpuLower::current()->caMap()->getConcreteMappedID(
      sub_loop, IdMappingMode::LOOP);

  // Insert sub loops from this tv
  auto main_loop_entry_it =
      concrete_main_loop_to_subloop_map_.find(concrete_main_loop);
  TORCH_INTERNAL_ASSERT(
      main_loop_entry_it != concrete_main_loop_to_subloop_map_.end(),
      "unknown main loop ",
      main_loop->toString());
  main_loop_entry_it->second.pushBack(concrete_sub_loop);

  // Insert interleaved tvs.
  auto tv_entry_it =
      concrete_main_loop_to_interleaved_tv_.find(concrete_main_loop);
  TORCH_INTERNAL_ASSERT(
      tv_entry_it != concrete_main_loop_to_interleaved_tv_.end(),
      "unknown main loop");
  tv_entry_it->second.pushBack(tv);
}

void InterleaveLoopInfo::collectInterleavedSubLoops() {
  for (auto tv : used_tvs_) {
    IterDomain* main_loop = nullptr;
    for (auto leaf_id : tv->domain()->domain()) {
      if (isMainLoop(leaf_id)) {
        TORCH_INTERNAL_ASSERT(
            main_loop == nullptr,
            tv,
            "has nested main loop ",
            main_loop->toString(),
            " and ",
            leaf_id->toString(),
            " which is not yet supported");

        auto maybe_subloop = getMaybeSubloop(tv, leaf_id);
        TORCH_INTERNAL_ASSERT(
            maybe_subloop.has_value(),
            tv->toString(),
            " cannot be interleaved within ",
            leaf_id);
        insertEntry(tv, leaf_id, maybe_subloop.value());
        main_loop = leaf_id;
      }
    }
  }
}

// Validation of double buffering topology of interleaved expressions:
void InterleaveLoopInfo::validate() {
  // Validate expression consistency after interleaving
  for (auto& main_loop_entry : concrete_main_loop_to_interleaved_tv_) {
    validateMainLoop(main_loop_entry.first, main_loop_entry.second);
  }
}

bool InterleaveLoopInfo::isExitTv(
    TensorView* tv,
    const TensorViewVector& interleaved_tvs) {
  // Check for exit tv's:
  //  TODO: This is a simplified version of the analysis
  //   where all the interleaved tv has have an unrolled
  //   serial loop on the right of a CA axis, so when a
  //   tv is an "exit-tv", we only need to check that
  //   non of the consumers of this tv is an interleaved tv.

  // Output is always an exit
  if (tv->isFusionOutput()) {
    return true;
  }

  for (auto use : fusion_->unordered_uses(tv)) {
    // Check if any immediate consumer of tv is interleaved.
    for (auto consumer_tv :
         ir_utils::filterByType<TensorView>(use->outputs())) {
      if (interleaved_tvs.has(consumer_tv)) {
        return false;
      }
    }
  }

  // No immediate consumer of tv is interleaved so the tv is an exit tv.
  return true;
}

void InterleaveLoopInfo::validateMainLoop(
    IterDomain* concrete_main_loop,
    const TensorViewVector& interleaved_tvs) {
  // All the expressions that are inside the main loop or subloop can
  //  only be 2 cases:
  // 1. It's double buffered.
  // 2. It's inlined into the subloop.
  // 3. It's not a producer of any other interleaved tv's, i.e. it is an exit
  // tv.
  for (auto tv : interleaved_tvs.vector()) {
    if (isExitTv(tv, interleaved_tvs)) {
      continue;
    }
    // Double buffered tv doesn't need to be checked.
    if (tv->isDoubleBuffered() || tv->isCircularBuffered()) {
      continue;
    }

    // Check that the subloop is on the left of CA axis:
    auto& concrete_subloops =
        concrete_main_loop_to_subloop_map_.at(concrete_main_loop);
    bool subloop_found = false;
    for (auto id_it = tv->domain()->domain().begin();
         id_it != tv->domain()->domain().begin() + tv->getComputeAtPosition();
         id_it++) {
      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          *id_it, IdMappingMode::LOOP);
      if (concrete_subloops.has(concrete_id)) {
        subloop_found = true;
        break;
      }
    }
    TORCH_INTERNAL_ASSERT(
        subloop_found,
        "unsupported interleaved tv ",
        tv->toString(),
        " it needs to be either double buffered, or an exit of interleaved region or inlined beyond subloops");
  }
}

namespace {
struct InterLeaveConfig {
  int64_t number_of_units = 1;
  std::unordered_map<IterDomain*, int64_t> concrete_id_to_extent_;
};

class LoopInterLeaver : kir::ExprMutator {
 public:
  static std::vector<Expr*> run(std::vector<Expr*> exprs) {
    // Interleave main loops one at a time.
    for (auto& it : GpuLower::current()
                        ->interleavedLoopInfo()
                        .concreteMainLoopToSubloopMap()) {
      LoopInterLeaver interleaver;
      interleaver.concrete_main_loop_ = it.first;

      interleaver.concrete_subloop_set_ = std::unordered_set<IterDomain*>(
          it.second.vector().begin(), it.second.vector().end());
      interleaver.traverseAndInsert(exprs);
      exprs = interleaver.exprs_;
    }
    return exprs;
  }

 private:
  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* fl) final {
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        fl->iter_domain(), IdMappingMode::LOOP);

    // Only interleave main loop is necessary
    if (concrete_main_loop_ == concrete_loop_id &&
        fl->doubleBufferLoopStage() == DoubleBufferLoopStage::Main) {
      handleMainLoop(fl);
    } else {
      kir::ExprMutator::handle(fl);
    }
  }

  bool isInterleavedSubloop(Expr* expr) {
    if (auto loop = dynamic_cast<kir::ForLoop*>(expr)) {
      auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
          loop->iter_domain(), IdMappingMode::LOOP);
      if (concrete_subloop_set_.count(concrete_loop_id) &&
          loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Epilog) {
        return true;
      }
    }
    return false;
  }

  void clearSubLoops(
      std::vector<kir::ForLoop*>& interleaved_subloops,
      kir::ForLoop* main_loop) {
    for (auto fl : interleaved_subloops) {
      registerRemove(fl, &main_loop->body());
    }
    interleaved_subloops.clear();
  }

  void handleMainLoop(kir::ForLoop* fl) {
    std::vector<kir::ForLoop*> interleaved_subloops;
    Expr* last_expr = nullptr;

    for (auto expr : fl->body().exprs()) {
      last_expr = expr;
      if (auto loop = dynamic_cast<kir::ForLoop*>(expr)) {
        if (loop->doubleBufferLoopStage() == DoubleBufferLoopStage::Prolog ||
            loop->doubleBufferLoopStage() ==
                DoubleBufferLoopStage::UpperProlog ||
            loop->doubleBufferLoopStage() ==
                DoubleBufferLoopStage::LowerProlog) {
          continue;
        } else if (
            isInterleavedSubloop(expr) &&
            loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Epilog) {
          interleaved_subloops.push_back(expr->as<kir::ForLoop>());
          continue;
        }
      }

      // TODO: generic detect
      if (expr->isA<kir::Allocate>()) {
        continue;
      }

      if (!interleaved_subloops.empty()) {
        realizeInterleavedSubloops(last_expr, interleaved_subloops, true, fl);
        clearSubLoops(interleaved_subloops, fl);
      }
    }

    // It's possible, actually common that all exprs within
    //  the main loop are subloops, so we will need to run
    //  another realization step after visiting the whole main
    //  loop.
    if (!interleaved_subloops.empty()) {
      realizeInterleavedSubloops(last_expr, interleaved_subloops, false, fl);
      clearSubLoops(interleaved_subloops, fl);
    }
  }

  // TODO: use common infra
  Expr* cloneMaybeLoopNest(Expr* expr) {
    auto fl = dynamic_cast<kir::ForLoop*>(expr);
    if (!fl) {
      return expr;
    }

    TORCH_INTERNAL_ASSERT(!expr->isA<kir::IfThenElse>(), "unsupported");
    auto cloned_fl = IrBuilder::create<kir::ForLoop>(fl);

    for (auto loop_expr : fl->body().exprs()) {
      cloned_fl->body().push_back(cloneMaybeLoopNest(loop_expr));
    }

    return cloned_fl;
  }

  void handle(kir::IfThenElse*) final {
    TORCH_INTERNAL_ASSERT(
        false, "LoopInterleaving: no support yet post IfThenElse lowering");
  }

  void realizeInterleavedSubloops(
      Expr* insert_point,
      std::vector<kir::ForLoop*> sub_loops,
      bool insert_before,
      kir::ForLoop* main_loop) {
    std::vector<kir::ForLoop*> interleave_units;
    auto config = getInterleaveConfig(sub_loops);

    for (int idx : c10::irange(config.number_of_units)) {
      for (auto sub_loop : sub_loops) {
        auto concrete_loop_id =
            GpuLower::current()->caMap()->getConcreteMappedID(
                sub_loop->iter_domain(), IdMappingMode::LOOP);

        auto concrete_extent =
            config.concrete_id_to_extent_.at(concrete_loop_id);

        auto interleave_unit = ceilDiv(concrete_extent, config.number_of_units);

        int start_idx = idx * interleave_unit;

        auto stop_idx = std::min(start_idx + interleave_unit, concrete_extent);

        // No longer need to generate more of this sub loop if
        //  start is already out of bound.
        if (start_idx < concrete_extent) {
          auto start_val = SimplifyingIrBuilder::create<Int>(start_idx);
          auto stop_val = SimplifyingIrBuilder::create<Int>(stop_idx);
          interleave_units.push_back(
              makeInterleavedUnit(sub_loop, start_val, stop_val));
        }
      }
    }

    if (insert_before) {
      for (auto unit : interleave_units) {
        registerInsertBefore(insert_point, unit, &main_loop->body());
      }
    } else {
      // Need to insert in reverse order when inserting after.
      for (auto it = interleave_units.rbegin(); it != interleave_units.rend();
           it++) {
        registerInsertAfter(insert_point, *it, &main_loop->body());
      }
    }
  }

  kir::ForLoop* makeInterleavedUnit(kir::ForLoop* fl, Val* start, Val* stop) {
    // Create an outer loop with the same loop expressions but
    //  different start and stop.
    auto outer_loop = IrBuilder::create<kir::ForLoop>(
        fl->iter_domain(),
        fl->index(),
        start,
        stop,
        fl->step(),
        fl->vectorize(),
        fl->vectorize_shift(),
        fl->isUnrolled(),
        fl->loopTransformInfo().interLeaveUnit());

    for (auto expr : fl->body().exprs()) {
      outer_loop->body().push_back(cloneMaybeLoopNest(expr));
    }

    return outer_loop;
  }

  InterLeaveConfig getInterleaveConfig(
      const std::vector<kir::ForLoop*> sub_loops_) {
    TORCH_INTERNAL_ASSERT(
        !sub_loops_.empty(), "Cannot generate config for empty subloops");
    InterLeaveConfig interleave_config;
    ExpressionEvaluator const_evaluator(sub_loops_[0]->iter_domain()->fusion());

    int64_t max_extent = -1;
    int64_t min_extent = -1;

    for (auto fl : sub_loops_) {
      auto maybe_value = const_evaluator.evaluate(fl->stop());
      TORCH_INTERNAL_ASSERT(
          maybe_value.has_value(), "non constant interleaving not supported");
      auto value = maybe_value.value().as<int64_t>();

      // Collect max or min value while converting concrete extents
      //  from the subloop iterdomains.
      max_extent = max_extent == -1 ? value : std::max(max_extent, value);
      min_extent = min_extent == -1 ? value : std::min(min_extent, value);

      // TODO: check if this concretize step is necessary, all the iterdomain
      // attached
      //  to a for loop should be the concrete id.
      auto concrete_loop_domain =
          GpuLower::current()->caMap()->getConcreteMappedID(
              fl->iter_domain(), IdMappingMode::LOOP);

      // Collect concrete extents of each of the subloops.
      interleave_config.concrete_id_to_extent_[concrete_loop_domain] = value;
    }

    // Calculate interleave factor, simple heuristic as ceilDiv(max, min):
    interleave_config.number_of_units = min_extent;

    return interleave_config;
  }

 private:
  // Marks the current main loop this pass
  //  is processing.
  IterDomain* concrete_main_loop_ = nullptr;

  // Set of subloop concrete IterDomains that will
  //  be interleaved within main loop.
  std::unordered_set<IterDomain*> concrete_subloop_set_;
};
} // namespace

std::vector<Expr*> interLeaveDoubleBufferUnrolledLoops(
    const std::vector<Expr*>& exprs) {
  return LoopInterLeaver::run(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
