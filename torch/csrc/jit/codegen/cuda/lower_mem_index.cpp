#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_mem_index.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

AddressRecord::AddressRecord(
    TensorView* data_tv,
    TensorView* address_tv,
    std::vector<IterDomain*> allocation_ids,
    TensorView* reference_tv,
    ReadWrite direction,
    IterDomain* loop_concrete_serial_id)
    : address_tv_(address_tv_),
      data_tv_(data_tv),
      reference_tv_(reference_tv),
      allocation_ids_(allocation_ids),
      access_direction_(direction),
      loop_concrete_serial_id_(loop_concrete_serial_id) {}

AddressRecordKey AddressRecord::key() const {
  return AddressRecordKey(reference_tv_, data_tv_);
}

void AddressComputeInfo::build(Fusion* fusion) {
  for (auto expr : fusion->unordered_exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
      ;
    }
    for (auto consumer_tv :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (consumer_tv->shouldLiftWriteAddress()) {
        // Build write address record if consumer index should
        //  be lifted.
        makeAddressRecord(consumer_tv, consumer_tv);
      }
      for (auto producer_tv :
           ir_utils::filterByType<TensorView>(expr->inputs())) {
        if (producer_tv->shouldLiftReadAddress()) {
          // Build read address record if producer index should
          //  be lifted.
          makeAddressRecord(producer_tv, consumer_tv);
        }
      }
    }
  }
}

namespace {

std::pair<std::vector<IterDomain*>, std::unordered_set<IterDomain*>>
getContigMergeFrontier(
    std::vector<IterDomain*> domains,
    std::vector<bool> contiguity,
    const std::unordered_map<IterDomain*, Expr*> id_to_use_map) {
  std::list<IterDomain*> domain_list{domains.begin(), domains.end()};
  std::unordered_set<IterDomain*> contiguity_id_set;

  TORCH_INTERNAL_ASSERT(domains.size() == contiguity.size());

  for (auto idx : c10::irange(domains.size())) {
    if (contiguity[idx]) {
      contiguity_id_set.insert(domains[idx]);
    }
  }

  // Track if any update has been found,
  //  keep iterating until no more update is found.
  bool update = true;

  while (update) {
    update = false;
    auto domain_it = domain_list.begin();

    while (domain_it != domain_list.end()) {
      auto first_domain = *domain_it;
      auto second_domain_it = std::next(domain_it);
      if (second_domain_it == domain_list.end()) {
        // No next domain to merge with.
        //  no need to keep moving forward
        //  in this round.
        break;
      }
      auto second_domain = *second_domain_it;

      if (!contiguity_id_set.count(first_domain) ||
          !contiguity_id_set.count(second_domain)) {
        // Only contiguous pairs of iterdomains are considered.
        continue;
      }

      auto first_domain_use_it = id_to_use_map.find(first_domain);
      TORCH_INTERNAL_ASSERT(first_domain_use_it != id_to_use_map.end());

      auto merge = dynamic_cast<Merge*>(first_domain_use_it->second);

      // See if this is contiguous merge
      if (merge && merge->inner() == second_domain) {
        update = true;
        auto merged_id = merge->out();
        // Insert the merged id and contiguity
        domain_list.insert(domain_it, merged_id);
        contiguity_id_set.erase(first_domain);
        contiguity_id_set.erase(second_domain);
        contiguity_id_set.insert(merged_id);

        // Erase the merged id.
        domain_list.erase(domain_it, std::next(second_domain_it));

        // Start over for simplicity, actually
        //  just decrement the domain it by one would be enough.
        break;
      }
    }
  }

  std::vector<IterDomain*> result_domain{
      domain_list.begin(), domain_list.end()};
  return std::make_pair(result_domain, contiguity_id_set);
}

std::unordered_map<IterDomain*, Expr*> getIdToUseMap(TensorDomain* td) {
  std::vector<Val*> all_id_vals{td->domain().begin(), td->domain().end()};
  auto all_inputs = InputsOf::outputs(td->fusion(), all_id_vals);
  std::unordered_map<IterDomain*, Expr*> id_to_use_map;

  auto all_exprs = DependencyCheck::getAllExprsBetween(
      {all_inputs.begin(), all_inputs.end()}, all_id_vals);

  for (auto expr : all_exprs) {
    for (auto id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      id_to_use_map[id] = expr;
    }
  }

  return id_to_use_map;
}

// TODO: unify with contiguity pass
std::unordered_set<IterDomain*> getInitialContiguousRootIdsOnReferenceTv(
    TensorView* data_tv,
    TensorView* reference_tv) {
  std::unordered_set<IterDomain*> contig_ids;

  std::pair<std::vector<IterDomain*>, std::unordered_set<IterDomain*>>
      contig_merge_front;

  // Compute the merge frontier for contiguity info
  if (data_tv == reference_tv) {
    contig_merge_front = getContigMergeFrontier(
        reference_tv->getRootDomain(),
        reference_tv->domain()->contiguity(),
        getIdToUseMap(reference_tv->domain()));
  } else {
    std::unordered_set<IterDomain*> data_contig_domains;
    for (auto root_idx : c10::irange(data_tv->getRootDomain().size())) {
      if (data_tv->domain()->contiguity()[root_idx]) {
        data_contig_domains.insert(data_tv->getRootDomain()[root_idx]);
      }
    }

    auto c2p_root_map =
        PairwiseRootDomainMap(data_tv, reference_tv)
            .mapConsumerToProducer(reference_tv->domain(), data_tv->domain());

    auto root_size = reference_tv->getRootDomain().size();
    std::vector<bool> contiguity_vec(root_size, false);

    for (auto root_idx = root_size - 1; root_idx >= 0; root_idx--) {
      auto root_id = reference_tv->getRootDomain()[root_idx];
      auto data_id_it = c2p_root_map.find(root_id);

      if (data_id_it == c2p_root_map.end()) {
        // Skip the rest if id did not map across to the producer side.
        break;
      }

      if (data_contig_domains.count(data_id_it->second)) {
        contiguity_vec[root_idx] = true;
      } else {
        // Skip the rest if we find any domain
        //  that isn't contiguous from the inner most.
        break;
      }
    }

    contig_merge_front = getContigMergeFrontier(
        reference_tv->getRootDomain(),
        contiguity_vec,
        getIdToUseMap(reference_tv->domain()));
  }

  // Only record the contiguous ids from the innermost
  //  for the current optimization.
  std::unordered_set<IterDomain*> contig_ids;
  for (int root_idx = contig_merge_front.first.size() - 1; root_idx >= 0;
       root_idx--) {
    auto root_id = contig_merge_front.first[root_idx];
    if (contig_merge_front.second.count(root_id)) {
      contig_ids.insert(root_id);
    } else {
      break;
    }
  }

  return contig_ids;
}

bool isSeparable(
    TensorView* tv,
    IterDomain* id,
    const std::unordered_set<IterDomain*>& contig_merged_ids) {
  auto id_def = id->definition();
  while (id_def != nullptr) {
    if (auto split = dynamic_cast<Split*>(id_def)) {
      // Traverse backward towards the root domains.
      id = split->in();
      id_def = id->definition();

      // Already hitting contiguous root so stop
      //  traversing up.
      if (contig_merged_ids.count(id)) {
        return true;
      }

      // Check for non-divisible split:
      auto id_extent = id->extent()->getInt();
      auto factor_extent = split->factor()->getInt();

      if (!id_extent.has_value() || !factor_extent.has_value()) {
        // Non constant sized tiling cannot be lifted.
        return false;
      }

      if (id_extent.value() % factor_extent.value() != 0) {
        // Non-divisible split cannot be separated.
        return false;
      }
    }

    // TODO: check swizzle conditions when merged in.
    return false;
  }

  // This covers the base case where we start
  //  with a contigous merged output of root.
  return contig_merged_ids.count(id);
}

void validateSeparability(
    TensorView* reference_tv,
    const std::vector<IterDomain*>& alloc_ids,
    const std::unordered_set<IterDomain*>& contig_merged_ids) {
  std::unordered_set<IterDomain*> alloc_id_set{
      alloc_ids.begin(), alloc_ids.end()};

  for (auto id : reference_tv->domain()->domain()) {
    if (!alloc_id_set.count(id) && !id->isParallelized()) {
      TORCH_INTERNAL_ASSERT(
          isSeparable(reference_tv, id, contig_merged_ids),
          "Unsupported lift address since",
          id->toString(),
          " cannot be separated");
    }
  }
}

} // namespace

void AddressComputeInfo::makeAddressRecord(
    TensorView* data_tv,
    TensorView* reference_tv) {
  // Signify if this is caching shared mem access,
  //  assume it'd be global access if other wise.
  bool is_shared_mem_access = data_tv->getMemoryType() == MemoryType::Shared;

  auto& ca_map = GpuLower::current()->caMap();

  // Get allocation ids for the lifted index.
  std::deque<IterDomain*> alloc_ids;

  auto contig_merged_ids =
      getInitialContiguousRootIdsOnReferenceTv(data_tv, reference_tv);

  // Mark the boundary iterdomain at the compute at
  //  position, where iterdomains on the left would
  //  be in shared loop.
  auto ref_ca_id = reference_tv->axis(reference_tv->getComputeAtPosition());

  // Serial domain which the cached indexing will
  //  be lifted out of.
  // TODO: some extra work to support
  //  global scope.
  IterDomain* serial_id = nullptr;

  // compute id's to cache index on:
  for (auto ref_id_it = reference_tv->domain()->domain().rbegin();
       ref_id_it != reference_tv->domain()->domain().rend();
       ref_id_it++) {
    auto ref_id = *ref_id_it;

    if (ref_id == ref_ca_id) {
      // TODO:
      // Temporarily no support for lifting
      //  beyond the computeAt axis. The axes
      //  mapping tends to be more complex and
      //  would need extra information to resolve.
      break;
    }

    // Skip parallel dims as they do not
    //  contribute to register lifetime.
    if (ref_id->isParallelized()) {
      continue;
    }

    // Current not attempting to lift outside of
    //  non-constant sized serial loops, due to
    //  limited profitability. Might want to
    //  revisit when we see useful cases.
    if (!ref_id->extent()->isConstScalar()) {
      break;
    }

    if (is_shared_mem_access &&
        isSeparable(reference_tv, ref_id, contig_merged_ids)) {
      // This is an optimization step that will be built out.
      // It is related to the GPU memory indexing modes,
      //  and generally we should feel free to lift shared
      //  memory indexing variables as much as possible but
      //  would generally want to cache global memory index
      //  more.
      // TODO: there are exceptions to this rule of thumb
      //  and they could matter in extremely register limited
      //  scenarios. Will be built out in follow ups.
      continue;
    }

    // We are visiting the iterdomains from reference
    //  tv in reverse order but we want the outputs
    //  to be in original order.
    alloc_ids.push_front(ref_id);
  }

  std::vector<IterDomain*> alloc_ids_vec{alloc_ids.begin(), alloc_ids.end()};

  validateSeparability(data_tv, alloc_ids_vec, contig_merged_ids);

  // Create address record:
  auto address_tv = makeAddressTv(alloc_ids_vec, !is_shared_mem_access);

  // Assuming we are only having two scenarios,
  //  either accessing a consumer in the consumer's loop,
  //  or accessing the producer in producer's loop.
  auto access_direction = reference_tv == data_tv
      ? AddressRecord::ReadWrite::WRITE
      : AddressRecord::ReadWrite::READ;

  TORCH_INTERNAL_ASSERT(
      serial_id != nullptr, "no support yet for global scope hoisting");

  auto concrete_serial_id =
      ca_map->getConcreteMappedID(serial_id, IdMappingMode::LOOP);

  // Create and insert lifting record
  auto new_record_ptr = std::make_unique<AddressRecord>(
      data_tv,
      address_tv,
      alloc_ids_vec,
      reference_tv,
      access_direction,
      concrete_serial_id);

  address_tv_to_address_record_[address_tv] = new_record_ptr.get();

  // Add address lift record
  index_lift_record_.insert({new_record_ptr->key(), std::move(new_record_ptr)});
}

c10::optional<AddressRecord*> AddressComputeInfo::getMaybeLiftedAddress(
    TensorView* data_tv,
    TensorView* reference_tv) {
  // Use data tv as the reference if
  //  reference is missing.
  if (reference_tv == nullptr) {
    reference_tv = data_tv;
  }

  auto address_record_it =
      index_lift_record_.find(AddressRecordKey(reference_tv, data_tv));

  if (address_record_it == index_lift_record_.end()) {
    return c10::nullopt;
  }

  return address_record_it->second.get();
}

TensorView* AddressComputeInfo::makeAddressTv(
    std::vector<IterDomain*> address_domains,
    bool is_global_address) {
  DataType dtype = is_global_address ? DataType::Int : DataType::Int32;
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          address_domains, std::vector<bool>(address_domains.size(), true)),
      dtype);
}

c10::optional<AddressRecord*> AddressComputeInfo::getMaybeRecordForAddressTv(
    TensorView* tv) {
  auto record_it = address_tv_to_address_record_.find(tv);
  if (record_it == address_tv_to_address_record_.end()) {
    return c10::nullopt;
  }
  return record_it->second;
}

namespace {

struct AddressComputeInsertionInfo {
  std::vector<kir::ForLoop*> loop_nest;
  AddressRecord* address_compute_record;
};

class MemoryAddressComputeInserter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    MemoryAddressComputeInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  explicit MemoryAddressComputeInserter(const std::vector<Expr*>& exprs) {
    traverseAndInsert(exprs);
  }

  void handle(kir::ForLoop* loop) final {
    // Descend into the inner loop nest first
    //  to register lifting inner loops.
    kir::ExprMutator::handle(loop);
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        loop->iter_domain(), IdMappingMode::LOOP);

    // Currently assumes that this pass is run
    //  almost right after the loopnest generation,
    //  i.e. not any loop fission should have happened
    //  before.
    auto insertion_info_vec_it =
        pending_address_compute_insertions_.find(concrete_loop_id);
    if (insertion_info_vec_it != pending_address_compute_insertions_.end()) {
      insertAddressComputes(loop, *(insertion_info_vec_it->second));
      pending_address_compute_insertions_.erase(concrete_loop_id);
    }
  }

  void insertAddressComputes(
      kir::ForLoop* loop,
      const std::vector<AddressComputeInsertionInfo>& insertion_infos) {
    for (auto& insertion_info : insertion_infos) {
      // allocate address tensor:
      auto alloc = IrBuilder::create<kir::Allocate>(
          insertion_info.address_compute_record->addressTensor(),
          MemoryType::Local,
          insertion_info.address_compute_record->allocationIterDomains());
      registerInsertBefore(loop, alloc);

      // clone loop nest:
      auto outermost_innermost =
          scope_utils::makeLoopNest(insertion_info.loop_nest);

      // make the address compute op:
      auto address_initialize_op = IrBuilder::create<kir::AddressCompute>(
          kir::AddressCompute::AddressComputeOpType::BASE_ADDRESS,
          insertion_info.address_compute_record->addressTensor(),
          insertion_info.address_compute_record->dataTensor());

      // put address compute in forloop
      outermost_innermost.second->body().push_back(address_initialize_op);

      // put the new loopnest before the hoisted loop
      registerInsertBefore(loop, outermost_innermost.first);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(
        false, "this pass cannot yet run after ite insertion");
  }

  std::vector<kir::ForLoop*> createAddressComputeLoop(
      AddressRecord* address_record) {
    // Find the loop in the current loop nest that maps the concrete serial loop
    //  on record
    auto concrete_loop_id = address_record->getConcreteSerialLoopId();
    auto serial_loop_it = std::find_if(
        for_loops_.begin(),
        for_loops_.end(),
        [concrete_loop_id](kir::ForLoop* for_loop) {
          return GpuLower::current()->caMap()->areMapped(
              for_loop->iter_domain(), concrete_loop_id, IdMappingMode::LOOP);
        });
    TORCH_INTERNAL_ASSERT(
        serial_loop_it != for_loops_.end(),
        "invalid address record, serial loop not found");

    // make an empty outmost for loop:
    auto original_serial_loop = *serial_loop_it;
    auto cloned_serial_loop = IrBuilder::create<kir::ForLoop>(
        original_serial_loop->iter_domain(),
        // index
        GpuLower::current()->kernel()->zeroVal(),
        // Start
        GpuLower::current()->kernel()->zeroVal(),
        // Stop
        GpuLower::current()->kernel()->oneVal(),
        false,
        nullptr,
        original_serial_loop->isUnrollRequired());

    std::vector<kir::ForLoop*> loop_vector;
    loop_vector.push_back(cloned_serial_loop);

    // Copy the rest of the loop nest.
    loop_vector.insert(
        loop_vector.end(), std::next(serial_loop_it), for_loops_.end());

    return loop_vector;
  }

  AddressComputeInsertionInfo makeInsertionInfo(AddressRecord* address_record) {
    AddressComputeInsertionInfo address_info;
    address_info.loop_nest = createAddressComputeLoop(address_record);
    address_info.address_compute_record = address_record;
    return address_info;
  }

  void addAddressComputeInsertionInfo(
      IterDomain* serial_id,
      AddressComputeInsertionInfo insertion_info) {
    std::vector<AddressComputeInsertionInfo>* insertion_info_vec_ptr = nullptr;
    auto exisiting_info_it =
        pending_address_compute_insertions_.find(serial_id);
    if (exisiting_info_it == pending_address_compute_insertions_.end()) {
      // Make a new vector of address compute record if needed.
      auto new_address_compute_record =
          std::make_unique<std::vector<AddressComputeInsertionInfo>>();
      insertion_info_vec_ptr = new_address_compute_record.get();
      pending_address_compute_insertions_[serial_id] =
          std::move(new_address_compute_record);
    } else {
      insertion_info_vec_ptr = exisiting_info_it->second.get();
    }
    insertion_info_vec_ptr->push_back(insertion_info);
  }

  void handle(Expr* expr) final {
    if (ir_utils::isTvOp(expr)) {
      auto& address_compute_info = GpuLower::current()->addressComputeInfo();

      // Register memory writes to insert
      for (auto consumer_tv :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto maybe_consumer_address_record =
            address_compute_info.getMaybeLiftedAddress(consumer_tv);

        if (maybe_consumer_address_record.has_value()) {
          auto insert_info =
              makeInsertionInfo(maybe_consumer_address_record.value());
          addAddressComputeInsertionInfo(
              maybe_consumer_address_record.value()->getConcreteSerialLoopId(),
              insert_info);
        }

        // Register memory reads to insert
        for (auto producer_tv :
             ir_utils::filterByType<TensorView>(expr->inputs())) {
          auto maybe_producer_address_record =
              address_compute_info.getMaybeLiftedAddress(
                  producer_tv, consumer_tv);
          if (maybe_producer_address_record.has_value()) {
            auto insert_info =
                makeInsertionInfo(maybe_producer_address_record.value());
            addAddressComputeInsertionInfo(
                maybe_producer_address_record.value()
                    ->getConcreteSerialLoopId(),
                insert_info);
          }
        }
      }
    } else {
      kir::ExprMutator::handle(expr);
    }
  }

 private:
  using AddressComputeInsertionInfoPtr =
      std::unique_ptr<std::vector<AddressComputeInsertionInfo>>;
  std::unordered_map<IterDomain*, AddressComputeInsertionInfoPtr>
      pending_address_compute_insertions_;
};

} // namespace

} // namespace

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch