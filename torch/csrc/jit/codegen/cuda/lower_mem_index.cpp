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
    : address_tv_(address_tv),
      data_tv_(data_tv),
      reference_tv_(reference_tv),
      access_direction_(direction),
      allocation_ids_(allocation_ids),
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
      // Didn't find a merged domain candidate, just
      //  increment
      domain_it++;
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
    const TensorView* data_tv,
    const TensorView* reference_tv) {
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

// Get all the nodes merged with a sequence of merge ops.
//  ordered by their inner/outer position at the ops.
std::vector<IterDomain*> getFlattenedMergedIds(IterDomain* merged_id) {
  std::vector<IterDomain*> result;

  if (merged_id->definition() != nullptr) {
    if (auto merge = dynamic_cast<Merge*>(merged_id->definition())) {
      auto left_leaves = getFlattenedMergedIds(merge->outer());
      auto right_leaves = getFlattenedMergedIds(merge->inner());

      result.insert(result.end(), left_leaves.begin(), left_leaves.end());
      result.insert(result.end(), right_leaves.begin(), right_leaves.end());
    }
  }

  return result;
}

// Get all the nodes merged with a sequence of merge ops.
//  ordered by their inner/outer position at the ops.
std::vector<IterDomain*> getFlattenedSplitIds(
    IterDomain* split_id,
    std::unordered_set<IterDomain*> stop_set = {}) {
  std::vector<IterDomain*> result;

  auto maybe_single_use =
      ir_utils::getMaybeSingleUse(split_id, ir_utils::isIterDomainOp);

  // Assuming this means the iter domain is a leaf domain,
  //  the invariant that each iter domain should be consumed by only
  //  one iter domain op should be validated in another pass.
  if (maybe_single_use.has_value()) {
    if (auto split = dynamic_cast<Split*>(maybe_single_use.value())) {
      auto process_node = [&result, &stop_set](IterDomain* id) {
        if (stop_set.count(id)) {
          result.push_back(id);
        } else {
          auto leaves = getFlattenedSplitIds(id, stop_set);
          result.insert(result.end(), leaves.begin(), leaves.end());
        }
      };

      process_node(split->outer());
      process_node(split->inner());
      return result;
    }
  }

  // base case where the current node is even a leaf or
  //  not a split consumer:
  result.push_back(split_id);
  return result;
}

//! Returns the stride of the leading dimension given the vector of leaf
//! iterdomains.
c10::optional<int64_t> getMaybeLeadingStride(
    std::vector<IterDomain*> leaf_ids,
    IterDomain* leading_id = nullptr) {
  if (leaf_ids.empty()) {
    return c10::nullopt;
  }
  if (leading_id == nullptr) {
    leading_id = leaf_ids.at(0);
  }

  bool leading_id_found = false;
  // Utility to compute the effective stride of id and prev_id:
  c10::optional<int64_t> result = c10::nullopt;
  for (auto leaf_it = leaf_ids.begin(); leaf_it != leaf_ids.end(); leaf_it++) {
    auto leaf_id = *leaf_it;
    if (leading_id_found) {
      if (!leaf_id->extent()->isConstInt()) {
        // Non-constant dimensions cannot be supported.
        return c10::nullopt;
      }
      result.value() *= leaf_id->extent()->evaluateInt();
    } else {
      if (leaf_id == leading_id) {
        leading_id_found = true;
        result = 1;
      }
    }
  }
  return result;
}

bool isSeparable(
    const TensorView* tv,
    IterDomain* id,
    const std::unordered_set<IterDomain*>& contig_merged_ids) {
  auto id_def = id->definition();
  IterDomain* prev_id = nullptr;
  while (id_def != nullptr) {
    if (auto split = dynamic_cast<Split*>(id_def)) {
      // Traverse backward towards the root domains.
      prev_id = id;
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
    } else if (auto merge = dynamic_cast<Merge*>(id_def)) {
      // Support only the simplest case for lifting across merge.

      if (prev_id == nullptr) {
        // A merge on the leaf domain not yet supported.
        return false;
      }

      if (auto prev_split = dynamic_cast<Split*>(prev_id->definition())) {
        auto split_leaves = getFlattenedSplitIds(id, {prev_id});
        auto merged_leaves = getFlattenedMergedIds(id);

        if (prev_id != split_leaves.at(0) || id != merged_leaves.at(0)) {
          // Only support the case when both ids are leading dims.
          return false;
        }

        // Check the actual static dimensions, which would be the
        //  only way for now to prove that we'd not need to wrap
        //  around when incrementing this id:
        auto maybe_id_stride = getMaybeLeadingStride(merged_leaves);
        auto maybe_prev_id_stride = getMaybeLeadingStride(split_leaves);

        if (maybe_id_stride.has_value() && maybe_prev_id_stride.has_value()) {
          if (maybe_id_stride.value() <= maybe_id_stride.value()) {
            return true;
          }
        }
      }

      return false;
    } else {
      return false;
    }
  }

  // This covers the base case where we start
  //  with a contigous merged output of root.
  return contig_merged_ids.count(id);
}

bool isSwizzleProducer(IterDomain* id) {
  auto single_use = ir_utils::getMaybeSingleUse(id, ir_utils::isIterDomainOp);
  if (!single_use.has_value()) {
    // Base case check.
    return false;
  }

  if (single_use.value()->isA<Swizzle2D>()) {
    return true;
  }

  for (auto id :
       ir_utils::filterByType<IterDomain>(single_use.value()->outputs())) {
    // The current id is swizzled if any subsequent consumer iterdomains
    //  is a producer of swizzle op.
    if (isSwizzleProducer(id)) {
      return true;
    }
  }

  return false;
}

bool isSeparableSmemSwizzledProducerIndex(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    IterDomain* id,
    const std::unordered_set<IterDomain*>& contig_merged_ids) {
  // Collect exact iterdomains in producer tv on the right of the producer
  //  ca_axis, which would be the nodes that'd be materialzied in shared mem
  //  tile.
  std::unordered_map<IterDomain*, IterDomain*> exact_to_producer_id_map;
  auto all_producer_id_vals = DependencyCheck::getAllValsBetween(
      {producer_tv->getMaybeRFactorDomain().begin(),
       producer_tv->getMaybeRFactorDomain().end()},
      {producer_tv->domain()->domain().begin() +
           producer_tv->getComputeAtPosition(),
       producer_tv->domain()->domain().end()});

  for (auto producer_id :
       ir_utils::filterByType<IterDomain>(all_producer_id_vals)) {
    exact_to_producer_id_map.emplace(std::make_pair(
        ir_utils::caMapExactConcreteId(producer_id), producer_id));
  }

  if (id->definition() == nullptr) {
    // If id is in root domain this should be supported.
    return true;
  }

  if (auto split = dynamic_cast<Split*>(id->definition())) {
    auto exact_producer_it = exact_to_producer_id_map.find(
        ir_utils::caMapExactConcreteId(split->in()));

    // Find a tile split node that this needs to assume.
    while (exact_producer_it == exact_to_producer_id_map.end()) {
      auto in_def = split->in()->definition();
      if (in_def == nullptr) {
        return false;
      }
      split = dynamic_cast<Split*>(in_def);
      if (split == nullptr) {
        return false;
      }
      exact_producer_it = exact_to_producer_id_map.find(
          ir_utils::caMapExactConcreteId(split->in()));
    }

    if (!exact_producer_it->second->extent()->isConstInt()) {
      // TODO: more levels of splits and merges should not be too hard
      //  to support but would probably want to do after the next round
      //  of indexing cleanup.
      return false;
    }

    auto producer_in_id = exact_producer_it->second;
    auto split_leaves = getFlattenedSplitIds(split->in(), {id});
    auto producer_split_leaves = getFlattenedSplitIds(producer_in_id);

    // Check that the leading dimension of producer split leaves are not
    // directly swizzled:
    if (isSwizzleProducer(producer_split_leaves.at(0))) {
      // No support for lifting swizzled id.
      return false;
    }

    auto maybe_producer_stride = getMaybeLeadingStride(producer_split_leaves);
    auto maybe_consumer_stride = getMaybeLeadingStride(split_leaves, id);

    if (maybe_consumer_stride.has_value() &&
        maybe_producer_stride.has_value()) {
      if (maybe_consumer_stride.value() >= maybe_producer_stride.value()) {
        // Only supports the case where the leading dimension maps to
        //  the leading dimension of the producer side.
        return true;
      }
    }
  }

  // TODO: more flexible checking would need to be built out.
  return false;
}

void validateSeparability(
    const TensorView* reference_tv,
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
  bool is_data_read = data_tv != reference_tv;

  auto& ca_map = GpuLower::current()->caMap();

  // Get allocation ids for the lifted index.
  std::deque<IterDomain*> alloc_ids;

  auto contig_merged_ids =
      getInitialContiguousRootIdsOnReferenceTv(data_tv, reference_tv);

  // Mark the boundary iterdomain at the compute at
  //  position, where iterdomains on the left would
  //  be in shared loop.
  auto ref_ca_id = data_tv->axis(data_tv->getComputeAtPosition() - 1);

  if (data_tv->isFusionInput()) {
    // Reading input tensor looks at the consumer
    //  scheduling so use the CA of comsumer.
    ref_ca_id = reference_tv->axis(reference_tv->getComputeAtPosition() - 1);
  } else if (data_tv->isFusionOutput()) {
    ref_ca_id = data_tv->axis(data_tv->getMaxProducerPosition() - 1);
  } else if (data_tv != reference_tv) {
    bool found_ref_ca_id = false;
    for (auto id : reference_tv->domain()->domain()) {
      if (GpuLower::current()->caMap()->areMapped(
              id, ref_ca_id, IdMappingMode::LOOP)) {
        ref_ca_id = id;
        found_ref_ca_id = true;
        break;
      }
    }

    TORCH_INTERNAL_ASSERT(found_ref_ca_id);
  }

  // Serial domain which the cached indexing will
  //  be lifted out of.
  // TODO: some extra work to support
  //  lifting all the way to global scope.
  IterDomain* serial_id = nullptr;

  // compute id's to cache index on:
  auto ref_id_it = reference_tv->domain()->domain().rbegin();
  while (ref_id_it != reference_tv->domain()->domain().rend()) {
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
      ref_id_it++;
      continue;
    }

    // Current not attempting to lift outside of
    //  non-constant sized serial loops, due to
    //  limited profitability. Might want to
    //  revisit when we see useful cases.
    if (!ref_id->extent()->isConstScalar()) {
      serial_id = ref_id;
      break;
    }

    // TODO: re-enable index re-use in shared mem.
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
      if (
          // Checking reference tv is enough for non-swizzled producer.
          !is_data_read || !data_tv->hasSwizzleOp() ||
          // Check supported lifting in the case of swizzled producer.
          isSeparableSmemSwizzledProducerIndex(
              data_tv, reference_tv, ref_id, contig_merged_ids)) {
        ref_id_it++;
        continue;
      }
    }

    // We are visiting the iterdomains from reference
    //  tv in reverse order but we want the outputs
    //  to be in original order.
    alloc_ids.push_front(ref_id);
    ref_id_it++;
  }

  while (ref_id_it != reference_tv->domain()->domain().rend() &&
         serial_id == nullptr) {
    auto ref_id = *ref_id_it;
    ref_id_it++;
    if (!ref_id->isParallelized()) {
      serial_id = ref_id;
    }
  }

  TORCH_INTERNAL_ASSERT(
      serial_id != nullptr,
      "Cannot target serial loop to lift indexing out of");

  std::vector<IterDomain*> alloc_ids_vec{alloc_ids.begin(), alloc_ids.end()};

  // TODO: validate SMEM and GMEM are different, build them out.
  // validateSeparability(data_tv, alloc_ids_vec, contig_merged_ids);

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
  index_lift_record_.insert(
      std::make_pair(new_record_ptr->key(), std::move(new_record_ptr)));
}

c10::optional<AddressRecord*> AddressComputeInfo::getMaybeLiftedAddress(
    const TensorView* data_tv,
    const TensorView* reference_tv) {
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
  DataType dtype = is_global_address ? DataType::Index : DataType::Int32;
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          address_domains, std::vector<bool>(address_domains.size(), true)),
      dtype);
}

c10::optional<AddressRecord*> AddressComputeInfo::getMaybeRecordForAddressTv(
    const TensorView* tv) {
  auto record_it = address_tv_to_address_record_.find(tv);
  if (record_it == address_tv_to_address_record_.end()) {
    return c10::nullopt;
  }
  return record_it->second;
}

c10::optional<kir::ForLoop*> AddressRecord::getMaybeSerialLoop(
    std::vector<kir::ForLoop*> loops) {
  auto concrete_loop_id = getConcreteSerialLoopId();

  auto serial_loop_it = std::find_if(
      loops.begin(), loops.end(), [concrete_loop_id](kir::ForLoop* for_loop) {
        return GpuLower::current()->caMap()->areMapped(
            for_loop->iter_domain(), concrete_loop_id, IdMappingMode::LOOP);
      });

  if (serial_loop_it != loops.end()) {
    return *serial_loop_it;
  }

  return c10::nullopt;
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
      std::vector<Val*> alloc_shape;
      std::transform(
          insertion_info.address_compute_record->allocationIterDomains()
              .begin(),
          insertion_info.address_compute_record->allocationIterDomains().end(),
          std::back_inserter(alloc_shape),
          [](IterDomain* id) { return id->extent(); });

      // allocate address tensor:
      auto alloc = IrBuilder::create<kir::Allocate>(
          insertion_info.address_compute_record->addressTensor(),
          MemoryType::Local,
          alloc_shape);
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

  //! Create an replica of the original loop except that
  //!  the extent and index are zeroed.
  kir::ForLoop* createZeroedLoop(kir::ForLoop* original_loop) {
    auto start = original_loop->iter_domain()->isThread()
        ? original_loop->start()
        : GpuLower::current()->kernel()->zeroVal();
    auto stop = original_loop->iter_domain()->isThread()
        ? original_loop->iter_domain()->extent()
        : GpuLower::current()->kernel()->oneVal();
    return IrBuilder::create<kir::ForLoop>(
        original_loop->iter_domain(),
        // index
        GpuLower::current()->kernel()->zeroVal(),
        // Start
        start,
        // Stop
        stop,
        // Step
        GpuLower::current()->kernel()->oneVal(),
        false,
        nullptr,
        original_loop->isUnrollRequired(),
        original_loop->loopTransformInfo().baseIndexLoop());
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
    serial_loop_it++;
    auto cloned_serial_loop = createZeroedLoop(original_serial_loop);

    std::vector<kir::ForLoop*> loop_vector;
    loop_vector.push_back(cloned_serial_loop);

    while (serial_loop_it != for_loops_.end()) {
      auto loop = *serial_loop_it;
      serial_loop_it++;
      if (std::any_of(
              address_record->allocationIterDomains().begin(),
              address_record->allocationIterDomains().end(),
              [loop](IterDomain* allocated_id) {
                return GpuLower::current()->caMap()->areMapped(
                    allocated_id, loop->iter_domain(), IdMappingMode::LOOP);
              })) {
        loop_vector.push_back(loop);
      } else {
        // Only materialize the loops that correspond to
        //  allocated dimensions of address tensor.
        loop_vector.push_back(createZeroedLoop(loop));
      }
    }

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
    if (ir_utils::isTvOp(expr) && !ir_utils::isTensorScalarFillOp(expr)) {
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

// TODO: add insert address compute pass
std::vector<Expr*> preComputeLiftedAddress(const std::vector<Expr*>& exprs) {
  return MemoryAddressComputeInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch