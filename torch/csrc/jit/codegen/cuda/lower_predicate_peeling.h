#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Stage, Factor pair:
struct PeeledTileEntry {
  PredicatePeelStage peel_stage = PredicatePeelStage::NoApplicable;
  Val* inner_factor = nullptr;
  kir::ForLoop* for_loop = nullptr;
};

class PredicatePeelingInfo {
 public:
  bool shouldPeelLoop(kir::ForLoop* forloop) const;

  void build(Fusion* fusion);

  c10::optional<PeeledTileEntry> getMaybePeeledTileEntry(
      const std::vector<kir::ForLoop*>& loops,
      IterDomain* root_id);

  bool hasPeeledId(const TensorView* tv) const;

 private:
  std::unordered_set<IterDomain*> concrete_id_of_peeled_loops_;
};

namespace PredicatePeeling {

// User space check that makes sure the loop can
//  actually be peeled to remove predicates.
bool supportedPeelingLoop(IterDomain* id);

std::vector<Expr*> peelPredicatedLoop(const std::vector<Expr*> exprs);

Val* getSplitTileOffset(IterDomain* id, Val* tile_factor);

Val* getSplitTileMainOffset(IterDomain* id, Val* tile_factor);

} // namespace PredicatePeeling

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
