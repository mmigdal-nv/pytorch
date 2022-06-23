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

class PredicatePeelingInfo {
 public:
  bool shouldPeelLoop(kir::ForLoop* forloop);

  void markLoopPeeled(kir::ForLoop* forloop);

 private:
 private:
  std::unordered_set<IterDomain*> concrete_loops_to_peel_;
};

namespace PredicatePeeling {

// User space check that makes sure the loop can
//  actually be peeled to remove predicates.
bool supportedPeelingLoop(IterDomain* id);

std::vector<Expr*> peelPredicatedLoop(const std::vector<Expr*> exprs);

} // namespace PredicatePeeling

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
