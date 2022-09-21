#pragma once

#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class InterleaveLoopInfo {
  using ConcreteIdVector = VectorOfUniqueEntries<IterDomain*>;
  using TensorViewVector = VectorOfUniqueEntries<TensorView*>;

 public:
  void build(Fusion* fusion);

  void validate();

  const auto& concreteMainLoopToSubloopMap() const {
    return concrete_main_loop_to_subloop_map_;
  }

  const auto& concreteMainLoopToFactorMap() const {
    return concrete_main_loop_to_number_of_units_;
  }

 private:
  //! Build phase 1: check all the tv's for
  //!  main_loops where subloops are interleaved.
  void collectInterleaveMainLoops();

  //! Build phase2: collect all tv's that are
  //!  computed within interleaved loops.
  void collectInterleavedSubLoops();

  void insertEntry(TensorView* tv, IterDomain* main_loop, IterDomain* sub_loop);

  bool isMainLoop(IterDomain* id);

  bool isSubLoopof(IterDomain* id, IterDomain* concrete_main_id);

  void validateMainLoop(
      IterDomain* main_loop,
      const TensorViewVector& interleaved_tvs);

  bool isExitTv(TensorView* tv, const TensorViewVector& interleaved_tvs);

 private:
  std::unordered_map<IterDomain*, ConcreteIdVector>
      concrete_main_loop_to_subloop_map_;
  std::unordered_map<IterDomain*, TensorViewVector>
      concrete_main_loop_to_interleaved_tv_;
  std::unordered_map<IterDomain*, int> concrete_main_loop_to_number_of_units_;

  //! Short-cut to the fusion this info keeps track of.
  Fusion* fusion_ = nullptr;

  //! Cached used math vals from fusion_;
  std::vector<TensorView*> used_tvs_;
};

void validateInterleaving(Fusion* fusion);

std::vector<Expr*> interLeaveDoubleBufferUnrolledLoops(
    const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
