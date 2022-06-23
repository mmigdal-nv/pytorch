#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_predicate_peeling.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool PredicatePeeling::supportedPeelingLoop(IterDomain* id) {
  if (id->isParallelized()) {
    return false;
  }

  auto id_def = id->definition();

  if (id_def == nullptr) {
    // This case is not profitable so skip peeling support.
    return false;
  } else if (auto split = dynamic_cast<Split*>(id_def)) {
    auto split_in = split->in();

    // This is typical case that we want to peel.
    //  where we advance a serial iteration through
    //  constant sized tiles.
    return split_in->definition() == nullptr && id == split->outer() &&
        split->factor()->isConstInt();
  }

  // TODO:
  //  should extend to use separability defined in anohter PR.
  return false;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
