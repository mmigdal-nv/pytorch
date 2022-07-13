#include <torch/csrc/jit/codegen/cuda/scheduler/matmul.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/mma_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// Move the broadcast axes to the left on the inner 3 dimensions
// e.g. [... I0, B, I1] -> [... B, I0, I1]
//  should probably be only used to order innermost mnk axes.
void moveInnerBroadcastLeft(TensorView* tv, int number_of_inner_pos = 3) {
  TORCH_INTERNAL_ASSERT(tv->nDims() >= number_of_inner_pos);
  std::vector<int> broadcast_pos;
  std::vector<int> nonbroadcast_pos;

  for (auto i : c10::irange(number_of_inner_pos)) {
    auto axis_idx = i - number_of_inner_pos;
    auto id = tv->axis(axis_idx);
    if (id->isBroadcast()) {
      broadcast_pos.push_back(axis_idx);
    } else {
      nonbroadcast_pos.push_back(axis_idx);
    }
  }

  auto combined_pos_vec = broadcast_pos;
  combined_pos_vec.insert(
      combined_pos_vec.end(), nonbroadcast_pos.begin(), nonbroadcast_pos.end());

  std::unordered_map<int, int> order_map;
  for (auto i : c10::irange(number_of_inner_pos)) {
    order_map[combined_pos_vec.at(i)] = i - number_of_inner_pos;
  }

  // Apply ordering.
  tv->reorder(order_map);
}

} // namespace

void scheduleMatmul(
    TensorView* c,
    TensorView* a,
    TensorView* b,
    MmaBuilder& mma_builder,
    MatMulTileOptions& gemm_tile) {
  // Currently only support a, b, c as fusion inputs/outputs
  //  aka. no prolog and epilog fusion yet.
  TORCH_CHECK(
      c->isFusionOutput() && a->isFusionInput() && b->isFusionInput(),
      "not supporting matmul fusion yet");
  TORCH_CHECK(c->definition() && c->definition()->isA<MmaOp>());

  mma_builder.configureMma(c);

  // Setup register and shared memory stages:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  auto ar = a->cacheAfter();
  auto br = b->cacheAfter();
  auto acw = ar->cacheAfter();
  auto acr =
      acw->cacheAfter(mma_builder.operand(MmaOptions::Operand::A).ldMatrix());
  auto bcw = br->cacheAfter();
  auto bcr =
      bcw->cacheAfter(mma_builder.operand(MmaOptions::Operand::B).ldMatrix());
  auto cc = c->cacheBefore();
  mma_builder.accumulatorTv(cc);

  // Option can only be pulled from this stage, so had to add the temporary
  //  assertion here.
  auto mma_options = mma_builder.build();
  // The load schedule for volta is different, will enable in a follow up.
  TORCH_CHECK(isTuring(mma_options.macro) || isAmpere(mma_options.macro));

  // Make a CTA tile
  // ------------------------------------------------------------------
  scheduler_utils::matmul_utils::canonicalizeMmaTvOrdering(cc);
  // [... M,N,K]
  scheduler_utils::matmul_utils::makeTile(cc, gemm_tile.cta_tile.toVector());

  // [Mo, No, Ko, Mi, Ni, Ki]
  // Propagate tiling globally
  scheduler_utils::matmul_utils::transformPropagateToAllFrom(cc, -1);

  // Schedule warp tile
  scheduler_utils::matmul_utils::scheduleWarpTileWithReduction(cc, gemm_tile);

  // Propagate warp tile to main loop and epilog/output tvs
  scheduler_utils::matmul_utils::transformPropagateWithin(
      cc, -1, {acw, bcw}, {c});

  // Schedule prolog:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  scheduler_utils::matmul_utils::orderTiledConcreteIdAsRoot(acw);
  // [... M, K]
  acw->merge(-2);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      acw, gemm_tile, 8);

  // [... N, K]
  scheduler_utils::matmul_utils::orderTiledConcreteIdAsRoot(bcw);
  bcw->merge(-2);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      bcw, gemm_tile, 8);

  // Propagate prolog tensors
  //  propagate up the DAG, and propagate parallel type.
  scheduler_utils::matmul_utils::transformPropagateWithin(
      acw, -1, {a}, {}, true);
  scheduler_utils::matmul_utils::transformPropagateWithin(
      bcw, -1, {b}, {}, true);

  // Set computeAt, setup the loop nesting structure on the kernel.
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  // CTA tile:
  a->computeAt(c, 2);
  b->computeAt(c, 2);

  // Prolog:
  a->computeAt(cc, 3);
  b->computeAt(cc, 3);

  // Main Loop:
  acr->computeAt(cc, -4);
  bcr->computeAt(cc, -4);

  // Add mma swizzle:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  auto mma = dynamic_cast<MmaOp*>(cc->definition());
  TORCH_INTERNAL_ASSERT(mma != nullptr);
  auto ab = mma->inA()->as<TensorView>();
  auto bb = mma->inB()->as<TensorView>();

  moveInnerBroadcastLeft(ab);
  ab->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());

  moveInnerBroadcastLeft(bb);
  bb->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  cc->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  // Propagate mma input swizzle up the DAG
  scheduler_utils::matmul_utils::transformPropagateWithin(
      ab, -1, {acw}, {}, true);
  scheduler_utils::matmul_utils::transformPropagateWithin(
      bb, -1, {bcw}, {}, true);

  // Set memory type:
  acw->setMemoryType(MemoryType::Shared);
  bcw->setMemoryType(MemoryType::Shared);

  // Set parallelization:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------

  // Vectorize smem loads:
  acr->axis(-1)->parallelize(ParallelType::Vectorize);
  bcr->axis(-1)->parallelize(ParallelType::Vectorize);

  //  0   1  2  3    4   5  6  7  8  9  10
  // [Mo No Ko Mwo  Nwo Kw Mw Nw (Mi Ni Ki)]
  cc->axis(0)->parallelize(ParallelType::BIDx);
  cc->axis(1)->parallelize(ParallelType::BIDy);
  cc->axis(3)->parallelize(ParallelType::TIDz);
  cc->axis(4)->parallelize(ParallelType::TIDy);

  // Propagate mma output swizzle and parallelization down the DAG
  scheduler_utils::matmul_utils::transformPropagateWithin(
      cc, -1, {}, {c}, true, true);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
