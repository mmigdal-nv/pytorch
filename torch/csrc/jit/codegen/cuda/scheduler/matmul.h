#pragma once

#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/mma_type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Starting point for a matmul scheduler parameters:
class MatmulParam {
 public:
  MatmulParam(MmaBuilder builder) : mma_builder(builder) {}

  struct DoubleBufferOptions {
    bool double_buffer_smem_write = false;
    bool double_buffer_smem_read = false;
    int smem_double_buffer_stage = 2;
  };

  //! (Ampere+) Use cp.async to load operands.
  bool async_gmem_load_operands = false;

  //! Specifies the tiling hierarchy on block,
  //!  warp, and instruction levels.
  MatMulTileOptions tile_sizes;

  //! Parameters for configuring mma ops.
  MmaBuilder mma_builder;

  //! Specify which tensor we double buffer.
  DoubleBufferOptions double_buffer_options;

  //! Tunable spec to enable/disable lifting
  //!  memory indexing math out of the main
  //!  loop on the generated kernel.
  //! (All defaults to on).
  //! Note: eventually this part of logic
  //!  will be merged into automatic
  //!  indexing math allocation/placement pass.
  struct IndexLiftingOptions {
    bool lift_smem_read_address = true;
    bool lift_smem_write_address = true;
    bool lift_gmem_read_address = true;
    // TODO: add gmem_write address for
    //  latency bound kernels.
  } index_lift_options;
};

//! Prototype auto scheduling function.
//!  Currently only support a pure matmul with no
//!   fused prolog or epilog.
//!
//! TODO:
//!   - will support a range of fusions in a follow up
//!   - will formalize scheduling decisions into
//! matmul params data structure.
TORCH_CUDA_CU_API void scheduleMatmul(
    TensorView* c_tv,
    TensorView* a_tv,
    TensorView* b_tv,
    MatmulParam& params);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
