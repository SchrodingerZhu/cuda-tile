// RUN: cuda-tile-opt %s --pass-pipeline='builtin.module(cuda_tile.module(cuda_tile.entry(alternate-reduction-loop)))' --split-input-file | FileCheck %s

// Test: Basic reduction loop transformation
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @basic_reduction
  cuda_tile.module @basic_reduction {
    entry @kernel_0() {
      %cst_0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
      %cst_10_i32 = constant <i32: 10> : !cuda_tile.tile<i32>
      %cst_1_i32 = constant <i32: 1> : !cuda_tile.tile<i32>
      %cst_0_f32 = constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
      %init_m = constant <f32: 0xFF800000> : !cuda_tile.tile<64x1xf32>
      %init_l = constant <f32: 0.0> : !cuda_tile.tile<64x1xf32>

      // CHECK: for {{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}})
      // CHECK:   for {{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}})
      // The outer loop is the alternating loop with TILE_X iterations
      // The inner loop contains the original reduction body

      %for:3 = for %loopIdx in (%cst_0_i32 to %cst_10_i32, step %cst_1_i32) : tile<i32> iter_values(%acc = %cst_0_f32, %l_i = %init_l, %m_i = %init_m) -> (tile<64x64xf32>, tile<64x1xf32>, tile<64x1xf32>) {
        // Simulate a reduction pattern
        %reduce = reduce %acc dim=1 identities=[0xFF800000 : f32] : tile<64x64xf32> -> tile<64xf32>
        (%reduce_lhs: tile<f32>, %reduce_rhs: tile<f32>) {
          %max = maxf %reduce_lhs, %reduce_rhs : tile<f32>
          yield %max : tile<f32>
        }
        %reshape = reshape %reduce : tile<64xf32> -> tile<64x1xf32>
        %new_m = maxf %m_i, %reshape : tile<64x1xf32>
        continue %acc, %l_i, %new_m : tile<64x64xf32>, tile<64x1xf32>, tile<64x1xf32>
      }
      return
    }
  }
}

// -----

// Test: Loop with MMA operations (should also be transformed)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @mma_reduction
  cuda_tile.module @mma_reduction {
    entry @kernel_0() {
      %cst_0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
      %cst_8_i32 = constant <i32: 8> : !cuda_tile.tile<i32>
      %cst_1_i32 = constant <i32: 1> : !cuda_tile.tile<i32>
      %cst_0_f32 = constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
      %lhs = constant dense<0.0> : tile<64x64xf16>
      %rhs = constant dense<0.0> : tile<64x64xf16>

      // CHECK: for {{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}})
      // CHECK:   for {{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}})

      %for = for %loopIdx in (%cst_0_i32 to %cst_8_i32, step %cst_1_i32) : tile<i32> iter_values(%acc = %cst_0_f32) -> (tile<64x64xf32>) {
        %mma_result = mmaf %lhs, %rhs, %acc : tile<64x64xf16>, tile<64x64xf16>, tile<64x64xf32>
        continue %mma_result : tile<64x64xf32>
      }
      return
    }
  }
}

// -----

// Test: Loop without reduction (should NOT be transformed)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @no_reduction
  cuda_tile.module @no_reduction {
    entry @kernel_0() {
      %cst_0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
      %cst_10_i32 = constant <i32: 10> : !cuda_tile.tile<i32>
      %cst_1_i32 = constant <i32: 1> : !cuda_tile.tile<i32>
      %cst_5_i32 = constant <i32: 5> : !cuda_tile.tile<i32>

      // CHECK: for {{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}})
      // CHECK-NOT: for {{.*}} in (%{{.*}} to %{{.*}}, step %{{.*}})
      // No nested loop should be created since there's no reduction

      for %loopIdx in (%cst_0_i32 to %cst_10_i32, step %cst_1_i32) : tile<i32> {
        %add = addi %loopIdx, %cst_5_i32 : tile<i32>
        continue
      }
      return
    }
  }
}
