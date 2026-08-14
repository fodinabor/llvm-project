// RUN: mlir-translate --mlir-to-mim %s | FileCheck %s

// A small PyTorch linear + ReLU model: checks that linalg lowers to the
// high-level SSA-world axioms of MimIR's tensor plugin.

// CHECK-LABEL: fun extern main (

// linalg.matmul becomes %tensor.dot_product over a float ring, contracting
// dim 1 of the input with dim 0 of the (transposed) weight; the zero-filled
// `outs` tensor is elided. The transpose is a map_reduce with a swapped read
// map, its input the dense resource weights (first element 0xBE4F96D4).
// CHECK: %tensor.dot_product (%math.F (23, 8), 0:(%math.F (23, 8)), ring_fadd_{{[0-9]+}}, ring_fmul_{{[0-9]+}}) ‹2; 2› (1, 0) (tt, ff, (), ())
// CHECK-SAME: %tensor.map_reduce 1
// CHECK-SAME: transpose_copy
// CHECK-SAME: 3192886996:(%math.F (23, 8))

// The bias addition: a two-input map_reduce whose second input is the
// rank-1 bias (first element 0x3E62E4F8) read through a broadcast map.
// CHECK: %tensor.map_reduce 2 (%math.F (23, 8), 2, 0)
// CHECK-SAME: linalg_body_
// CHECK-SAME: 1046668536:(%math.F (23, 8))

// The ReLU: a single-input elementwise map_reduce.
// CHECK: %tensor.map_reduce 1 (%math.F (23, 8), 2, 0)
// CHECK-SAME: linalg_body_

// The fold functions and ring operations.
// CHECK: fun transpose_copy_{{[0-9]+}}
// CHECK: lam ring_fadd_{{[0-9]+}}
// CHECK: %math.arith.add (23, 8) 0
// CHECK: lam ring_fmul_{{[0-9]+}}
// CHECK: %math.arith.mul (23, 8) 0
// CHECK: fun linalg_body_{{[0-9]+}}
// CHECK: %math.arith.add (23, 8) 0
// CHECK: fun linalg_body_{{[0-9]+}}
// CHECK: %math.cmp.UGle (23, 8) 0

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_4_4_torch.float32> : tensor<4x4xf32>
    %0 = tensor.empty() : tensor<4x4xf32>
    %transposed = linalg.transpose ins(%cst_1 : tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) permutation = [1, 0] 
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%arg0, %transposed : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %cst : tensor<4x4xf32>, tensor<4xf32>) outs(%0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.addf %in, %in_2 : f32
      linalg.yield %5 : f32
    } -> tensor<4x4xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.cmpf ugt, %in, %cst_0 : f32
      %6 = arith.select %5, %in, %cst_0 : f32
      linalg.yield %6 : f32
    } -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_4_torch.float32: "0x04000000F8E4623E661AD9BE067BB63EDA5297BE",
      torch_tensor_4_4_torch.float32: "0x04000000D4964FBE8036C5BCCCD37A3E7C1687BEF88D893DF47CE9BE78BD85BD701AE93EC4CDF73E0602D7BEF80CA7BDD26F8E3E16809C3E3C53533E3824D8BEEEBFC93E"
    }
  }
#-}
