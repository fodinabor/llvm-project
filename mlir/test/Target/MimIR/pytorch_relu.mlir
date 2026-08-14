// RUN: mlir-translate --mlir-to-mim %s -split-input-file | FileCheck %s
// Translating tensor/linalg ops to MimIR is not implemented yet.
// XFAIL: *

// CHECK: main

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
