// RUN: mlir-translate --mlir-to-mim %s | FileCheck %s

// linalg.fill of a scalar becomes a pack.
// CHECK-LABEL: fun extern test_fill (
// CHECK: (_{{[0-9]+}}, ‹2; ‹3; 1065353216:(%math.F (23, 8))››);
func.func @test_fill() -> tensor<2x3xf32> {
  %cst = arith.constant 1.0 : f32
  %0 = tensor.empty() : tensor<2x3xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// A rank-3 transpose with a non-trivial permutation: `dim(result, i) =
// dim(input, perm[i])`, so tensor<1x2x3> with perm [2, 0, 1] yields
// tensor<3x1x2>, and the read map is the inverse permutation.
// CHECK-LABEL: fun extern test_transpose (
// CHECK: %tensor.map_reduce 1 (%math.F (23, 8), 3, 0)
// CHECK-SAME: transpose_copy
func.func @test_transpose(%arg0 : tensor<1x2x3xf32>) -> tensor<3x1x2xf32> {
  %0 = tensor.empty() : tensor<3x1x2xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<1x2x3xf32>)
      outs(%0 : tensor<3x1x2xf32>) permutation = [2, 0, 1]
  return %1 : tensor<3x1x2xf32>
}

// A non-square matmul with a zero-filled accumulator becomes a plain
// %tensor.dot_product, contracting dim 1 of the left with dim 0 of the right
// operand.
// CHECK-LABEL: fun extern test_matmul (
// CHECK: %tensor.dot_product (%math.F (23, 8), 0:(%math.F (23, 8)), ring_fadd_{{[0-9]+}}, ring_fmul_{{[0-9]+}}) ‹2; 2› (1, 0) (tt, ff, (), ())
func.func @test_matmul(%a : tensor<2x3xf32>, %b : tensor<3x4xf32>) -> tensor<2x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
  %2 = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>)
      outs(%1 : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %2 : tensor<2x4xf32>
}

#map_in = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>

// A row-sum reduction: one parallel and one reduction loop; the fold is
// seeded with the splat value of the zero-filled `outs` operand.
// CHECK-LABEL: fun extern test_reduce (
// CHECK: %tensor.map_reduce 1 (%math.F (23, 8), 1, 1) (2, (2, 3))
// CHECK-SAME: linalg_body_{{[0-9]+}}, 0:(%math.F (23, 8))
func.func @test_reduce(%arg0 : tensor<2x3xf32>) -> tensor<2xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<2xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32>
  %2 = linalg.generic {indexing_maps = [#map_in, #map_out],
                       iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2x3xf32>) outs(%1 : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<2xf32>
  return %2 : tensor<2xf32>
}
