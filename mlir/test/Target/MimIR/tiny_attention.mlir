// RUN: mlir-translate --mlir-to-mim %s | FileCheck %s

// A tiny single-head attention block: Q/K/V projections, scores against the
// transposed keys, a sigmoid gate, and the output projection — five
// %tensor.dot_product contractions plus transpose/sigmoid map_reduces.

// CHECK-LABEL: fun extern main (
// CHECK: %tensor.dot_product
// CHECK: %tensor.dot_product
// CHECK: %tensor.map_reduce 1
// CHECK-SAME: transpose_copy
// CHECK: %tensor.dot_product
// CHECK: %tensor.map_reduce 1
// CHECK-SAME: linalg_body
// CHECK: %tensor.dot_product
// CHECK: %tensor.dot_product
// CHECK: %tensor.dot_product

// The sigmoid body: 1 / (1 + exp(-x)), with negf normalized to `0 - x`.
// CHECK: fun linalg_body_{{[0-9]+}}
// CHECK: %math.arith.sub (23, 8) 0 (0:(%math.F (23, 8)),
// CHECK: %math.exp.lbb
// CHECK: %math.arith.add (23, 8) 0 (1065353216:(%math.F (23, 8)),
// CHECK: %math.arith.div (23, 8) 0 (1065353216:(%math.F (23, 8)),

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %cst = arith.constant dense_resource<torch_tensor_8_8_torch.float32> : tensor<8x8xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_8_8_torch.float32_1> : tensor<8x8xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_8_8_torch.float32_2> : tensor<8x8xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_8_8_torch.float32_3> : tensor<8x8xf32>
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %2 = linalg.matmul ins(%arg0, %cst : tensor<4x8xf32>, tensor<8x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %3 = linalg.matmul ins(%arg0, %cst_2 : tensor<4x8xf32>, tensor<8x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %4 = linalg.matmul ins(%arg0, %cst_3 : tensor<4x8xf32>, tensor<8x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %5 = tensor.empty() : tensor<8x4xf32>
    %transposed = linalg.transpose ins(%3 : tensor<4x8xf32>) outs(%5 : tensor<8x4xf32>) permutation = [1, 0] 
    %6 = tensor.empty() : tensor<4x4xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %8 = linalg.matmul ins(%2, %transposed : tensor<4x8xf32>, tensor<8x4xf32>) outs(%7 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<4x4xf32>) outs(%6 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.negf %in : f32
      %13 = math.exp %12 : f32
      %14 = arith.addf %13, %cst_1 : f32
      %15 = arith.divf %cst_1, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<4x4xf32>
    %10 = linalg.matmul ins(%9, %4 : tensor<4x4xf32>, tensor<4x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %11 = linalg.matmul ins(%10, %cst_4 : tensor<4x8xf32>, tensor<8x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %11 : tensor<4x8xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_8_torch.float32: "0x040000007517FA3E5F1135BF7644A13F55B0863DBB591FC0F13289BF4469323F00C8E33F3CCA3ABEA0D2EDBE4F2F7D3E76C6BA3E45B9CEBFECDAFB3ED2AE1A3F7B9DA33F9AC5383E012DA4BF6DF2193FD9D4233FB81E163F0DE01DBF7DCE5A3F70928A3F4A4D89BF7C7E5E3F721593BFBC016C3F9A03FCBE4BD2913EE1E0EC3E6F243BBF96E2DE3F5D5768BE777246BF23C7FE3FA15A7A3F9FA457BFC3B3803EA5EF44BF0F36D8BF3ADA2E3F8820B1BE768DFC3E7F096E3EF63FBBBF03F0B73DD4436E3FD25AA63EF375103D13B3543F5BD27D3F56A05EBF0352B03ED87826BE86281BBF87D8D2BF70D378BFC777F6BE884D373FBC30BB3FA1BC0BBFE034BC3F3D510C3F",
      torch_tensor_8_8_torch.float32_1: "0x040000007306173FBC7DB1BFB4A815C0975498BF664748BFD977C2BF20DF07BF512C02BFDE73E13FFACC8CBF735ED93E47A6A33F19D7C73EAFC7AA3FC88B313E3368933D9EAA3B3E50F8353F3D602640A1C68A3FE0FAF23E88FB043F6943913FAC54F9BFEA101ABF69C292BFFAA85EBE08F31140D8DAFDBF954D9A3E0A7FD7BFD442D4BE21B267BE8022233F470C8F3FE72BBB3D5C63E43F9BE9853E3EFC01BBDB27883D926B03BF5C342BC072172F3D4DD3EB3FAB08423FDEF282BF8DDF753FC14B5CBFE0A6203F4D5885BF9CD426BF1D8F52BFB527ABBE884C7CBEDBEE123E54488FBF635F483F310EB53E10DA643E573B723EFFF603406A7D20BFE1E8E83BDDDCA33F",
      torch_tensor_8_8_torch.float32_2: "0x040000004361A9BD48A136BF4DBDACBE99B66CBF151944BE356431BF8048B4BF40A0393FF9A313BF162C7CBFB438E43FF7B0B2BF5C55DABFA9DB80BFAEC728BFE364A8BEAC3286BE426B5D3FFE26B13FB8CFA03F9C7FB93EDEE5ADBFDE0B4EBF29A089BFAF82BFBFDABA013EF2488EBE30D7C23FC287F0BF8957573DEAAAD1BEDA7E143FC43F24BFAF3F80BF4A73983DD074873D76A8DE3E91EBE33FA3495BBEFBC5583FA261A73F35B31DC0DF2BE7BF637FC33F5E1C0CBF65C75A3E679B29BF11B3083FC9DCD83E292A0FC033E796BF75756CBF1069A53F75CE12C0C42F22BF27D2AABD9ECE7B3F98093FBF7420D0BE0AA48FBF346C2EBF39AD024074A104BFF9E97FBE",
      torch_tensor_8_8_torch.float32_3: "0x0400000017B1C03F48E90EBF4E79653F497FA4BFBE2609C0B672FABE75D93F3F8A09933F92CB26BF918AC0BF12FC8ABFE4C8813F3BA7B83F8ACC50BFB2ED5FBFC41FFF3F6DA259BD6E8E673E054926BFBCFE27BF37B3863F4AEAA03FBA536F3F53A217BFBAFDE83F52B884BE57D9A73FF250C33EC70B583FECAA063FC4C41B3F985B96BF074722C0E265D2BEDCFB0D405A29AF3F2284503F8894873EBA520640745C763FB1AA6A3F99051C3FC4A00EBF0291C03ED62D7ABFE1D82C3FDF027C3E150C76BE51AC23BF29B29DBFB45D4D3F694B043F5B85923F2B438FBDB8BD9A3F567DA0BE7521AB3FDA0C2E3EE9D19ABF136FDABE7CF6013F0DDCECBF776E653E368C2A40"
    }
  }
#-}
