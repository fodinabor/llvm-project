// RUN: mlir-translate --mlir-to-mim %s | FileCheck %s

// CHECK-LABEL: fun extern test_unary (
// CHECK: %math.exp.lbb (52, 11) 0
// CHECK: %math.exp.Lbb (52, 11) 0
// CHECK: %math.tri.aHFf (52, 11) 0
// CHECK: %math.rt.sq (52, 11) 0
// CHECK: %math.abs (52, 11) 0
// CHECK: %math.round.f (52, 11) 0
func.func @test_unary(%arg0 : f64) -> f64 {
  %0 = math.exp %arg0 : f64
  %1 = math.log %0 : f64
  %2 = math.tanh %1 : f64
  %3 = math.sqrt %2 : f64
  %4 = math.absf %3 : f64
  %5 = math.floor %4 : f64
  return %5 : f64
}

// CHECK-LABEL: fun extern test_pow (
// CHECK: %math.pow (23, 8) 0
func.func @test_pow(%arg0 : f32, %arg1 : f32) -> f32 {
  %0 = math.powf %arg0, %arg1 : f32
  return %0 : f32
}

// rsqrt is composed as 1 / sqrt(x).
// CHECK-LABEL: fun extern test_rsqrt (
// CHECK: %math.rt.sq (23, 8) 0
// CHECK: %math.arith.div (23, 8) 0
func.func @test_rsqrt(%arg0 : f32) -> f32 {
  %0 = math.rsqrt %arg0 : f32
  return %0 : f32
}
