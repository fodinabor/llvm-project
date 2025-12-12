// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: test_addi
func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.addi %arg0, %arg1 : i64
  return %0 : i64
}


// CHECK-LABEL: test_addui_extended
// func.func @test_addui_extended(%arg0 : i64, %arg1 : i64) -> i64 {
//   %sum, %overflow = arith.addui_extended %arg0, %arg1 : i64, i1
//   return %sum : i64
// }

// CHECK-LABEL: test_subi
func.func @test_subi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.subi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_muli
func.func @test_muli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.muli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_mulsi_extended
// func.func @test_mulsi_extended(%arg0 : i32, %arg1 : i32) -> i32 {
//   %low, %high = arith.mulsi_extended %arg0, %arg1 : i32
//   return %high : i32
// }

// CHECK-LABEL: test_mului_extended
// func.func @test_mului_extended(%arg0 : i32, %arg1 : i32) -> i32 {
//   %low, %high = arith.mului_extended %arg0, %arg1 : i32
//   return %high : i32
// }

// CHECK-LABEL: test_divui
// func.func @test_divui(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.divui %arg0, %arg1 : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_divui_exact
// func.func @test_divui_exact(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.divui %arg0, %arg1 exact : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_divsi
// func.func @test_divsi(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.divsi %arg0, %arg1 : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_divsi_exact
// func.func @test_divsi_exact(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.divsi %arg0, %arg1 exact : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_remui
// func.func @test_remui(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.remui %arg0, %arg1 : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_remsi
// func.func @test_remsi(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.remsi %arg0, %arg1 : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_andi
func.func @test_andi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.andi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_ori
func.func @test_ori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.ori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_xori
func.func @test_xori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.xori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_ceildivsi
// func.func @test_ceildivsi(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.ceildivsi %arg0, %arg1 : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_floordivsi
// func.func @test_floordivsi(%arg0 : i64, %arg1 : i64) -> i64 {
//   %0 = arith.floordivsi %arg0, %arg1 : i64
//   return %0 : i64
// }

// CHECK-LABEL: test_shli
func.func @test_shli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrui
func.func @test_shrui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrui_exact
func.func @test_shrui_exact(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrui %arg0, %arg1 exact : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrsi
func.func @test_shrsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: test_shrsi_exact
func.func @test_shrsi_exact(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrsi %arg0, %arg1 exact : i64
  return %0 : i64
}

// CHECK-LABEL: test_negf
func.func @test_negf(%arg0 : f64) -> f64 {
  %0 = arith.negf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: test_addf
func.func @test_addf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.addf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_subf
func.func @test_subf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.subf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_mulf
func.func @test_mulf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.mulf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_divf
func.func @test_divf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_remf
func.func @test_remf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.remf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: test_extui
func.func @test_extui(%arg0 : i32) -> i64 {
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: test_extsi
func.func @test_extsi(%arg0 : i32) -> i64 {
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: test_extf
func.func @test_extf(%arg0 : f32) -> f64 {
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

// CHECK-LABEL: test_trunci
func.func @test_trunci(%arg0 : i32) -> i16 {
  %0 = arith.trunci %arg0 : i32 to i16
  return %0 : i16
}

// CHECK-LABEL: test_truncf
func.func @test_truncf(%arg0 : f32) -> bf16 {
  %0 = arith.truncf %arg0 : f32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: test_truncf_rounding_mode
func.func @test_truncf_rounding_mode(%arg0 : f64) -> (f32, f32, f32, f32, f32) {
  %0 = arith.truncf %arg0 to_nearest_even : f64 to f32
  %1 = arith.truncf %arg0 downward : f64 to f32
  %2 = arith.truncf %arg0 upward : f64 to f32
  %3 = arith.truncf %arg0 toward_zero : f64 to f32
  %4 = arith.truncf %arg0 to_nearest_away : f64 to f32
  return %0, %1, %2, %3, %4 : f32, f32, f32, f32, f32
}

// CHECK-LABEL: test_uitofp
func.func @test_uitofp(%arg0 : i32) -> f32 {
  %0 = arith.uitofp %arg0 : i32 to f32
 return %0 : f32
}

// CHECK-LABEL: test_sitofp
func.func @test_sitofp(%arg0 : i16) -> f64 {
  %0 = arith.sitofp %arg0 : i16 to f64
  return %0 : f64
}

// CHECK-LABEL: test_fptoui
func.func @test_fptoui(%arg0 : bf16) -> i8 {
  %0 = arith.fptoui %arg0 : bf16 to i8
  return %0 : i8
}

// CHECK-LABEL: test_fptosi
func.func @test_fptosi(%arg0 : f64) -> i64 {
  %0 = arith.fptosi %arg0 : f64 to i64
  return %0 : i64
}

// CHECK-LABEL: test_index_cast0
func.func @test_index_cast0(%arg0 : i32) -> index {
  %0 = arith.index_cast %arg0 : i32 to index
  return %0 : index
}

// CHECK-LABEL: test_index_cast1
func.func @test_index_cast1(%arg0 : index) -> i64 {
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}


// CHECK-LABEL: test_index_castui0
func.func @test_index_castui0(%arg0 : i32) -> index {
  %0 = arith.index_castui %arg0 : i32 to index
  return %0 : index
}

// CHECK-LABEL: test_indexui_cast1
func.func @test_indexui_cast1(%arg0 : index) -> i64 {
  %0 = arith.index_castui %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: test_bitcast0
func.func @test_bitcast0(%arg0 : i64) -> f64 {
  %0 = arith.bitcast %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: test_bitcast1
func.func @test_bitcast1(%arg0 : f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: test_cmpi
func.func @test_cmpi(%arg0 : i64, %arg1 : i64) -> i1 {
  %0 = arith.cmpi ne, %arg0, %arg1 : i64
  return %0 : i1
}

// CHECK-LABEL: test_cmpf
func.func @test_cmpf(%arg0 : f64, %arg1 : f64) -> i1 {
  %0 = arith.cmpf oeq, %arg0, %arg1 : f64
  return %0 : i1
}

// CHECK-LABEL: test_index_cast
func.func @test_index_cast(%arg0 : index) -> i64 {
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: func @bitcast(
func.func @bitcast(%arg : f32) -> i32 {
  %res = arith.bitcast %arg : f32 to i32
  return %res : i32
}

// CHECK-LABEL: test_constant
func.func @test_constant() -> () {
  // CHECK: %c42_i32 = arith.constant 42 : i32
  %0 = "arith.constant"(){value = 42 : i32} : () -> i32

  // CHECK: %c42_i32_0 = arith.constant 42 : i32
  %1 = arith.constant 42 : i32

  // CHECK: %c43 = arith.constant {crazy = "func.foo"} 43 : index
  %2 = arith.constant {crazy = "func.foo"} 43: index

  // CHECK: %cst = arith.constant 4.300000e+01 : bf16
  %3 = arith.constant 43.0 : bf16

  // CHECK: %true = arith.constant true
  %7 = arith.constant true

  // CHECK: %false = arith.constant false
  %8 = arith.constant false

  // CHECK: %c-1_i128 = arith.constant -1 : i128
  //%9 = arith.constant 340282366920938463463374607431768211455 : i128

  // CHECK: %c85070591730234615865843651857942052864_i128 = arith.constant 85070591730234615865843651857942052864 : i128
  //%10 = arith.constant 85070591730234615865843651857942052864 : i128

  return
}

// CHECK-LABEL: func @maximum
func.func @maximum(%f1: f32, %f2: f32,
               %i1: i32, %i2: i32) {
  %maximum_float = arith.maximumf %f1, %f2 : f32
  %maxnum_float = arith.maxnumf %f1, %f2 : f32
  %max_signed = arith.maxsi %i1, %i2 : i32
  %max_unsigned = arith.maxui %i1, %i2 : i32
  return
}

// CHECK-LABEL: func @minimum
func.func @minimum(%f1: f32, %f2: f32,
               %i1: i32, %i2: i32) {
  %minimum_float = arith.minimumf %f1, %f2 : f32
  %minnum_float = arith.minnumf %f1, %f2 : f32
  %min_signed = arith.minsi %i1, %i2 : i32
  %min_unsigned = arith.minui %i1, %i2 : i32
  return
}

// CHECK-LABEL: @fastmath
func.func @fastmath(%arg0: f32, %arg1: f32, %arg2: i32) -> i1 {
// CHECK: {{.*}} = arith.addf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.subf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.mulf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.divf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.remf %arg0, %arg1 fastmath<fast> : f32
// CHECK: {{.*}} = arith.negf %arg0 fastmath<fast> : f32
  %0 = arith.addf %arg0, %arg1 fastmath<fast> : f32
  %1 = arith.subf %arg0, %0 fastmath<fast> : f32
  %2 = arith.mulf %arg0, %1 fastmath<fast> : f32
  %3 = arith.divf %arg0, %2 fastmath<fast> : f32
  %4 = arith.remf %arg0, %3 fastmath<fast> : f32
  %5 = arith.negf %4 fastmath<fast> : f32
  %51 = arith.subf %5, %5 fastmath<fast> : f32
// CHECK: {{.*}} = arith.addf %arg0, %arg1 : f32
  %6 = arith.addf %arg0, %arg1 fastmath<none> : f32
// CHECK: {{.*}} = arith.addf %arg0, %arg1 fastmath<nnan,ninf> : f32
  %7 = arith.addf %arg0, %6 fastmath<nnan,ninf> : f32
// CHECK: {{.*}} = arith.mulf %arg0, %arg1 fastmath<fast> : f32
  %8 = arith.mulf %arg0, %7 fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn> : f32
// CHECK: {{.*}} = arith.cmpf oeq, %arg0, %arg1 fastmath<fast> : f32
  %9 = arith.cmpf oeq, %51, %8 fastmath<fast> : f32
  return %9 : i1
}

// CHECK-LABEL: @intflags_func
func.func @intflags_func(%arg0: i64, %arg1: i64) {
  // CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} overflow<nsw> : i64
  %0 = arith.addi %arg0, %arg1 overflow<nsw> : i64
  // CHECK: %{{.*}} = arith.subi %{{.*}}, %{{.*}} overflow<nuw> : i64
  %1 = arith.subi %arg0, %arg1 overflow<nuw> : i64
  // CHECK: %{{.*}} = arith.muli %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
  %2 = arith.muli %arg0, %arg1 overflow<nsw, nuw> : i64
  // CHECK: %{{.*}} = arith.shli %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
  %3 = arith.shli %arg0, %arg1 overflow<nsw, nuw> : i64
  // CHECK: %{{.*}} = arith.trunci %{{.*}} overflow<nsw, nuw> : i64 to i32
  %4 = arith.trunci %arg0 overflow<nsw, nuw> : i64 to i32
  return
}
