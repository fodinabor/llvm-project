// RUN: mlir-translate --mlir-to-mim %s | FileCheck %s

// Note: MimIR is a graph IR — values that do not contribute to a function's
// result are not part of the output. Hence all tested ops feed the return.

// CHECK-LABEL: fun extern test_addi (
// CHECK: %core.wrap.add 0 0
func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.addi %arg0, %arg1 : i64
  return %0 : i64
}

// TODO: not supported yet: arith.addui_extended, arith.mulsi_extended,
// arith.mului_extended (multi-result ops).
// TODO: not supported yet: arith.ceildivsi, arith.floordivsi (no direct
// %core.div counterpart).

// Division by zero is a visible side effect: %core.div consumes and produces
// the %mem.M token that is threaded through every function.
// CHECK-LABEL: fun extern test_divsi (
// CHECK: %core.div.sdiv
func.func @test_divsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.divsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_divui (
// CHECK: %core.div.udiv
func.func @test_divui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.divui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_remsi (
// CHECK: %core.div.srem
func.func @test_remsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.remsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_remui (
// CHECK: %core.div.urem
func.func @test_remui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.remui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_subi (
// CHECK: %core.wrap.sub 0 0
func.func @test_subi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.subi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_muli (
// CHECK: %core.wrap.mul 0 0
func.func @test_muli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.muli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_andi (
// CHECK: %core.bit2.and_ 0 0
func.func @test_andi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.andi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_ori (
// CHECK: %core.bit2.or_ 0 0
func.func @test_ori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.ori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_xori (
// CHECK: %core.bit2.xor_ 0 0
func.func @test_xori(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.xori %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_shli (
// CHECK: %core.wrap.shl 0 0
func.func @test_shli(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shli %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_shrui (
// CHECK: %core.shr.l 0
func.func @test_shrui(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrui %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_shrui_exact (
// CHECK: %core.shr.l 0
func.func @test_shrui_exact(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrui %arg0, %arg1 exact : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_shrsi (
// CHECK: %core.shr.a 0
func.func @test_shrsi(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrsi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_shrsi_exact (
// CHECK: %core.shr.a 0
func.func @test_shrsi_exact(%arg0 : i64, %arg1 : i64) -> i64 {
  %0 = arith.shrsi %arg0, %arg1 exact : i64
  return %0 : i64
}

// negf is normalized to `0 - x` by MimIR.
// CHECK-LABEL: fun extern test_negf (
// CHECK: %math.arith.sub (52, 11) 0 (0:(%math.F (52, 11)),
func.func @test_negf(%arg0 : f64) -> f64 {
  %0 = arith.negf %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_addf (
// CHECK: %math.arith.add (52, 11) 0
func.func @test_addf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.addf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_subf (
// CHECK: %math.arith.sub (52, 11) 0
func.func @test_subf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.subf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_mulf (
// CHECK: %math.arith.mul (52, 11) 0
func.func @test_mulf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.mulf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_divf (
// CHECK: %math.arith.div (52, 11) 0
func.func @test_divf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.divf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_remf (
// CHECK: %math.arith.rem (52, 11) 0
func.func @test_remf(%arg0 : f64, %arg1 : f64) -> f64 {
  %0 = arith.remf %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_extui (
// CHECK: %core.conv.u i32 0
func.func @test_extui(%arg0 : i32) -> i64 {
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_extsi (
// CHECK: %core.conv.s i32 0
func.func @test_extsi(%arg0 : i32) -> i64 {
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_extf (
// CHECK: %math.conv.f2f (23, 8) (52, 11)
func.func @test_extf(%arg0 : f32) -> f64 {
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_trunci (
// CHECK: %core.conv.u i32 i16
func.func @test_trunci(%arg0 : i32) -> i16 {
  %0 = arith.trunci %arg0 : i32 to i16
  return %0 : i16
}

// CHECK-LABEL: fun extern test_truncf (
// CHECK: %math.conv.f2f (23, 8) (7, 8)
func.func @test_truncf(%arg0 : f32) -> bf16 {
  %0 = arith.truncf %arg0 : f32 to bf16
  return %0 : bf16
}

// All five truncf variants hash to the same conversion (rounding modes are not
// translated yet), so a single f2f remains and the return packs it five times.
// CHECK-LABEL: fun extern test_truncf_rounding_mode (
// CHECK: %math.conv.f2f (52, 11) (23, 8)
// CHECK: (_{{[0-9]+}}, _[[V:[0-9]+]], _[[V]], _[[V]], _[[V]], _[[V]]);
func.func @test_truncf_rounding_mode(%arg0 : f64) -> (f32, f32, f32, f32, f32) {
  %0 = arith.truncf %arg0 to_nearest_even : f64 to f32
  %1 = arith.truncf %arg0 downward : f64 to f32
  %2 = arith.truncf %arg0 upward : f64 to f32
  %3 = arith.truncf %arg0 toward_zero : f64 to f32
  %4 = arith.truncf %arg0 to_nearest_away : f64 to f32
  return %0, %1, %2, %3, %4 : f32, f32, f32, f32, f32
}

// CHECK-LABEL: fun extern test_uitofp (
// CHECK: %math.conv.u2f i32 (23, 8)
func.func @test_uitofp(%arg0 : i32) -> f32 {
  %0 = arith.uitofp %arg0 : i32 to f32
 return %0 : f32
}

// CHECK-LABEL: fun extern test_sitofp (
// CHECK: %math.conv.s2f i16 (52, 11)
func.func @test_sitofp(%arg0 : i16) -> f64 {
  %0 = arith.sitofp %arg0 : i16 to f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_fptoui (
// CHECK: %math.conv.f2u (7, 8) i8
func.func @test_fptoui(%arg0 : bf16) -> i8 {
  %0 = arith.fptoui %arg0 : bf16 to i8
  return %0 : i8
}

// CHECK-LABEL: fun extern test_fptosi (
// CHECK: %math.conv.f2s (52, 11) 0
func.func @test_fptosi(%arg0 : f64) -> i64 {
  %0 = arith.fptosi %arg0 : f64 to i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_index_cast0 (
// CHECK: %core.conv.s i32 0
func.func @test_index_cast0(%arg0 : i32) -> index {
  %0 = arith.index_cast %arg0 : i32 to index
  return %0 : index
}

// index and i64 translate to the same type; the cast folds away.
// CHECK-LABEL: fun extern test_index_cast1 (
// CHECK-NOT: %core.conv
func.func @test_index_cast1(%arg0 : index) -> i64 {
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_index_castui0 (
// CHECK: %core.conv.u i32 0
func.func @test_index_castui0(%arg0 : i32) -> index {
  %0 = arith.index_castui %arg0 : i32 to index
  return %0 : index
}

// CHECK-LABEL: fun extern test_indexui_cast1 (
// CHECK-NOT: %core.conv
func.func @test_indexui_cast1(%arg0 : index) -> i64 {
  %0 = arith.index_castui %arg0 : index to i64
  return %0 : i64
}

// CHECK-LABEL: fun extern test_bitcast0 (
// CHECK: %core.bitcast I64 (%math.F (52, 11))
func.func @test_bitcast0(%arg0 : i64) -> f64 {
  %0 = arith.bitcast %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: fun extern test_bitcast1 (
// CHECK: %core.bitcast (%math.F (23, 8)) I32
func.func @test_bitcast1(%arg0 : f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: fun extern test_cmpi (
// CHECK: %core.icmp.XYGLe 0
func.func @test_cmpi(%arg0 : i64, %arg1 : i64) -> i1 {
  %0 = arith.cmpi ne, %arg0, %arg1 : i64
  return %0 : i1
}

// CHECK-LABEL: fun extern test_cmpf (
// CHECK: %math.cmp.uglE (52, 11) 0
func.func @test_cmpf(%arg0 : f64, %arg1 : f64) -> i1 {
  %0 = arith.cmpf oeq, %arg0, %arg1 : f64
  return %0 : i1
}

// CHECK-LABEL: fun extern test_select (
// CHECK: (_{{[0-9]+}}, _{{[0-9]+}})#(_{{[0-9]+}})
func.func @test_select(%cond : i1, %a : i64, %b : i64) -> i64 {
  %0 = arith.select %cond, %a, %b : i64
  return %0 : i64
}

// CHECK-LABEL: fun extern bitcast (
// CHECK: %core.bitcast (%math.F (23, 8)) I32
func.func @bitcast(%arg : f32) -> i32 {
  %res = arith.bitcast %arg : f32 to i32
  return %res : i32
}

// CHECK-LABEL: fun extern test_constant (
func.func @test_constant() -> (i32, index, bf16, i1, i1) {
  %0 = "arith.constant"(){value = 42 : i32} : () -> i32
  %1 = arith.constant {crazy = "func.foo"} 43: index
  %2 = arith.constant 43.0 : bf16
  %3 = arith.constant true
  %4 = arith.constant false
  return %0, %1, %2, %3, %4 : i32, index, bf16, i1, i1
}

// CHECK-LABEL: fun extern maximum (
// CHECK: %math.extrema.iM (23, 8) 0
// CHECK: %math.extrema.IM (23, 8) 0
// CHECK: %core.extrema.SM i32
// CHECK: %core.extrema.sM i32
func.func @maximum(%f1: f32, %f2: f32,
               %i1: i32, %i2: i32) -> (f32, f32, i32, i32) {
  %maximum_float = arith.maximumf %f1, %f2 : f32
  %maxnum_float = arith.maxnumf %f1, %f2 : f32
  %max_signed = arith.maxsi %i1, %i2 : i32
  %max_unsigned = arith.maxui %i1, %i2 : i32
  return %maximum_float, %maxnum_float, %max_signed, %max_unsigned
      : f32, f32, i32, i32
}

// CHECK-LABEL: fun extern minimum (
// CHECK: %math.extrema.im (23, 8) 0
// CHECK: %math.extrema.Im (23, 8) 0
// CHECK: %core.extrema.Sm i32
// CHECK: %core.extrema.sm i32
func.func @minimum(%f1: f32, %f2: f32,
               %i1: i32, %i2: i32) -> (f32, f32, i32, i32) {
  %minimum_float = arith.minimumf %f1, %f2 : f32
  %minnum_float = arith.minnumf %f1, %f2 : f32
  %min_signed = arith.minsi %i1, %i2 : i32
  %min_unsigned = arith.minui %i1, %i2 : i32
  return %minimum_float, %minnum_float, %min_signed, %min_unsigned
      : f32, f32, i32, i32
}

// Fastmath flags map to %math.Mode bits (nnan|ninf = 3, all = 127). Note that
// MimIR folds `x - x` to 0 under fast math, collapsing the fast chain.
// CHECK-LABEL: fun extern fastmath (
// CHECK: %math.arith.add (23, 8) 0
// CHECK: %math.arith.add (23, 8) 3
// CHECK: %math.arith.mul (23, 8) 127
// CHECK: %math.cmp.uglE (23, 8) 127
func.func @fastmath(%arg0: f32, %arg1: f32, %arg2: i32) -> i1 {
  %0 = arith.addf %arg0, %arg1 fastmath<fast> : f32
  %1 = arith.subf %arg0, %0 fastmath<fast> : f32
  %2 = arith.mulf %arg0, %1 fastmath<fast> : f32
  %3 = arith.divf %arg0, %2 fastmath<fast> : f32
  %4 = arith.remf %arg0, %3 fastmath<fast> : f32
  %5 = arith.negf %4 fastmath<fast> : f32
  %51 = arith.subf %5, %5 fastmath<fast> : f32
  %6 = arith.addf %arg0, %arg1 fastmath<none> : f32
  %7 = arith.addf %arg0, %6 fastmath<nnan,ninf> : f32
  %8 = arith.mulf %arg0, %7 fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn> : f32
  %9 = arith.cmpf oeq, %51, %8 fastmath<fast> : f32
  return %9 : i1
}

// Overflow flags map to %core.Mode bits (nsw = 1, nuw = 2, nsw|nuw = 3).
// CHECK-LABEL: fun extern intflags_func (
// CHECK: %core.wrap.add 0 1
// CHECK: %core.wrap.sub 0 2
// CHECK: %core.wrap.mul 0 3
// CHECK: %core.wrap.shl 0 3
// CHECK: %core.conv.s 0 i32
func.func @intflags_func(%arg0: i64, %arg1: i64) -> (i64, i64, i64, i64, i32) {
  %0 = arith.addi %arg0, %arg1 overflow<nsw> : i64
  %1 = arith.subi %arg0, %arg1 overflow<nuw> : i64
  %2 = arith.muli %arg0, %arg1 overflow<nsw, nuw> : i64
  %3 = arith.shli %arg0, %arg1 overflow<nsw, nuw> : i64
  %4 = arith.trunci %arg0 overflow<nsw, nuw> : i64 to i32
  return %0, %1, %2, %3, %4 : i64, i64, i64, i64, i32
}
