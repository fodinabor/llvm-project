// RUN: mlir-translate --mlir-to-mim %s | FileCheck %s

// scf.if with a result: both regions jump to a join continuation carrying
// (mem, result).
// CHECK-LABEL: fun extern test_if (
// CHECK: con if.join_[[JOIN:[0-9]+]] [_{{[0-9]+}}: %mem.M 0, _{{[0-9]+}}: I64]
// CHECK: con if.then_[[THEN:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: if.join_[[JOIN]]
// CHECK: con if.else_[[ELSE:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: if.join_[[JOIN]]
// CHECK: (if.else_[[ELSE]], if.then_[[THEN]])#
func.func @test_if(%cond : i1, %a : i64, %b : i64) -> i64 {
  %0 = scf.if %cond -> (i64) {
    %1 = arith.addi %a, %b : i64
    scf.yield %1 : i64
  } else {
    %2 = arith.subi %a, %b : i64
    scf.yield %2 : i64
  }
  return %0 : i64
}

// scf.for with an iter_arg: the header carries (mem, iv, acc), the body
// yields back to the header with the incremented induction variable, and the
// exit receives (mem, acc).
// CHECK-LABEL: fun extern test_for (
// CHECK: con for.head_[[HEAD:[0-9]+]] [_{{[0-9]+}}: %mem.M 0, _{{[0-9]+}}: I64, _{{[0-9]+}}: I64]
// CHECK: con for.body_{{[0-9]+}} [_{{[0-9]+}}: %mem.M 0]
// CHECK: %core.wrap.add
// CHECK: for.head_[[HEAD]]
// CHECK: for.exit_[[EXIT:[0-9]+]]
// CHECK: %core.icmp.xYgLe
// CHECK: con for.exit_[[EXIT]] [_{{[0-9]+}}: %mem.M 0, _{{[0-9]+}}: I64]
// CHECK: for.head_[[HEAD]]
func.func @test_for(%lb : index, %ub : index, %step : index, %init : i64) -> i64 {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i64) {
    %ic = arith.index_cast %i : index to i64
    %next = arith.addi %acc, %ic : i64
    scf.yield %next : i64
  }
  return %r : i64
}

// A nested loop summing a 2-d iteration space.
// CHECK-LABEL: fun extern test_nested (
// CHECK: con for.head_[[H1:[0-9]+]]
// CHECK: con for.head_[[H2:[0-9]+]]
// CHECK: con for.exit_
// CHECK: con for.exit_
func.func @test_nested(%n : index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> (index) {
    %inner = scf.for %j = %c0 to %n step %c1 iter_args(%acc2 = %acc) -> (index) {
      %s = arith.addi %acc2, %j : index
      scf.yield %s : index
    }
    scf.yield %inner : index
  }
  return %r : index
}
