// RUN: mlir-translate --mlir-to-mim %s | FileCheck %s

// Unconditional branch with block arguments.
// CHECK-LABEL: fun extern test_br (
// CHECK: con test_br.bb1_[[BB:[0-9]+]] [_{{[0-9]+}}: %mem.M 0, _{{[0-9]+}}: I64]
// CHECK: test_br.bb1_[[BB]] (_{{[0-9]+}}, _{{[0-9]+}});
func.func @test_br(%arg0 : i64) -> i64 {
  cf.br ^bb1(%arg0 : i64)
^bb1(%x : i64):
  return %x : i64
}

// Conditional branch, successors without arguments: branches directly on the
// block continuations.
// CHECK-LABEL: fun extern test_cond_br (
// CHECK: con test_cond_br.bb1_[[TBB:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: con test_cond_br.bb2_[[FBB:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: (test_cond_br.bb2_[[FBB]], test_cond_br.bb1_[[TBB]])#
func.func @test_cond_br(%cond : i1, %a : i64, %b : i64) -> i64 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  return %a : i64
^bb2:
  return %b : i64
}

// Conditional branch with successor arguments: the successors are wrapped in
// parameterless continuations forwarding the arguments.
// CHECK-LABEL: fun extern test_cond_br_args (
// CHECK: con test_cond_br_args.bb1_[[JOIN:[0-9]+]] [_{{[0-9]+}}: %mem.M 0, _{{[0-9]+}}: I64]
// CHECK: con _[[TW:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: test_cond_br_args.bb1_[[JOIN]] (_{{[0-9]+}}, _{{[0-9]+}});
// CHECK: con _[[FW:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: test_cond_br_args.bb1_[[JOIN]] (_{{[0-9]+}}, _{{[0-9]+}});
// CHECK: (_[[FW]], _[[TW]])#
func.func @test_cond_br_args(%cond : i1, %a : i64, %b : i64) -> i64 {
  cf.cond_br %cond, ^bb1(%a : i64), ^bb1(%b : i64)
^bb1(%x : i64):
  return %x : i64
}

// A loop: block arguments become the loop-carried values of the header
// continuation.
// CHECK-LABEL: fun extern test_loop (
// CHECK: con test_loop.bb1_[[HEADER:[0-9]+]] [_{{[0-9]+}}: %mem.M 0, _{{[0-9]+}}: I64, _{{[0-9]+}}: I64]
// CHECK: con test_loop.bb2_[[BODY:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: %core.wrap.add
// CHECK: test_loop.bb1_[[HEADER]]
// CHECK: con test_loop.bb3_[[EXIT:[0-9]+]] [_{{[0-9]+}}: %mem.M 0]
// CHECK: %core.icmp.xYgLe
// CHECK: (test_loop.bb3_[[EXIT]], test_loop.bb2_[[BODY]])#
// CHECK: test_loop.bb1_[[HEADER]]
func.func @test_loop(%n : i64) -> i64 {
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  cf.br ^header(%zero, %zero : i64, i64)
^header(%i : i64, %acc : i64):
  %cont = arith.cmpi slt, %i, %n : i64
  cf.cond_br %cont, ^body, ^exit
^body:
  %acc2 = arith.addi %acc, %i : i64
  %i2 = arith.addi %i, %one : i64
  cf.br ^header(%i2, %acc2 : i64, i64)
^exit:
  return %acc : i64
}

// Direct call: CPS with a fresh return continuation receiving the results.
// CHECK-LABEL: fun extern test_callee (
func.func @test_callee(%arg0 : i64) -> i64 {
  return %arg0 : i64
}

// CHECK-LABEL: fun extern test_call (
// CHECK: con ret.test_callee_[[RET:[0-9]+]] [_{{[0-9]+}}: %mem.M 0, _{{[0-9]+}}: I64]
// CHECK: %core.wrap.add
// CHECK: test_callee (_{{[0-9]+}}, _{{[0-9]+}}, ret.test_callee_[[RET]]);
func.func @test_call(%arg0 : i64) -> i64 {
  %0 = call @test_callee(%arg0) : (i64) -> i64
  %1 = arith.addi %0, %arg0 : i64
  return %1 : i64
}
