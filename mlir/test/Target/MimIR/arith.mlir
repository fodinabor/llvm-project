// RUN: mlir-translate --mlir-to-mimir %s -split-input-file | FileCheck %s

func.func @recurse_fold_traits(%arg0 : i32) -> i32 {
  %cst0 = arith.constant 0 : i32
  %res = arith.addi %cst0, %arg0 : i32
  return %res : i32
}

func.func @dont_fold(%arg0 : i32) -> i32 {
  %cst0 = arith.constant 2 : i32
  %res = arith.addi %cst0, %arg0 : i32
  return %res : i32
}

