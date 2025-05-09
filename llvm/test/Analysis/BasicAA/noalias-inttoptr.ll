; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "p:64:64"

declare void @escape(ptr %ptr)

; Verify that unescaped noalias parameter does not alias inttoptr
define void @test1(ptr noalias %P, i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test1:
  ; CHECK: NoAlias:	i8* %P, i8* %Q
  %Q = inttoptr i64 %Q_as_int to ptr
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}

; Verify that unescaped alloca does not alias inttoptr
define void @test2(i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test2:
  ; CHECK: NoAlias:	i8* %P, i8* %Q
  %P = alloca i8
  %Q = inttoptr i64 %Q_as_int to ptr
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}

; Verify that escaped noalias parameter may alias inttoptr
define void @test3(ptr noalias %P, i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test3:
  ; CHECK: MayAlias:	i8* %P, i8* %Q
  call void @escape(ptr %P)
  %Q = inttoptr i64 %Q_as_int to ptr
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}

; Verify that escaped alloca may alias inttoptr
define void @test4(i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test4:
  ; CHECK: MayAlias:	i8* %P, i8* %Q
  %P = alloca i8
  call void @escape(ptr %P)
  %Q = inttoptr i64 %Q_as_int to ptr
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}


; Verify that global may alias inttoptr
@G = external global i8
define void @test5(i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test5:
  ; CHECK: MayAlias:	i8* %Q, i8* @G
  %Q = inttoptr i64 %Q_as_int to ptr
  store i8 0, ptr @G
  store i8 1, ptr %Q
  ret void
}

; Verify that extractvalue of a coerced ptr argument array are NoAlias a function local object
define void @test_extractvalue([2 x ptr] %Q.coerce) {
  ; CHECK-LABEL: Function: test_extractvalue:
  ; CHECK: NoAlias:    i8* %P, i8* %Q
  %P = alloca i8
  %Q = extractvalue [2 x ptr] %Q.coerce, 1
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}

; Same as test_extractvalue with an escape of %P
define void @test_extractvalue_escape([2 x ptr] %Q.coerce) {
  ; CHECK-LABEL: Function: test_extractvalue_escape:
  ; CHECK: NoAlias:    i8* %P, i8* %Q
  %P = alloca i8
  call void @escape(ptr %P)
  %Q = extractvalue [2 x ptr] %Q.coerce, 1
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}

; Verify that extractvalue of a coerced ptr argument array are NoAlias a function local object
define void @test_extractvalue_int([2 x i64] %Q.coerce) {
  ; CHECK-LABEL: Function: test_extractvalue_int:
  ; CHECK: NoAlias:    i8* %P, i8* %Q
  %P = alloca i8
  %Q_as_int = extractvalue [2 x i64] %Q.coerce, 1
  %Q = inttoptr i64 %Q_as_int to ptr
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}

; Same as test_extractvalue_int with an escape of %P
define void @test_extractvalue_int_escape([2 x i64] %Q.coerce) {
  ; CHECK-LABEL: Function: test_extractvalue_int_escape:
  ; CHECK: MayAlias:    i8* %P, i8* %Q
  %P = alloca i8
  call void @escape(ptr %P)
  %Q_as_int = extractvalue [2 x i64] %Q.coerce, 1
  %Q = inttoptr i64 %Q_as_int to ptr
  store i8 0, ptr %P
  store i8 1, ptr %Q
  ret void
}