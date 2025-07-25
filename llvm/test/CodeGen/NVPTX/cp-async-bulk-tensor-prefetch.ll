; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx80| FileCheck --check-prefixes=CHECK-PTX %s
; RUN: %if ptxas-12.3 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx80| %ptxas-verify -arch=sm_90 %}

target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %tm, i32 %d0, i64 %ch, i1 %flag);
declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(ptr %tm, i32 %d0, i32 %d1, i64 %ch, i1 %flag);
declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(ptr %tm, i32 %d0, i32 %d1, i32 %d2, i64 %ch, i1 %flag);
declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 %ch, i1 %flag);
declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 %ch, i1 %flag);

declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %tm, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i64 %ch, i1 %f1);
declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i64 %ch, i1 %f1);
declare void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(ptr %tm, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i64 %ch, i1 %f1);

; CHECK-LABEL: cp_async_bulk_tensor_prefetch_tile_1d
define void @cp_async_bulk_tensor_prefetch_tile_1d(ptr %tmap, i32 %d0, i64 %ch) {
; CHECK-PTX-LABEL: cp_async_bulk_tensor_prefetch_tile_1d(
; CHECK-PTX:       {
; CHECK-PTX-NEXT:    .reg .b32 %r<2>;
; CHECK-PTX-NEXT:    .reg .b64 %rd<3>;
; CHECK-PTX-EMPTY:
; CHECK-PTX-NEXT:  // %bb.0:
; CHECK-PTX-NEXT:    ld.param.b64 %rd1, [cp_async_bulk_tensor_prefetch_tile_1d_param_0];
; CHECK-PTX-NEXT:    ld.param.b32 %r1, [cp_async_bulk_tensor_prefetch_tile_1d_param_1];
; CHECK-PTX-NEXT:    ld.param.b64 %rd2, [cp_async_bulk_tensor_prefetch_tile_1d_param_2];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.1d.L2.global.tile [%rd1, {%r1}];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint [%rd1, {%r1}], %rd2;
; CHECK-PTX-NEXT:    ret;
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %tmap, i32 %d0, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d(ptr %tmap, i32 %d0, i64 %ch, i1 1)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_prefetch_tile_2d
define void @cp_async_bulk_tensor_prefetch_tile_2d(i32 %flag, ptr %tmap, i32 %d0, i32 %d1, i64 %ch) {
; CHECK-PTX-LABEL: cp_async_bulk_tensor_prefetch_tile_2d(
; CHECK-PTX:       {
; CHECK-PTX-NEXT:    .reg .b32 %r<3>;
; CHECK-PTX-NEXT:    .reg .b64 %rd<3>;
; CHECK-PTX-EMPTY:
; CHECK-PTX-NEXT:  // %bb.0:
; CHECK-PTX-NEXT:    ld.param.b64 %rd1, [cp_async_bulk_tensor_prefetch_tile_2d_param_1];
; CHECK-PTX-NEXT:    ld.param.b32 %r1, [cp_async_bulk_tensor_prefetch_tile_2d_param_2];
; CHECK-PTX-NEXT:    ld.param.b32 %r2, [cp_async_bulk_tensor_prefetch_tile_2d_param_3];
; CHECK-PTX-NEXT:    ld.param.b64 %rd2, [cp_async_bulk_tensor_prefetch_tile_2d_param_4];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%rd1, {%r1, %r2}];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [%rd1, {%r1, %r2}], %rd2;
; CHECK-PTX-NEXT:    ret;
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(ptr %tmap, i32 %d0, i32 %d1, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d(ptr %tmap, i32 %d0, i32 %d1, i64 %ch, i1 1)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_prefetch_3d
define void @cp_async_bulk_tensor_prefetch_3d(i32 %flag, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i64 %ch) {
; CHECK-PTX-LABEL: cp_async_bulk_tensor_prefetch_3d(
; CHECK-PTX:       {
; CHECK-PTX-NEXT:    .reg .b16 %rs<2>;
; CHECK-PTX-NEXT:    .reg .b32 %r<4>;
; CHECK-PTX-NEXT:    .reg .b64 %rd<3>;
; CHECK-PTX-EMPTY:
; CHECK-PTX-NEXT:  // %bb.0:
; CHECK-PTX-NEXT:    ld.param.b64 %rd1, [cp_async_bulk_tensor_prefetch_3d_param_1];
; CHECK-PTX-NEXT:    ld.param.b32 %r1, [cp_async_bulk_tensor_prefetch_3d_param_2];
; CHECK-PTX-NEXT:    ld.param.b32 %r2, [cp_async_bulk_tensor_prefetch_3d_param_3];
; CHECK-PTX-NEXT:    ld.param.b32 %r3, [cp_async_bulk_tensor_prefetch_3d_param_4];
; CHECK-PTX-NEXT:    ld.param.b64 %rd2, [cp_async_bulk_tensor_prefetch_3d_param_6];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%rd1, {%r1, %r2, %r3}];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%rd1, {%r1, %r2, %r3}], %rd2;
; CHECK-PTX-NEXT:    ld.param.b16 %rs1, [cp_async_bulk_tensor_prefetch_3d_param_5];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.3d.L2.global.im2col [%rd1, {%r1, %r2, %r3}], {%rs1};
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.3d.L2.global.im2col.L2::cache_hint [%rd1, {%r1, %r2, %r3}], {%rs1}, %rd2;
; CHECK-PTX-NEXT:    ret;
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i64 %ch, i1 1)

  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i16 %im2col0, i64 %ch, i1 1)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_prefetch_4d
define void @cp_async_bulk_tensor_prefetch_4d(i32 %flag, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i64 %ch) {
; CHECK-PTX-LABEL: cp_async_bulk_tensor_prefetch_4d(
; CHECK-PTX:       {
; CHECK-PTX-NEXT:    .reg .b16 %rs<3>;
; CHECK-PTX-NEXT:    .reg .b32 %r<5>;
; CHECK-PTX-NEXT:    .reg .b64 %rd<3>;
; CHECK-PTX-EMPTY:
; CHECK-PTX-NEXT:  // %bb.0:
; CHECK-PTX-NEXT:    ld.param.b64 %rd1, [cp_async_bulk_tensor_prefetch_4d_param_1];
; CHECK-PTX-NEXT:    ld.param.b32 %r1, [cp_async_bulk_tensor_prefetch_4d_param_2];
; CHECK-PTX-NEXT:    ld.param.b32 %r2, [cp_async_bulk_tensor_prefetch_4d_param_3];
; CHECK-PTX-NEXT:    ld.param.b32 %r3, [cp_async_bulk_tensor_prefetch_4d_param_4];
; CHECK-PTX-NEXT:    ld.param.b32 %r4, [cp_async_bulk_tensor_prefetch_4d_param_5];
; CHECK-PTX-NEXT:    ld.param.b64 %rd2, [cp_async_bulk_tensor_prefetch_4d_param_8];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%rd1, {%r1, %r2, %r3, %r4}];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint [%rd1, {%r1, %r2, %r3, %r4}], %rd2;
; CHECK-PTX-NEXT:    ld.param.b16 %rs1, [cp_async_bulk_tensor_prefetch_4d_param_6];
; CHECK-PTX-NEXT:    ld.param.b16 %rs2, [cp_async_bulk_tensor_prefetch_4d_param_7];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.4d.L2.global.im2col [%rd1, {%r1, %r2, %r3, %r4}], {%rs1, %rs2};
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.4d.L2.global.im2col.L2::cache_hint [%rd1, {%r1, %r2, %r3, %r4}], {%rs1, %rs2}, %rd2;
; CHECK-PTX-NEXT:    ret;
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i64 %ch, i1 1)

  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i16 %im2col0, i16 %im2col1, i64 %ch, i1 1)
  ret void
}

; CHECK-LABEL: cp_async_bulk_tensor_prefetch_5d
define void @cp_async_bulk_tensor_prefetch_5d(i32 %flag, ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i64 %ch) {
; CHECK-PTX-LABEL: cp_async_bulk_tensor_prefetch_5d(
; CHECK-PTX:       {
; CHECK-PTX-NEXT:    .reg .b16 %rs<4>;
; CHECK-PTX-NEXT:    .reg .b32 %r<6>;
; CHECK-PTX-NEXT:    .reg .b64 %rd<3>;
; CHECK-PTX-EMPTY:
; CHECK-PTX-NEXT:  // %bb.0:
; CHECK-PTX-NEXT:    ld.param.b64 %rd1, [cp_async_bulk_tensor_prefetch_5d_param_1];
; CHECK-PTX-NEXT:    ld.param.b32 %r1, [cp_async_bulk_tensor_prefetch_5d_param_2];
; CHECK-PTX-NEXT:    ld.param.b32 %r2, [cp_async_bulk_tensor_prefetch_5d_param_3];
; CHECK-PTX-NEXT:    ld.param.b32 %r3, [cp_async_bulk_tensor_prefetch_5d_param_4];
; CHECK-PTX-NEXT:    ld.param.b32 %r4, [cp_async_bulk_tensor_prefetch_5d_param_5];
; CHECK-PTX-NEXT:    ld.param.b32 %r5, [cp_async_bulk_tensor_prefetch_5d_param_6];
; CHECK-PTX-NEXT:    ld.param.b64 %rd2, [cp_async_bulk_tensor_prefetch_5d_param_10];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%rd1, {%r1, %r2, %r3, %r4, %r5}];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint [%rd1, {%r1, %r2, %r3, %r4, %r5}], %rd2;
; CHECK-PTX-NEXT:    ld.param.b16 %rs1, [cp_async_bulk_tensor_prefetch_5d_param_7];
; CHECK-PTX-NEXT:    ld.param.b16 %rs2, [cp_async_bulk_tensor_prefetch_5d_param_8];
; CHECK-PTX-NEXT:    ld.param.b16 %rs3, [cp_async_bulk_tensor_prefetch_5d_param_9];
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.5d.L2.global.im2col [%rd1, {%r1, %r2, %r3, %r4, %r5}], {%rs1, %rs2, %rs3};
; CHECK-PTX-NEXT:    cp.async.bulk.prefetch.tensor.5d.L2.global.im2col.L2::cache_hint [%rd1, {%r1, %r2, %r3, %r4, %r5}], {%rs1, %rs2, %rs3}, %rd2;
; CHECK-PTX-NEXT:    ret;
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i64 %ch, i1 1)

  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i64 %ch, i1 0)
  tail call void @llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d(ptr %tmap, i32 %d0, i32 %d1, i32 %d2, i32 %d3, i32 %d4, i16 %im2col0, i16 %im2col1, i16 %im2col2, i64 %ch, i1 1)
  ret void
}
