; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=amdgcn < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() nounwind readnone

define amdgpu_kernel void @test_umul24_i32(ptr addrspace(1) %out, i32 %a, i32 %b) {
; SI-LABEL: test_umul24_i32:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s7, 0xf000
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s2, s2, 0xffffff
; SI-NEXT:    s_and_b32 s3, s3, 0xffffff
; SI-NEXT:    s_mul_i32 s2, s2, s3
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_mov_b32 s4, s0
; SI-NEXT:    s_mov_b32 s5, s1
; SI-NEXT:    v_mov_b32_e32 v0, s2
; SI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i32:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s7, 0xf000
; VI-NEXT:    s_mov_b32 s6, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_mov_b32 s4, s0
; VI-NEXT:    s_mov_b32 s5, s1
; VI-NEXT:    s_and_b32 s0, s2, 0xffffff
; VI-NEXT:    s_and_b32 s1, s3, 0xffffff
; VI-NEXT:    s_mul_i32 s0, s0, s1
; VI-NEXT:    v_mov_b32_e32 v0, s0
; VI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i32:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    s_mov_b32 s7, 0xf000
; GFX9-NEXT:    s_mov_b32 s6, -1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_mov_b32 s4, s0
; GFX9-NEXT:    s_mov_b32 s5, s1
; GFX9-NEXT:    s_and_b32 s0, s2, 0xffffff
; GFX9-NEXT:    s_and_b32 s1, s3, 0xffffff
; GFX9-NEXT:    s_mul_i32 s0, s0, s1
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX9-NEXT:    s_endpgm
entry:
  %0 = shl i32 %a, 8
  %a_24 = lshr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = lshr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  store i32 %2, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umul24_i16_sext(ptr addrspace(1) %out, i16 %a, i16 %b) {
; SI-LABEL: test_umul24_i16_sext:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dword s2, s[4:5], 0xb
; SI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_lshr_b32 s4, s2, 16
; SI-NEXT:    s_mul_i32 s2, s2, s4
; SI-NEXT:    s_sext_i32_i16 s4, s2
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i16_sext:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dword s6, s[4:5], 0x2c
; VI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_lshr_b32 s4, s6, 16
; VI-NEXT:    s_mul_i32 s6, s6, s4
; VI-NEXT:    s_sext_i32_i16 s4, s6
; VI-NEXT:    v_mov_b32_e32 v0, s4
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i16_sext:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dword s6, s[4:5], 0x2c
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_lshr_b32 s4, s6, 16
; GFX9-NEXT:    s_mul_i32 s6, s6, s4
; GFX9-NEXT:    s_sext_i32_i16 s4, s6
; GFX9-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
entry:
  %mul = mul i16 %a, %b
  %ext = sext i16 %mul to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umul24_i16_vgpr_sext(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; SI-LABEL: test_umul24_i16_vgpr_sext:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s7, 0xf000
; SI-NEXT:    s_mov_b32 s10, 0
; SI-NEXT:    v_lshlrev_b32_e32 v2, 1, v0
; SI-NEXT:    v_mov_b32_e32 v3, 0
; SI-NEXT:    v_lshlrev_b32_e32 v0, 1, v1
; SI-NEXT:    s_mov_b32 s11, s7
; SI-NEXT:    v_mov_b32_e32 v1, v3
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b64 s[8:9], s[2:3]
; SI-NEXT:    buffer_load_ushort v2, v[2:3], s[8:11], 0 addr64
; SI-NEXT:    buffer_load_ushort v0, v[0:1], s[8:11], 0 addr64
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_mov_b32 s4, s0
; SI-NEXT:    s_mov_b32 s5, s1
; SI-NEXT:    s_waitcnt vmcnt(0)
; SI-NEXT:    v_mul_u32_u24_e32 v0, v2, v0
; SI-NEXT:    v_bfe_i32 v0, v0, 0, 16
; SI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i16_vgpr_sext:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    v_lshlrev_b32_e32 v0, 1, v0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    v_add_u32_e32 v2, vcc, s2, v0
; VI-NEXT:    v_addc_u32_e32 v3, vcc, 0, v3, vcc
; VI-NEXT:    v_lshlrev_b32_e32 v0, 1, v1
; VI-NEXT:    v_mov_b32_e32 v1, s3
; VI-NEXT:    v_add_u32_e32 v0, vcc, s2, v0
; VI-NEXT:    v_addc_u32_e32 v1, vcc, 0, v1, vcc
; VI-NEXT:    flat_load_ushort v2, v[2:3]
; VI-NEXT:    flat_load_ushort v0, v[0:1]
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    v_mul_lo_u16_e32 v0, v2, v0
; VI-NEXT:    v_bfe_i32 v0, v0, 0, 16
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i16_vgpr_sext:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    v_lshlrev_b32_e32 v0, 1, v0
; GFX9-NEXT:    v_lshlrev_b32_e32 v1, 1, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    global_load_ushort v2, v0, s[2:3]
; GFX9-NEXT:    global_load_ushort v3, v1, s[2:3]
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mul_lo_u16_e32 v0, v2, v3
; GFX9-NEXT:    v_bfe_i32 v0, v0, 0, 16
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.y = call i32 @llvm.amdgcn.workitem.id.y()
  %ptr_a = getelementptr i16, ptr addrspace(1) %in, i32 %tid.x
  %ptr_b = getelementptr i16, ptr addrspace(1) %in, i32 %tid.y
  %a = load i16, ptr addrspace(1) %ptr_a
  %b = load i16, ptr addrspace(1) %ptr_b
  %mul = mul i16 %a, %b
  %val = sext i16 %mul to i32
  store i32 %val, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umul24_i16(ptr addrspace(1) %out, i16 %a, i16 %b) {
; SI-LABEL: test_umul24_i16:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dword s2, s[4:5], 0xb
; SI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_lshr_b32 s4, s2, 16
; SI-NEXT:    s_mul_i32 s2, s2, s4
; SI-NEXT:    s_and_b32 s4, s2, 0xffff
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i16:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dword s6, s[4:5], 0x2c
; VI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_lshr_b32 s4, s6, 16
; VI-NEXT:    s_mul_i32 s6, s6, s4
; VI-NEXT:    s_and_b32 s4, s6, 0xffff
; VI-NEXT:    v_mov_b32_e32 v0, s4
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i16:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dword s6, s[4:5], 0x2c
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_lshr_b32 s4, s6, 16
; GFX9-NEXT:    s_mul_i32 s6, s6, s4
; GFX9-NEXT:    s_and_b32 s4, s6, 0xffff
; GFX9-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
entry:
  %mul = mul i16 %a, %b
  %ext = zext i16 %mul to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umul24_i16_vgpr(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; SI-LABEL: test_umul24_i16_vgpr:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s7, 0xf000
; SI-NEXT:    s_mov_b32 s10, 0
; SI-NEXT:    v_lshlrev_b32_e32 v2, 1, v0
; SI-NEXT:    v_mov_b32_e32 v3, 0
; SI-NEXT:    v_lshlrev_b32_e32 v0, 1, v1
; SI-NEXT:    s_mov_b32 s11, s7
; SI-NEXT:    v_mov_b32_e32 v1, v3
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b64 s[8:9], s[2:3]
; SI-NEXT:    buffer_load_ushort v2, v[2:3], s[8:11], 0 addr64
; SI-NEXT:    buffer_load_ushort v0, v[0:1], s[8:11], 0 addr64
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_mov_b32 s4, s0
; SI-NEXT:    s_mov_b32 s5, s1
; SI-NEXT:    s_waitcnt vmcnt(0)
; SI-NEXT:    v_mul_u32_u24_e32 v0, v2, v0
; SI-NEXT:    v_and_b32_e32 v0, 0xffff, v0
; SI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i16_vgpr:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    v_lshlrev_b32_e32 v0, 1, v0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    v_add_u32_e32 v2, vcc, s2, v0
; VI-NEXT:    v_addc_u32_e32 v3, vcc, 0, v3, vcc
; VI-NEXT:    v_lshlrev_b32_e32 v0, 1, v1
; VI-NEXT:    v_mov_b32_e32 v1, s3
; VI-NEXT:    v_add_u32_e32 v0, vcc, s2, v0
; VI-NEXT:    v_addc_u32_e32 v1, vcc, 0, v1, vcc
; VI-NEXT:    flat_load_ushort v2, v[2:3]
; VI-NEXT:    flat_load_ushort v0, v[0:1]
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    v_mul_lo_u16_e32 v0, v2, v0
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i16_vgpr:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    v_lshlrev_b32_e32 v0, 1, v0
; GFX9-NEXT:    v_lshlrev_b32_e32 v1, 1, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    global_load_ushort v2, v0, s[2:3]
; GFX9-NEXT:    global_load_ushort v3, v1, s[2:3]
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mul_lo_u16_e32 v0, v2, v3
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.y = call i32 @llvm.amdgcn.workitem.id.y()
  %ptr_a = getelementptr i16, ptr addrspace(1) %in, i32 %tid.x
  %ptr_b = getelementptr i16, ptr addrspace(1) %in, i32 %tid.y
  %a = load i16, ptr addrspace(1) %ptr_a
  %b = load i16, ptr addrspace(1) %ptr_b
  %mul = mul i16 %a, %b
  %val = zext i16 %mul to i32
  store i32 %val, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umul24_i8_vgpr(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b) {
; SI-LABEL: test_umul24_i8_vgpr:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    v_mov_b32_e32 v3, v0
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0xd
; SI-NEXT:    s_mov_b32 s11, 0xf000
; SI-NEXT:    s_mov_b32 s14, 0
; SI-NEXT:    v_mov_b32_e32 v4, 0
; SI-NEXT:    s_mov_b32 s15, s11
; SI-NEXT:    v_mov_b32_e32 v2, v4
; SI-NEXT:    s_mov_b64 s[6:7], s[14:15]
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b64 s[12:13], s[2:3]
; SI-NEXT:    buffer_load_ubyte v0, v[3:4], s[12:15], 0 addr64
; SI-NEXT:    buffer_load_ubyte v1, v[1:2], s[4:7], 0 addr64
; SI-NEXT:    s_mov_b32 s10, -1
; SI-NEXT:    s_mov_b32 s8, s0
; SI-NEXT:    s_mov_b32 s9, s1
; SI-NEXT:    s_waitcnt vmcnt(0)
; SI-NEXT:    v_mul_u32_u24_e32 v0, v0, v1
; SI-NEXT:    v_bfe_i32 v0, v0, 0, 8
; SI-NEXT:    buffer_store_dword v0, off, s[8:11], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i8_vgpr:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    s_load_dwordx2 s[4:5], s[4:5], 0x34
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    v_add_u32_e32 v2, vcc, s2, v0
; VI-NEXT:    v_addc_u32_e32 v3, vcc, 0, v3, vcc
; VI-NEXT:    v_mov_b32_e32 v4, s5
; VI-NEXT:    v_add_u32_e32 v0, vcc, s4, v1
; VI-NEXT:    v_addc_u32_e32 v1, vcc, 0, v4, vcc
; VI-NEXT:    flat_load_ubyte v2, v[2:3]
; VI-NEXT:    flat_load_ubyte v0, v[0:1]
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt vmcnt(0)
; VI-NEXT:    v_mul_lo_u16_e32 v0, v2, v0
; VI-NEXT:    v_bfe_i32 v0, v0, 0, 8
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i8_vgpr:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    s_load_dwordx2 s[6:7], s[4:5], 0x34
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    global_load_ubyte v2, v0, s[2:3]
; GFX9-NEXT:    global_load_ubyte v3, v1, s[6:7]
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mul_lo_u16_e32 v0, v2, v3
; GFX9-NEXT:    v_bfe_i32 v0, v0, 0, 8
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
entry:
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.y = call i32 @llvm.amdgcn.workitem.id.y()
  %a.ptr = getelementptr i8, ptr addrspace(1) %a, i32 %tid.x
  %b.ptr = getelementptr i8, ptr addrspace(1) %b, i32 %tid.y
  %a.l = load i8, ptr addrspace(1) %a.ptr
  %b.l = load i8, ptr addrspace(1) %b.ptr
  %mul = mul i8 %a.l, %b.l
  %ext = sext i8 %mul to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umulhi24_i32_i64(ptr addrspace(1) %out, i32 %a, i32 %b) {
; SI-LABEL: test_umulhi24_i32_i64:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s7, 0xf000
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b32 s4, s0
; SI-NEXT:    s_mov_b32 s5, s1
; SI-NEXT:    v_mov_b32_e32 v0, s3
; SI-NEXT:    v_mul_hi_u32_u24_e32 v0, s2, v0
; SI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umulhi24_i32_i64:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s7, 0xf000
; VI-NEXT:    s_mov_b32 s6, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, s3
; VI-NEXT:    s_mov_b32 s4, s0
; VI-NEXT:    s_mov_b32 s5, s1
; VI-NEXT:    v_mul_hi_u32_u24_e32 v0, s2, v0
; VI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umulhi24_i32_i64:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    s_mov_b32 s7, 0xf000
; GFX9-NEXT:    s_mov_b32 s6, -1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_mov_b32 s4, s0
; GFX9-NEXT:    s_mov_b32 s5, s1
; GFX9-NEXT:    s_and_b32 s0, s2, 0xffffff
; GFX9-NEXT:    s_and_b32 s1, s3, 0xffffff
; GFX9-NEXT:    s_mul_hi_u32 s0, s0, s1
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX9-NEXT:    s_endpgm
entry:
  %a.24 = and i32 %a, 16777215
  %b.24 = and i32 %b, 16777215
  %a.24.i64 = zext i32 %a.24 to i64
  %b.24.i64 = zext i32 %b.24 to i64
  %mul48 = mul i64 %a.24.i64, %b.24.i64
  %mul48.hi = lshr i64 %mul48, 32
  %mul24hi = trunc i64 %mul48.hi to i32
  store i32 %mul24hi, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umulhi24(ptr addrspace(1) %out, i64 %a, i64 %b) {
; SI-LABEL: test_umulhi24:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s3, s[4:5], 0xd
; SI-NEXT:    s_mov_b32 s7, 0xf000
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_mov_b32 s4, s0
; SI-NEXT:    s_mov_b32 s5, s1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    v_mov_b32_e32 v0, s3
; SI-NEXT:    v_mul_hi_u32_u24_e32 v0, s2, v0
; SI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umulhi24:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s3, s[4:5], 0x34
; VI-NEXT:    s_mov_b32 s7, 0xf000
; VI-NEXT:    s_mov_b32 s6, -1
; VI-NEXT:    s_mov_b32 s4, s0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, s3
; VI-NEXT:    s_mov_b32 s5, s1
; VI-NEXT:    v_mul_hi_u32_u24_e32 v0, s2, v0
; VI-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umulhi24:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_load_dword s3, s[4:5], 0x34
; GFX9-NEXT:    s_mov_b32 s7, 0xf000
; GFX9-NEXT:    s_mov_b32 s6, -1
; GFX9-NEXT:    s_mov_b32 s4, s0
; GFX9-NEXT:    s_mov_b32 s5, s1
; GFX9-NEXT:    s_and_b32 s0, s2, 0xffffff
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_and_b32 s1, s3, 0xffffff
; GFX9-NEXT:    s_mul_hi_u32 s0, s0, s1
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX9-NEXT:    s_endpgm
entry:
  %a.24 = and i64 %a, 16777215
  %b.24 = and i64 %b, 16777215
  %mul48 = mul i64 %a.24, %b.24
  %mul48.hi = lshr i64 %mul48, 32
  %mul24.hi = trunc i64 %mul48.hi to i32
  store i32 %mul24.hi, ptr addrspace(1) %out
  ret void
}

; Multiply with 24-bit inputs and 64-bit output.
define amdgpu_kernel void @test_umul24_i64(ptr addrspace(1) %out, i64 %a, i64 %b) {
; SI-LABEL: test_umul24_i64:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_load_dword s3, s[4:5], 0xd
; SI-NEXT:    s_mov_b32 s7, 0xf000
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_mov_b32 s4, s0
; SI-NEXT:    s_mov_b32 s5, s1
; SI-NEXT:    s_and_b32 s0, s2, 0xffffff
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s1, s3, 0xffffff
; SI-NEXT:    v_mov_b32_e32 v0, s3
; SI-NEXT:    s_mul_i32 s0, s0, s1
; SI-NEXT:    v_mul_hi_u32_u24_e32 v1, s2, v0
; SI-NEXT:    v_mov_b32_e32 v0, s0
; SI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i64:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_load_dword s3, s[4:5], 0x34
; VI-NEXT:    s_mov_b32 s7, 0xf000
; VI-NEXT:    s_mov_b32 s6, -1
; VI-NEXT:    s_mov_b32 s4, s0
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, s3
; VI-NEXT:    s_mov_b32 s5, s1
; VI-NEXT:    v_mul_hi_u32_u24_e32 v1, s2, v0
; VI-NEXT:    v_mul_u32_u24_e32 v0, s2, v0
; VI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[4:7], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i64:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_load_dword s3, s[4:5], 0x34
; GFX9-NEXT:    s_mov_b32 s7, 0xf000
; GFX9-NEXT:    s_mov_b32 s6, -1
; GFX9-NEXT:    s_mov_b32 s4, s0
; GFX9-NEXT:    s_mov_b32 s5, s1
; GFX9-NEXT:    s_and_b32 s0, s2, 0xffffff
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_and_b32 s1, s3, 0xffffff
; GFX9-NEXT:    s_mul_hi_u32 s2, s0, s1
; GFX9-NEXT:    s_mul_i32 s0, s0, s1
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s2
; GFX9-NEXT:    buffer_store_dwordx2 v[0:1], off, s[4:7], 0
; GFX9-NEXT:    s_endpgm
entry:
  %tmp0 = shl i64 %a, 40
  %a_24 = lshr i64 %tmp0, 40
  %tmp1 = shl i64 %b, 40
  %b_24 = lshr i64 %tmp1, 40
  %tmp2 = mul i64 %a_24, %b_24
  store i64 %tmp2, ptr addrspace(1) %out
  ret void
}

define i64 @test_umul48_i64(i64 %lhs, i64 %rhs) {
; GCN-LABEL: test_umul48_i64:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_mul_hi_u32_u24_e32 v1, v0, v2
; GCN-NEXT:    v_mul_u32_u24_e32 v0, v0, v2
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %lhs24 = and i64 %lhs, 16777215
  %rhs24 = and i64 %rhs, 16777215
  %mul = mul i64 %lhs24, %rhs24
  ret i64 %mul
}

define <2 x i64> @test_umul48_v2i64(<2 x i64> %lhs, <2 x i64> %rhs) {
; GCN-LABEL: test_umul48_v2i64:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_mul_hi_u32_u24_e32 v1, v0, v4
; GCN-NEXT:    v_mul_u32_u24_e32 v0, v0, v4
; GCN-NEXT:    v_mul_hi_u32_u24_e32 v3, v2, v6
; GCN-NEXT:    v_mul_u32_u24_e32 v2, v2, v6
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %lhs24 = and <2 x i64> %lhs, <i64 16777215, i64 16777215>
  %rhs24 = and <2 x i64> %rhs, <i64 16777215, i64 16777215>
  %mul = mul <2 x i64> %lhs24, %rhs24
  ret <2 x i64> %mul
}

define amdgpu_kernel void @test_umul24_i64_square(ptr addrspace(1) %out, [8 x i32], i64 %a) {
; SI-LABEL: test_umul24_i64_square:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dword s6, s[4:5], 0x13
; SI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s4, s6, 0xffffff
; SI-NEXT:    s_mul_i32 s4, s4, s4
; SI-NEXT:    v_mul_hi_u32_u24_e64 v1, s6, s6
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i64_square:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dword s6, s[4:5], 0x4c
; VI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mul_hi_u32_u24_e64 v1, s6, s6
; VI-NEXT:    v_mul_u32_u24_e64 v0, s6, s6
; VI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i64_square:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dword s6, s[4:5], 0x4c
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_and_b32 s4, s6, 0xffffff
; GFX9-NEXT:    s_mul_hi_u32 s5, s4, s4
; GFX9-NEXT:    s_mul_i32 s4, s4, s4
; GFX9-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-NEXT:    v_mov_b32_e32 v1, s5
; GFX9-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
entry:
  %tmp0 = shl i64 %a, 40
  %a.24 = lshr i64 %tmp0, 40
  %tmp2 = mul i64 %a.24, %a.24
  store i64 %tmp2, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umulhi16_i32(ptr addrspace(1) %out, i32 %a, i32 %b) {
; SI-LABEL: test_umulhi16_i32:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s7, 0xf000
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s2, s2, 0xffff
; SI-NEXT:    s_and_b32 s3, s3, 0xffff
; SI-NEXT:    s_mul_i32 s2, s2, s3
; SI-NEXT:    s_lshr_b32 s2, s2, 16
; SI-NEXT:    s_mov_b32 s6, -1
; SI-NEXT:    s_mov_b32 s4, s0
; SI-NEXT:    s_mov_b32 s5, s1
; SI-NEXT:    v_mov_b32_e32 v0, s2
; SI-NEXT:    buffer_store_short v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umulhi16_i32:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s7, 0xf000
; VI-NEXT:    s_mov_b32 s6, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    s_mov_b32 s4, s0
; VI-NEXT:    s_mov_b32 s5, s1
; VI-NEXT:    s_and_b32 s0, s2, 0xffff
; VI-NEXT:    s_and_b32 s1, s3, 0xffff
; VI-NEXT:    s_mul_i32 s0, s0, s1
; VI-NEXT:    s_lshr_b32 s0, s0, 16
; VI-NEXT:    v_mov_b32_e32 v0, s0
; VI-NEXT:    buffer_store_short v0, off, s[4:7], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umulhi16_i32:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[4:5], 0x24
; GFX9-NEXT:    v_mov_b32_e32 v0, 0
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_and_b32 s2, s2, 0xffff
; GFX9-NEXT:    s_and_b32 s3, s3, 0xffff
; GFX9-NEXT:    s_mul_i32 s2, s2, s3
; GFX9-NEXT:    v_mov_b32_e32 v1, s2
; GFX9-NEXT:    global_store_short_d16_hi v0, v1, s[0:1]
; GFX9-NEXT:    s_endpgm
entry:
  %a.16 = and i32 %a, 65535
  %b.16 = and i32 %b, 65535
  %mul = mul i32 %a.16, %b.16
  %hi = lshr i32 %mul, 16
  %mulhi = trunc i32 %hi to i16
  store i16 %mulhi, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umul24_i33(ptr addrspace(1) %out, i33 %a, i33 %b) {
; SI-LABEL: test_umul24_i33:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x9
; SI-NEXT:    s_load_dword s6, s[4:5], 0xb
; SI-NEXT:    s_load_dword s4, s[4:5], 0xd
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_and_b32 s5, s6, 0xffffff
; SI-NEXT:    s_and_b32 s7, s4, 0xffffff
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    v_mul_hi_u32_u24_e32 v0, s6, v0
; SI-NEXT:    s_mul_i32 s5, s5, s7
; SI-NEXT:    v_and_b32_e32 v1, 1, v0
; SI-NEXT:    v_mov_b32_e32 v0, s5
; SI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umul24_i33:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dword s6, s[4:5], 0x34
; VI-NEXT:    s_load_dword s7, s[4:5], 0x2c
; VI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v1, s6
; VI-NEXT:    v_mul_u32_u24_e32 v0, s7, v1
; VI-NEXT:    v_mul_hi_u32_u24_e32 v1, s7, v1
; VI-NEXT:    v_and_b32_e32 v1, 1, v1
; VI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umul24_i33:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dword s6, s[4:5], 0x2c
; GFX9-NEXT:    s_load_dword s7, s[4:5], 0x34
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_and_b32 s4, s6, 0xffffff
; GFX9-NEXT:    s_and_b32 s5, s7, 0xffffff
; GFX9-NEXT:    s_mul_i32 s6, s4, s5
; GFX9-NEXT:    s_mul_hi_u32 s4, s4, s5
; GFX9-NEXT:    s_and_b32 s4, s4, 1
; GFX9-NEXT:    v_mov_b32_e32 v0, s6
; GFX9-NEXT:    v_mov_b32_e32 v1, s4
; GFX9-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
entry:
  %tmp0 = shl i33 %a, 9
  %a_24 = lshr i33 %tmp0, 9
  %tmp1 = shl i33 %b, 9
  %b_24 = lshr i33 %tmp1, 9
  %tmp2 = mul i33 %a_24, %b_24
  %ext = zext i33 %tmp2 to i64
  store i64 %ext, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_umulhi24_i33(ptr addrspace(1) %out, i33 %a, i33 %b) {
; SI-LABEL: test_umulhi24_i33:
; SI:       ; %bb.0: ; %entry
; SI-NEXT:    s_load_dword s6, s[4:5], 0xd
; SI-NEXT:    s_load_dword s7, s[4:5], 0xb
; SI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x9
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    v_mov_b32_e32 v0, s6
; SI-NEXT:    v_mul_hi_u32_u24_e32 v0, s7, v0
; SI-NEXT:    v_and_b32_e32 v0, 1, v0
; SI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: test_umulhi24_i33:
; VI:       ; %bb.0: ; %entry
; VI-NEXT:    s_load_dword s6, s[4:5], 0x34
; VI-NEXT:    s_load_dword s7, s[4:5], 0x2c
; VI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; VI-NEXT:    s_mov_b32 s3, 0xf000
; VI-NEXT:    s_mov_b32 s2, -1
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, s6
; VI-NEXT:    v_mul_hi_u32_u24_e32 v0, s7, v0
; VI-NEXT:    v_and_b32_e32 v0, 1, v0
; VI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: test_umulhi24_i33:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_load_dword s6, s[4:5], 0x2c
; GFX9-NEXT:    s_load_dword s7, s[4:5], 0x34
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GFX9-NEXT:    s_mov_b32 s3, 0xf000
; GFX9-NEXT:    s_mov_b32 s2, -1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_and_b32 s4, s6, 0xffffff
; GFX9-NEXT:    s_and_b32 s5, s7, 0xffffff
; GFX9-NEXT:    s_mul_hi_u32 s4, s4, s5
; GFX9-NEXT:    s_and_b32 s4, s4, 1
; GFX9-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX9-NEXT:    s_endpgm
entry:
  %tmp0 = shl i33 %a, 9
  %a_24 = lshr i33 %tmp0, 9
  %tmp1 = shl i33 %b, 9
  %b_24 = lshr i33 %tmp1, 9
  %tmp2 = mul i33 %a_24, %b_24
  %hi = lshr i33 %tmp2, 32
  %trunc = trunc i33 %hi to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
}


; Make sure the created any_extend is ignored to use the real bits
; being multiplied.
define i17 @test_umul24_anyextend_i24_src0_src1(i24 %a, i24 %b) {
; GCN-LABEL: test_umul24_anyextend_i24_src0_src1:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_mul_u32_u24_e32 v0, 0xea, v0
; GCN-NEXT:    v_mul_u32_u24_e32 v1, 0x39b, v1
; GCN-NEXT:    v_mul_u32_u24_e32 v0, v0, v1
; GCN-NEXT:    v_and_b32_e32 v0, 0x1fffe, v0
; GCN-NEXT:    v_mul_u32_u24_e32 v0, 0x63, v0
; GCN-NEXT:    s_setpc_b64 s[30:31]
entry:
  %aa = mul i24 %a, 234
  %bb = mul i24 %b, 923
  %a_32 = zext i24 %aa to i32
  %b_32 = zext i24 %bb to i32
  %mul = mul i32 %a_32, %b_32
  %trunc = trunc i32 %mul to i17
  %arst = mul i17 %trunc, 99
  ret i17 %arst
}

define i17 @test_umul24_anyextend_i23_src0_src1(i23 %a, i23 %b) {
; GCN-LABEL: test_umul24_anyextend_i23_src0_src1:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_and_b32_e32 v0, 0x7fffff, v0
; GCN-NEXT:    v_and_b32_e32 v1, 0x7fffff, v1
; GCN-NEXT:    v_mul_u32_u24_e32 v0, 0xea, v0
; GCN-NEXT:    v_mul_u32_u24_e32 v1, 0x39b, v1
; GCN-NEXT:    v_and_b32_e32 v0, 0x7ffffe, v0
; GCN-NEXT:    v_and_b32_e32 v1, 0x7fffff, v1
; GCN-NEXT:    v_mul_u32_u24_e32 v0, v0, v1
; GCN-NEXT:    v_and_b32_e32 v0, 0x1fffe, v0
; GCN-NEXT:    v_mul_u32_u24_e32 v0, 0x63, v0
; GCN-NEXT:    s_setpc_b64 s[30:31]
entry:
  %aa = mul i23 %a, 234
  %bb = mul i23 %b, 923
  %a_32 = zext i23 %aa to i32
  %b_32 = zext i23 %bb to i32
  %mul = mul i32 %a_32, %b_32
  %trunc = trunc i32 %mul to i17
  %arst = mul i17 %trunc, 99
  ret i17 %arst
}
