#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * gfx10.3 (RDNA2 / Oberon) GPU register file. The AGC PM4 SET_*_REG packets
 * write into a unified register space; we mirror it as a flat dword array
 * indexed by absolute register offset (base + packet offset), exactly like the
 * PS4 Liverpool path (gpu/ps4/liverpool.h). Draw handlers read the render
 * target / shader pointers / primitive state out of it.
 *
 * The offsets below are gfx10.3 values (context 0xA000 / sh 0x2C00 /
 * uconfig 0xC000), verified against published gfx10.3 register maps
 * (src/graphics/guest_gpu/pm4.h). They differ from the PS4's GCN-gen2 values:
 * CB_COLOR moved and gained 64-bit BASE_EXT high-bit registers, there is no
 * hardware VS stage (vertex work runs as a merged ES/GS NGG shader), and the
 * SET_*_REG offset dword carries a gfx10 register selector in bits [30:28]
 * that must be masked off (see kRegSelectorMask).
 */

#include <array>
#include "base/arch.h"

namespace gpu::ps5 {

// The unified register file is sparse but small enough to store flat. 0xD000
// dwords covers config(0x2000)/sh(0x2C00)/context(0xA000)/uconfig(0xC000).
constexpr u32 kRegFileSize = 0xD000;

// gfx10 SET_*_REG offset-dword selector bits. AGC (and real gfx10 hardware)
// encode a register-space selector in [30:28] of the offset dword; strip it to
// recover the plain register offset (gfx10.3 register-offset normalization).
constexpr u32 kRegSelectorMask = 0x70000000u;

// --- key context registers (absolute dword offset = kContextRegBase + n) ---
// Color buffer 0 (the render target). CB_COLORn are a 15-dword stride apart on
// gfx10 (not 0xF-per-slot-with-gaps like GCN); the low 32 bits of the base are
// in _BASE, the high bits in the separate _BASE_EXT register (64-bit address).
constexpr u32 mmCB_COLOR0_BASE = 0xA318;
constexpr u32 mmCB_COLOR0_VIEW = 0xA31B;
constexpr u32 mmCB_COLOR0_INFO =
    0xA31C;  // FORMAT[6:2], NUMBER_TYPE[10:8], COMP_SWAP[12:11]
constexpr u32 mmCB_COLOR0_ATTRIB =
    0xA31D;  // NUM_SAMPLES[14:12], NUM_FRAGMENTS[16:15]
constexpr u32 kCbColorStride = 0xF;
// The fast-clear colour, in the target's own format, and the DCC metadata
// the target is compressed with (base << 8, high bits in _DCC_BASE_EXT).
constexpr u32 mmCB_COLOR0_CLEAR_WORD0 = 0xA323;
constexpr u32 mmCB_COLOR0_CLEAR_WORD1 = 0xA324;
constexpr u32 mmCB_COLOR0_DCC_BASE = 0xA325;
constexpr u32 mmCB_COLOR0_DCC_BASE_EXT = 0xA3A8;
// gfx10 64-bit high-bit extension registers (stride 1 across the 8 slots).
constexpr u32 mmCB_COLOR0_BASE_EXT = 0xA390;  // high bits of the RT base
constexpr u32 mmCB_COLOR0_ATTRIB2 =
    0xA3B0;  // MIP0_HEIGHT[13:0], MIP0_WIDTH[27:14]
constexpr u32 mmCB_COLOR0_ATTRIB3 =
    0xA3B8;  // COLOR_SW_MODE[18:14] (gfx10 swizzle mode)
constexpr u32 kCbColorExtStride = 0x1;
// Screen scissor gives the render area (width/height).
constexpr u32 mmPA_SC_SCREEN_SCISSOR_TL = 0xA00C;
constexpr u32 mmPA_SC_SCREEN_SCISSOR_BR = 0xA00D;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_TL = 0xA090;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_BR = 0xA091;
// Viewport 0 scale/offset (float). gfx10 packs all 6 fields per viewport at a
// stride of 6 dwords (X/Y/Z scale+offset interleaved).
constexpr u32 mmPA_CL_VPORT_XSCALE = 0xA10F;
constexpr u32 mmPA_CL_VPORT_XOFFSET = 0xA110;
constexpr u32 mmPA_CL_VPORT_YSCALE = 0xA111;
constexpr u32 mmPA_CL_VPORT_YOFFSET = 0xA112;
constexpr u32 mmPA_CL_VPORT_ZSCALE = 0xA113;
constexpr u32 mmPA_CL_VPORT_ZOFFSET = 0xA114;
// Render-target mask (which CB targets are written).
constexpr u32 mmCB_TARGET_MASK = 0xA08E;
constexpr u32 mmCB_SHADER_MASK = 0xA08F;
// Per-MRT blend control. CB_BLENDn_CONTROL are 1 dword apart. Layout (gfx10,
// same field split as GCN gen2):
//  [4:0] color_src_factor  [7:5] color_func   [12:8] color_dst_factor
//  [20:16] alpha_src_factor [23:21] alpha_func [28:24] alpha_dst_factor
//  [29] separate_alpha_blend  [30] enable
constexpr u32 mmCB_BLEND0_CONTROL = 0xA1E0;
constexpr u32 kCbBlendStride = 0x1;
// The operand of the CONSTANT_COLOR/CONSTANT_ALPHA blend factors, one dword
// per channel (R at 0xA105, then G, B, A).
constexpr u32 mmCB_BLEND_RED = 0xA105;
// Overall color-buffer mode (ROP3 / blend disable). MODE field is [6:4].
constexpr u32 mmCB_COLOR_CONTROL = 0xA202;

// --- depth/stencil (DB) state ---
// DB_RENDER_CONTROL: DEPTH_CLEAR_ENABLE[0], STENCIL_CLEAR_ENABLE[1].
constexpr u32 mmDB_RENDER_CONTROL = 0xA000;
// DB_DEPTH_CONTROL: STENCIL_ENABLE[0] Z_ENABLE[1] Z_WRITE_ENABLE[2] ZFUNC[6:4].
constexpr u32 mmDB_DEPTH_CONTROL = 0xA200;
// DB_Z_INFO: FORMAT[1:0] (0=invalid/off, 1=Z16, 3=Z32_FLOAT).
constexpr u32 mmDB_Z_INFO = 0xA010;
// Depth surface base (byte addr = value << 8); gfx10 adds high-bit ext regs.
constexpr u32 mmDB_Z_READ_BASE = 0xA012;
constexpr u32 mmDB_Z_WRITE_BASE = 0xA014;
constexpr u32 mmDB_Z_READ_BASE_HI = 0xA01A;
constexpr u32 mmDB_Z_WRITE_BASE_HI = 0xA01C;
// The depth surface's HTILE metadata (base << 8): a title fast-clears depth
// by filling this, not the Z plane.
constexpr u32 mmDB_HTILE_DATA_BASE = 0xA005;
constexpr u32 mmDB_HTILE_DATA_BASE_HI = 0xA01E;
// Fast-clear depth value (float) used when the buffer is bound with
// loadOp=CLEAR.
constexpr u32 mmDB_DEPTH_CLEAR = 0xA00B;
// Primitive-setup: cull + winding. CULL_FRONT[0] CULL_BACK[1] FACE[2] (0=CCW
// front).
constexpr u32 mmPA_SU_SC_MODE_CNTL = 0xA205;
// Clip control. DX_CLIP_SPACE_DEF[19]: 1 = clip z in [0,w] (Vulkan's
// convention), 0 = OpenGL's [-w,w], which the recompiled VS then has to remap.
constexpr u32 mmPA_CL_CLIP_CNTL = 0xA204;

// --- SPI shader-interface (context) ---
constexpr u32 mmSPI_VS_OUT_CONFIG = 0xA1B1;  // # of VS output params
// SPI_PS_INPUT_CNTL_0..31: OFFSET[4:0] names which VS parameter export each PS
// input slot reads. Only defined below NUM_INTERP.
constexpr u32 mmSPI_PS_INPUT_CNTL_0 = 0xA191;
constexpr u32 mmSPI_PS_INPUT_ENA = 0xA1B3;   // interpolants the PS reads
constexpr u32 mmSPI_PS_INPUT_ADDR = 0xA1B4;
constexpr u32 mmSPI_PS_IN_CONTROL = 0xA1B6;  // NUM_INTERP
constexpr u32 mmSPI_SHADER_POS_FORMAT = 0xA1C3;
constexpr u32 mmSPI_SHADER_Z_FORMAT = 0xA1C4;
constexpr u32 mmSPI_SHADER_COL_FORMAT = 0xA1C5;  // per-MRT export format
constexpr u32 mmPA_CL_VS_OUT_CNTL = 0xA207;
// NGG / geometry-engine stage select (which HW stages run).
constexpr u32 mmVGT_SHADER_STAGES_EN = 0xA2D5;

// Primitive type + index type moved to uconfig on gfx9+.
constexpr u32 mmVGT_PRIMITIVE_TYPE = 0xC242;
constexpr u32 mmVGT_INDEX_TYPE = 0xC243;
constexpr u32 mmGE_CNTL = 0xC25B;
constexpr u32 mmVGT_GS_ONCHIP_CNTL = 0xA291;
constexpr u32 mmVGT_GS_MAX_VERT_OUT = 0xA2CE;
constexpr u32 mmGE_NGG_SUBGRP_CNTL = 0xA2D3;

// --- shader (SH) registers (absolute = kShRegBase + n) ---
// Pixel shader program address + resources + user data.
constexpr u32 mmSPI_SHADER_PGM_LO_PS = 0x2C08;
constexpr u32 mmSPI_SHADER_PGM_HI_PS = 0x2C09;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_PS = 0x2C0A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_PS =
    0x2C0B;  // USER_SGPR[5:1], USER_SGPR_MSB[27]
constexpr u32 mmSPI_SHADER_USER_DATA_PS_0 =
    0x2C0C;  // 32 user-data SGPRs (0x0C..0x2B)

// gfx10.3 has no hardware VS stage: vertex work runs as a merged ES(front)/GS
// (back) NGG primitive shader. AGC writes the vertex program address into BOTH
// the ES and GS PGM_LO registers. We read the vertex shader from the GS block
// (SPI_SHADER_PGM_LO_GS) and its user data from SPI_SHADER_USER_DATA_GS_0.
constexpr u32 mmSPI_SHADER_PGM_LO_GS = 0x2C88;
constexpr u32 mmSPI_SHADER_USER_DATA_ADDR_LO_GS = 0x2C82;
constexpr u32 mmSPI_SHADER_USER_DATA_ADDR_HI_GS = 0x2C83;
constexpr u32 mmSPI_SHADER_PGM_HI_GS = 0x2C89;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_GS = 0x2C8A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_GS = 0x2C8B;
constexpr u32 mmSPI_SHADER_USER_DATA_GS_0 =
    0x2C8C;  // 32 user-data SGPRs (0x8C..0xAB)
constexpr u32 mmSPI_SHADER_PGM_LO_ES =
    0x2CC8;  // ES front half (== GS addr)
constexpr u32 mmSPI_SHADER_PGM_HI_ES = 0x2CC9;
constexpr u32 mmSPI_SHADER_USER_DATA_ES_0 = 0x2CCC;

// Compute program registers.
constexpr u32 mmCOMPUTE_START_X = 0x2E04;
constexpr u32 mmCOMPUTE_START_Y = 0x2E05;
constexpr u32 mmCOMPUTE_START_Z = 0x2E06;
constexpr u32 mmCOMPUTE_NUM_THREAD_X = 0x2E07;  // u16 full | u16 partial
constexpr u32 mmCOMPUTE_NUM_THREAD_Y = 0x2E08;
constexpr u32 mmCOMPUTE_NUM_THREAD_Z = 0x2E09;
constexpr u32 mmCOMPUTE_PGM_LO = 0x2E0C;     // CS addr[39:8]
constexpr u32 mmCOMPUTE_PGM_HI = 0x2E0D;     // CS addr[47:40] in [7:0]
constexpr u32 mmCOMPUTE_PGM_RSRC1 = 0x2E12;  // W32_EN[30] (wave32)
constexpr u32 mmCOMPUTE_PGM_RSRC2 =
    0x2E13;  // user_sgpr[5:1], tgid_en[9:7], lds[23:15]
constexpr u32 mmCOMPUTE_USER_DATA_0 = 0x2E40;  // 16 user-data SGPRs

// The full GPU register state. A draw is rendered from a snapshot of this.
struct Regs {
  std::array<u32, kRegFileSize> data{};

  u32& operator[](u32 off) { return data[off]; }
  u32 operator[](u32 off) const {
    return off < kRegFileSize ? data[off] : 0;
  }

  // A run of registers as a pointer: the user-data windows a shader stage is
  // launched with are read that way, and the file is flat, so they are
  // contiguous.
  const u32* At(u32 off) const { return &data[off]; }

  // 48-bit GPU shader address from a PGM_LO/HI register pair:
  // addr = (LO << 8) | ((HI & 0xFF) << 40)  (LO holds bits [39:8], HI [47:40]).
  u64 ShaderAddr(u32 lo_reg) const {
    u64 lo = data[lo_reg];
    u64 hi = data[lo_reg + 1] & 0xFF;
    return (lo << 8) | (hi << 40);
  }
  // Full 64-bit color-target base from the gfx10 _BASE (low) + _BASE_EXT (high)
  // register pair; the stored value is 256-byte aligned (<< 8).
  u64 CbColorBase(int rt = 0) const {
    u64 lo = data[mmCB_COLOR0_BASE + rt * kCbColorStride];
    u64 hi = data[mmCB_COLOR0_BASE_EXT + rt * kCbColorExtStride];
    return ((hi << 32) | lo) << 8;
  }
};

}  // namespace gpu::ps5
