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
#include <cstdint>

namespace gpu::ps5 {

// The unified register file is sparse but small enough to store flat. 0xD000
// dwords covers config(0x2000)/sh(0x2C00)/context(0xA000)/uconfig(0xC000).
constexpr uint32_t kRegFileSize = 0xD000;

// gfx10 SET_*_REG offset-dword selector bits. AGC (and real gfx10 hardware)
// encode a register-space selector in [30:28] of the offset dword; strip it to
// recover the plain register offset (gfx10.3 register-offset normalization).
constexpr uint32_t kRegSelectorMask = 0x70000000u;

// --- key context registers (absolute dword offset = kContextRegBase + n) ---
// Color buffer 0 (the render target). CB_COLORn are a 15-dword stride apart on
// gfx10 (not 0xF-per-slot-with-gaps like GCN); the low 32 bits of the base are
// in _BASE, the high bits in the separate _BASE_EXT register (64-bit address).
constexpr uint32_t mmCB_COLOR0_BASE = 0xA318;
constexpr uint32_t mmCB_COLOR0_VIEW = 0xA31B;
constexpr uint32_t mmCB_COLOR0_INFO = 0xA31C;    // FORMAT[6:2], NUMBER_TYPE[10:8], COMP_SWAP[12:11]
constexpr uint32_t mmCB_COLOR0_ATTRIB = 0xA31D;  // NUM_SAMPLES[14:12], NUM_FRAGMENTS[16:15]
constexpr uint32_t kCbColorStride = 0xF;
// gfx10 64-bit high-bit extension registers (stride 1 across the 8 slots).
constexpr uint32_t mmCB_COLOR0_BASE_EXT = 0xA390;   // high bits of the RT base
constexpr uint32_t mmCB_COLOR0_ATTRIB2 = 0xA3B0;    // MIP0_HEIGHT[13:0], MIP0_WIDTH[27:14]
constexpr uint32_t mmCB_COLOR0_ATTRIB3 = 0xA3B8;    // COLOR_SW_MODE[18:14] (gfx10 swizzle mode)
constexpr uint32_t kCbColorExtStride = 0x1;
// Screen scissor gives the render area (width/height).
constexpr uint32_t mmPA_SC_SCREEN_SCISSOR_TL = 0xA00C;
constexpr uint32_t mmPA_SC_SCREEN_SCISSOR_BR = 0xA00D;
constexpr uint32_t mmPA_SC_GENERIC_SCISSOR_TL = 0xA090;
constexpr uint32_t mmPA_SC_GENERIC_SCISSOR_BR = 0xA091;
// Viewport 0 scale/offset (float). gfx10 packs all 6 fields per viewport at a
// stride of 6 dwords (X/Y/Z scale+offset interleaved).
constexpr uint32_t mmPA_CL_VPORT_XSCALE = 0xA10F;
constexpr uint32_t mmPA_CL_VPORT_XOFFSET = 0xA110;
constexpr uint32_t mmPA_CL_VPORT_YSCALE = 0xA111;
constexpr uint32_t mmPA_CL_VPORT_YOFFSET = 0xA112;
constexpr uint32_t mmPA_CL_VPORT_ZSCALE = 0xA113;
constexpr uint32_t mmPA_CL_VPORT_ZOFFSET = 0xA114;
// Render-target mask (which CB targets are written).
constexpr uint32_t mmCB_TARGET_MASK = 0xA08E;
constexpr uint32_t mmCB_SHADER_MASK = 0xA08F;
// Per-MRT blend control. CB_BLENDn_CONTROL are 1 dword apart. Layout (gfx10,
// same field split as GCN gen2):
//  [4:0] color_src_factor  [7:5] color_func   [12:8] color_dst_factor
//  [20:16] alpha_src_factor [23:21] alpha_func [28:24] alpha_dst_factor
//  [29] separate_alpha_blend  [30] enable
constexpr uint32_t mmCB_BLEND0_CONTROL = 0xA1E0;
constexpr uint32_t kCbBlendStride = 0x1;
// Overall color-buffer mode (ROP3 / blend disable). MODE field is [6:4].
constexpr uint32_t mmCB_COLOR_CONTROL = 0xA202;

// --- depth/stencil (DB) state ---
// DB_DEPTH_CONTROL: STENCIL_ENABLE[0] Z_ENABLE[1] Z_WRITE_ENABLE[2] ZFUNC[6:4].
constexpr uint32_t mmDB_DEPTH_CONTROL = 0xA200;
// DB_Z_INFO: FORMAT[1:0] (0=invalid/off, 1=Z16, 3=Z32_FLOAT).
constexpr uint32_t mmDB_Z_INFO = 0xA010;
// Depth surface base (byte addr = value << 8); gfx10 adds high-bit ext regs.
constexpr uint32_t mmDB_Z_READ_BASE = 0xA012;
constexpr uint32_t mmDB_Z_WRITE_BASE = 0xA014;
constexpr uint32_t mmDB_Z_READ_BASE_HI = 0xA01A;
constexpr uint32_t mmDB_Z_WRITE_BASE_HI = 0xA01C;
// Fast-clear depth value (float) used when the buffer is bound with loadOp=CLEAR.
constexpr uint32_t mmDB_DEPTH_CLEAR = 0xA00B;
// Primitive-setup: cull + winding. CULL_FRONT[0] CULL_BACK[1] FACE[2] (0=CCW front).
constexpr uint32_t mmPA_SU_SC_MODE_CNTL = 0xA205;

// --- SPI shader-interface (context) ---
constexpr uint32_t mmSPI_VS_OUT_CONFIG = 0xA1B1;    // # of VS output params
constexpr uint32_t mmSPI_PS_INPUT_ENA = 0xA1B3;     // interpolants the PS reads
constexpr uint32_t mmSPI_PS_INPUT_ADDR = 0xA1B4;
constexpr uint32_t mmSPI_PS_IN_CONTROL = 0xA1B6;    // NUM_INTERP
constexpr uint32_t mmSPI_SHADER_POS_FORMAT = 0xA1C3;
constexpr uint32_t mmSPI_SHADER_Z_FORMAT = 0xA1C4;
constexpr uint32_t mmSPI_SHADER_COL_FORMAT = 0xA1C5;  // per-MRT export format
constexpr uint32_t mmPA_CL_VS_OUT_CNTL = 0xA207;
// NGG / geometry-engine stage select (which HW stages run).
constexpr uint32_t mmVGT_SHADER_STAGES_EN = 0xA2D5;

// Primitive type + index type moved to uconfig on gfx9+.
constexpr uint32_t mmVGT_PRIMITIVE_TYPE = 0xC242;
constexpr uint32_t mmVGT_INDEX_TYPE = 0xC243;
constexpr uint32_t mmGE_CNTL = 0xC25B;

// --- shader (SH) registers (absolute = kShRegBase + n) ---
// Pixel shader program address + resources + user data.
constexpr uint32_t mmSPI_SHADER_PGM_LO_PS = 0x2C08;
constexpr uint32_t mmSPI_SHADER_PGM_HI_PS = 0x2C09;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC1_PS = 0x2C0A;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC2_PS = 0x2C0B;  // USER_SGPR[5:1], USER_SGPR_MSB[27]
constexpr uint32_t mmSPI_SHADER_USER_DATA_PS_0 = 0x2C0C;  // 32 user-data SGPRs (0x0C..0x2B)

// gfx10.3 has no hardware VS stage: vertex work runs as a merged ES(front)/GS
// (back) NGG primitive shader. AGC writes the vertex program address into BOTH
// the ES and GS PGM_LO registers. We read the vertex shader from the GS block
// (SPI_SHADER_PGM_LO_GS) and its user data from SPI_SHADER_USER_DATA_GS_0.
constexpr uint32_t mmSPI_SHADER_PGM_LO_GS = 0x2C88;
constexpr uint32_t mmSPI_SHADER_PGM_HI_GS = 0x2C89;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC1_GS = 0x2C8A;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC2_GS = 0x2C8B;
constexpr uint32_t mmSPI_SHADER_USER_DATA_GS_0 = 0x2C8C;  // 32 user-data SGPRs (0x8C..0xAB)
constexpr uint32_t mmSPI_SHADER_PGM_LO_ES = 0x2CC8;       // ES front half (== GS addr)
constexpr uint32_t mmSPI_SHADER_PGM_HI_ES = 0x2CC9;
constexpr uint32_t mmSPI_SHADER_USER_DATA_ES_0 = 0x2CCC;

// Compute program registers.
constexpr uint32_t mmCOMPUTE_NUM_THREAD_X = 0x2E07;  // u16 full | u16 partial
constexpr uint32_t mmCOMPUTE_NUM_THREAD_Y = 0x2E08;
constexpr uint32_t mmCOMPUTE_NUM_THREAD_Z = 0x2E09;
constexpr uint32_t mmCOMPUTE_PGM_LO = 0x2E0C;   // CS addr[39:8]
constexpr uint32_t mmCOMPUTE_PGM_HI = 0x2E0D;   // CS addr[47:40] in [7:0]
constexpr uint32_t mmCOMPUTE_PGM_RSRC1 = 0x2E12;  // W32_EN[30] (wave32)
constexpr uint32_t mmCOMPUTE_PGM_RSRC2 = 0x2E13;  // user_sgpr[5:1], tgid_en[9:7], lds[23:15]
constexpr uint32_t mmCOMPUTE_USER_DATA_0 = 0x2E40;  // 16 user-data SGPRs

// The full GPU register state. A draw is rendered from a snapshot of this.
struct Regs {
  std::array<uint32_t, kRegFileSize> data{};

  uint32_t &operator[](uint32_t off) { return data[off]; }
  uint32_t operator[](uint32_t off) const { return off < kRegFileSize ? data[off] : 0; }

  // 48-bit GPU shader address from a PGM_LO/HI register pair:
  // addr = (LO << 8) | ((HI & 0xFF) << 40)  (LO holds bits [39:8], HI [47:40]).
  uint64_t shaderAddr(uint32_t loReg) const {
    uint64_t lo = data[loReg];
    uint64_t hi = data[loReg + 1] & 0xFF;
    return (lo << 8) | (hi << 40);
  }
  // Full 64-bit color-target base from the gfx10 _BASE (low) + _BASE_EXT (high)
  // register pair; the stored value is 256-byte aligned (<< 8).
  uint64_t cbColorBase(int rt = 0) const {
    uint64_t lo = data[mmCB_COLOR0_BASE + rt * kCbColorStride];
    uint64_t hi = data[mmCB_COLOR0_BASE_EXT + rt * kCbColorExtStride];
    return ((hi << 32) | lo) << 8;
  }
};

}  // namespace gpu::ps5
