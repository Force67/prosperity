#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Liverpool GPU register file. The PM4 SET_*_REG packets write into a unified
 * register space; we mirror it as a flat dword array indexed by absolute
 * register offset (base + packet offset). Draw handlers read the relevant
 * registers (render target, shader pointers, primitive state) out of it.
 *
 * Register offsets below are GCN gen2 (Sea Islands / Liverpool) values, the
 * same the PS4 Gnm driver programs. Linux GFX 7.2 register offsets and field
 * layouts:
 * https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_d.h
 * https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_sh_mask.h
 */

#include <array>
#include "base/arch.h"

namespace gpu {

// The unified register file is sparse but small enough to store flat. 0xD000
// dwords covers config(0x2000)/sh(0x2C00)/context(0xA000)/uconfig(0xC000).
constexpr u32 kRegFileSize = 0xD000;

// --- key context registers (absolute dword offset = kContextRegBase + n) ---
// Color buffer 0 (the render target). CB_COLORn are a 0xF-dword stride apart.
constexpr u32 mmCB_COLOR0_BASE = 0xA318;
constexpr u32 mmCB_COLOR0_PITCH = 0xA319;
constexpr u32 mmCB_COLOR0_SLICE = 0xA31A;
constexpr u32 mmCB_COLOR0_VIEW = 0xA31B;
constexpr u32 mmCB_COLOR0_INFO = 0xA31C;    // format/number-type
constexpr u32 mmCB_COLOR0_ATTRIB = 0xA31D;  // tiling/dims
// Fast-clear colour, already encoded in the target's own format. GNM's clear
// helper programs these and then issues a RECT_LIST draw with no pixel shader;
// the colour is here, not in vertex data. See the clear-rect path in
// vk_draw_recomp.cc.
constexpr u32 mmCB_COLOR0_CLEAR_WORD0 = 0xA323;  // CB_COLOR0_BASE + 0xB
constexpr u32 mmCB_COLOR0_CLEAR_WORD1 = 0xA324;
constexpr u32 kCbColorStride = 0xF;
// Screen scissor gives the render area (width/height).
constexpr u32 mmPA_SC_SCREEN_SCISSOR_TL = 0xA00C;
constexpr u32 mmPA_SC_SCREEN_SCISSOR_BR = 0xA00D;
// Clip control. DX_CLIP_SPACE_DEF[19]: 1 = clip z in [0,w] (Vulkan/D3D
// convention, what GNM sets for a normal projection), 0 = z in [-w,w] (GL).
constexpr u32 mmPA_CL_CLIP_CNTL = 0xA204;
// Per-viewport scissor (viewport 0). The hardware intersects this with the
// screen/window/generic scissors; it is the one a deferred renderer moves
// per draw to bound each light volume.
constexpr u32 mmPA_SC_VPORT_SCISSOR_0_TL = 0xA094;
constexpr u32 mmPA_SC_VPORT_SCISSOR_0_BR = 0xA095;
constexpr u32 mmPA_SC_WINDOW_SCISSOR_TL = 0xA081;
constexpr u32 mmPA_SC_WINDOW_SCISSOR_BR = 0xA082;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_TL = 0xA090;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_BR = 0xA091;
// Viewport 0 scale/offset (float).
// DB_RENDER_CONTROL: DEPTH_CLEAR_ENABLE[0], STENCIL_CLEAR_ENABLE[1],
// DEPTH_COPY[2], STENCIL_COPY[3], RESUMMARIZE_ENABLE[4]. A draw issued with a
// clear bit set is not a draw: the hardware fills the depth/stencil plane with
// DB_DEPTH_CLEAR / DB_STENCIL_CLEAR over the drawn rect and ignores the
// shader's output.
constexpr u32 mmDB_RENDER_CONTROL = 0xA000;
constexpr u32 mmPA_CL_VPORT_XSCALE = 0xA10F;
constexpr u32 mmPA_CL_VPORT_XOFFSET = 0xA110;
constexpr u32 mmPA_CL_VPORT_YSCALE = 0xA111;
constexpr u32 mmPA_CL_VPORT_YOFFSET = 0xA112;
constexpr u32 mmPA_CL_VPORT_ZSCALE = 0xA113;
constexpr u32 mmPA_CL_VPORT_ZOFFSET = 0xA114;
// Render-target mask (which CB targets are written).
constexpr u32 mmCB_TARGET_MASK = 0xA08E;
constexpr u32 mmCB_SHADER_MASK = 0xA08F;
// Per-MRT blend control. CB_BLENDn_CONTROL are 1 dword apart. Layout (GCN
// gen2):
//  [0:4] color_src_factor  [5:7] color_func   [8:12] color_dst_factor
//  [16:20] alpha_src_factor [21:23] alpha_func [24:28] alpha_dst_factor
//  [29] separate_alpha_blend  [30] enable
constexpr u32 mmCB_BLEND0_CONTROL = 0xA1E0;
constexpr u32 kCbBlendStride = 0x1;
// The operand of the CONSTANT_COLOR/CONSTANT_ALPHA blend factors, one dword
// per channel (R at 0xA105, then G, B, A).
constexpr u32 mmCB_BLEND_RED = 0xA105;
// Pixel-shader system-value VGPR layout (barycentrics, position, face, etc.).
constexpr u32 mmSPI_PS_INPUT_CNTL_0 = 0xA191;  // ..._31 at 0xA1B0
constexpr u32 mmSPI_PS_INPUT_ENA = 0xA1B3;
// SPI_PS_IN_CONTROL.NUM_INTERP[5:0]: how many of the 32 input-control slots are
// meaningful. Slots at or above it are don't-care, so their zero says nothing.
constexpr u32 mmSPI_PS_IN_CONTROL = 0xA1B6;
// Overall color-buffer mode (ROP3 / blend disable). MODE field is [4:6].
constexpr u32 mmCB_COLOR_CONTROL = 0xA202;
// Primitive type for the draw (VGT_PRIMITIVE_TYPE is a uconfig reg on gen2).
// Which shader stages the pipeline runs: bits [1:0] select the LS/HS
// tessellation path, [3:2] the ES/GS path. Anything but 0 means the shader at
// the VS slot is not the stage that exports vertices.
constexpr u32 mmVGT_SHADER_STAGES_EN = 0xA2D5;
constexpr u32 mmVGT_PRIMITIVE_TYPE = 0xC242;
constexpr u32 mmVGT_NUM_INDICES = 0xC24C;

// --- depth/stencil (DB) state ---
// DB_DEPTH_CONTROL: STENCIL_ENABLE[0] Z_ENABLE[1] Z_WRITE_ENABLE[2] ZFUNC[6:4].
constexpr u32 mmDB_DEPTH_CONTROL = 0xA200;
// DB_Z_INFO: FORMAT[1:0] (0=invalid/off, 1=Z16, 3=Z32_FLOAT).
constexpr u32 mmDB_Z_INFO = 0xA010;
constexpr u32 mmDB_STENCIL_INFO = 0xA011;
// Depth surface base (byte addr = value << 8). Z_WRITE is what the draw renders
// to.
constexpr u32 mmDB_Z_READ_BASE = 0xA012;
constexpr u32 mmDB_STENCIL_READ_BASE = 0xA013;
constexpr u32 mmDB_Z_WRITE_BASE = 0xA014;
constexpr u32 mmDB_STENCIL_WRITE_BASE = 0xA015;
// DB_DEPTH_SIZE: PITCH_TILE_MAX[10:0], HEIGHT_TILE_MAX[21:11] (both in 8-texel
// tiles, minus one). DB_DEPTH_SLICE: SLICE_TILE_MAX[21:0], the tiles in one
// slice minus one -- which is what says whether a Z surface has more than one.
constexpr u32 mmDB_HTILE_DATA_BASE = 0xA005;
constexpr u32 mmDB_DEPTH_SIZE = 0xA016;
constexpr u32 mmDB_DEPTH_SLICE = 0xA017;
// Fast-clear depth value (float) used when the buffer is bound with
// loadOp=CLEAR.
constexpr u32 mmDB_DEPTH_CLEAR = 0xA00B;
constexpr u32 mmDB_STENCIL_CLEAR = 0xA00A;
constexpr u32 mmDB_STENCIL_CONTROL = 0xA10B;
constexpr u32 mmDB_STENCILREFMASK = 0xA10C;
constexpr u32 mmDB_STENCILREFMASK_BF = 0xA10D;
// Primitive-setup: cull + winding. CULL_FRONT[0] CULL_BACK[1] FACE[2] (0=CCW
// front).
constexpr u32 mmPA_SU_SC_MODE_CNTL = 0xA205;

// --- shader (SH) registers (absolute = kShRegBase + n) ---
// Pixel shader program address + resources + user data.
constexpr u32 mmSPI_SHADER_PGM_LO_PS = 0x2C08;
constexpr u32 mmSPI_SHADER_PGM_HI_PS = 0x2C09;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_PS = 0x2C0A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_PS = 0x2C0B;
constexpr u32 mmSPI_SHADER_USER_DATA_PS_0 = 0x2C0C;  // 16 user-data SGPRs
// Vertex shader program address + resources + user data.
constexpr u32 mmSPI_SHADER_PGM_LO_VS = 0x2C48;
constexpr u32 mmSPI_SHADER_PGM_HI_VS = 0x2C49;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_VS = 0x2C4A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_VS = 0x2C4B;
constexpr u32 mmSPI_SHADER_USER_DATA_VS_0 = 0x2C4C;  // 16 user-data SGPRs

// Compute program registers (SET_SH_REG relative 0x200..; absolute =
// 0x2C00+rel).
constexpr u32 mmCOMPUTE_NUM_THREAD_X = 0x2E07;  // u16 full | u16 partial
constexpr u32 mmCOMPUTE_NUM_THREAD_Y = 0x2E08;
constexpr u32 mmCOMPUTE_NUM_THREAD_Z = 0x2E09;
constexpr u32 mmCOMPUTE_PGM_LO = 0x2E0C;  // CS addr[39:8]
constexpr u32 mmCOMPUTE_PGM_HI = 0x2E0D;  // CS addr[47:40] in [7:0]
constexpr u32 mmCOMPUTE_PGM_RSRC1 = 0x2E12;
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

  // The register run starting at `off`, for the blocks read as arrays: the
  // 16-dword user-data SGPRs a shader's descriptors hang off, and the float
  // registers (viewport, clear values) copied out by value.
  const u32* At(u32 off) const { return &data[off]; }

  // 48-bit GPU address from a LO/HI register pair (HI holds the top bits << 0,
  // i.e. addr = ((u64)HI << 32 | LO) << 8 for shader program pointers).
  u64 ShaderAddr(u32 lo_reg) const {
    u64 lo = data[lo_reg];
    u64 hi = data[lo_reg + 1] & 0xFF;
    return ((hi << 32) | lo) << 8;
  }
  u64 CbColorBase(int rt = 0) const {
    return static_cast<u64>(data[mmCB_COLOR0_BASE + rt * kCbColorStride])
           << 8;
  }
};

}  // namespace gpu
