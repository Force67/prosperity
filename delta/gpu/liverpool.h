#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Liverpool GPU register file. The PM4 SET_*_REG packets write into a unified
 * register space; we mirror it as a flat dword array indexed by absolute
 * register offset (base + packet offset). Draw handlers read the relevant
 * registers (render target, shader pointers, primitive state) out of it.
 *
 * Register offsets below are GCN gen2 (Sea Islands / Liverpool) values, the same
 * the PS4 Gnm driver programs. Verified against the AMD PM4 / GCN docs.
 */

#include <array>
#include <cstdint>

namespace gpu {

// The unified register file is sparse but small enough to store flat. 0xD000
// dwords covers config(0x2000)/sh(0x2C00)/context(0xA000)/uconfig(0xC000).
constexpr uint32_t kRegFileSize = 0xD000;

// --- key context registers (absolute dword offset = kContextRegBase + n) ---
// Color buffer 0 (the render target). CB_COLORn are a 0xF-dword stride apart.
constexpr uint32_t mmCB_COLOR0_BASE = 0xA318;
constexpr uint32_t mmCB_COLOR0_PITCH = 0xA319;
constexpr uint32_t mmCB_COLOR0_SLICE = 0xA31A;
constexpr uint32_t mmCB_COLOR0_VIEW = 0xA31B;
constexpr uint32_t mmCB_COLOR0_INFO = 0xA31C;   // format/number-type
constexpr uint32_t mmCB_COLOR0_ATTRIB = 0xA31D; // tiling/dims
constexpr uint32_t kCbColorStride = 0xF;
// Screen scissor gives the render area (width/height).
constexpr uint32_t mmPA_SC_SCREEN_SCISSOR_TL = 0xA00C;
constexpr uint32_t mmPA_SC_SCREEN_SCISSOR_BR = 0xA00D;
constexpr uint32_t mmPA_SC_GENERIC_SCISSOR_TL = 0xA090;
constexpr uint32_t mmPA_SC_GENERIC_SCISSOR_BR = 0xA091;
// Viewport 0 scale/offset (float).
constexpr uint32_t mmPA_CL_VPORT_XSCALE = 0xA10F;
// Render-target mask (which CB targets are written).
constexpr uint32_t mmCB_TARGET_MASK = 0xA08E;
constexpr uint32_t mmCB_SHADER_MASK = 0xA08F;
// Per-MRT blend control. CB_BLENDn_CONTROL are 1 dword apart. Layout (GCN gen2):
//  [0:4] color_src_factor  [5:7] color_func   [8:12] color_dst_factor
//  [16:20] alpha_src_factor [21:23] alpha_func [24:28] alpha_dst_factor
//  [29] separate_alpha_blend  [30] enable
constexpr uint32_t mmCB_BLEND0_CONTROL = 0xA1E0;
constexpr uint32_t kCbBlendStride = 0x1;
// Overall color-buffer mode (ROP3 / blend disable). MODE field is [4:6].
constexpr uint32_t mmCB_COLOR_CONTROL = 0xA202;
// Primitive type for the draw (VGT_PRIMITIVE_TYPE is a uconfig reg on gen2).
constexpr uint32_t mmVGT_PRIMITIVE_TYPE = 0xC242;
constexpr uint32_t mmVGT_NUM_INDICES = 0xC24C;

// --- shader (SH) registers (absolute = kShRegBase + n) ---
// Pixel shader program address + resources + user data.
constexpr uint32_t mmSPI_SHADER_PGM_LO_PS = 0x2C08;
constexpr uint32_t mmSPI_SHADER_PGM_HI_PS = 0x2C09;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC1_PS = 0x2C0A;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC2_PS = 0x2C0B;
constexpr uint32_t mmSPI_SHADER_USER_DATA_PS_0 = 0x2C0C;  // 16 user-data SGPRs
// Vertex shader program address + resources + user data.
constexpr uint32_t mmSPI_SHADER_PGM_LO_VS = 0x2C48;
constexpr uint32_t mmSPI_SHADER_PGM_HI_VS = 0x2C49;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC1_VS = 0x2C4A;
constexpr uint32_t mmSPI_SHADER_PGM_RSRC2_VS = 0x2C4B;
constexpr uint32_t mmSPI_SHADER_USER_DATA_VS_0 = 0x2C4C;  // 16 user-data SGPRs

// The full GPU register state. A draw is rendered from a snapshot of this.
struct Regs {
  std::array<uint32_t, kRegFileSize> data{};

  uint32_t &operator[](uint32_t off) { return data[off]; }
  uint32_t operator[](uint32_t off) const { return off < kRegFileSize ? data[off] : 0; }

  // 48-bit GPU address from a LO/HI register pair (HI holds the top bits << 0,
  // i.e. addr = ((u64)HI << 32 | LO) << 8 for shader program pointers).
  uint64_t shaderAddr(uint32_t loReg) const {
    uint64_t lo = data[loReg];
    uint64_t hi = data[loReg + 1] & 0xFF;
    return ((hi << 32) | lo) << 8;
  }
  uint64_t cbColorBase(int rt = 0) const {
    return static_cast<uint64_t>(data[mmCB_COLOR0_BASE + rt * kCbColorStride]) << 8;
  }
};

}  // namespace gpu
