#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"
#include "base/containers/array.h"

namespace gpu::ps5 {

constexpr u32 kRegFileSize = 0xD000;

constexpr u32 kRegSelectorMask = 0x70000000u;

constexpr u32 mmCB_COLOR0_BASE = 0xA318;
constexpr u32 mmCB_COLOR0_VIEW = 0xA31B;
constexpr u32 mmCB_COLOR0_INFO = 0xA31C;
constexpr u32 mmCB_COLOR0_ATTRIB = 0xA31D;
constexpr u32 kCbColorStride = 0xF;
constexpr u32 mmCB_COLOR0_CLEAR_WORD0 = 0xA323;
constexpr u32 mmCB_COLOR0_CLEAR_WORD1 = 0xA324;
constexpr u32 mmCB_COLOR0_DCC_BASE = 0xA325;
constexpr u32 mmCB_COLOR0_DCC_BASE_EXT = 0xA3A8;
constexpr u32 mmCB_COLOR0_BASE_EXT = 0xA390;
constexpr u32 mmCB_COLOR0_ATTRIB2 = 0xA3B0;
constexpr u32 mmCB_COLOR0_ATTRIB3 = 0xA3B8;
constexpr u32 kCbColorExtStride = 0x1;
constexpr u32 mmPA_SC_SCREEN_SCISSOR_TL = 0xA00C;
constexpr u32 mmPA_SC_SCREEN_SCISSOR_BR = 0xA00D;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_TL = 0xA090;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_BR = 0xA091;
constexpr u32 mmPA_CL_VPORT_XSCALE = 0xA10F;
constexpr u32 mmPA_CL_VPORT_XOFFSET = 0xA110;
constexpr u32 mmPA_CL_VPORT_YSCALE = 0xA111;
constexpr u32 mmPA_CL_VPORT_YOFFSET = 0xA112;
constexpr u32 mmPA_CL_VPORT_ZSCALE = 0xA113;
constexpr u32 mmPA_CL_VPORT_ZOFFSET = 0xA114;
constexpr u32 mmCB_TARGET_MASK = 0xA08E;
constexpr u32 mmCB_SHADER_MASK = 0xA08F;
constexpr u32 mmCB_BLEND0_CONTROL = 0xA1E0;
constexpr u32 kCbBlendStride = 0x1;
constexpr u32 mmCB_BLEND_RED = 0xA105;
constexpr u32 mmCB_COLOR_CONTROL = 0xA202;

constexpr u32 mmDB_RENDER_CONTROL = 0xA000;
constexpr u32 mmDB_DEPTH_CONTROL = 0xA200;
constexpr u32 mmDB_DEPTH_VIEW = 0xA002;
constexpr u32 mmDB_Z_INFO = 0xA010;
constexpr u32 mmDB_Z_READ_BASE = 0xA012;
constexpr u32 mmDB_Z_WRITE_BASE = 0xA014;
constexpr u32 mmDB_Z_READ_BASE_HI = 0xA01A;
constexpr u32 mmDB_Z_WRITE_BASE_HI = 0xA01C;
constexpr u32 mmDB_STENCIL_INFO = 0xA011;
constexpr u32 mmDB_STENCIL_WRITE_BASE = 0xA015;
constexpr u32 mmDB_STENCIL_WRITE_BASE_HI = 0xA01D;
constexpr u32 mmDB_STENCIL_CLEAR = 0xA00A;
constexpr u32 mmDB_STENCIL_CONTROL = 0xA10B;
constexpr u32 mmDB_STENCILREFMASK = 0xA10C;
constexpr u32 mmDB_STENCILREFMASK_BF = 0xA10D;
constexpr u32 mmDB_HTILE_DATA_BASE = 0xA005;
constexpr u32 mmDB_HTILE_DATA_BASE_HI = 0xA01E;
constexpr u32 mmDB_DEPTH_CLEAR = 0xA00B;
constexpr u32 mmPA_SU_SC_MODE_CNTL = 0xA205;
constexpr u32 mmPA_CL_CLIP_CNTL = 0xA204;

constexpr u32 mmSPI_VS_OUT_CONFIG = 0xA1B1;
constexpr u32 mmSPI_PS_INPUT_CNTL_0 = 0xA191;
constexpr u32 mmSPI_PS_INPUT_ENA = 0xA1B3;
constexpr u32 mmSPI_PS_INPUT_ADDR = 0xA1B4;
constexpr u32 mmSPI_PS_IN_CONTROL = 0xA1B6;
constexpr u32 mmSPI_SHADER_POS_FORMAT = 0xA1C3;
constexpr u32 mmSPI_SHADER_Z_FORMAT = 0xA1C4;
constexpr u32 mmSPI_SHADER_COL_FORMAT = 0xA1C5;
constexpr u32 mmPA_CL_VS_OUT_CNTL = 0xA207;
constexpr u32 mmVGT_SHADER_STAGES_EN = 0xA2D5;

constexpr u32 mmVGT_PRIMITIVE_TYPE = 0xC242;
constexpr u32 mmVGT_INDEX_TYPE = 0xC243;
constexpr u32 mmGE_CNTL = 0xC25B;
constexpr u32 mmVGT_GS_ONCHIP_CNTL = 0xA291;
constexpr u32 mmVGT_GS_MAX_VERT_OUT = 0xA2CE;
constexpr u32 mmGE_NGG_SUBGRP_CNTL = 0xA2D3;

constexpr u32 mmSPI_SHADER_PGM_LO_PS = 0x2C08;
constexpr u32 mmSPI_SHADER_PGM_HI_PS = 0x2C09;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_PS = 0x2C0A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_PS = 0x2C0B;
constexpr u32 mmSPI_SHADER_USER_DATA_PS_0 = 0x2C0C;

constexpr u32 mmSPI_SHADER_PGM_LO_GS = 0x2C88;
constexpr u32 mmSPI_SHADER_USER_DATA_ADDR_LO_GS = 0x2C82;
constexpr u32 mmSPI_SHADER_USER_DATA_ADDR_HI_GS = 0x2C83;
constexpr u32 mmSPI_SHADER_PGM_HI_GS = 0x2C89;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_GS = 0x2C8A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_GS = 0x2C8B;
constexpr u32 mmSPI_SHADER_USER_DATA_GS_0 = 0x2C8C;
constexpr u32 mmSPI_SHADER_PGM_LO_ES = 0x2CC8;
constexpr u32 mmSPI_SHADER_PGM_HI_ES = 0x2CC9;
constexpr u32 mmSPI_SHADER_USER_DATA_ES_0 = 0x2CCC;

constexpr u32 mmCOMPUTE_START_X = 0x2E04;
constexpr u32 mmCOMPUTE_START_Y = 0x2E05;
constexpr u32 mmCOMPUTE_START_Z = 0x2E06;
constexpr u32 mmCOMPUTE_NUM_THREAD_X = 0x2E07;
constexpr u32 mmCOMPUTE_NUM_THREAD_Y = 0x2E08;
constexpr u32 mmCOMPUTE_NUM_THREAD_Z = 0x2E09;
constexpr u32 mmCOMPUTE_PGM_LO = 0x2E0C;
constexpr u32 mmCOMPUTE_PGM_HI = 0x2E0D;
constexpr u32 mmCOMPUTE_PGM_RSRC1 = 0x2E12;
constexpr u32 mmCOMPUTE_PGM_RSRC2 = 0x2E13;
constexpr u32 mmCOMPUTE_USER_DATA_0 = 0x2E40;

struct Regs {
  base::Array<u32, kRegFileSize> data{};

  u32& operator[](u32 off) { return data[off]; }
  u32 operator[](u32 off) const { return off < kRegFileSize ? data[off] : 0; }

  const u32* At(u32 off) const { return &data[off]; }

  u64 ShaderAddr(u32 lo_reg) const {
    u64 lo = data[lo_reg];
    u64 hi = data[lo_reg + 1] & 0xFF;
    return (lo << 8) | (hi << 40);
  }
  u64 CbColorBase(int rt = 0) const {
    u64 lo = data[mmCB_COLOR0_BASE + rt * kCbColorStride];
    u64 hi = data[mmCB_COLOR0_BASE_EXT + rt * kCbColorExtStride];
    return ((hi << 32) | lo) << 8;
  }
};

}  // namespace gpu::ps5
