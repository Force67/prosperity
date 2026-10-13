#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"
#include "base/containers/array.h"

namespace gpu {

constexpr u32 kRegFileSize = 0xD000;

constexpr u32 mmCB_COLOR0_BASE = 0xA318;
constexpr u32 mmCB_COLOR0_PITCH = 0xA319;
constexpr u32 mmCB_COLOR0_SLICE = 0xA31A;
constexpr u32 mmCB_COLOR0_VIEW = 0xA31B;
constexpr u32 mmCB_COLOR0_INFO = 0xA31C;
constexpr u32 mmCB_COLOR0_ATTRIB = 0xA31D;
constexpr u32 mmCB_COLOR0_CMASK = 0xA31F;
constexpr u32 mmCB_COLOR0_CLEAR_WORD0 = 0xA323;
constexpr u32 mmCB_COLOR0_CLEAR_WORD1 = 0xA324;
constexpr u32 kCbColorStride = 0xF;
constexpr u32 mmPA_SC_SCREEN_SCISSOR_TL = 0xA00C;
constexpr u32 mmPA_SC_SCREEN_SCISSOR_BR = 0xA00D;
constexpr u32 mmPA_CL_CLIP_CNTL = 0xA204;
constexpr u32 mmPA_SC_VPORT_SCISSOR_0_TL = 0xA094;
constexpr u32 mmPA_SC_VPORT_SCISSOR_0_BR = 0xA095;
constexpr u32 mmPA_SC_WINDOW_SCISSOR_TL = 0xA081;
constexpr u32 mmPA_SC_WINDOW_SCISSOR_BR = 0xA082;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_TL = 0xA090;
constexpr u32 mmPA_SC_GENERIC_SCISSOR_BR = 0xA091;
constexpr u32 mmDB_RENDER_CONTROL = 0xA000;
constexpr u32 mmPA_CL_VPORT_XSCALE = 0xA10F;
constexpr u32 mmPA_CL_VPORT_XOFFSET = 0xA110;
constexpr u32 mmPA_CL_VPORT_YSCALE = 0xA111;
constexpr u32 mmPA_CL_VPORT_YOFFSET = 0xA112;
constexpr u32 mmPA_CL_VPORT_ZSCALE = 0xA113;
constexpr u32 mmPA_CL_VPORT_ZOFFSET = 0xA114;
constexpr u32 mmCB_TARGET_MASK = 0xA08E;
constexpr u32 mmCB_SHADER_MASK = 0xA08F;
constexpr u32 mmSPI_SHADER_COL_FORMAT = 0xA1C5;
constexpr u32 mmCB_BLEND0_CONTROL = 0xA1E0;
constexpr u32 kCbBlendStride = 0x1;
constexpr u32 mmCB_BLEND_RED = 0xA105;
constexpr u32 mmSPI_PS_INPUT_CNTL_0 = 0xA191;
constexpr u32 mmSPI_PS_INPUT_ENA = 0xA1B3;
// Slots at or above NUM_INTERP are don't-care, so their zero says nothing.
constexpr u32 mmSPI_PS_IN_CONTROL = 0xA1B6;
constexpr u32 mmCB_COLOR_CONTROL = 0xA202;
constexpr u32 mmVGT_SHADER_STAGES_EN = 0xA2D5;
constexpr u32 mmVGT_GS_OUT_PRIM_TYPE = 0xA29B;
constexpr u32 mmVGT_ESGS_RING_ITEMSIZE = 0xA2AB;
constexpr u32 mmVGT_GSVS_RING_ITEMSIZE = 0xA2AC;
constexpr u32 mmVGT_GS_MAX_VERT_OUT = 0xA2CE;
constexpr u32 mmVGT_GS_INSTANCE_CNT = 0xA2E4;
constexpr u32 mmVGT_PRIMITIVE_TYPE = 0xC242;
constexpr u32 mmVGT_NUM_INDICES = 0xC24C;

constexpr u32 mmDB_DEPTH_CONTROL = 0xA200;
constexpr u32 mmDB_Z_INFO = 0xA010;
constexpr u32 mmDB_STENCIL_INFO = 0xA011;
constexpr u32 mmDB_Z_READ_BASE = 0xA012;
constexpr u32 mmDB_STENCIL_READ_BASE = 0xA013;
constexpr u32 mmDB_Z_WRITE_BASE = 0xA014;
constexpr u32 mmDB_STENCIL_WRITE_BASE = 0xA015;
constexpr u32 mmDB_HTILE_DATA_BASE = 0xA005;
constexpr u32 mmDB_DEPTH_SIZE = 0xA016;
constexpr u32 mmDB_DEPTH_SLICE = 0xA017;
constexpr u32 mmDB_DEPTH_CLEAR = 0xA00B;
constexpr u32 mmDB_STENCIL_CLEAR = 0xA00A;
constexpr u32 mmDB_STENCIL_CONTROL = 0xA10B;
constexpr u32 mmDB_STENCILREFMASK = 0xA10C;
constexpr u32 mmDB_STENCILREFMASK_BF = 0xA10D;
constexpr u32 mmPA_SU_SC_MODE_CNTL = 0xA205;

constexpr u32 mmSPI_SHADER_PGM_LO_PS = 0x2C08;
constexpr u32 mmSPI_SHADER_PGM_HI_PS = 0x2C09;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_PS = 0x2C0A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_PS = 0x2C0B;
constexpr u32 mmSPI_SHADER_USER_DATA_PS_0 = 0x2C0C;
constexpr u32 mmSPI_SHADER_PGM_LO_VS = 0x2C48;
constexpr u32 mmSPI_SHADER_PGM_HI_VS = 0x2C49;
constexpr u32 mmSPI_SHADER_PGM_RSRC1_VS = 0x2C4A;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_VS = 0x2C4B;
constexpr u32 mmSPI_SHADER_USER_DATA_VS_0 = 0x2C4C;
constexpr u32 mmSPI_SHADER_PGM_LO_GS = 0x2C88;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_GS = 0x2C8B;
constexpr u32 mmSPI_SHADER_USER_DATA_GS_0 = 0x2C8C;
constexpr u32 mmSPI_SHADER_PGM_LO_ES = 0x2CC8;
constexpr u32 mmSPI_SHADER_PGM_RSRC2_ES = 0x2CCB;
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
    return ((hi << 32) | lo) << 8;
  }
  u64 CbColorBase(int rt = 0) const {
    return static_cast<u64>(data[mmCB_COLOR0_BASE + rt * kCbColorStride]) << 8;
  }
};

}  // namespace gpu
