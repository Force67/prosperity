#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/ps5/agc_regs.h"
#include "gpu/render/command.h"

namespace gpu::ps5 {

struct DrawPacket {
  u32 op = 0;
  const u32* body = nullptr;
  u32 count = 0;
  u32 index_type = 0;
  u64 index_base = 0;
  u32 index_max = 0;
  u32 num_instances = 1;
  u64 indirect_args = 0;
};

bool BuildDrawInfo(const Regs& regs,
                   const DrawPacket& packet,
                   render::DrawInfo& d);

void TexFromImage(const gcn::TImage& image, render::DrawInfo::DrawTex& tex);

}  // namespace gpu::ps5
