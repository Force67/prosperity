#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/render/command.h"
#include "gpu/render/renderer.h"

namespace gpu::ps4 {

struct DrawPacket {
  u32 op = 0;
  const u32* body = nullptr;
  u32 count = 0;
  u32 index_type = 0;
  u64 index_base = 0;
  u64 indirect_base = 0;
  u32 index_max = 0;
  u32 num_instances = 1;
  u32 frame = 0;
};

bool BuildDrawInfo(render::Renderer& renderer,
                   const Regs& regs,
                   const DrawPacket& packet,
                   render::DrawInfo& d);

void PrefetchDrawShaders(const Regs& regs);

void FillDrawTex(render::DrawInfo::DrawTex& dt, const gcn::TImage& t);

}  // namespace gpu::ps4
