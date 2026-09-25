#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * How Liverpool register state becomes one renderer draw.
 *
 * The registers say almost nothing directly: a draw's vertex streams, constant
 * buffers and textures live behind descriptors the shaders load out of their
 * user-data SGPRs, so producing a DrawInfo is as much recovering what the
 * shaders are about to read (gpu/gcn resource tracking) as reading registers.
 * The packet walker only says which draw packet arrived.
 */

#include "base/arch.h"

#include "gpu/ps4/liverpool.h"
#include "gpu/rhi/command.h"
#include "gpu/rhi/renderer.h"

namespace gpu::ps4 {

// The parts of a draw that are not in the register file: the packet itself plus
// the index/instance state the IT_* packets before it set up.
struct DrawPacket {
  u32 op = 0;
  const u32* body = nullptr;
  u32 count = 0;
  u32 index_type = 0;     // IT_INDEX_TYPE: 0 = 16-bit, 1 = 32-bit
  u64 index_base = 0;     // IT_INDEX_BASE (DRAW_INDEX_2 carries its own)
  u64 indirect_base = 0;  // IT_SET_BASE(1): where indirect args live
  u32 num_instances = 1;  // IT_NUM_INSTANCES
  u32 frame = 0;          // presented frames + 1, for the frame gates
};

// Fill `d` from the packet and the live registers. False when nothing resolved
// a vertex source: a draw whose shaders or descriptors we could not follow is
// dropped rather than rendered from garbage.
bool BuildDrawInfo(rhi::Renderer& renderer,
                   const Regs& regs,
                   const DrawPacket& packet,
                   rhi::DrawInfo& d);

// Start compiling, on worker threads, the shaders a draw with these registers
// would need. For a look ahead through a command buffer before it runs.
void PrefetchDrawShaders(const Regs& regs);

}  // namespace gpu::ps4
