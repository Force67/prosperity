#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * How gfx10.3 register state becomes one renderer draw.
 *
 * The registers say almost nothing directly: a draw's vertex streams, constant
 * buffers and textures live behind descriptors the shaders load out of their
 * user-data SGPRs, so producing a DrawInfo is as much recovering what the
 * shaders are about to read (gpu/ps5/rdna resource tracking) as reading
 * registers. The packet walker only says which draw packet arrived.
 *
 * Two things are harder here than on the PS4. gfx10.3 has no hardware VS stage,
 * so the vertex program is the merged ES/GS NGG shader and can be bound through
 * either half; and a title's context state may arrive entirely through AGC
 * state blocks, which leave the viewport and target-size registers unwritten,
 * so the bound target's own dimensions have to stand in.
 */

#include "base/arch.h"

#include "gpu/ps5/agc_regs.h"
#include "gpu/rhi/command.h"

namespace gpu::ps5 {

// The parts of a draw that are not in the register file: the packet itself plus
// the index/instance state the IT_* packets before it set up.
struct DrawPacket {
  u32 op = 0;
  const u32* body = nullptr;
  u32 count = 0;
  u32 index_type = 0;     // IT_INDEX_TYPE: 0 = 16-bit, 1 = 32-bit, 2 = 8-bit
  u64 index_base = 0;     // IT_INDEX_BASE (DRAW_INDEX_2 carries its own)
  u32 index_max = 0;      // IT_INDEX_BUFFER_SIZE, bounding an offset draw
  u32 num_instances = 1;  // IT_NUM_INSTANCES
};

// Fill `d` from the packet and the live registers. False when no usable shader
// pair resolved: a draw whose shaders or descriptors we could not follow is
// dropped rather than rendered from garbage.
bool BuildDrawInfo(const Regs& regs,
                   const DrawPacket& packet,
                   rhi::DrawInfo& d);

}  // namespace gpu::ps5
