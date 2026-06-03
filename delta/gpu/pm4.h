#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * PM4 command-buffer packet decoding. The PS4 GPU (Liverpool, GCN gen2) consumes
 * a stream of PM4 packets built by libSceGnmDriver. We walk that stream, track
 * the GPU register state the packets set, and translate draws to Vulkan.
 */

#include <cstdint>

namespace gpu {

// PM4 packet type is the top 2 bits of the header dword.
enum class Pm4Type : uint32_t { type0 = 0, type1 = 1, type2 = 2, type3 = 3 };

inline Pm4Type pm4Type(uint32_t hdr) {
  return static_cast<Pm4Type>(hdr >> 30);
}
// Dword count of the packet body (after the header).
inline uint32_t pm4Count(uint32_t hdr) { return ((hdr >> 16) & 0x3FFF) + 1; }
// Type-3 IT opcode.
inline uint32_t pm4Opcode(uint32_t hdr) { return (hdr >> 8) & 0xFF; }
// Type-0 base register (dword offset).
inline uint32_t pm4Type0Reg(uint32_t hdr) { return hdr & 0xFFFF; }

// PM4 IT_ opcodes (type-3). Subset the PS4 GPU actually uses.
enum Pm4It : uint32_t {
  IT_NOP = 0x10,
  IT_CLEAR_STATE = 0x12,
  IT_INDEX_BUFFER_SIZE = 0x13,
  IT_DISPATCH_DIRECT = 0x15,
  IT_DISPATCH_INDIRECT = 0x16,
  IT_SET_PREDICATION = 0x20,
  IT_COND_EXEC = 0x22,
  IT_INDEX_BASE = 0x26,
  IT_DRAW_INDEX_2 = 0x27,
  IT_CONTEXT_CONTROL = 0x28,
  IT_INDEX_TYPE = 0x2A,
  IT_DRAW_INDEX_AUTO = 0x2D,
  IT_NUM_INSTANCES = 0x2F,
  IT_DRAW_INDEX_MULTI_AUTO = 0x30,
  IT_DRAW_INDEX_OFFSET_2 = 0x35,
  IT_DRAW_INDEX_INDIRECT = 0x24,
  IT_WAIT_REG_MEM = 0x3C,
  IT_INDIRECT_BUFFER = 0x3F,
  IT_COPY_DATA = 0x40,
  IT_EVENT_WRITE = 0x46,
  IT_EVENT_WRITE_EOP = 0x47,
  IT_EVENT_WRITE_EOS = 0x48,
  IT_RELEASE_MEM = 0x49,
  IT_DMA_DATA = 0x50,
  IT_ACQUIRE_MEM = 0x58,
  IT_REWIND = 0x59,
  IT_SET_CONFIG_REG = 0x68,
  IT_SET_CONTEXT_REG = 0x69,
  IT_SET_SH_REG = 0x76,
  IT_SET_UCONFIG_REG = 0x79,
  IT_WRITE_DATA = 0x37,
  IT_WAIT_ON_CE_COUNTER = 0x86,
};

// Base dword offsets the SET_*_REG packets are relative to (the register at
// payload[0] is `base + payload[0]`). These index the unified Liverpool register
// file (see liverpool.h).
enum Pm4RegBase : uint32_t {
  kConfigRegBase = 0x2000,
  kShRegBase = 0x2C00,
  kContextRegBase = 0xA000,
  kUConfigRegBase = 0xC000,
};

}  // namespace gpu
