#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"

namespace gpu {

enum class Pm4Type : u32 { kType0 = 0, kType1 = 1, kType2 = 2, kType3 = 3 };

inline Pm4Type Pm4TypeOf(u32 hdr) {
  return static_cast<Pm4Type>(hdr >> 30);
}
inline u32 Pm4Count(u32 hdr) {
  return ((hdr >> 16) & 0x3FFF) + 1;
}
inline u32 Pm4Opcode(u32 hdr) {
  return (hdr >> 8) & 0xFF;
}
inline u32 Pm4Type0Reg(u32 hdr) {
  return hdr & 0xFFFF;
}

// NOLINTBEGIN(readability-identifier-naming)
enum Pm4It : u32 {
  IT_NOP = 0x10,
  IT_SET_BASE = 0x11,
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
  IT_DRAW_INDIRECT = 0x24,
  IT_DRAW_INDEX_INDIRECT = 0x25,
  IT_WAIT_REG_MEM = 0x3C,
  IT_INDIRECT_BUFFER = 0x3F,
  IT_INDIRECT_BUFFER_CNST = 0x33,
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
  IT_SET_SH_REG_INDEX = 0x9B,
  IT_WRITE_DATA = 0x37,
  IT_LOAD_CONST_RAM = 0x80,
  IT_WRITE_CONST_RAM = 0x81,
  IT_DUMP_CONST_RAM = 0x83,
  IT_INCREMENT_CE_COUNTER = 0x84,
  IT_INCREMENT_DE_COUNTER = 0x85,
  IT_WAIT_ON_CE_COUNTER = 0x86,
  IT_WAIT_ON_DE_COUNTER_DIFF = 0x88,
  IT_DUMP_CONST_RAM_OFFSET = 0x9E,
};
// NOLINTEND(readability-identifier-naming)

enum Pm4RegBase : u32 {
  kConfigRegBase = 0x2000,
  kShRegBase = 0x2C00,
  kContextRegBase = 0xA000,
  kUConfigRegBase = 0xC000,
};

constexpr u32 Pm4SetRegAddress(u32 base, u32 offset_and_index) {
  return base + (offset_and_index & 0xffff);
}

// The DCB and CNST variants both carry count-field 2 on the wire.
inline u32 Pm4IbHeader(u32 op) {
  return (3u << 30) | (2u << 16) | (op << 8);
}

}  // namespace gpu
