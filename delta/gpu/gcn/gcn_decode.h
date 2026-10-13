#pragma once

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/shared_pointer.h"

namespace gpu::gcn {

enum class IsaMode : u8 { kBase, kNeo };

enum class Enc : u8 {
  kUnknown,
  kSop1,
  kSop2,
  kSopk,
  kSopc,
  kSopp,
  kSmrd,
  kVop1,
  kVop2,
  kVop3,
  kVop3p,
  kVopc,
  kVintrp,
  kDs,
  kMubuf,
  kMtbuf,
  kMimg,
  kExp,
  kFlat,
};

enum class InstExtension : u8 {
  kNone,
  kLiteral,
  kSdwa,
  kDpp,
  kDpp8,
  kDpp8Fi,
};

struct Inst {
  IsaMode isa = IsaMode::kBase;
  Enc enc = Enc::kUnknown;
  u32 opcode = 0;
  u32 raw[5] = {};
  u32 size = 1;
  u32 pc = 0;
  bool has_literal = false;
  u32 literal = 0;
  InstExtension extension = InstExtension::kNone;
};

using Program = base::Vector<Inst>;

struct SmrdOffset {
  u32 dwords = 0;
  u32 sgpr = 0;
  bool in_sgpr = false;
};

inline SmrdOffset DecodeSmrdOffset(const Inst& inst) {
  const u32 w = inst.raw[0], field = w & 0xFF;
  SmrdOffset o;
  if ((w >> 8) & 1)
    o.dwords = field;
  else if (field == 0xFF && inst.has_literal)
    o.dwords = inst.literal;
  else {
    o.in_sgpr = true;
    o.sgpr = field;
  }
  return o;
}

class DescriptorVersions {
 public:
  static constexpr u32 kInline = 0xFFFFFFFFu;

  u32 Of(u32 sgpr, u32 dwords) const {
    if (sgpr >= kSgprs || !len_[sgpr])
      return kInline;
    const u32 first = start_[sgpr];
    return sgpr + dwords <= first + len_[sgpr] ? pc_[sgpr] : kInline;
  }

  void Note(const Inst& inst) {
    if (inst.enc != Enc::kSmrd)
      return;
    const u32 op = inst.opcode;
    const u32 n = op <= 0x04                 ? 1u << op
                  : op >= 0x08 && op <= 0x0c ? 1u << (op - 0x08)
                                             : 0u;
    if (!n)
      return;
    const u32 sdst = (inst.raw[0] >> 15) & 0x7F;
    const u32 end = base::Min(sdst + n, kSgprs);
    for (u32 s = sdst; s < end; s++)
      if (len_[s]) {
        const u32 first = start_[s], last = first + len_[s];
        for (u32 t = first; t < last; t++)
          len_[t] = 0;
      }
    for (u32 s = sdst; s < end; s++) {
      start_[s] = static_cast<u8>(sdst);
      len_[s] = static_cast<u8>(n);
      pc_[s] = inst.pc;
    }
  }

 private:
  static constexpr u32 kSgprs = 128;
  u8 len_[kSgprs] = {};
  u8 start_[kSgprs];
  u32 pc_[kSgprs];
};

IsaMode DefaultIsaMode();
void SetDefaultIsaMode(IsaMode mode);

Program Decode(const u32* code,
               u32 max_dwords,
               bool stop_at_endpgm = true,
               IsaMode mode = DefaultIsaMode());

u32 CodeLength(const u32* code, u32 max_dwords);

Program DecodeShader(const u32* code,
                     u32 max_dwords,
                     IsaMode mode = DefaultIsaMode());

base::Vector<u8> ComputeReachability(const Program& program);

base::SharedPointer<const Program> CachedProgram(u64 addr, u32 max_dwords);

u64 CachedCodeHash(u64 addr, u32 max_dwords);

bool CallsFetchShader(const Program& program);

void NextProgramCacheGeneration();

}  // namespace gpu::gcn
