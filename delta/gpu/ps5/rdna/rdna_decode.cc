/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/ps5/rdna/rdna_decode.h"
#include "base/arch.h"
#include "guest/session.h"

#include <cstdio>
#include "base/algorithm.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/memory/shared_pointer.h"

namespace gpu::rdna {
namespace {

bool BinaryInfoAt(const u32* code, u32 off) {
  const char* s = reinterpret_cast<const char*>(code) + off;
  return s[0] == 'O' && s[1] == 'r' && s[2] == 'b' && s[3] == 'S' &&
         s[4] == 'h' && s[5] == 'd' && s[6] == 'r';
}

bool Scalar2HasLit(u32 w) {
  return (w & 0xFF) == 255 || ((w >> 8) & 0xFF) == 255;
}
bool Scalar1HasLit(u32 w) {
  return (w & 0xFF) == 255;
}
gpu::gcn::InstExtension ValuExtension(Enc enc, u32 w) {
  switch (w & 0x1FF) {
    case 233:
      return enc == Enc::kVop1 || enc == Enc::kVop2 || enc == Enc::kVopc
                 ? gpu::gcn::InstExtension::kDpp8
                 : gpu::gcn::InstExtension::kNone;
    case 234:
      return enc == Enc::kVop1 || enc == Enc::kVop2 || enc == Enc::kVopc
                 ? gpu::gcn::InstExtension::kDpp8Fi
                 : gpu::gcn::InstExtension::kNone;
    case 249:
      return gpu::gcn::InstExtension::kSdwa;
    case 250:
      return enc == Enc::kVop1 || enc == Enc::kVop2 || enc == Enc::kVopc
                 ? gpu::gcn::InstExtension::kDpp
                 : gpu::gcn::InstExtension::kNone;
    case 255:
      return gpu::gcn::InstExtension::kLiteral;
    default:
      return gpu::gcn::InstExtension::kNone;
  }
}

Enc Classify(u32 w, u32& opcode) {
  if ((w & 0x80000000u) == 0u) {
    u32 vop2_op = (w >> 25) & 0x3F;
    if (vop2_op == 0x3F) {
      opcode = (w >> 9) & 0xFF;
      return Enc::kVop1;
    }
    if (vop2_op == 0x3E) {
      opcode = (w >> 17) & 0xFF;
      return Enc::kVopc;
    }
    opcode = vop2_op;
    return Enc::kVop2;
  }
  if ((w & 0xC0000000u) == 0x80000000u) {
    u32 sub = (w >> 23) & 0x7F;
    if (sub == 0x7D) {
      opcode = (w >> 8) & 0xFF;
      return Enc::kSop1;
    }
    if (sub == 0x7E) {
      opcode = (w >> 16) & 0x7F;
      return Enc::kSopc;
    }
    if (sub == 0x7F) {
      opcode = (w >> 16) & 0x7F;
      return Enc::kSopp;
    }
    if (sub >= 0x60) {
      opcode = (w >> 23) & 0x1F;
      return Enc::kSopk;
    }
    opcode = (w >> 23) & 0x7F;
    return Enc::kSop2;
  }
  switch (w >> 26) {
    case 0x32:
      opcode = (w >> 16) & 0x3;
      return Enc::kVintrp;
    case 0x33:
      if ((w >> 24) != 0xCC || (w & (1u << 23))) {
        opcode = 0;
        return Enc::kUnknown;
      }
      opcode = (w >> 16) & 0x7F;
      return Enc::kVop3p;
    case 0x35:
      opcode = (w >> 16) & 0x3FF;
      return Enc::kVop3;
    case 0x36:
      opcode = (w >> 18) & 0xFF;
      return Enc::kDs;
    case 0x37:
      opcode = (w >> 18) & 0x7F;
      return Enc::kFlat;
    case 0x38:
      opcode = ((w >> 18) & 0x7F) | (((w >> 25) & 1) << 7);
      return Enc::kMubuf;
    case 0x3A:
      opcode = (w >> 16) & 0x7;
      return Enc::kMtbuf;
    case 0x3C:
      opcode = ((w >> 18) & 0x7F) | ((w & 1) << 7);
      return Enc::kMimg;
    case 0x3D:
      opcode = (w >> 18) & 0xFF;
      return Enc::kSmrd;
    case 0x31:
      if (w & 0xF) {
        opcode = 0;
        return Enc::kUnknown;
      }
      opcode = (w >> 4) & 0x3F;
      return Enc::kExp;
    case 0x3E:
      opcode = (w >> 4) & 0x3F;
      return Enc::kExp;
    default:
      opcode = 0;
      return Enc::kUnknown;
  }
}

u32 BaseSize(Enc e, u32 w) {
  if ((w >> 26) == 0x31)
    return 2;
  switch (e) {
    case Enc::kVop3:
    case Enc::kVop3p:
    case Enc::kSmrd:
    case Enc::kDs:
    case Enc::kFlat:
    case Enc::kMubuf:
    case Enc::kMtbuf:
    case Enc::kExp:
      return 2;
    case Enc::kMimg: {
      u32 nsa = (w >> 1) & 0x3;
      return 2 + nsa;
    }
    default:
      return 1;
  }
}

u32 Vop3SourceCountImpl(Enc enc, u32 op) {
  if (enc == Enc::kVop3p) {
    switch (op) {
      case 0x00:
      case 0x09:
      case 0x0e:
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
      case 0x17:
      case 0x18:
      case 0x19:
      case 0x20:
      case 0x21:
      case 0x22:
        return 3;
      default:
        return 2;
    }
  }
  if (op < 0x100)
    return 2;
  if (op == 0x101 || (op >= 0x128 && op <= 0x12a))
    return 3;
  if (op >= 0x100 && op < 0x140)
    return 2;
  if (op >= 0x180 && op < 0x200)
    return 1;
  if (op >= 0x270 && op <= 0x272)
    return 2;
  return 3;
}

base::Vector<u8> ComputeRdnaReachability(const Program& program) {
  base::Vector<u8> reachable(program.size(), 0);
  if (program.empty())
    return reachable;

  bool has_call = false;
  for (const Inst& inst : program)
    if (inst.enc == Enc::kSopk && inst.opcode == 0x16) {
      has_call = true;
      break;
    }
  const auto branch_kind = [has_call](const Inst& inst) {
    if (inst.enc == Enc::kSopk) {
      if (inst.opcode == 0x16 || inst.opcode == 0x1b || inst.opcode == 0x1c)
        return 2;
      return 0;
    }
    if (inst.enc == Enc::kSop1) {
      if (inst.opcode == 0x20 && has_call)
        return 3;
      if (inst.opcode >= 0x20 && inst.opcode <= 0x22)
        return 4;
      return 0;
    }
    if (inst.enc != Enc::kSopp)
      return 0;
    switch (inst.opcode) {
      case 0x01:
      case 0x1b:
      case 0x1e:
      case 0x1f:
        return 3;
      case 0x02:
        return 1;
      case 0x04:
      case 0x05:
      case 0x06:
      case 0x07:
      case 0x08:
      case 0x09:
        return 2;
      case 0x17:
      case 0x18:
      case 0x19:
      case 0x1a:
        return 0;
      default:
        return 0;
    }
  };

  const u32 max_pc = program.back().pc + program.back().size;
  base::Vector<u32> starts{0};
  for (const Inst& inst : program) {
    const int kind = branch_kind(inst);
    if (!kind)
      continue;
    starts.push_back(inst.pc + inst.size);
    if (kind == 1 || kind == 2) {
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xffff);
      starts.push_back(static_cast<u32>(static_cast<i32>(inst.pc) +
                                        static_cast<i32>(inst.size) + simm));
    }
  }
  base::Sort(starts.begin(), starts.end());
  starts.erase(base::Unique(starts.begin(), starts.end()), starts.end());
  starts.erase(base::RemoveIf(starts.begin(), starts.end(),
                              [max_pc](u32 pc) { return pc >= max_pc; }),
               starts.end());
  const auto block_of = [&](u32 pc) {
    u32 block = 0;
    for (u32 i = 0; i < starts.size() && starts[i] <= pc; i++)
      block = i;
    return block;
  };

  base::Vector<u8> block_reachable(starts.size(), 0);
  base::Vector<u32> worklist{0};
  while (!worklist.empty()) {
    const u32 block = worklist.back();
    worklist.pop_back();
    if (block >= starts.size() || block_reachable[block])
      continue;
    block_reachable[block] = 1;
    const u32 block_end =
        block + 1 < starts.size() ? starts[block + 1] : max_pc;
    bool terminated = false;
    for (const Inst& inst : program) {
      if (inst.pc < starts[block] || inst.pc >= block_end)
        continue;
      const int kind = branch_kind(inst);
      if (!kind)
        continue;
      terminated = true;
      if (kind == 3)
        break;
      if (kind == 4) {
        base::Fill(block_reachable.begin(), block_reachable.end(), 1);
        worklist.clear();
        break;
      }
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xffff);
      const u32 target = static_cast<u32>(static_cast<i32>(inst.pc) +
                                          static_cast<i32>(inst.size) + simm);
      if (target < max_pc)
        worklist.push_back(block_of(target));
      if (kind == 2)
        worklist.push_back(block + 1);
      break;
    }
    if (!terminated)
      worklist.push_back(block + 1);
  }

  for (u32 i = 0; i < program.size(); i++)
    reachable[i] = block_reachable[block_of(program[i].pc)];
  return reachable;
}

}  // namespace

u32 Vop3SourceCount(Enc enc, u32 op) {
  return Vop3SourceCountImpl(enc, op);
}

u32 CodeLength(const u32* code, u32 max_dwords) {
  if (!code || max_dwords < 2)
    return 0;
  if (code[0] == 0xBEEB03FFu) {
    u64 d = static_cast<u64>(code[1] + 1) * 2;
    if (d >= 2 && d + 2 <= max_dwords &&
        BinaryInfoAt(code, static_cast<u32>(d) * 4))
      return static_cast<u32>(d);
  }
  for (u32 d = 1; d + 2 <= max_dwords; d++)
    if (BinaryInfoAt(code, d * 4))
      return d;
  return 0;
}

Program Decode(const u32* code, u32 max_dwords, bool stop_at_endpgm) {
  Program out;
  if (!code)
    return out;
  u32 furthest_target = 0;
  u32 i = 0;
  while (i < max_dwords) {
    Inst in;
    in.pc = i;
    in.raw[0] = code[i];
    in.enc = Classify(code[i], in.opcode);
    in.size = BaseSize(in.enc, code[i]);
    if (i + in.size > max_dwords) {
      in.enc = Enc::kUnknown;
      in.opcode = 0;
      in.size = max_dwords - i;
      for (u32 d = 1; d < in.size; d++)
        in.raw[d] = code[i + d];
      out.push_back(in);
      break;
    }
    for (u32 d = 1; d < in.size; d++)
      in.raw[d] = code[i + d];
    if (in.enc == Enc::kMtbuf)
      in.opcode |= ((in.raw[1] >> 21) & 1) << 3;

    bool lit = false;
    switch (in.enc) {
      case Enc::kSop2:
      case Enc::kSopc:
        lit = Scalar2HasLit(code[i]);
        break;
      case Enc::kSop1:
        lit = Scalar1HasLit(code[i]);
        break;
      case Enc::kSopk:
        lit = in.opcode == 0x15;
        break;
      case Enc::kVop2:
        in.extension = ValuExtension(in.enc, code[i]);
        lit = in.extension != gpu::gcn::InstExtension::kNone ||
              in.opcode == 0x20 || in.opcode == 0x21 || in.opcode == 0x2C ||
              in.opcode == 0x2D || in.opcode == 0x37 || in.opcode == 0x38;
        break;
      case Enc::kVop1:
      case Enc::kVopc:
        in.extension = ValuExtension(in.enc, code[i]);
        lit = in.extension != gpu::gcn::InstExtension::kNone;
        break;
      case Enc::kVop3:
      case Enc::kVop3p:
        for (u32 src = 0; src < Vop3SourceCount(in.enc, in.opcode); src++)
          lit |= ((in.raw[1] >> (src * 9)) & 0x1ff) == 255;
        break;
      default:
        break;
    }
    if (lit && i + in.size < max_dwords) {
      const u32 extra = code[i + in.size];
      if (in.extension != gpu::gcn::InstExtension::kNone &&
          in.extension != gpu::gcn::InstExtension::kLiteral) {
        in.raw[in.size] = extra;
      } else {
        in.has_literal = true;
        in.literal = extra;
        in.extension = gpu::gcn::InstExtension::kLiteral;
      }
      in.size += 1;
    } else if (lit) {
      in.enc = Enc::kUnknown;
      in.opcode = 0;
    }
    if (in.size == 0)
      in.size = 1;

    out.push_back(in);

    if (in.enc == Enc::kSopp) {
      const bool branch = in.opcode == 0x02 ||
                          (in.opcode >= 0x04 && in.opcode <= 0x09) ||
                          (in.opcode >= 0x17 && in.opcode <= 0x1A);
      if (branch) {
        const int offset = static_cast<i16>(code[i] & 0xFFFF);
        const long long target = static_cast<long long>(i) + in.size + offset;
        if (target > 0 && static_cast<u64>(target) > furthest_target &&
            static_cast<u64>(target) < max_dwords)
          furthest_target = static_cast<u32>(target);
      }
      if (in.opcode == 0x1F ||
          (stop_at_endpgm && i + in.size > furthest_target &&
           (in.opcode == 0x01 || in.opcode == 0x1B || in.opcode == 0x1E)))
        break;
    }
    i += in.size;
  }
  return out;
}

Program DecodeShader(const u32* code, u32 max_dwords) {
  u32 len = CodeLength(code, max_dwords);
  if (len && len <= max_dwords)
    return Decode(code, len, false);
  return Decode(code, max_dwords, true);
}

Program ReachableProgram(const Program& program) {
  const base::Vector<u8> reachable = ComputeRdnaReachability(program);
  Program out;
  out.reserve(program.size());
  for (u32 i = 0; i < program.size(); i++)
    if (reachable[i])
      out.push_back(program[i]);
  return out;
}

namespace {

u64 g_generation = 1;

u64 HashCode(const u32* code, u32 dwords) {
  u64 h = 1469598103934665603ull;
  for (u32 i = 0; i < dwords; i++)
    h = (h ^ code[i]) * 1099511628211ull;
  return h;
}

}  // namespace

void NextProgramGeneration() {
  g_generation++;
}

base::SharedPointer<const Program> CachedReachableProgram(const u32* code,
                                                          u32 max_dwords) {
  struct Entry {
    u64 hash = 0;
    u32 hashed_dwords = 0;
    u64 generation = 0;
    base::SharedPointer<const Program> program;
  };
  static base::HashMap<u64, Entry> cache;
  static const guest::SessionReset reset_cache(
      [] { guest::ResetResource(cache); });

  if (!code)
    return base::MakeShared<const Program>();
  const u64 addr = reinterpret_cast<u64>(code);

  auto it = cache.find(addr);
  if (it != cache.end() && it->second.generation == g_generation)
    return it->second.program;
  if (it != cache.end() &&
      it->second.hash == HashCode(code, it->second.hashed_dwords)) {
    it->second.generation = g_generation;
    return it->second.program;
  }

  u32 len = CodeLength(code, max_dwords);
  if (!len) {
    len = max_dwords;
    for (const Inst& inst : Decode(code, max_dwords, true))
      if (inst.enc == Enc::kSopp && inst.opcode == 0x01) {
        len = inst.pc + inst.size;
        break;
      }
  }
  const u32 hashed = len;
  const u64 hash = HashCode(code, hashed);
  if (it != cache.end() && it->second.hash == hash &&
      it->second.hashed_dwords == hashed) {
    it->second.generation = g_generation;
    return it->second.program;
  }

  if (cache.size() > 16384)
    cache.clear();
  auto program = base::MakeShared<const Program>(
      ReachableProgram(DecodeShader(code, max_dwords)));
  cache[addr] = {hash, hashed, g_generation, program};
  return program;
}

u64 CachedCodeHash(const u32* code, u32 max_dwords) {
  struct Entry {
    u64 hash = 0;
    u32 len = 0;
    u64 generation = 0;
  };
  static base::HashMap<u64, Entry> cache;
  static const guest::SessionReset reset_cache(
      [] { guest::ResetResource(cache); });

  if (!code)
    return 0;
  const u64 addr = reinterpret_cast<u64>(code);
  auto it = cache.find(addr);
  if (it != cache.end() && it->second.generation == g_generation)
    return it->second.hash;
  if (it != cache.end() && it->second.hash == HashCode(code, it->second.len)) {
    it->second.generation = g_generation;
    return it->second.hash;
  }

  u32 len = CodeLength(code, max_dwords);
  if (!len) {
    len = max_dwords;
    for (const Inst& inst : Decode(code, max_dwords, true)) {
      const bool ends = inst.enc == Enc::kSopp && inst.opcode == 0x01;
      if (ends) {
        len = inst.pc + inst.size;
        break;
      }
    }
  }
  const u64 hash = HashCode(code, len);
  if (cache.size() > 16384)
    cache.clear();
  cache[addr] = {hash, len, g_generation};
  return hash;
}

}  // namespace gpu::rdna
