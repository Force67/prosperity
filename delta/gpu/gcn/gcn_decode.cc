/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN GFX7 instruction decoder. See gcn_decode.h.
 */

#include "gpu/gcn/gcn_decode.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace gpu::gcn {
namespace {

// The 'OrbShdr' ShaderBinaryInfo signature at byte offset `off` in `code`?
bool OrbShdrAt(const uint32_t* code, uint32_t off) {
  const char* s = reinterpret_cast<const char*>(code) + off;
  return std::memcmp(s, "OrbShdr", 7) == 0;
}

// A 32-bit-encoded scalar/vector op carries a trailing 32-bit literal when a
// source-operand field selects LITERAL_CONST (255).
bool Sop2HasLiteral(uint32_t w) {
  return (w & 0xFF) == 255 || ((w >> 8) & 0xFF) == 255;
}
bool Sop1HasLiteral(uint32_t w) {
  return (w & 0xFF) == 255;
}
InstExtension VopExtension(uint32_t w, IsaMode mode) {
  switch (w & 0x1ff) {
    case 249:
      return mode == IsaMode::kNeo ? InstExtension::kSdwa
                                   : InstExtension::kNone;
    case 250:
      return mode == IsaMode::kNeo ? InstExtension::kDpp : InstExtension::kNone;
    case 255:
      return InstExtension::kLiteral;
    default:
      return InstExtension::kNone;
  }
}

// Encoding classification by the fixed top bits. Returns the family and fills
// the encoding-relative opcode.
Enc Classify(uint32_t w, IsaMode mode, uint32_t& opcode) {
  if ((w >> 30) == 0x2) {  // 10b => scalar
    const uint32_t top9 = w >> 23;
    if (top9 == 0x17F) {
      opcode = (w >> 16) & 0x7F;
      return Enc::kSopp;
    }
    if (top9 == 0x17E) {
      opcode = (w >> 16) & 0x7F;
      return Enc::kSopc;
    }
    if (top9 == 0x17D) {
      opcode = (w >> 8) & 0xFF;
      return Enc::kSop1;
    }
    if ((w >> 28) == 0xB) {
      opcode = (w >> 23) & 0x1F;
      return Enc::kSopk;
    }
    opcode = (w >> 23) & 0x7F;
    return Enc::kSop2;
  }
  if ((w >> 27) == 0x18) {
    opcode = (w >> 22) & 0x1F;
    return Enc::kSmrd;
  }
  if ((w >> 26) == 0x34) {
    opcode = (w >> 17) & 0x1FF;
    if (mode == IsaMode::kNeo)
      opcode |= ((w >> 16) & 1) << 9;
    return Enc::kVop3;
  }
  if (mode == IsaMode::kNeo && (w >> 26) == 0x33) {
    opcode = (w >> 16) & 0x7f;
    return Enc::kVop3p;
  }
  if ((w >> 26) == 0x32) {
    opcode = (w >> 16) & 0x3;
    return Enc::kVintrp;
  }
  if ((w >> 26) == 0x36) {
    opcode = (w >> 18) & 0xFF;
    return Enc::kDs;
  }
  if ((w >> 26) == 0x37) {
    opcode = (w >> 18) & 0x7F;
    return Enc::kFlat;
  }
  if ((w >> 26) == 0x38) {
    opcode = (w >> 18) & 0x7F;
    return Enc::kMubuf;
  }
  if ((w >> 26) == 0x3A) {
    opcode = (w >> 16) & 0x7;
    return Enc::kMtbuf;
  }
  if ((w >> 26) == 0x3C) {
    opcode = (w >> 18) & 0x7F;
    return Enc::kMimg;
  }
  if ((w >> 26) == 0x3E) {
    opcode = (w >> 16) & 0x3F;
    return Enc::kExp;
  }
  if ((w >> 25) == 0x3F) {
    opcode = (w >> 9) & 0xFF;
    return Enc::kVop1;
  }
  if ((w >> 25) == 0x3E) {
    opcode = (w >> 17) & 0xFF;
    return Enc::kVopc;
  }
  if ((w >> 31) == 0x0) {
    opcode = (w >> 25) & 0x3F;
    return Enc::kVop2;
  }
  opcode = 0;
  return Enc::kUnknown;
}

// Dwords occupied (excluding any trailing literal).
uint32_t BaseSize(Enc e) {
  switch (e) {
    case Enc::kVop3:
    case Enc::kVop3p:
    case Enc::kDs:
    case Enc::kMubuf:
    case Enc::kMtbuf:
    case Enc::kMimg:
    case Enc::kExp:
    case Enc::kFlat:
      return 2;
    default:
      return 1;
  }
}

bool HasTrailingLiteral(const Inst& inst, uint32_t w) {
  switch (inst.enc) {
    case Enc::kSop2:
    case Enc::kSopc:
      return Sop2HasLiteral(w);
    case Enc::kSop1:
      return Sop1HasLiteral(w);
    case Enc::kSopk:
      return inst.opcode == 0x15;  // s_setreg_imm32_b32
    case Enc::kSmrd:
      // With IMM=0, SOFFSET uses the scalar-source encoding; 255 selects a
      // trailing literal byte offset instead of an SGPR.
      return ((w >> 8) & 1) == 0 && (w & 0xFF) == 255;
    case Enc::kVop2:
      // V_MADMK_F32 (0x20) and V_MADAK_F32 (0x21) always carry a trailing
      // 32-bit literal (the K constant), independent of src0=LITERAL_CONST.
      return VopExtension(w, inst.isa) == InstExtension::kLiteral ||
             inst.opcode == 0x20 || inst.opcode == 0x21 ||
             (inst.isa == IsaMode::kNeo &&
              (inst.opcode == 0x37 || inst.opcode == 0x38));
    case Enc::kVop1:
    case Enc::kVopc:
      return VopExtension(w, inst.isa) == InstExtension::kLiteral;
    case Enc::kVop3:
    case Enc::kVop3p: {
      if (inst.isa != IsaMode::kNeo)
        return false;
      const uint32_t w1 = inst.raw[1];
      return (w1 & 0x1ff) == 255 || ((w1 >> 9) & 0x1ff) == 255 ||
             ((w1 >> 18) & 0x1ff) == 255;
    }
    default:
      return false;
  }
}

// FNV-1a over the code dwords; validates program-cache entries.
uint64_t HashCode(const uint32_t* code, uint32_t dwords) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (uint32_t i = 0; i < dwords; i++)
    h = (h ^ code[i]) * 0x100000001b3ull;
  return h;
}

}  // namespace

namespace {
std::atomic<IsaMode> g_default_isa_mode{IsaMode::kBase};
}

IsaMode DefaultIsaMode() {
  return g_default_isa_mode.load(std::memory_order_acquire);
}

void SetDefaultIsaMode(IsaMode mode) {
  g_default_isa_mode.store(mode, std::memory_order_release);
  NextProgramCacheGeneration();
}

uint32_t CodeLength(const uint32_t* code, uint32_t max_dwords) {
  if (!code || max_dwords < 2)
    return 0;
  // Fast path: the toolchain emits "s_mov_b32 vcc_hi, #imm" (0xBEEB03FF) as the
  // first instruction, where the ShaderBinaryInfo footer sits at
  // code[(imm+1)*2].
  if (code[0] == 0xBEEB03FFu) {
    const uint64_t d = (static_cast<uint64_t>(code[1]) + 1) * 2;
    if (d >= 2 && d + 2 <= max_dwords &&
        OrbShdrAt(code, static_cast<uint32_t>(d) * 4))
      return static_cast<uint32_t>(d);
  }
  // General case: scan (dword-aligned) for the footer signature. The GCN code
  // ends exactly where its footer begins, so the footer's offset is the length.
  for (uint32_t d = 1; d + 2 <= max_dwords; d++)
    if (OrbShdrAt(code, d * 4))
      return d;
  return 0;
}

Program Decode(const uint32_t* code,
               uint32_t max_dwords,
               bool stop_at_endpgm,
               IsaMode mode) {
  Program out;
  if (!code)
    return out;
  uint32_t i = 0;
  while (i < max_dwords) {
    Inst inst;
    inst.isa = mode;
    inst.pc = i;
    inst.raw[0] = code[i];
    inst.enc = Classify(code[i], mode, inst.opcode);
    inst.size = BaseSize(inst.enc);
    if (inst.size == 2) {
      if (i + 1 >= max_dwords) {
        inst.enc = Enc::kUnknown;
        inst.opcode = 0;
        inst.size = 1;
      } else {
        inst.raw[1] = code[i + 1];
      }
    }

    if (mode == IsaMode::kNeo && inst.enc == Enc::kMtbuf)
      inst.opcode |= ((inst.raw[1] >> 21) & 1) << 3;

    const InstExtension vop_extension = inst.enc == Enc::kVop1 ||
                                                inst.enc == Enc::kVop2 ||
                                                inst.enc == Enc::kVopc
                                            ? VopExtension(code[i], mode)
                                            : InstExtension::kNone;
    if (vop_extension == InstExtension::kSdwa ||
        vop_extension == InstExtension::kDpp) {
      if (i + inst.size >= max_dwords) {
        inst.enc = Enc::kUnknown;
        inst.opcode = 0;
      } else {
        inst.extension = vop_extension;
        inst.raw[1] = code[i + inst.size];
        inst.size += 1;
      }
    } else if (HasTrailingLiteral(inst, code[i])) {
      if (i + inst.size >= max_dwords) {
        inst.enc = Enc::kUnknown;
        inst.opcode = 0;
      } else {
        inst.has_literal = true;
        inst.literal = code[i + inst.size];
        inst.extension = InstExtension::kLiteral;
        inst.size += 1;
      }
    }
    if (inst.size == 0)
      inst.size = 1;  // safety: never stall

    out.push_back(inst);

    // s_endpgm (SOPP opcode 1) terminates a basic block. When bounded by the
    // real code length it is not an end-of-stream marker, so keep decoding: a
    // block reached only after an early-out s_endpgm must still be lifted.
    if (stop_at_endpgm && inst.enc == Enc::kSopp && inst.opcode == 1)
      break;
    i += inst.size;
  }
  return out;
}

Program DecodeShader(const uint32_t* code, uint32_t max_dwords, IsaMode mode) {
  const uint32_t len = CodeLength(code, max_dwords);
  if (len && len <= max_dwords)
    return Decode(code, len, /*stop_at_endpgm=*/false, mode);
  return Decode(code, max_dwords, /*stop_at_endpgm=*/true, mode);
}

std::vector<uint8_t> ComputeReachability(const Program& program) {
  std::vector<uint8_t> reachable(program.size(), 0);
  if (program.empty())
    return reachable;

  // 0=ordinary, 1=unconditional relative, 2=conditional relative, 3=end,
  // 4=indirect control flow. Indirect targets cannot be recovered statically.
  const auto branch_kind = [](const Inst& inst) {
    if (inst.enc == Enc::kSopk && inst.opcode == 0x11)
      return 2;  // s_cbranch_i_fork
    if (inst.enc == Enc::kSop2 && inst.opcode == 0x2b)
      return 4;  // s_cbranch_g_fork
    if (inst.enc == Enc::kSop1) {
      switch (inst.opcode) {
        case 0x20:  // s_setpc_b64
        case 0x21:  // s_swappc_b64
        case 0x22:  // s_rfe_b64
        case 0x32:  // s_cbranch_join
          return 4;
        default:
          return 0;
      }
    }
    if (inst.enc != Enc::kSopp)
      return 0;
    switch (inst.opcode) {
      case 0x01:
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
      case 0x17:
      case 0x18:
      case 0x19:
      case 0x1a:
        return 2;
      default:
        return 0;
    }
  };
  const uint32_t max_pc = program.back().pc + program.back().size;
  std::vector<uint32_t> starts{0};
  for (const Inst& inst : program) {
    const int kind = branch_kind(inst);
    if (!kind)
      continue;
    starts.push_back(inst.pc + inst.size);
    if (kind == 1 || kind == 2) {
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      starts.push_back(static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                             static_cast<int32_t>(inst.size) +
                                             simm));
    }
  }
  std::sort(starts.begin(), starts.end());
  starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
  starts.erase(std::remove_if(starts.begin(), starts.end(),
                              [max_pc](uint32_t pc) { return pc >= max_pc; }),
               starts.end());
  const auto block_of = [&](uint32_t pc) {
    uint32_t block = 0;
    for (uint32_t i = 0; i < starts.size(); i++) {
      if (starts[i] > pc)
        break;
      block = i;
    }
    return block;
  };

  std::vector<uint8_t> block_reachable(starts.size(), 0);
  std::vector<uint32_t> worklist{0};
  while (!worklist.empty()) {
    const uint32_t block = worklist.back();
    worklist.pop_back();
    if (block >= starts.size() || block_reachable[block])
      continue;
    block_reachable[block] = 1;
    const uint32_t block_end =
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
        // An indirect target may be any decoded block. Conservatively retain
        // all blocks rather than misreporting valid code as dead.
        std::fill(block_reachable.begin(), block_reachable.end(), 1);
        worklist.clear();
        break;
      }
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      const uint32_t target =
          static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                static_cast<int32_t>(inst.size) + simm);
      if (target < max_pc)
        worklist.push_back(block_of(target));
      if (kind == 2)
        worklist.push_back(block + 1);
      break;
    }
    if (!terminated)
      worklist.push_back(block + 1);
  }

  for (uint32_t i = 0; i < program.size(); i++)
    reachable[i] = block_reachable[block_of(program[i].pc)];
  return reachable;
}

namespace {
uint64_t g_prog_cache_generation = 1;
}  // namespace

void NextProgramCacheGeneration() {
  g_prog_cache_generation++;
}

std::shared_ptr<const Program> CachedProgram(uint64_t addr,
                                             uint32_t max_dwords) {
  struct Entry {
    uint64_t hash = 0;
    uint32_t hashed_dwords = 0;
    uint64_t generation = 0;
    IsaMode mode = IsaMode::kBase;
    std::shared_ptr<const Program> program;
  };
  static std::unordered_map<uint64_t, Entry> cache;

  const auto* code = reinterpret_cast<const uint32_t*>(addr);
  if (!code)
    return std::make_shared<const Program>();
  const IsaMode mode = DefaultIsaMode();

  // Fast path: already revalidated this generation (frame). A draw touches the
  // same shader several times (textures, cbuffers, attributes), so skipping the
  // footer scan + code hash on repeats is what keeps this per-draw-affordable.
  auto it = cache.find(addr);
  if (it != cache.end() && it->second.generation == g_prog_cache_generation &&
      it->second.mode == mode)
    return it->second.program;

  // Hash the real code span (footer-bounded when available) so an in-place
  // rewrite at the same address invalidates the entry.
  const uint32_t len = CodeLength(code, max_dwords);
  const uint32_t hashed = len ? len : (max_dwords < 64 ? max_dwords : 64);
  const uint64_t hash = HashCode(code, hashed);

  if (it != cache.end() && it->second.mode == mode && it->second.hash == hash &&
      it->second.hashed_dwords == hashed) {
    it->second.generation = g_prog_cache_generation;
    return it->second.program;
  }

  if (cache.size() > 512)
    cache.clear();  // unbounded-growth backstop
  auto program =
      std::make_shared<const Program>(DecodeShader(code, max_dwords, mode));
  cache[addr] = {hash, hashed, g_prog_cache_generation, mode, program};
  return program;
}

uint64_t CachedCodeHash(uint64_t addr, uint32_t max_dwords) {
  if (!addr)
    return 0;
  struct Entry {
    uint64_t hash = 0;
    uint64_t generation = 0;
  };
  static std::unordered_map<uint64_t, Entry> cache;

  auto it = cache.find(addr);
  if (it != cache.end() && it->second.generation == g_prog_cache_generation)
    return it->second.hash;

  const auto* code = reinterpret_cast<const uint32_t*>(addr);
  uint32_t len = CodeLength(code, max_dwords);
  if (!len) {
    // No footer: hash up to the terminator instead of a fixed window, or the
    // bytes that happen to follow the shader make every instance of the same
    // code hash differently.
    const auto program = CachedProgram(addr, max_dwords);
    len = max_dwords;
    for (const Inst& inst : *program) {
      const bool ends =
          (inst.enc == Enc::kSopp && inst.opcode == 0x01) ||  // s_endpgm
          (inst.enc == Enc::kSop1 &&
           (inst.opcode == 0x20 || inst.opcode == 0x21));  // s_setpc/s_swappc
      if (ends) {
        len = inst.pc + inst.size;
        break;
      }
    }
  }
  const uint64_t hash = HashCode(code, len);
  if (cache.size() > 4096)
    cache.clear();  // unbounded-growth backstop
  cache[addr] = {hash, g_prog_cache_generation};
  return hash;
}

bool CallsFetchShader(const Program& program) {
  for (const Inst& inst : program)
    if (inst.enc == Enc::kSop1 && inst.opcode == 0x21)  // s_swappc_b64
      return true;
  return false;
}

}  // namespace gpu::gcn
