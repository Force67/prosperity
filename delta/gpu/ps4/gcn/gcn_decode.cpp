/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN GFX7 instruction decoder. See gcn_decode.h.
 */

#include "gcn_decode.h"

#include <algorithm>
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
bool Sop1HasLiteral(uint32_t w) { return (w & 0xFF) == 255; }
// VOP src0 is 9 bits [8:0]; 255 selects a literal.
bool VopHasLiteral(uint32_t w) { return (w & 0x1FF) == 255; }

// Encoding classification by the fixed top bits. Returns the family and fills
// the encoding-relative opcode.
Enc Classify(uint32_t w, uint32_t& opcode) {
  if ((w >> 30) == 0x2) {  // 10b => scalar
    const uint32_t top9 = w >> 23;
    if (top9 == 0x17F) { opcode = (w >> 16) & 0x7F; return Enc::kSopp; }
    if (top9 == 0x17E) { opcode = (w >> 16) & 0x7F; return Enc::kSopc; }
    if (top9 == 0x17D) { opcode = (w >> 8) & 0xFF; return Enc::kSop1; }
    if ((w >> 28) == 0xB) { opcode = (w >> 23) & 0x1F; return Enc::kSopk; }
    opcode = (w >> 23) & 0x7F;
    return Enc::kSop2;
  }
  if ((w >> 27) == 0x18) { opcode = (w >> 22) & 0x1F; return Enc::kSmrd; }
  if ((w >> 26) == 0x34) { opcode = (w >> 17) & 0x1FF; return Enc::kVop3; }
  if ((w >> 26) == 0x32) { opcode = (w >> 16) & 0x3; return Enc::kVintrp; }
  if ((w >> 26) == 0x36) { opcode = (w >> 18) & 0xFF; return Enc::kDs; }
  if ((w >> 26) == 0x38) { opcode = (w >> 18) & 0x7F; return Enc::kMubuf; }
  if ((w >> 26) == 0x3A) { opcode = (w >> 16) & 0x7; return Enc::kMtbuf; }
  if ((w >> 26) == 0x3C) { opcode = (w >> 18) & 0x7F; return Enc::kMimg; }
  if ((w >> 26) == 0x3E) { opcode = (w >> 16) & 0x3F; return Enc::kExp; }
  if ((w >> 25) == 0x3F) { opcode = (w >> 9) & 0xFF; return Enc::kVop1; }
  if ((w >> 25) == 0x3E) { opcode = (w >> 17) & 0xFF; return Enc::kVopc; }
  if ((w >> 31) == 0x0) { opcode = (w >> 25) & 0x3F; return Enc::kVop2; }
  opcode = 0;
  return Enc::kUnknown;
}

// Dwords occupied (excluding any trailing literal).
uint32_t BaseSize(Enc e) {
  switch (e) {
    case Enc::kVop3:
    case Enc::kDs:
    case Enc::kMubuf:
    case Enc::kMtbuf:
    case Enc::kMimg:
    case Enc::kExp:
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
    case Enc::kSmrd:
      // With IMM=0, SOFFSET uses the scalar-source encoding; 255 selects a
      // trailing literal byte offset instead of an SGPR.
      return ((w >> 8) & 1) == 0 && (w & 0xFF) == 255;
    case Enc::kVop2:
      // V_MADMK_F32 (0x20) and V_MADAK_F32 (0x21) always carry a trailing
      // 32-bit literal (the K constant), independent of src0=LITERAL_CONST.
      return VopHasLiteral(w) || inst.opcode == 0x20 || inst.opcode == 0x21;
    case Enc::kVop1:
    case Enc::kVopc:
      return VopHasLiteral(w);
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

uint32_t CodeLength(const uint32_t* code, uint32_t max_dwords) {
  if (!code || max_dwords < 2) return 0;
  // Fast path: the toolchain emits "s_mov_b32 vcc_hi, #imm" (0xBEEB03FF) as the
  // first instruction, where the ShaderBinaryInfo footer sits at code[(imm+1)*2].
  if (code[0] == 0xBEEB03FFu) {
    const uint64_t d = (static_cast<uint64_t>(code[1]) + 1) * 2;
    if (d >= 2 && d + 2 <= max_dwords && OrbShdrAt(code, static_cast<uint32_t>(d) * 4))
      return static_cast<uint32_t>(d);
  }
  // General case: scan (dword-aligned) for the footer signature. The GCN code
  // ends exactly where its footer begins, so the footer's offset is the length.
  for (uint32_t d = 1; d + 2 <= max_dwords; d++)
    if (OrbShdrAt(code, d * 4)) return d;
  return 0;
}

Program Decode(const uint32_t* code, uint32_t max_dwords, bool stop_at_endpgm) {
  Program out;
  if (!code) return out;
  uint32_t i = 0;
  while (i < max_dwords) {
    Inst inst;
    inst.pc = i;
    inst.raw[0] = code[i];
    inst.enc = Classify(code[i], inst.opcode);
    inst.size = BaseSize(inst.enc);
    if (inst.size == 2 && i + 1 < max_dwords) inst.raw[1] = code[i + 1];

    if (HasTrailingLiteral(inst, code[i]) && i + inst.size < max_dwords) {
      inst.has_literal = true;
      inst.literal = code[i + inst.size];
      inst.size += 1;
    }
    if (inst.size == 0) inst.size = 1;  // safety: never stall

    out.push_back(inst);

    // s_endpgm (SOPP opcode 1) terminates a basic block. When bounded by the
    // real code length it is not an end-of-stream marker, so keep decoding: a
    // block reached only after an early-out s_endpgm must still be lifted.
    if (stop_at_endpgm && inst.enc == Enc::kSopp && inst.opcode == 1) break;
    i += inst.size;
  }
  return out;
}

Program DecodeShader(const uint32_t* code, uint32_t max_dwords) {
  const uint32_t len = CodeLength(code, max_dwords);
  if (len && len <= max_dwords)
    return Decode(code, len, /*stop_at_endpgm=*/false);
  return Decode(code, max_dwords, /*stop_at_endpgm=*/true);
}

std::vector<uint8_t> ComputeReachability(const Program& program) {
  std::vector<uint8_t> reachable(program.size(), 0);
  if (program.empty()) return reachable;

  const auto branch_kind = [](const Inst& inst) {
    if (inst.enc != Enc::kSopp) return 0;
    switch (inst.opcode) {
      case 0x01: return 8;  // endpgm
      case 0x02: return 1;  // unconditional
      case 0x04: return 2;  // scc0
      case 0x05: return 3;  // scc1
      case 0x06: return 4;  // vccz
      case 0x07: return 5;  // vccnz
      case 0x08: return 6;  // execz
      case 0x09: return 7;  // execnz
      default: return 0;
    }
  };
  const uint32_t max_pc = program.back().pc + program.back().size;
  std::vector<uint32_t> starts{0};
  for (const Inst& inst : program) {
    const int kind = branch_kind(inst);
    if (!kind) continue;
    starts.push_back(inst.pc + inst.size);
    if (kind >= 1 && kind <= 7) {
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      starts.push_back(static_cast<uint32_t>(
          static_cast<int32_t>(inst.pc) + static_cast<int32_t>(inst.size) +
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
      if (starts[i] > pc) break;
      block = i;
    }
    return block;
  };

  std::vector<uint8_t> block_reachable(starts.size(), 0);
  std::vector<uint32_t> worklist{0};
  while (!worklist.empty()) {
    const uint32_t block = worklist.back();
    worklist.pop_back();
    if (block >= starts.size() || block_reachable[block]) continue;
    block_reachable[block] = 1;
    const uint32_t block_end =
        block + 1 < starts.size() ? starts[block + 1] : max_pc;
    bool terminated = false;
    for (const Inst& inst : program) {
      if (inst.pc < starts[block] || inst.pc >= block_end) continue;
      const int kind = branch_kind(inst);
      if (!kind) continue;
      terminated = true;
      if (kind == 8) break;
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      const uint32_t target = static_cast<uint32_t>(
          static_cast<int32_t>(inst.pc) + static_cast<int32_t>(inst.size) +
          simm);
      if (target < max_pc) worklist.push_back(block_of(target));
      if (kind != 1) worklist.push_back(block + 1);
      break;
    }
    if (!terminated) worklist.push_back(block + 1);
  }

  for (uint32_t i = 0; i < program.size(); i++)
    reachable[i] = block_reachable[block_of(program[i].pc)];
  return reachable;
}

namespace {
uint64_t g_progCacheGeneration = 1;
}  // namespace

void NextProgramCacheGeneration() { g_progCacheGeneration++; }

std::shared_ptr<const Program> CachedProgram(uint64_t addr,
                                             uint32_t max_dwords) {
  struct Entry {
    uint64_t hash = 0;
    uint32_t hashed_dwords = 0;
    uint64_t generation = 0;
    std::shared_ptr<const Program> program;
  };
  static std::unordered_map<uint64_t, Entry> cache;

  const auto* code = reinterpret_cast<const uint32_t*>(addr);
  if (!code) return std::make_shared<const Program>();

  // Fast path: already revalidated this generation (frame). A draw touches the
  // same shader several times (textures, cbuffers, attributes), so skipping the
  // footer scan + code hash on repeats is what keeps this per-draw-affordable.
  auto it = cache.find(addr);
  if (it != cache.end() && it->second.generation == g_progCacheGeneration)
    return it->second.program;

  // Hash the real code span (footer-bounded when available) so an in-place
  // rewrite at the same address invalidates the entry.
  const uint32_t len = CodeLength(code, max_dwords);
  const uint32_t hashed = len ? len : (max_dwords < 64 ? max_dwords : 64);
  const uint64_t hash = HashCode(code, hashed);

  if (it != cache.end() && it->second.hash == hash &&
      it->second.hashed_dwords == hashed) {
    it->second.generation = g_progCacheGeneration;
    return it->second.program;
  }

  if (cache.size() > 512) cache.clear();  // unbounded-growth backstop
  auto program = std::make_shared<const Program>(DecodeShader(code, max_dwords));
  cache[addr] = {hash, hashed, g_progCacheGeneration, program};
  return program;
}

const char* Mnemonic(const Inst& inst) {
  switch (inst.enc) {
    case Enc::kSop1:
      switch (inst.opcode) {
        case 0x03: return "s_mov_b32";
        case 0x04: return "s_mov_b64";
        default: return "s_op1";
      }
    case Enc::kSop2:
      switch (inst.opcode) {
        case 0x00: return "s_add_u32";
        case 0x01: return "s_sub_u32";
        case 0x02: return "s_add_i32";
        default: return "s_op2";
      }
    case Enc::kSopp:
      switch (inst.opcode) {
        case 0x00: return "s_nop";
        case 0x01: return "s_endpgm";
        case 0x0a: return "s_barrier";
        case 0x0c: return "s_waitcnt";
        default: return "s_opp";
      }
    case Enc::kSopk: return "s_movk/sopk";
    case Enc::kSopc: return "s_cmp";
    case Enc::kSmrd:
      switch (inst.opcode) {
        case 0x00: return "s_load_dword";
        case 0x01: return "s_load_dwordx2";
        case 0x02: return "s_load_dwordx4";
        case 0x03: return "s_load_dwordx8";
        case 0x04: return "s_load_dwordx16";
        case 0x08: return "s_buffer_load_dword";
        case 0x0a: return "s_buffer_load_dwordx4";
        case 0x0c: return "s_buffer_load_dwordx16";
        default: return "smrd";
      }
    case Enc::kVop1: return "v_op1";
    case Enc::kVop2: return "v_op2";
    case Enc::kVop3: return "v_op3";
    case Enc::kVopc: return "v_cmp";
    case Enc::kVintrp: return "v_interp";
    case Enc::kDs: return "ds";
    case Enc::kMubuf: return "mubuf";
    case Enc::kMtbuf: return "tbuffer";
    case Enc::kMimg: return "image";
    case Enc::kExp: return "exp";
    default: return "?";
  }
}

void Disassemble(const uint32_t* code, uint32_t max_dwords, const char* tag) {
  const Program program = Decode(code, max_dwords);
  std::fprintf(stderr, "[gcn] %s: %zu instructions\n", tag, program.size());
  for (const Inst& inst : program) {
    std::fprintf(stderr, "[gcn]   %04x: %08x", inst.pc, inst.raw[0]);
    if (inst.size >= 2) std::fprintf(stderr, " %08x", inst.raw[1]);
    std::fprintf(stderr, "  %s op=%#x", Mnemonic(inst), inst.opcode);
    if (inst.has_literal) std::fprintf(stderr, " lit=%#x", inst.literal);
    std::fprintf(stderr, "\n");
  }
}

}  // namespace gpu::gcn
