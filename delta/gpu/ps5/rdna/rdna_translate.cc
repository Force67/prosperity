/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) -> SPIR-V translator. Reuses the shared gpu::gcn SPIR-V
 * backend (the register-file model in Translator, the scalar/vector ALU
 * emitters, exports, and constant-buffer plumbing); only the RDNA2-specific
 * per-instruction field decode + opcode remap and the SMEM constant-buffer path
 * live here. Control-flow lowering is a self-contained copy of the GFX7
 * while/switch state machine (it dispatches through the RDNA2 emitter), so the
 * PS4 backend is untouched.
 */

#include "gpu/ps5/rdna/rdna_translate.h"
#include "base/arch.h"
#include "memory_debug/memory_debug.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
namespace gpu::rdna {
bool HasNggTransfer(const u32*) {
  return false;
}
bool HasNggPrimitiveExports(const u32*) {
  return false;
}
gpu::gcn::Recompiled Recompile(const u32*,
                               const u32*,
                               const u32*,
                               const u32*,
                               u32,
                               bool,
                               u32,
                               u32,
                               const u32*,
                               u32,
                               const NggConfig*) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kShaderCache);
  return {};
}
u64 FetchPlanHash(u64) {
  return 0;
}
}  // namespace gpu::rdna
#else

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/strings/to_string.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/gcn_audit.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/gcn/spirv/translator.h"
#include "gpu/guest_memory.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_emit.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kExpTrace, "DELTA_GPU_EXPTRACE", false);
DELTA_OPTION(u32, kCfgMaxIter, "DELTA_GPU_CFG_MAXITER", 16384);
DELTA_OPTION(bool, kLoopTrace, "DELTA_GPU_LOOPTRACE", false);
DELTA_OPTION(bool, kAllowNan, "DELTA_GPU_ALLOWNAN", false);
DELTA_OPTION(const char*, kDbgPos, "DELTA_GPU_DBGPOS", nullptr);
DELTA_OPTION(float, kRawPos, "DELTA_GPU_RAWPOS", 0.f);
DELTA_OPTION(bool, kDrawCensus, "DELTA_GPU_DRAWCENSUS", false);
DELTA_OPTION(u64, kNullPs, "DELTA_GPU_NULL_PS", 0);
DELTA_OPTION(const char*, kSpvDump, "DELTA_GPU_SPVDUMP", nullptr);
DELTA_OPTION(bool, kAgcTrace, "DELTA_AGC_TRACE", false);
DELTA_OPTION(bool, kGpuDebugalpha, "DELTA_GPU_DEBUGALPHA", false);
DELTA_OPTION(bool, kGpuForcecolor, "DELTA_GPU_FORCECOLOR", false);
DELTA_OPTION(u32, kGpuPsVgpr, "DELTA_GPU_PSVGPR", 0);
DELTA_OPTION(u32, kGpuPsVgprBlock, "DELTA_GPU_PSVGPR_BLOCK", 0);
DELTA_OPTION(u64, kGpuPsVgprPs, "DELTA_GPU_PSVGPR_PS", 0);
DELTA_OPTION(u32, kNggWaveInfo, "DELTA_PS5_WAVEINFO", 0x0101);
DELTA_OPTION(bool, kGpuForcequad, "DELTA_GPU_FORCEQUAD", false);
DELTA_OPTION(bool, kGpuNokill, "DELTA_GPU_NOKILL", false);
DELTA_OPTION(bool, kGpuPosprobe, "DELTA_GPU_POSPROBE", false);
DELTA_OPTION(bool, kGpuPosunset, "DELTA_GPU_POSUNSET", false);
DELTA_OPTION(bool, kLdsMark, "DELTA_GPU_LDSMARK", false);
DELTA_OPTION(int, kGpuPstex, "DELTA_GPU_PSTEX", 0);
DELTA_OPTION(bool, kWaveMasks, "DELTA_PS5_WAVEMASK", true);
DELTA_OPTION(bool, kGpuPsuv, "DELTA_GPU_PSUV", false);
DELTA_OPTION(u32, kGpuPsuvAttr, "DELTA_GPU_PSUV_ATTR", 0);
DELTA_OPTION(bool, kGpuShtrace, "DELTA_GPU_SHTRACE", false);
DELTA_OPTION(bool, kGpuSpirvCfg, "DELTA_GPU_SPIRV_CFG", false);
}  // namespace

namespace gpu::rdna {
namespace {

using gpu::gcn::Enc;
using gpu::gcn::Id;
using gpu::gcn::Inst;
using gpu::gcn::kMaxCbufBindings;
using gpu::gcn::Program;
using gpu::gcn::Recompiled;
using gpu::gcn::ShaderCbuf;
using gpu::gcn::StageContext;
using gpu::gcn::Translator;

bool ShDbg() {
  static const bool kOn = kGpuShtrace || kAgcTrace;
  return kOn;
}

// GFX7 ranges are correct here. Do not 'correct' them to gfx8/gfx9.
bool RdnaMubufStore(u32 op) {
  return (op >= 0x04 && op <= 0x07) || (op >= 0x84 && op <= 0x87) ||
         (op >= 0x18 && op <= 0x1f);
}

bool RdnaMubufAtomicNoReturn(const Inst& inst) {
  return inst.opcode >= 0x30 && inst.opcode <= 0x60 &&
         !((inst.raw[0] >> 14) & 1);
}

bool RdnaMtbufStore(u32 op) {
  return (op >= 0x04 && op <= 0x07) || (op >= 0x0c && op <= 0x0f);
}

struct Flat {
  u32 op;
  u32 seg;
  u32 addr;
  u32 data;
  u32 saddr;
  u32 vdst;
  i32 offset;
  bool dlc;
  bool lds;
  bool glc;
  bool slc;
  bool saddr_null;
};

Flat DecodeFlat(const Inst& inst) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 seg = (w >> 14) & 0x3, off = w & 0xFFF;
  const u32 saddr = (w1 >> 16) & 0x7F;
  return {
      .op = (w >> 18) & 0x7F,
      .seg = seg,
      .addr = w1 & 0xFF,
      .data = (w1 >> 8) & 0xFF,
      .saddr = saddr,
      .vdst = (w1 >> 24) & 0xFF,
      .offset = seg == 0 ? static_cast<i32>(off & 0x7FF)
                         : (static_cast<i32>(off << 20) >> 20),
      .dlc = ((w >> 12) & 1) != 0,
      .lds = ((w >> 13) & 1) != 0,
      .glc = ((w >> 16) & 1) != 0,
      .slc = ((w >> 17) & 1) != 0,
      .saddr_null = saddr == 0x7D || saddr == 0x7F,
  };
}

u32 FlatLoadDwords(u32 op) {
  switch (op) {
    case 0x08:
    case 0x09:
    case 0x0a:
    case 0x0b:
    case 0x0c:
      return 1;
    case 0x0d:
      return 2;
    case 0x0e:
      return 4;
    case 0x0f:
      return 3;
    default:
      return 0;
  }
}

bool FlatIsStore(u32 op) {
  return op >= 0x18 && op <= 0x1f;
}

bool FlatServableLoad(const Flat& f) {
  return f.seg == 2 && !f.saddr_null && !f.lds && FlatLoadDwords(f.op) &&
         f.offset >= 0;
}

constexpr u32 kFlatBaseTag = 0x100;

bool BufLoadIsVertexFetch(const Inst& in, bool chained) {
  const bool idxen = (in.raw[0] >> 13) & 1;
  return idxen && chained;
}

base::HashMap<u32, base::Pair<u32, u32> > MapTableChainedLoads(
    const Program& insts) {
  base::HashMap<u32, base::Pair<u32, u32> > out;
  i32 root[128];
  u32 slot[128] = {};
  for (int i = 0; i < 128; i++)
    root[i] = -1;
  base::HashMap<u32, u32> next_slot;
  for (const Inst& in : insts) {
    if (in.enc == Enc::kSop1 && in.opcode == 0x20)
      break;
    if (in.enc == Enc::kSopp && in.opcode == 1)
      break;
    if (in.enc == Enc::kSmrd && in.opcode <= 0x04) {
      const u32 sdst = (in.raw[0] >> 6) & 0x7F;
      const u32 sbase = (in.raw[0] & 0x3F) * 2;
      const u32 nreg = in.opcode == 0   ? 1
                       : in.opcode == 1 ? 2
                       : in.opcode == 2 ? 4
                       : in.opcode == 3 ? 8
                                        : 16;
      const u32 s = in.opcode == 2 ? next_slot[sbase]++ : 0;
      for (u32 k = 0; k < nreg && sdst + k < 128; k++) {
        root[sdst + k] = static_cast<i32>(sbase);
        slot[sdst + k] = s;
      }
      continue;
    }
    if ((in.enc != Enc::kMubuf && in.enc != Enc::kMtbuf) || in.opcode > 0x03)
      continue;
    const u32 srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
    if (srsrc < 128 && root[srsrc] >= 0)
      out[in.pc] = {static_cast<u32>(root[srsrc]), slot[srsrc]};
  }
  return out;
}

const char* EncName(Enc e) {
  switch (e) {
    case Enc::kSop1:
      return "sop1";
    case Enc::kSop2:
      return "sop2";
    case Enc::kSopk:
      return "sopk";
    case Enc::kSopc:
      return "sopc";
    case Enc::kSopp:
      return "sopp";
    case Enc::kSmrd:
      return "smem";
    case Enc::kVop1:
      return "vop1";
    case Enc::kVop2:
      return "vop2";
    case Enc::kVop3:
      return "vop3";
    case Enc::kVop3p:
      return "vop3p";
    case Enc::kVopc:
      return "vopc";
    case Enc::kVintrp:
      return "vintrp";
    case Enc::kDs:
      return "ds";
    case Enc::kMubuf:
      return "mubuf";
    case Enc::kMtbuf:
      return "mtbuf";
    case Enc::kMimg:
      return "mimg";
    case Enc::kExp:
      return "exp";
    case Enc::kFlat:
      return "flat";
    default:
      return "UNKNOWN";
  }
}

void DumpProgram(const Program& prog, const char* tag) {
  BASE_LOGI("gcnspv", "=== {} decode: {} insts ===", tag, prog.size());
  for (const Inst& in : prog) {
    base::String line;
    base::FormatTo(line, "  pc={:04x} len={} {:<6} op={:#05x}  {:08x}", in.pc,
                   in.size, EncName(in.enc), in.opcode, in.raw[0]);
    if (in.size >= 2)
      base::FormatTo(line, " {:08x}", in.raw[1]);
    if (in.has_literal)
      base::FormatTo(line, " lit={:08x}", in.literal);
    BASE_LOGI("gcnspv", "{}", line.c_str());
  }
}

u32 RemapVop2(u32 op) {
  return op == 0x01 ? 0x00 : op;
}

bool RdnaSharedVop2(u32 op) {
  return (op >= 0x03 && op <= 0x05) || (op >= 0x08 && op <= 0x0c) ||
         (op >= 0x0f && op <= 0x19) || (op >= 0x1a && op <= 0x1d) ||
         (op >= 0x22 && op <= 0x24) || op == 0x2f;
}

bool UnsupportedRdnaScalarSource(u32 field) {
  return (field >= 108 && field <= 123) || (field >= 209 && field <= 239) ||
         field == 249 || field == 250 || field == 254;
}

bool UnsupportedRdnaValuSource(u32 field) {
  if (field >= 256)
    return false;
  return (field >= 108 && field <= 123) || (field >= 209 && field <= 239) ||
         field == 249 || field == 250 || field == 254;
}

bool UnsupportedRdnaSdwaSource(u32 field) {
  return UnsupportedRdnaValuSource(field) || field == 255;
}

bool UsesUnsupportedRdnaSource(const Inst& inst) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  switch (inst.enc) {
    case Enc::kSop1:
      return UnsupportedRdnaScalarSource(w & 0xff);
    case Enc::kSop2:
    case Enc::kSopc:
      return UnsupportedRdnaScalarSource(w & 0xff) ||
             UnsupportedRdnaScalarSource((w >> 8) & 0xff);
    case Enc::kSmrd:
      return UnsupportedRdnaScalarSource((w1 >> 25) & 0x7f);
    case Enc::kVop1: {
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        const u32 src0 = (w1 & 0xff) + (((w1 >> 23) & 1) ? 0 : 256);
        return UnsupportedRdnaSdwaSource(src0);
      }
      return UnsupportedRdnaValuSource(w & 0x1ff);
    }
    case Enc::kVop2:
    case Enc::kVopc: {
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        const u32 src0 = (w1 & 0xff) + (((w1 >> 23) & 1) ? 0 : 256);
        const u32 src1 = ((w >> 9) & 0xff) + (((w1 >> 31) & 1) ? 0 : 256);
        return UnsupportedRdnaSdwaSource(src0) ||
               UnsupportedRdnaSdwaSource(src1);
      }
      return UnsupportedRdnaValuSource(w & 0x1ff);
    }
    case Enc::kVop3:
    case Enc::kVop3p: {
      const u32 count = Vop3SourceCount(inst.enc, inst.opcode);
      for (u32 source = 0; source < count; source++)
        if (UnsupportedRdnaValuSource((w1 >> (source * 9)) & 0x1ff))
          return true;
      return false;
    }
    default:
      return false;
  }
}

Id RdnaF16Bits(Translator& t, u32 field, u32 literal) {
  static constexpr u16 kInline[] = {0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                    0xc000, 0x4400, 0xc400, 0x3118};
  if (field >= 240 && field <= 248)
    return t.U32(kInline[field - 240]);
  return t.SrcRaw(field, literal);
}

bool RdnaVop3HasSdst(u32 op) {
  return op == 0x128 || op == 0x129 || op == 0x12A || op == 0x30F ||
         op == 0x310 || op == 0x319 || op == 0x16D || op == 0x16E ||
         op == 0x176 || op == 0x177;
}

bool RdnaEmitVop3Int(Translator& t,
                     u32 op,
                     u32 vdst,
                     u32 sdst,
                     Id s0,
                     Id s1,
                     Id s2) {
  const auto carry = [&](spv::Op operation, Id a, Id b, Id carry_in = 0) {
    Id pair = t.m.Emit(operation, t.PairType(), {a, b});
    Id value = t.m.CompositeExtract(t.t_u, pair, 0);
    Id flag = t.m.CompositeExtract(t.t_u, pair, 1);
    if (carry_in) {
      pair =
          t.m.Emit(operation, t.PairType(), {value, t.And(carry_in, t.U32(1))});
      value = t.m.CompositeExtract(t.t_u, pair, 0);
      flag = t.Or(flag, t.m.CompositeExtract(t.t_u, pair, 1));
    }
    t.SetVg(vdst, value);
    t.SetLaneFlag(sdst, flag);
  };
  switch (op) {
    case 0x30F:
      carry(spv::Op::OpIAddCarry, s0, s1);
      return true;
    case 0x310:
      carry(spv::Op::OpISubBorrow, s0, s1);
      return true;
    case 0x319:
      carry(spv::Op::OpISubBorrow, s1, s0);
      return true;
    case 0x128:
      carry(spv::Op::OpIAddCarry, s0, s1, s2);
      return true;
    case 0x129:
      carry(spv::Op::OpISubBorrow, s0, s1, s2);
      return true;
    case 0x12A:
      carry(spv::Op::OpISubBorrow, s1, s0, s2);
      return true;
    case 0x125:
    case 0x37F:
      t.SetVg(vdst, t.Add(s0, s1));
      return true;
    case 0x126:
    case 0x376:
      t.SetVg(vdst, t.Sub(s0, s1));
      return true;
    case 0x127:
      t.SetVg(vdst, t.Sub(s1, s0));
      return true;
    case 0x11E:
      t.SetVg(vdst, t.Not(t.Xor(s0, s1)));
      return true;
    case 0x345:
      t.SetVg(vdst, t.Add(t.Xor(s0, s1), s2));
      return true;
    case 0x346:
      t.SetVg(vdst, t.Add(t.Shl(s0, s1), s2));
      return true;
    case 0x347:
      t.SetVg(vdst, t.Shl(t.Add(s0, s1), s2));
      return true;
    case 0x36D:
      t.SetVg(vdst, t.Add(t.Add(s0, s1), s2));
      return true;
    case 0x36F:
      t.SetVg(vdst, t.Or(t.Shl(s0, s1), s2));
      return true;
    case 0x371:
      t.SetVg(vdst, t.Or(t.And(s0, s1), s2));
      return true;
    case 0x372:
      t.SetVg(vdst, t.Or(t.Or(s0, s1), s2));
      return true;
    default:
      return false;
  }
}

bool RdnaEmitVop3p(Translator& t,
                   u32 op,
                   u32 vdst,
                   u32 s0,
                   u32 s1,
                   u32 s2,
                   u32 lit) {
  auto unpack = [&](u32 f) {
    return t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                       {RdnaF16Bits(t, f, lit)});
  };
  auto pack = [&](Id v2) {
    return t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16, {v2});
  };
  const Id a = unpack(s0), b = unpack(s1);
  switch (op) {
    case 0x0F:
      t.SetVg(vdst, pack(t.m.Emit(spv::Op::OpFAdd, t.t_v2, {a, b})));
      return true;
    case 0x10:
      t.SetVg(vdst, pack(t.m.Emit(spv::Op::OpFMul, t.t_v2, {a, b})));
      return true;
    case 0x11:
      t.SetVg(vdst, pack(t.m.ExtInst(t.t_v2, GLSLstd450FMin, {a, b})));
      return true;
    case 0x12:
      t.SetVg(vdst, pack(t.m.ExtInst(t.t_v2, GLSLstd450FMax, {a, b})));
      return true;
    case 0x0E: {
      t.SetVg(vdst,
              pack(t.m.ExtInst(t.t_v2, GLSLstd450Fma, {a, b, unpack(s2)})));
      return true;
    }
    default:
      return false;
  }
}

bool RdnaEmitSop2(Translator& t, const Inst& inst) {
  const u32 w = inst.raw[0], op = inst.opcode;
  if (op < 0x32 || op > 0x36)
    return false;
  const u32 sdst = (w >> 16) & 0x7F;
  const Id a = t.SrcRaw(w & 0xFF, inst.literal);
  const Id b = t.SrcRaw((w >> 8) & 0xFF, inst.literal);
  const auto mul_hi = [&](spv::Op wide) {
    return t.m.CompositeExtract(t.t_u, t.m.Emit(wide, t.PairType(), {a, b}), 1);
  };
  switch (op) {
    case 0x32:
      t.SetSdst(sdst, 0,
                t.Or(t.And(a, t.U32(0xFFFF)),
                     t.Shl(t.And(b, t.U32(0xFFFF)), t.U32(16))));
      return true;
    case 0x33:
      t.SetSdst(sdst, 0,
                t.Or(t.And(a, t.U32(0xFFFF)), t.And(b, t.U32(0xFFFF0000u))));
      return true;
    case 0x34:
      t.SetSdst(sdst, 0,
                t.Or(t.Shr(a, t.U32(16)), t.And(b, t.U32(0xFFFF0000u))));
      return true;
    case 0x35:
      t.SetSdst(sdst, 0, mul_hi(spv::Op::OpUMulExtended));
      return true;
    case 0x36:
      t.SetSdst(sdst, 0, mul_hi(spv::Op::OpSMulExtended));
      return true;
    default:
      return false;
  }
}

struct CbufDef {
  u32 source, offset, count;
};

bool Overlaps(u32 first, u32 count, u32 other_first, u32 other_count) {
  return first < other_first + other_count && other_first < first + count;
}

void InvalidateCbufDefs(base::HashMap<u32, CbufDef>& loads, ScalarWrite write) {
  if (!write.count)
    return;
  for (auto it = loads.begin(); it != loads.end();) {
    if (Overlaps(it->first, it->second.count, write.first, write.count))
      it = loads.erase(it);
    else
      ++it;
  }
}

void InvalidateDescSrc(base::HashMap<u32, u64>& src, ScalarWrite write) {
  if (!write.count)
    return;
  for (auto it = src.begin(); it != src.end();) {
    if (Overlaps(it->first, 4, write.first, write.count))
      it = src.erase(it);
    else
      ++it;
  }
}

bool UsedAsBaseBeforeOverwrite(const Program& program,
                               u32 index,
                               u32 sdst,
                               u32 count) {
  for (u32 i = index + 1; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode) &&
        DecodeSmem(inst).sbase == sdst)
      return true;
    const ScalarWrite write = DecodeScalarWrite(inst);
    if (Overlaps(sdst, count, write.first, write.count))
      return false;
  }
  return false;
}

base::HashMap<u32, u64> BufferVersionKeys(const Program& program) {
  base::HashMap<u32, u64> out;
  u32 versions[136] = {};
  u32 generation = 1;
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode)) {
      const Smem smem = DecodeSmem(inst);
      const u32 dwords = smem.op >= 0x08 ? 4 : 2;
      u64 key = smem.sbase | (static_cast<u64>(dwords == 4) << 7);
      for (u32 i = 0; i < dwords; i++)
        key |= static_cast<u64>(versions[smem.sbase + i]) << (8 + i * 13);
      out.emplace(inst.pc, key);
    }
    const ScalarWrite write = DecodeScalarWrite(inst);
    for (u32 i = 0; i < write.count && write.first + i < 136; i++)
      versions[write.first + i] = generation;
    generation++;
  }
  return out;
}

u32 TraceCbufChain(u32 sbase,
                   const base::HashMap<u32, CbufDef>& loads,
                   u32 chain_off[3],
                   u32* len) {
  u32 cur = sbase, n = 0, tmp[3] = {};
  while (n < 3) {
    auto it = loads.find(cur);
    if (it == loads.end())
      break;
    tmp[n++] = it->second.offset;
    cur = it->second.source;
  }
  for (u32 i = 0; i < n; i++)
    chain_off[i] = tmp[n - 1 - i];
  *len = n;
  return cur;
}

static base::HashSet<u32> VmemDescriptorSgprs(const Program& program) {
  base::HashSet<u32> regs;
  for (const Inst& inst : program) {
    const bool buf = inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf;
    const bool img = inst.enc == Enc::kMimg;
    if (!buf && !img)
      continue;
    const u32 srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    for (u32 i = 0; i < 8; i++)
      regs.insert(srsrc + i);
    if (img && inst.opcode >= 0x20) {
      const u32 ssamp = ((inst.raw[1] >> 21) & 0x1F) * 4;
      for (u32 i = 0; i < 4; i++)
        regs.insert(ssamp + i);
    }
  }
  return regs;
}

bool RdnaPlanCbufs(const Program& program,
                   u32 first_binding,
                   base::Vector<ShaderCbuf>& cbufs,
                   base::HashMap<u32, u32>& bindings,
                   base::HashMap<u32, u32>& by_pc,
                   u32 binding_limit = kMaxCbufBindings) {
  base::HashMap<u32, CbufDef> loads;
  base::HashMap<u64, base::Vector<u32> > bindings_by_descriptor;
  base::HashMap<u32, base::Pair<u32, u32> > span;
  base::HashMap<u32, u64> desc_src;
  const auto version_keys = BufferVersionKeys(program);
  const auto descriptor_sgprs = VmemDescriptorSgprs(program);
  base::HashSet<u32> indexed_raw_descriptors;
  for (const Inst& inst : program)
    if (((inst.enc == Enc::kMubuf && inst.opcode >= 0x08 &&
          inst.opcode <= 0x0f) ||
         (inst.enc == Enc::kMtbuf && inst.opcode <= 0x03)) &&
        (inst.raw[0] & (1u << 13)))
      indexed_raw_descriptors.insert(((inst.raw[1] >> 16) & 0x1f) * 4);
  u32 inst_index = 0;
  for (const Inst& inst : program) {
    const u32 producer = inst_index++;
    if (inst.enc != Enc::kSmrd) {
      const ScalarWrite write = DecodeScalarWrite(inst);
      InvalidateCbufDefs(loads, write);
      InvalidateDescSrc(desc_src, write);
      continue;
    }
    const Smem smem = DecodeSmem(inst);
    const u32 op = smem.op;
    const bool sbufload = op >= 0x08 && op <= 0x0C, sload = op <= 0x04;
    if (!sbufload && !sload)
      continue;
    const u32 sbase = smem.sbase, sdst = smem.sdst;
    const u32 load_count = SmemLoadCount(op);
    const i32 off =
        sbufload ? static_cast<i32>(inst.raw[1] & 0xFFFFF) : smem.offset;
    if (sload && load_count == 4) {
      InvalidateDescSrc(desc_src, {sdst, load_count});
      desc_src[sdst] = version_keys.at(inst.pc) * 0x9E3779B97F4A7C15ull +
                       static_cast<u64>(static_cast<u32>(off));
    }
    const bool raw_descriptor =
        base::AnyOf(indexed_raw_descriptors.begin(),
                    indexed_raw_descriptors.end(), [&](u32 descriptor) {
                      return Overlaps(sdst, load_count, descriptor, 4);
                    });
    if (sload && descriptor_sgprs.count(sdst) && !raw_descriptor) {
      InvalidateCbufDefs(loads, {sdst, load_count});
      continue;
    }
    if (sload && smem.soffset == 125 && off < 0) {
      if (ShDbg() || kDrawCensus)
        BASE_LOGI("gcnspv", "cbuf plan reject pc={:#x} sload negative off={}",
                  inst.pc, off);
      return false;
    }
    if (sload && !raw_descriptor &&
        UsedAsBaseBeforeOverwrite(program, producer, sdst, load_count)) {
      InvalidateCbufDefs(loads, {sdst, load_count});
      loads[sdst] = {sbase, static_cast<u32>(off), load_count};
      continue;
    }
    const bool dynamic = smem.soffset != 125;
    const u32 lo = dynamic ? 0 : static_cast<u32>(off < 0 ? 0 : off) / 4;
    const u32 hi = dynamic ? gpu::gcn::kCbufDwords : lo + SmemLoadCount(op);

    u32 chain_off[3] = {}, chain_len = 0;
    const u32 root = TraceCbufChain(sbase, loads, chain_off, &chain_len);
    const auto src_it = desc_src.find(sbase);
    const u64 key =
        src_it != desc_src.end() ? src_it->second : version_keys.at(inst.pc);

    base::Vector<u32>& shared = bindings_by_descriptor[key];
    u32 binding = ~0u;
    for (u32 b : shared) {
      const auto& s = span[b];
      if (base::Max(s.second, hi) - base::Min(s.first, lo) <=
          gpu::gcn::kCbufDwords) {
        binding = b;
        break;
      }
    }
    if (binding == ~0u) {
      binding = first_binding + static_cast<u32>(cbufs.size());
      if (binding >= binding_limit) {
        if (ShDbg() || kDrawCensus)
          BASE_LOGI("gcnspv",
                    "cbuf plan reject pc={:#x} out of "
                    "bindings ({})",
                    inst.pc, binding);
        return false;
      }
      shared.push_back(binding);
      bindings.emplace(sbase, binding);
      ShaderCbuf cb;
      cb.binding = binding;
      cb.ud_sgpr = root;
      cb.first_dword = lo;
      cb.num_dwords = hi - lo;
      cb.chain_len = chain_len;
      for (u32 i = 0; i < 3; i++)
        cb.chain_off[i] = chain_off[i];
      cb.use_pc = inst.pc;
      cbufs.push_back(cb);
      span[binding] = {lo, hi};
      if (ShDbg())
        BASE_LOGI("gcnspv",
                  "cbuf bind={} pc={:#x} sbase={} root={} chain={} dw={}..{}",
                  binding, inst.pc, sbase, root, chain_len, lo, hi);
    } else {
      auto& s = span[binding];
      s.first = base::Min(s.first, lo);
      s.second = base::Max(s.second, hi);
      for (ShaderCbuf& cb : cbufs)
        if (cb.binding == binding) {
          cb.first_dword = s.first;
          cb.num_dwords = s.second - s.first;
        }
    }
    by_pc[inst.pc] = binding;
    InvalidateCbufDefs(loads, {sdst, load_count});
    InvalidateDescSrc(desc_src, {sdst, load_count});
  }
  return true;
}

bool DirectGraphics() {
  return gpu::gcn::g_direct_table != 0;
}

void PlanDirect(const Program& program, StageContext& sc) {
  sc.direct = true;
  u32 next = 0;
  for (const Inst& inst : program) {
    const u32 w1 = inst.raw[1];
    u32 kind = 0, sgpr = 0;
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode)) {
      const Smem smem = DecodeSmem(inst);
      kind = smem.op >= 0x08 ? 0 : 2;
      sgpr = smem.sbase;
    } else if (inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf) {
      sgpr = ((w1 >> 16) & 0x1F) * 4;
    } else {
      continue;
    }
    sc.cs_bind[inst.pc] = next;
    sc.cs_runtime_resources[next++] = {kind, sgpr};
  }
}

bool IsDirectMemory(const Inst& inst) {
  return inst.enc == Enc::kSmrd || inst.enc == Enc::kMubuf ||
         inst.enc == Enc::kMtbuf || inst.enc == Enc::kFlat;
}

void NoteCbufWindows(const base::Vector<ShaderCbuf>& cbufs, StageContext& sc) {
  for (const ShaderCbuf& cb : cbufs)
    if (cb.first_dword)
      sc.cbuf_first_dword[cb.binding] = cb.first_dword;
}

void RdnaPlanGfxBuffers(const Program& program,
                        u32 first_binding,
                        const base::HashSet<u32>* claimed,
                        base::Vector<gpu::gcn::ShaderBuffer>& buffers,
                        base::HashMap<u32, u32>& bindings) {
  base::HashMap<u32, u32> by_srsrc;
  for (const Inst& inst : program) {
    const auto writes = PossibleScalarWrites(inst);
    for (auto it = by_srsrc.begin(); it != by_srsrc.end();) {
      const bool flat = it->first >= kFlatBaseTag;
      const u32 first = flat ? it->first - kFlatBaseTag : it->first;
      const u32 count = flat ? 2u : 4u;
      bool overwritten = false;
      for (const auto& write : writes.range)
        overwritten |= write.count && first < write.first + write.count &&
                       write.first < first + count;
      if (overwritten)
        it = by_srsrc.erase(it);
      else
        ++it;
    }
    u32 srsrc;
    if (inst.enc == Enc::kFlat) {
      const Flat f = DecodeFlat(inst);
      if (!FlatServableLoad(f))
        continue;
      srsrc = kFlatBaseTag + f.saddr;
    } else {
      const bool raw = inst.opcode >= 0x08 && inst.opcode <= 0x0f;
      const bool format = inst.opcode <= 0x03;
      const bool typed = inst.enc == Enc::kMtbuf && format;
      if (!typed && (inst.enc != Enc::kMubuf || (!raw && !format)))
        continue;
      if (claimed && claimed->count(inst.pc))
        continue;
      srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    }
    const auto found = by_srsrc.find(srsrc);
    if (found != by_srsrc.end()) {
      bindings[inst.pc] = found->second;
      continue;
    }
    const u32 binding = first_binding + static_cast<u32>(buffers.size());
    if (binding >= gpu::gcn::MaxGfxBuffers()) {
      gpu::gcn::WarnUnsupported("mubuf.binding-count", binding + 1);
      continue;
    }
    by_srsrc[srsrc] = binding;
    bindings[inst.pc] = binding;
    buffers.push_back({binding, srsrc, inst.pc});
  }
}

base::HashSet<u32> LaunchExecMovPcs(const Program& program) {
  base::HashSet<u32> skip, written;
  for (const Inst& in : program) {
    u32 d0 = 0xFFFF, n = 1;
    switch (in.enc) {
      case Enc::kSop1: {
        const u32 sdst = (in.raw[0] >> 16) & 0x7F;
        const u32 src = in.raw[0] & 0xFF;
        const bool b64 = in.opcode == 0x04;
        if ((in.opcode == 0x03 || b64) && sdst == 126 && src <= 105 &&
            !written.count(src) && (!b64 || !written.count(src + 1))) {
          skip.insert(in.pc);
          if (ShDbg())
            BASE_LOGI("gcnspv", "drop launch-state exec mov @pc={:04x} (s{})",
                      in.pc, src);
        }
        d0 = sdst;
        n = b64 ? 2 : 1;
        break;
      }
      case Enc::kSop2:
      case Enc::kSopk:
        d0 = (in.raw[0] >> 16) & 0x7F;
        break;
      case Enc::kSmrd:
        if (in.opcode <= 0x0C) {
          d0 = (in.raw[0] >> 6) & 0x7F;
          n = SmemLoadCount(in.opcode);
        }
        break;
      default:
        break;
    }
    if (d0 <= 105)
      for (u32 k = 0; k < n; k++)
        written.insert(d0 + k);
  }
  return skip;
}

void RdnaEmitSmem(Translator& t, const Inst& inst, StageContext& sc) {
  const Smem smem = DecodeSmem(inst);
  const u32 op = smem.op;
  if (op <= 0x04 || (op >= 0x08 && op <= 0x0C)) {
    auto pc_it = sc.smem_cbuf_by_pc.find(inst.pc);
    auto base_it = sc.cbuf_bind.find(smem.sbase);
    if (pc_it == sc.smem_cbuf_by_pc.end() && base_it == sc.cbuf_bind.end()) {
      if (op >= 0x08)
        gpu::gcn::WarnUnsupported("smem.cbuf-unplanned", op, inst.raw[0],
                                  inst.raw[1]);
      return;
    }
    const u32 binding =
        pc_it != sc.smem_cbuf_by_pc.end() ? pc_it->second : base_it->second;
    const u32 immediate = op >= 0x08 ? inst.raw[1] & 0xFFFFC
                                     : static_cast<u32>(smem.offset) & ~3u;
    const Id soffset =
        smem.soffset == 125 ? t.U32(0) : t.SrcRaw(smem.soffset, 0);
    const Id byte_offset = t.Add(t.And(soffset, t.U32(~3u)), t.U32(immediate));
    const auto first_it = sc.cbuf_first_dword.find(binding);
    const u32 first =
        first_it == sc.cbuf_first_dword.end() ? 0 : first_it->second;
    Id dword0 = t.Shr(byte_offset, t.U32(2));
    if (first)
      dword0 = t.Sub(dword0, t.U32(first));
    const u32 n = SmemLoadCount(op);
    for (u32 k = 0; k < n; k++)
      t.SetSdst(smem.sdst, k, t.CbufDwordId(binding, t.Add(dword0, t.U32(k))));
    return;
  }
  gpu::gcn::WarnUnsupported("smem.rdna", op, inst.raw[0], inst.raw[1]);
}

thread_local u64 g_ps_addr = 0;

thread_local base::HashMap<u32, BufferResource> g_stage_bufs;

thread_local bool g_warned_store = false;

enum class BufNum : u8 {
  kUnorm,
  kSnorm,
  kUscaled,
  kSscaled,
  kUint,
  kSint,
  kFloat
};

struct BufFormat {
  u32 comps = 0;
  u32 bits[4] = {};
  BufNum num = BufNum::kUnorm;
};

bool DecodeBufFormat(u32 gfmt, BufFormat& out) {
  struct Run {
    u8 first, count, comps, bits[4];
  };
  static constexpr Run kRuns[] = {
      {1, 6, 1, {8}},
      {7, 7, 1, {16}},
      {14, 6, 2, {8, 8}},
      {20, 3, 1, {32}},
      {23, 7, 2, {16, 16}},
      {30, 7, 3, {10, 11, 11}},
      {37, 7, 3, {11, 11, 10}},
      {44, 6, 4, {10, 10, 10, 2}},
      {50, 6, 4, {2, 10, 10, 10}},
      {56, 6, 4, {8, 8, 8, 8}},
      {62, 3, 2, {32, 32}},
      {65, 7, 4, {16, 16, 16, 16}},
      {72, 3, 3, {32, 32, 32}},
      {75, 3, 4, {32, 32, 32, 32}},
  };
  for (const Run& r : kRuns) {
    if (gfmt < r.first || gfmt >= u32{r.first} + r.count)
      continue;
    const u32 i = gfmt - r.first;
    out.comps = r.comps;
    for (u32 c = 0; c < 4; c++)
      out.bits[c] = r.bits[c];
    out.num = r.count == 3 ? (i == 0   ? BufNum::kUint
                              : i == 1 ? BufNum::kSint
                                       : BufNum::kFloat)
              : i == 6     ? BufNum::kFloat
                           : static_cast<BufNum>(i);
    return true;
  }
  return false;
}

Id ConvertBufChannel(Translator& t, Id raw, u32 bits, BufNum num) {
  const bool is_signed =
      num == BufNum::kSnorm || num == BufNum::kSscaled || num == BufNum::kSint;
  Id value = raw;
  if (is_signed && bits < 32)
    value = t.m.Bitcast(
        t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                        {t.m.Bitcast(t.t_i, raw), t.U32(0), t.U32(bits)}));
  const auto as_float = [&](Id f) { return t.m.Bitcast(t.t_u, f); };
  const auto to_f = [&](bool sgn) {
    return sgn ? t.m.Emit(spv::Op::OpConvertSToF, t.t_f,
                          {t.m.Bitcast(t.t_i, value)})
               : t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {value});
  };
  switch (num) {
    case BufNum::kUint:
    case BufNum::kSint:
      return value;
    case BufNum::kFloat:
      if (bits == 32)
        return value;
      return as_float(t.m.CompositeExtract(
          t.t_f, t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {value}), 0));
    case BufNum::kUscaled:
      return as_float(to_f(false));
    case BufNum::kSscaled:
      return as_float(to_f(true));
    case BufNum::kUnorm:
      return as_float(t.FMul(
          to_f(false), t.F32(1.f / static_cast<float>((1ull << bits) - 1))));
    case BufNum::kSnorm:
      return as_float(t.Ext2(
          GLSLstd450FMax,
          t.FMul(to_f(true),
                 t.F32(1.f / static_cast<float>((1ull << (bits - 1)) - 1))),
          t.F32(-1.f)));
  }
  return value;
}

bool RdnaEmitBufFormatLoad(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const auto bind = sc.gfx_buf_bind.find(inst.pc);
  const auto res = g_stage_bufs.find(inst.pc);
  const bool typed = inst.enc == Enc::kMtbuf;
  if (bind == sc.gfx_buf_bind.end() ||
      (!typed && (res == g_stage_bufs.end() || !res->second.descriptor_valid)))
    return false;
  BufFormat fmt;
  const u32 format =
      typed ? (w >> 19) & 0x7F : (res->second.descriptor[3] >> 12) & 0x7F;
  if (!DecodeBufFormat(format, fmt))
    return false;
  if (fmt.num == BufNum::kFloat && fmt.bits[0] != 16 && fmt.bits[0] != 32)
    return false;

  const u32 soffset_field = (w1 >> 24) & 0xFF;
  u32 soffset = 0;
  if (!typed && soffset_field != 125 && soffset_field != 128) {
    if (!res->second.soffset_valid)
      return false;
    soffset = res->second.soffset;
  }
  const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
  const Id stride =
      typed ? t.And(t.Shr(t.Sg(srsrc + 1), t.U32(16)), t.U32(0x3FFF))
            : t.U32((res->second.descriptor[1] >> 16) & 0x3FFF);
  const u32 vdata = (w1 >> 8) & 0xFF, vaddr = w1 & 0xFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  Id byte_off =
      typed ? t.Add(t.U32(w & 0xFFF), t.SrcRaw(soffset_field, inst.literal))
            : t.U32((w & 0xFFF) + soffset);
  u32 va = vaddr;
  if (idxen)
    byte_off = t.Add(byte_off, t.Mul(t.Vg(va++), stride));
  if (offen)
    byte_off = t.Add(byte_off, t.Vg(va));

  const Id var = t.EnsureGfxBuffer(bind->second);
  const Id p_u = t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u);
  const auto dword = [&](Id index) {
    return t.m.Load(
        t.t_u,
        t.m.AccessChain(
            p_u, var,
            {t.U32(0), t.UMin(index, t.U32(gpu::gcn::kGfxBufferDwords - 1))}));
  };
  const Id first_bit = t.Shl(byte_off, t.U32(3));
  const u32 requested = (inst.opcode & 3) + 1;
  u32 channel_bit = 0;
  for (u32 c = 0; c < requested; c++) {
    if (c >= fmt.comps) {
      const bool integer = fmt.num == BufNum::kUint || fmt.num == BufNum::kSint;
      t.SetVg(vdata + c, t.U32(c != 3 ? 0u : (integer ? 1u : 0x3F800000u)));
      continue;
    }
    const u32 bits = fmt.bits[c];
    const Id bit = t.Add(first_bit, t.U32(channel_bit));
    const Id word = dword(t.Shr(bit, t.U32(5)));
    const Id raw = bits == 32
                       ? word
                       : t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u,
                                  {word, t.And(bit, t.U32(31)), t.U32(bits)});
    t.SetVg(vdata + c, ConvertBufChannel(t, raw, bits, fmt.num));
    channel_bit += bits;
  }
  return true;
}

void RdnaEmitFlat(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const Flat f = DecodeFlat(inst);
  if (FlatIsStore(f.op)) {
    gpu::gcn::AuditNote("flat.store.rdna", f.op);
    if (ShDbg() && !g_warned_store) {
      g_warned_store = true;
      BASE_LOGI("gcnspv", "dropped flat store op={:#x} @pc={:04x}", f.op,
                inst.pc);
    }
    return;
  }
  if (f.lds) {
    gpu::gcn::WarnUnsupported("flat.lds-dma.rdna", f.op, w, w1);
    return;
  }
  if (f.seg == 1) {
    gpu::gcn::WarnUnsupported("flat.scratch.rdna", f.op, w, w1);
    return;
  }
  if (f.seg != 2 || f.saddr_null) {
    gpu::gcn::WarnUnsupported("flat.vgpr-address.rdna", f.op, w, w1);
    return;
  }
  const u32 n = FlatLoadDwords(f.op);
  if (!n) {
    gpu::gcn::WarnUnsupported("flat.rdna", f.op, w, w1);
    return;
  }
  if (f.offset < 0) {
    gpu::gcn::WarnUnsupported("flat.negative-offset.rdna", f.op, w, w1);
    return;
  }
  const auto bind = sc.gfx_buf_bind.find(inst.pc);
  if (bind == sc.gfx_buf_bind.end()) {
    gpu::gcn::WarnUnsupported("flat.unplanned.rdna", f.op, w, w1);
    return;
  }
  const Id var = t.EnsureGfxBuffer(bind->second);
  const Id p_u = t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u);
  const auto dword = [&](Id index) {
    return t.m.Load(
        t.t_u,
        t.m.AccessChain(
            p_u, var,
            {t.U32(0), t.UMin(index, t.U32(gpu::gcn::kGfxBufferDwords - 1))}));
  };
  const Id byte_off =
      t.UMin(t.Add(t.Vg(f.addr), t.U32(static_cast<u32>(f.offset))),
             t.U32(gpu::gcn::kGfxBufferDwords * 4 - 4));
  if (f.op >= 0x08 && f.op <= 0x0b) {
    const u32 bits = f.op <= 0x09 ? 8u : 16u;
    const bool sext = f.op == 0x09 || f.op == 0x0b;
    const Id word = dword(t.Shr(byte_off, t.U32(2)));
    const Id shift = t.Shl(t.And(byte_off, t.U32(3)), t.U32(3));
    t.SetVg(f.vdst,
            sext ? t.m.Bitcast(
                       t.t_u,
                       t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                {t.m.Bitcast(t.t_i, word), shift, t.U32(bits)}))
                 : t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u,
                            {word, shift, t.U32(bits)}));
    return;
  }
  const Id dword0 = t.Shr(byte_off, t.U32(2));
  for (u32 k = 0; k < n; k++)
    t.SetVg(f.vdst + k, dword(t.Add(dword0, t.U32(k))));
}

void EmitExport(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
  const u32 v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF,
                    (w1 >> 24) & 0xFF};
  if (ShDbg())
    BASE_LOGI("gcnspv",
              "{}-exp target={} en={:#x} compr={} done={} vsrc={:08x}",
              sc.is_ps ? "ps" : "vs", target, en, compr, (w >> 11) & 1, w1);
  if (!en)
    return;
  if (sc.is_mesh) {
    const Id active = t.LaneActive(t.Exec());
    const Id body = t.m.NewBlock(), merge = t.m.NewBlock();
    t.m.SelectionMerge(merge);
    t.m.BranchConditional(active, body, merge);
    t.m.OpenBlock(body);
    if (target == 20) {
      t.m.Store(sc.mesh_primitive, t.Vg(v[0]));
    } else if (target == 12 || target >= 32) {
      const Id out =
          target == 12 ? sc.pos_out : gpu::gcn::VsParamOut(t, sc, target - 32);
      if (target >= 32)
        sc.max_param = base::Max(sc.max_param, target - 31);
      Id c[4];
      for (u32 i = 0; i < 4; ++i)
        c[i] = (en & (1u << i)) ? t.VgF(v[i])
                                : t.F32(target == 12 && i == 3 ? 1.f : 0.f);
      t.m.Store(out, t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
    } else if (target != 9) {
      gpu::gcn::WarnUnsupported("exp.mesh-target", target, w, w1);
    }
    t.m.Branch(merge);
    t.m.OpenBlock(merge);
    return;
  }
  if (sc.is_ps) {
    if (kExpTrace && target <= 7) {
      static u32 seen[256] = {};
      const u32 k = ((target & 7) << 5) | ((en & 0xF) << 1) | compr;
      if (seen[k]++ == 0)
        BASE_LOGI("exp", "ps {:#x} mrt{} en={:#x} compr={}",
                  (unsigned long)g_ps_addr, target, en, compr);
    }
    if (target <= 7) {
      sc.wrote_color = true;
      Id col;
      if (t.last_texel_var && t.last_texel) {
        t.m.Store(gpu::gcn::PsColorOut(t, sc, target),
                  t.m.Load(t.t_v4, t.last_texel_var));
        return;
      }
      if (compr) {
        Id c[4];
        if (en & 0x1) {
          const Id c01 =
              t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[0])});
          c[0] = t.m.CompositeExtract(t.t_f, c01, 0);
          c[1] = t.m.CompositeExtract(t.t_f, c01, 1);
        } else {
          c[0] = c[1] = t.F32(0.f);
        }
        if (en & 0x4) {
          const Id c23 =
              t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[1])});
          c[2] = t.m.CompositeExtract(t.t_f, c23, 0);
          c[3] = t.m.CompositeExtract(t.t_f, c23, 1);
        } else {
          c[2] = t.F32(0.f);
          c[3] = t.F32(1.f);
        }
        col = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
      } else {
        Id c[4];
        for (int i = 0; i < 4; i++)
          c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
        col = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
      }
      if (!kAllowNan) {
        const Id v = col;
        Id comps[4];
        for (int i = 0; i < 4; i++) {
          const Id c = t.m.CompositeExtract(t.t_f, v, i);
          const Id finite = t.m.Emit(spv::Op::OpFOrdEqual, t.t_bool, {c, c});
          const Id clamped = t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                                         {c, t.F32(-65504.f), t.F32(65504.f)});
          comps[i] =
              t.m.Emit(spv::Op::OpSelect, t.t_f, {finite, clamped, t.F32(0.f)});
        }
        col = t.m.CompositeConstruct(t.t_v4,
                                     {comps[0], comps[1], comps[2], comps[3]});
      }
      if (kGpuDebugalpha) {
        const Id a = t.m.CompositeExtract(t.t_f, col, 3);
        col = t.m.CompositeConstruct(t.t_v4, {a, a, a, t.F32(1.f)});
      }
      if (kGpuForcecolor)
        col = t.m.CompositeConstruct(
            t.t_v4, {t.F32(1.f), t.F32(0.f), t.F32(0.f), t.F32(1.f)});
      if (kGpuPsVgpr && (!kGpuPsVgprPs || g_ps_addr == (u64)kGpuPsVgprPs)) {
        const Id g = sc.vgpr_snap_var ? t.m.Load(t.t_f, sc.vgpr_snap_var)
                                      : t.VgF(kGpuPsVgpr - 1);
        col = t.m.CompositeConstruct(t.t_v4, {g, g, g, t.F32(1.f)});
      }
      const Id out_var = gpu::gcn::PsColorOut(t, sc, target);
      const auto live = [&](u32 i) {
        return compr ? (en & (i < 2 ? 0x1u : 0x4u)) != 0
                     : (en & (1u << i)) != 0;
      };
      bool all_channels =
          kGpuForcecolor || kGpuDebugalpha ||
          (kGpuPsVgpr && (!kGpuPsVgprPs || g_ps_addr == (u64)kGpuPsVgprPs));
      if (!all_channels) {
        all_channels = true;
        for (u32 i = 0; i < 4; i++)
          all_channels &= live(i);
      }
      if (all_channels) {
        t.m.Store(out_var, col);
      } else {
        const Id comp_ptr_ty =
            t.m.TypePointer(spv::StorageClass::Output, t.t_f);
        for (u32 i = 0; i < 4; i++)
          if (live(i))
            t.m.Store(t.m.AccessChain(comp_ptr_ty, out_var, {t.U32(i)}),
                      t.m.CompositeExtract(t.t_f, col, i));
      }
      if (sc.color_written_var)
        t.m.Store(sc.color_written_var, t.U32(1));
      return;
    }
    if (target == 8 && (en & 1)) {
      const Id depth =
          compr
              ? t.m.CompositeExtract(
                    t.t_f,
                    t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[0])}),
                    0)
              : t.VgF(v[0]);
      t.m.Store(gpu::gcn::PsDepthOut(t, sc), depth);
      return;
    }
    if (target != 9)
      gpu::gcn::WarnUnsupported("exp.ps-target", target, w, w1);
    return;
  }
  if (target == 12) {
    if (sc.dbg_pos_in && sc.dbg_pos_cbuf >= 0) {
      Id in[4];
      const Id val = t.m.Load(sc.dbg_pos_comps == 2   ? t.t_v2
                              : sc.dbg_pos_comps == 3 ? t.t_v3
                                                      : t.t_v4,
                              sc.dbg_pos_in);
      for (u32 i = 0; i < 4; i++)
        in[i] = i < sc.dbg_pos_comps ? t.m.CompositeExtract(t.t_f, val, i)
                                     : t.F32(i == 3 ? 1.f : 0.f);
      if (sc.dbg_pos_world >= 0) {
        Id w[4];
        for (u32 r = 0; r < 3; r++) {
          w[r] = t.F32(0.f);
          for (u32 c = 0; c < 4; c++) {
            const Id m = t.m.Bitcast(
                t.t_f,
                t.CbufDword(static_cast<u32>(sc.dbg_pos_world), r * 4 + c));
            w[r] = t.FAdd(w[r], t.FMul(m, in[c]));
          }
        }
        w[3] = t.F32(1.f);
        for (int i = 0; i < 4; i++)
          in[i] = w[i];
      }
      Id row[4];
      for (u32 r = 0; r < 4; r++) {
        row[r] = t.F32(0.f);
        for (u32 c = 0; c < 4; c++) {
          const Id m =
              t.m.Bitcast(t.t_f, t.CbufDword(static_cast<u32>(sc.dbg_pos_cbuf),
                                             sc.dbg_pos_dword + r * 4 + c));
          row[r] = t.FAdd(row[r], t.FMul(m, in[c]));
        }
      }
      t.m.Store(sc.pos_out, t.m.CompositeConstruct(
                                t.t_v4, {row[0], row[1], row[2], row[3]}));
      return;
    }
    Id c[4];
    for (int i = 0; i < 4; i++)
      c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
    t.m.Store(sc.pos_out,
              t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  } else if (target >= 32 && target <= 63) {
    const u32 p = target - 32;
    if (p + 1 > sc.max_param)
      sc.max_param = p + 1;
    const Id out_var = gpu::gcn::VsParamOut(t, sc, p);
    Id c[4];
    for (int i = 0; i < 4; i++)
      c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(0.f);
    t.m.Store(out_var,
              t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  } else if (target >= 13 && target <= 15) {
    gpu::gcn::NoteApproximated("exp.vs-pos-extra", target);
  } else if (target != 9 && target != 20) {
    gpu::gcn::WarnUnsupported("exp.vs-target", target, w, w1);
  }
}

struct SdwaMod {
  u32 src0 = 0, src1 = 0;
  u32 src0_sel = 6, src1_sel = 6;
  bool src0_sext = false, src1_sext = false;
  bool src0_neg = false, src0_abs = false;
  bool src1_neg = false, src1_abs = false;
  u32 dst_sel = 6;
  u32 dst_unused = 0;
  bool clamp = false;
  u32 omod = 0;
};

SdwaMod DecodeSdwa(const Inst& inst, u32 vsrc1, bool dpp) {
  const u32 m = inst.raw[1];
  SdwaMod s;
  s.src0 = (m & 0xFF) + ((!dpp && ((m >> 23) & 1)) ? 0u : 256u);
  s.src1 = vsrc1 + 256u;
  if (dpp)
    return s;
  s.dst_sel = (m >> 8) & 7;
  s.dst_unused = (m >> 11) & 3;
  s.clamp = (m >> 13) & 1;
  s.omod = (m >> 14) & 3;
  s.src0_sel = (m >> 16) & 7;
  s.src0_sext = (m >> 19) & 1;
  s.src0_neg = (m >> 20) & 1;
  s.src0_abs = (m >> 21) & 1;
  s.src1_sel = (m >> 24) & 7;
  s.src1_sext = (m >> 27) & 1;
  s.src1_neg = (m >> 28) & 1;
  s.src1_abs = (m >> 29) & 1;
  if ((m >> 31) & 1)
    s.src1 = vsrc1;
  return s;
}

Id SdwaFloatMod(Translator& t, Id raw, bool neg, bool abs) {
  if (!neg && !abs)
    return raw;
  Id f = t.m.Bitcast(t.t_f, raw);
  if (abs)
    f = t.m.ExtInst(t.t_f, GLSLstd450FAbs, {f});
  if (neg)
    f = t.m.Emit(spv::Op::OpFNegate, t.t_f, {f});
  return t.m.Bitcast(t.t_u, f);
}

Id SdwaSelect(Translator& t, Id raw, u32 sel, bool sext) {
  if (sel >= 6)
    return raw;
  const u32 bits = sel < 4 ? 8u : 16u;
  const u32 off = sel < 4 ? sel * 8u : (sel - 4) * 16u;
  const Id shifted = t.Shr(raw, t.U32(off));
  if (!sext)
    return t.And(shifted, t.U32(bits == 8 ? 0xFFu : 0xFFFFu));
  return t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_u,
                  {shifted, t.U32(0), t.U32(bits)});
}

void StoreSdwaResult(Translator& t, u32 vdst, const SdwaMod& sd, Id previous) {
  if (sd.dst_sel >= 6)
    return;
  const u32 bits = sd.dst_sel < 4 ? 8 : 16;
  const u32 offset = sd.dst_sel < 4 ? sd.dst_sel * 8 : (sd.dst_sel - 4) * 16;
  const u32 mask = bits == 8 ? 0xff : 0xffff;
  Id value = t.Vg(vdst);
  if (sd.dst_unused == 1)
    value = t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_u,
                     {value, t.U32(0), t.U32(bits)});
  else
    value = t.And(value, t.U32(mask));
  value = t.Shl(value, t.U32(offset));
  if (sd.dst_unused == 2)
    value = t.Or(value, t.And(previous, t.U32(~(mask << offset))));
  t.SetVg(vdst, value);
}

void ResolveValuSrc0(const Inst& inst, u32 src0, u32& field, u32& literal) {
  if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
    field = DecodeSdwa(inst, 0, false).src0;
    literal = 0;
  } else {
    field = src0;
    literal = inst.literal;
  }
}

struct DppLane {
  Id lane = 0;
  Id valid = 0;
  bool known = false;
};

DppLane RdnaDppLane(Translator& t, Id subid, u32 ctrl) {
  DppLane r;
  r.known = true;
  if (ctrl <= 0xFF) {
    const Id quad_base = t.And(subid, t.U32(0xFFFFFFFCu));
    const Id lane = t.And(subid, t.U32(3));
    const Id sel = t.And(t.Shr(t.U32(ctrl), t.Shl(lane, t.U32(1))), t.U32(3));
    r.lane = t.Or(quad_base, sel);
    return r;
  }
  const u32 amount = ctrl & 0xF;
  const Id row = t.And(subid, t.U32(0xFFFFFFF0u));
  const Id lane = t.And(subid, t.U32(15));
  if (ctrl >= 0x101 && ctrl <= 0x10F) {
    r.lane = t.Or(row, t.Add(lane, t.U32(amount)));
    r.valid = t.Ult(lane, t.U32(16u - amount));
    return r;
  }
  if (ctrl >= 0x111 && ctrl <= 0x11F) {
    r.lane = t.Or(row, t.Sub(lane, t.U32(amount)));
    r.valid = t.Uge(lane, t.U32(amount));
    return r;
  }
  if (ctrl >= 0x121 && ctrl <= 0x12F) {
    const Id high = t.Uge(lane, t.U32(amount));
    const Id minus = t.Sub(lane, t.U32(amount));
    const Id plus = t.Add(lane, t.U32(16u - amount));
    r.lane = t.Or(row, t.SelectB(high, minus, plus));
    return r;
  }
  if (ctrl == 0x140 || ctrl == 0x141) {
    const u32 lane_mask = ctrl == 0x141 ? 7u : 15u;
    const Id base = t.And(subid, t.U32(~lane_mask));
    const Id in = t.And(subid, t.U32(lane_mask));
    r.lane = t.Or(base, t.Sub(t.U32(lane_mask), in));
    return r;
  }
  if (ctrl >= 0x160 && ctrl <= 0x16F) {
    r.lane = t.Xor(subid, t.U32(amount));
    return r;
  }
  r.known = false;
  return r;
}

Id RdnaDppSrc0(Translator& t, StageContext& sc, const Inst& inst) {
  if (!sc.subgroup_local_id)
    return 0;
  const u32 mod = inst.raw[1];
  const u32 src_reg = mod & 0xFF;
  const u32 ctrl = (mod >> 8) & 0x1FF;
  const bool bound_ctrl = ((mod >> 19) & 1) != 0;
  const Id subid =
      t.CanExchange() ? t.WaveLane() : t.m.Load(t.t_u, sc.subgroup_local_id);
  const DppLane sel = RdnaDppLane(t, subid, ctrl);
  if (!sel.known)
    return 0;
  const Id own = t.Vg(src_reg);
  const Id scope = t.U32(static_cast<u32>(spv::Scope::Subgroup));
  Id value = t.CanExchange() ? t.WaveExchange(own, sel.lane)
                             : t.m.Emit(spv::Op::OpGroupNonUniformShuffle,
                                        t.t_u, {scope, own, sel.lane});
  if (sel.valid) {
    if (bound_ctrl)
      value = t.SelectB(sel.valid, value, t.U32(0));
    else
      t.vector_write_guard = sel.valid;
  }
  return value;
}

bool RdnaUsesDpp(const Program& program) {
  for (const Inst& inst : program)
    if (inst.extension == gpu::gcn::InstExtension::kDpp)
      return true;
  return false;
}

bool RdnaUsesLaneOps(const Program& program) {
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kVop3)
      continue;
    const u32 op = (inst.raw[0] >> 16) & 0x3FF;
    if (op == 0x365 || op == 0x366 || op == 0x377 || op == 0x378)
      return true;
  }
  return false;
}

DELTA_OPTION(u64, kCsVgprCs, "DELTA_GPU_CSVGPR_CS", 0);
DELTA_OPTION(u32, kCsVgprPc, "DELTA_GPU_CSVGPR_PC", 0);
DELTA_OPTION(u32, kCsVgpr, "DELTA_GPU_CSVGPR", 0);
thread_local u64 g_cs_addr = 0;

void RdnaEmitInstBody(Translator& t, const Inst& inst, StageContext& sc);

void RdnaEmitInst(Translator& t, const Inst& inst, StageContext& sc) {
  const bool probing = sc.is_cs && kCsVgprCs && g_cs_addr == kCsVgprCs;
  if (probing && !sc.cs_probe_var) {
    const Id arr = t.m.TypeArray(t.t_u, 4);
    sc.cs_probe_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, arr),
                     spv::StorageClass::Private, t.m.ConstNull(arr));
  }
  const Id p_priv = t.p_priv_u;
  if (probing && inst.enc == Enc::kMimg && inst.opcode == 0x08) {
    const u32 vdata = (inst.raw[1] >> 8) & 0xFF;
    const u32 n = __builtin_popcount((inst.raw[0] >> 8) & 0xF);
    for (u32 i = 0; i < n; i++)
      t.SetVg(vdata + i,
              t.m.Load(t.t_u,
                       t.m.AccessChain(p_priv, sc.cs_probe_var, {t.U32(i)})));
  }
  const Id write_guard = t.vector_write_guard;
  RdnaEmitInstBody(t, inst, sc);
  t.vector_write_guard = write_guard;
  if (probing && inst.pc == kCsVgprPc)
    for (u32 i = 0; i < 4; i++) {
      const Id value = kCsVgpr >= 2000   ? t.Vg(kCsVgpr - 2000 + i)
                       : kCsVgpr >= 1000 ? t.Sg(kCsVgpr - 1000 + i)
                                         : t.Vg(kCsVgpr + i);
      t.StorePrivate(t.m.AccessChain(p_priv, sc.cs_probe_var, {t.U32(i)}),
                     kCsVgpr >= 1000
                         ? t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpConvertUToF,
                                                       t.t_f, {value}))
                         : value);
    }
}

void RdnaEmitInstBody(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  Id dpp_src0 = 0;
  if (inst.extension == gpu::gcn::InstExtension::kDpp) {
    dpp_src0 = RdnaDppSrc0(t, sc, inst);
    if (!dpp_src0) {
      gpu::gcn::WarnUnsupported("dpp.rdna", inst.opcode, w, w1);
      return;
    }
  } else if (inst.extension == gpu::gcn::InstExtension::kDpp8 ||
             inst.extension == gpu::gcn::InstExtension::kDpp8Fi) {
    gpu::gcn::WarnUnsupported("dpp8.rdna", inst.opcode, w, w1);
    return;
  }
  if (!dpp_src0 && UsesUnsupportedRdnaSource(inst)) {
    gpu::gcn::WarnUnsupported("source.special.rdna", inst.opcode, w, w1);
    return;
  }
  if (sc.direct && IsDirectMemory(inst) &&
      (gpu::gcn::EmitDirectLoad(t, inst, sc) || EmitCsMemory(t, inst, sc)))
    return;
  if (sc.is_cs && EmitCsMemory(t, inst, sc))
    return;
  if (sc.is_mesh && inst.enc == Enc::kSopp && inst.opcode == 0x0a) {
    t.Barrier();
    return;
  }
  if (sc.is_mesh && inst.enc == Enc::kSopp && inst.opcode == 0x10 &&
      (w & 0xf) == 9) {
    const Id leader = t.InBlock(t.Eq(sc.mesh_local_index, t.U32(0)));
    const Id body = t.m.NewBlock(), merge = t.m.NewBlock();
    t.m.SelectionMerge(merge);
    t.m.BranchConditional(leader, body, merge);
    t.m.OpenBlock(body);
    const Id p = t.m.TypePointer(spv::StorageClass::Workgroup, t.t_u);
    const Id counts = t.Sg(124);
    t.m.Store(t.m.AccessChain(p, sc.mesh_counts, {t.U32(0)}),
              t.And(counts, t.U32(0x7ff)));
    t.m.Store(t.m.AccessChain(p, sc.mesh_counts, {t.U32(1)}),
              t.And(t.Shr(counts, t.U32(12)), t.U32(0x7ff)));
    t.m.Branch(merge);
    t.m.OpenBlock(merge);
    return;
  }
  if (sc.is_mesh && inst.enc == Enc::kDs) {
    if ((w >> 17) & 1) {
      gpu::gcn::WarnUnsupported("ds.mesh-gds", inst.opcode, w, w1);
      return;
    }
    const u32 op = inst.opcode;
    const bool writes = op != 0x35 && !(op >= 0x36 && op <= 0x38) &&
                        !(op >= 0x76 && op <= 0x78) && op != 0xfe && op != 0xff;
    if (!writes) {
      gpu::gcn::EmitDs(t, inst, sc);
      return;
    }
    const Id active = t.LaneActive(t.Exec());
    const Id body = t.m.NewBlock(), merge = t.m.NewBlock();
    t.m.SelectionMerge(merge);
    t.m.BranchConditional(active, body, merge);
    t.m.OpenBlock(body);
    gpu::gcn::EmitDs(t, inst, sc);
    t.m.Branch(merge);
    t.m.OpenBlock(merge);
    return;
  }
  switch (inst.enc) {
    case Enc::kSop1:
      if (sc.skip_launch_movs.count(inst.pc))
        break;
      if (inst.opcode == 0x20 || inst.opcode == 0x21) {
        gpu::gcn::WarnUnsupported("sop1.control-flow.rdna", inst.opcode, w, w1);
        break;
      }
      gpu::gcn::EmitSop1(t, inst);
      break;
    case Enc::kSop2:
      if (!RdnaEmitSop2(t, inst))
        gpu::gcn::EmitSop2(t, inst);
      break;
    case Enc::kSopc:
      gpu::gcn::EmitSopc(t, inst);
      break;
    case Enc::kSopk:
      if (inst.opcode >= 0x17 && inst.opcode <= 0x1A)
        break;
      if (inst.opcode == 0x10) {
        const u32 sdst = (w >> 16) & 0x7f;
        const u32 simm =
            static_cast<u32>(static_cast<i32>(static_cast<i16>(w & 0xffff)));
        t.SetSdst(sdst, 0, t.Mul(t.Sdst(sdst), t.U32(simm)));
        break;
      }
      gpu::gcn::EmitSopk(t, inst);
      break;
    case Enc::kSopp:
      if (inst.opcode == 0x0A) {
        if (sc.is_cs) {
          t.m.EmitVoid(spv::Op::OpControlBarrier,
                       {t.U32(2), t.U32(2), t.U32(0x108)});
        } else if (sc.lds_wave_base) {
          t.m.EmitVoid(
              spv::Op::OpControlBarrier,
              {t.U32(static_cast<u32>(spv::Scope::Subgroup)),
               t.U32(static_cast<u32>(spv::Scope::Subgroup)), t.U32(0x48)});
        } else {
          gpu::gcn::NoteApproximated("barrier.graphics", inst.opcode);
        }
      } else if (inst.opcode != 0x00 && inst.opcode != 0x01 &&
                 inst.opcode != 0x02 &&
                 !(inst.opcode >= 0x04 && inst.opcode <= 0x09) &&
                 inst.opcode != 0x0C && inst.opcode != 0x0E &&
                 inst.opcode != 0x0F && inst.opcode != 0x10 &&
                 !(inst.opcode >= 0x17 && inst.opcode <= 0x1A) &&
                 inst.opcode != 0x1E && inst.opcode != 0x1F &&
                 inst.opcode != 0x20 && inst.opcode != 0x21 &&
                 inst.opcode != 0x23) {
        gpu::gcn::WarnUnsupported("sopp.rdna", inst.opcode, w, w1);
      }
      break;
    case Enc::kSmrd:
      RdnaEmitSmem(t, inst, sc);
      break;
    case Enc::kVop1: {
      const u32 op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const u32 raw0 = w & 0x1FF;
      u32 src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (dpp_src0) {
        gpu::gcn::EmitVop1(t, op, vdst, t.m.Bitcast(t.t_f, dpp_src0));
        break;
      }
      if (raw0 == 249) {
        const SdwaMod sd = DecodeSdwa(inst, 0, false);
        if (sd.dst_sel > 6 || sd.dst_unused > 2 || sd.src0_sel > 6) {
          gpu::gcn::WarnUnsupported("vop1.sdwa-mod", op, w, w1);
          break;
        }
        const Id previous = sd.dst_sel < 6 ? t.Vg(vdst) : 0;
        gpu::gcn::EmitVop1(
            t, op, vdst,
            t.m.Bitcast(t.t_f,
                        SdwaFloatMod(t,
                                     SdwaSelect(t, t.SrcRaw(src0, 0),
                                                sd.src0_sel, sd.src0_sext),
                                     sd.src0_neg, sd.src0_abs)));
        if (sd.omod) {
          const float k = sd.omod == 1 ? 2.f : sd.omod == 2 ? 4.f : 0.5f;
          t.SetVgF(vdst, t.FMul(t.VgF(vdst), t.F32(k)));
        }
        if (sd.clamp)
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                                     {t.VgF(vdst), t.F32(0.f), t.F32(1.f)}));
        StoreSdwaResult(t, vdst, sd, previous);
        break;
      }
      if (op >= 0x42 && op <= 0x44) {
        const auto reg_at = [&](u32 base) {
          const Id idx =
              t.m.Emit(spv::Op::OpIAdd, t.t_u, {t.U32(base), t.Sg(124)});
          return t.m.AccessChain(
              t.p_priv_u, t.vgpr,
              {t.m.ExtInst(t.t_u, GLSLstd450UMin, {idx, t.U32(255)})});
        };
        if (op == 0x42)
          t.StorePrivate(reg_at(vdst), t.SrcRaw(src0, lit));
        else if (op == 0x43)
          t.SetVg(vdst, t.m.Load(t.t_u, reg_at(raw0 & 0xFF)));
        else
          t.StorePrivate(reg_at(vdst), t.m.Load(t.t_u, reg_at(raw0 & 0xFF)));
        break;
      }
      if (op == 0x0b) {
        gpu::gcn::EmitVop1(t, op, vdst,
                           t.m.Bitcast(t.t_f, RdnaF16Bits(t, src0, lit)));
      } else {
        gpu::gcn::EmitVop1(t, op, vdst, t.SrcF(src0, lit));
      }
      break;
    }
    case Enc::kVop2: {
      const u32 op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const u32 vsrc1 = (w >> 9) & 0xFF;
      const u32 raw0 = w & 0x1FF;
      u32 src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (raw0 == 249 && op >= 0x2b && op <= 0x2d) {
        gpu::gcn::WarnUnsupported("vop2.fma-sdwa", op, w, w1);
        break;
      }
      u32 src1 = 256 + vsrc1;
      Id sdwa0 = 0, sdwa1 = 0;
      bool sdwa_clamp = false;
      u32 sdwa_omod = 0;
      SdwaMod sdwa_dst;
      Id sdwa_previous = 0;
      if (raw0 == 249) {
        const SdwaMod sd = DecodeSdwa(inst, vsrc1, false);
        if (sd.dst_sel > 6 || sd.dst_unused > 2 || sd.src0_sel > 6 ||
            sd.src1_sel > 6) {
          gpu::gcn::WarnUnsupported("vop2.sdwa-mod", op, w, w1);
          break;
        }
        sdwa_dst = sd;
        sdwa_previous = sd.dst_sel < 6 ? t.Vg(vdst) : 0;
        sdwa_clamp = sd.clamp;
        sdwa_omod = sd.omod;
        src1 = sd.src1;
        sdwa0 = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src0, 0), sd.src0_sel, sd.src0_sext),
            sd.src0_neg, sd.src0_abs);
        sdwa1 = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src1, 0), sd.src1_sel, sd.src1_sext),
            sd.src1_neg, sd.src1_abs);
      }
      const Id s0u = dpp_src0 ? dpp_src0 : sdwa0 ? sdwa0 : t.SrcRaw(src0, lit);
      const Id s1u = sdwa1 ? sdwa1 : t.SrcRaw(src1, lit);
      switch (op) {
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x35:
        case 0x36:
        case 0x39:
        case 0x3a:
        case 0x3b: {
          const auto half = [&](Id bits) {
            return t.m.CompositeExtract(
                t.t_f, t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {bits}),
                0);
          };
          const Id a =
              half(dpp_src0 || sdwa0 ? s0u : RdnaF16Bits(t, src0, lit));
          const Id b = half(sdwa1 ? s1u : RdnaF16Bits(t, src1, lit));
          Id r;
          switch (op) {
            case 0x32:
              r = t.FAdd(a, b);
              break;
            case 0x33:
              r = t.FSub(a, b);
              break;
            case 0x34:
              r = t.FSub(b, a);
              break;
            case 0x35:
              r = t.FMul(a, b);
              break;
            case 0x36:
              r = t.m.ExtInst(t.t_f, GLSLstd450Fma, {a, b, half(t.Vg(vdst))});
              break;
            case 0x39:
              r = t.m.ExtInst(t.t_f, GLSLstd450NMax, {a, b});
              break;
            case 0x3a:
              r = t.m.ExtInst(t.t_f, GLSLstd450NMin, {a, b});
              break;
            default: {
              const Id e =
                  t.m.Bitcast(t.t_i, t.Sar(t.Shl(s1u, t.U32(16)), t.U32(16)));
              r = t.m.ExtInst(t.t_f, GLSLstd450Ldexp, {a, e});
              break;
            }
          }
          t.SetVg(vdst, t.And(t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16,
                                          {t.m.CompositeConstruct(
                                              t.t_v2, {r, t.F32(0.f)})}),
                              t.U32(0xFFFF)));
          break;
        }
        case 0x02:
        case 0x0D:
          gpu::gcn::WarnUnsupported("vop2.rdna", op, w, w1);
          break;
        case 0x1E:
          t.SetVg(vdst, t.Not(t.Xor(s0u, s1u)));
          break;
        case 0x1F:
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                     {t.m.Bitcast(t.t_f, s0u),
                                      t.m.Bitcast(t.t_f, s1u), t.VgF(vdst)}));
          break;
        case 0x25:
          t.SetVg(vdst, t.Add(s0u, s1u));
          break;
        case 0x26:
          t.SetVg(vdst, t.Sub(s0u, s1u));
          break;
        case 0x27:
          t.SetVg(vdst, t.Sub(s1u, s0u));
          break;
        case 0x28:
        case 0x29:
        case 0x2A: {
          const spv::Op operation =
              op == 0x28 ? spv::Op::OpIAddCarry : spv::Op::OpISubBorrow;
          const Id a = op == 0x2A ? s1u : s0u;
          const Id b = op == 0x2A ? s0u : s1u;
          Id pair = t.m.Emit(operation, t.PairType(), {a, b});
          Id value = t.m.CompositeExtract(t.t_u, pair, 0);
          Id flag = t.m.CompositeExtract(t.t_u, pair, 1);
          pair = t.m.Emit(operation, t.PairType(), {value, t.LaneFlag(106)});
          value = t.m.CompositeExtract(t.t_u, pair, 0);
          flag = t.Or(flag, t.m.CompositeExtract(t.t_u, pair, 1));
          t.SetVg(vdst, value);
          t.SetLaneFlag(106, flag);
          break;
        }
        case 0x2B:
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                     {t.m.Bitcast(t.t_f, s0u),
                                      t.m.Bitcast(t.t_f, s1u), t.VgF(vdst)}));
          break;
        case 0x20:
        case 0x2C:
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                     {t.m.Bitcast(t.t_f, s0u),
                                      t.m.Bitcast(t.t_f, t.U32(lit)),
                                      t.m.Bitcast(t.t_f, s1u)}));
          break;
        case 0x21:
        case 0x2D:
          t.SetVgF(vdst, t.m.ExtInst(
                             t.t_f, GLSLstd450Fma,
                             {t.m.Bitcast(t.t_f, s0u), t.m.Bitcast(t.t_f, s1u),
                              t.m.Bitcast(t.t_f, t.U32(lit))}));
          break;
        default:
          if (op == 0x01 || RdnaSharedVop2(op)) {
            gpu::gcn::EmitVop2(
                t, RemapVop2(op), vdst,
                sdwa0 ? t.m.Bitcast(t.t_f, sdwa0) : t.SrcF(src0, lit),
                sdwa1 ? t.m.Bitcast(t.t_f, sdwa1) : t.SrcF(src1, lit), lit);
          } else {
            gpu::gcn::WarnUnsupported("vop2.opcode.rdna", op, w, w1);
          }
          break;
      }
      if (sdwa_omod) {
        const float k = sdwa_omod == 1 ? 2.f : sdwa_omod == 2 ? 4.f : 0.5f;
        t.SetVgF(vdst, t.FMul(t.VgF(vdst), t.F32(k)));
      }
      if (sdwa_clamp)
        t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                                   {t.VgF(vdst), t.F32(0.f), t.F32(1.f)}));
      StoreSdwaResult(t, vdst, sdwa_dst, sdwa_previous);
      break;
    }
    case Enc::kVop3p: {
      const u32 op = inst.opcode, vdst = w & 0xFF;
      if (op >= 0x20 && op <= 0x22) {
        const u32 half_input = ((w1 >> 27) & 3) | ((w >> 12) & 4);
        const u32 half_select = (w >> 11) & 7;
        const u32 abs = (w >> 8) & 7, neg = (w1 >> 29) & 7;
        Id source[3];
        for (u32 i = 0; i < 3; i++) {
          const u32 field = (w1 >> (i * 9)) & 0x1ff;
          if (half_input & (1u << i)) {
            const Id pair = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                                        {RdnaF16Bits(t, field, inst.literal)});
            source[i] =
                t.m.CompositeExtract(t.t_f, pair, (half_select >> i) & 1);
          } else {
            source[i] = t.SrcF(field, inst.literal);
          }
          if (abs & (1u << i))
            source[i] = t.m.ExtInst(t.t_f, GLSLstd450FAbs, {source[i]});
          if (neg & (1u << i))
            source[i] = t.m.Emit(spv::Op::OpFNegate, t.t_f, {source[i]});
        }
        Id result = t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                {source[0], source[1], source[2]});
        if (w & (1u << 15))
          result = t.FClamp01(result);
        if (op == 0x20) {
          t.SetVgF(vdst, result);
        } else {
          const Id packed = t.m.ExtInst(
              t.t_u, GLSLstd450PackHalf2x16,
              {t.m.CompositeConstruct(t.t_v2, {result, t.F32(0.f)})});
          const u32 shift = op == 0x22 ? 16 : 0;
          t.SetVg(vdst,
                  t.Or(t.And(t.Vg(vdst), t.U32(~(0xffffu << shift))),
                       t.Shl(t.And(packed, t.U32(0xffff)), t.U32(shift))));
        }
        break;
      }
      if ((w & 0x0000FF00u) != 0x00004000u ||
          (w1 & 0xF8000000u) != 0x18000000u) {
        gpu::gcn::WarnUnsupported("vop3p.modifier", op, w, w1);
        break;
      }
      const u32 p0 = w1 & 0x1FF, p1 = (w1 >> 9) & 0x1FF;
      const u32 p2 = (w1 >> 18) & 0x1FF;
      if (!RdnaEmitVop3p(t, op, vdst, p0, p1, p2, inst.literal))
        gpu::gcn::WarnUnsupported("vop3p", op, w, w1);
      break;
    }
    case Enc::kVop3: {
      const u32 rdna_op = inst.opcode;
      u32 op = rdna_op;
      const u32 vdst = w & 0xFF;
      if (op == 0x101)
        op = 0x100;
      if (rdna_op == 0x360)
        op = 0x101;
      const u32 s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const u32 s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      const bool vop3b = RdnaVop3HasSdst(op);
      const u32 sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const u32 abs = vop3b ? 0 : ((w >> 8) & 7);
      const u32 op_sel = vop3b ? 0 : ((w >> 11) & 0xF);
      const bool clamp = (w >> 15) & 1;
      const u32 omod = (w1 >> 27) & 3;
      if (op == 0x102 || op == 0x10D) {
        gpu::gcn::WarnUnsupported("vop3.dot.rdna", op, w, w1);
        break;
      }
      if (op == 0x377 || op == 0x378) {
        if (s0 < 256 || s1 >= 256 || s2 >= 256 || abs || neg || omod || clamp ||
            (op_sel & ~3u) || !sc.subgroup_local_id) {
          gpu::gcn::WarnUnsupported("vop3.permlane", op, w, w1);
          break;
        }
        const Id lane = t.WaveLane();
        const Id selectors =
            t.SelectB(t.IsZero(t.And(lane, t.U32(8))),
                      t.SrcRaw(s1, inst.literal), t.SrcRaw(s2, inst.literal));
        const Id selected =
            t.And(t.Shr(selectors, t.Shl(t.And(lane, t.U32(7)), t.U32(2))),
                  t.U32(15));
        Id row = t.And(lane, t.U32(~15u));
        if (op == 0x378)
          row = t.Xor(row, t.U32(16));
        const Id source_lane = t.Or(row, selected);
        const Id scope = t.U32(static_cast<u32>(spv::Scope::Subgroup));
        const auto shuffle = [&](Id value) {
          if (t.CanExchange())
            return t.WaveExchange(value, source_lane);
          return t.m.Emit(spv::Op::OpGroupNonUniformShuffle, t.t_u,
                          {scope, value, source_lane});
        };
        Id result = shuffle(t.Vg(s0 - 256));
        if (!(op_sel & 1)) {
          const Id active =
              shuffle(t.SelectB(t.LaneActive(t.Exec()), t.U32(1), t.U32(0)));
          result = t.SelectB(t.IsNonZero(active), result,
                             (op_sel & 2) ? t.U32(0) : t.Vg(vdst));
        }
        t.SetVg(vdst, result);
        break;
      }
      if (op == 0x351 || op == 0x354 || op == 0x357 || op == 0x34b) {
        if (clamp || omod) {
          gpu::gcn::WarnUnsupported("vop3.f16-modifier", op, w, w1);
          break;
        }
        const u32 src[3] = {s0, s1, s2};
        Id v[3];
        for (u32 i = 0; i < 3; i++) {
          Id half = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                                {RdnaF16Bits(t, src[i], inst.literal)});
          if ((op_sel >> i) & 1)
            half =
                t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                            {t.Shr(t.SrcRaw(src[i], inst.literal), t.U32(16))});
          v[i] = t.m.CompositeExtract(t.t_f, half, 0);
          if ((abs >> i) & 1)
            v[i] = t.Ext1(GLSLstd450FAbs, v[i]);
          if ((neg >> i) & 1)
            v[i] = t.FNeg(v[i]);
        }
        const auto fmin = [&](Id a, Id b) {
          return t.m.ExtInst(t.t_f, GLSLstd450NMin, {a, b});
        };
        const auto fmax = [&](Id a, Id b) {
          return t.m.ExtInst(t.t_f, GLSLstd450NMax, {a, b});
        };
        Id r;
        if (op == 0x351)
          r = fmin(fmin(v[0], v[1]), v[2]);
        else if (op == 0x354)
          r = fmax(fmax(v[0], v[1]), v[2]);
        else if (op == 0x357)
          r = fmax(fmin(v[0], v[1]), fmin(fmax(v[0], v[1]), v[2]));
        else
          r = t.m.ExtInst(t.t_f, GLSLstd450Fma, {v[0], v[1], v[2]});
        const Id bits = t.And(
            t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16,
                        {t.m.CompositeConstruct(t.t_v2, {r, t.F32(0.f)})}),
            t.U32(0xFFFF));
        t.SetVg(vdst, (op_sel & 8) ? t.Or(t.And(t.Vg(vdst), t.U32(0xFFFF)),
                                          t.Shl(bits, t.U32(16)))
                                   : bits);
        break;
      }
      if (op_sel)
        gpu::gcn::WarnUnsupported("vop3.op-sel", op, w, w1);
      const Id mask_source = t.MaskWord(t.SrcRaw(s2, inst.literal),
                                        t.SrcRawHi(s2, inst.literal, false));
      if (t.lane_masks && op == 0x100) {
        t.SetVgF(vdst, t.SelectF(t.LaneActive(mask_source),
                                 t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                                 t.SrcF(s0, inst.literal, neg & 1, abs & 1)));
        break;
      }
      const Id third =
          (t.lane_masks || t.wave_masks) && op >= 0x128 && op <= 0x12a
              ? t.SelectB(t.LaneActive(mask_source), t.U32(1), t.U32(0))
              : t.SrcRaw(s2, inst.literal);
      if (RdnaEmitVop3Int(t, op, vdst, sdst, t.SrcRaw(s0, inst.literal),
                          t.SrcRaw(s1, inst.literal), third)) {
        if (neg || abs || clamp || omod || op_sel)
          gpu::gcn::WarnUnsupported("vop3.integer-modifier", op, w, w1);
        break;
      }
      if (op == 0x12B) {
        if (clamp || omod || op_sel || (neg & 4) || (abs & 4)) {
          gpu::gcn::WarnUnsupported("vop3.fmac-modifier", op, w, w1);
          break;
        }
        t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                   {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                                    t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                                    t.VgF(vdst)}));
        break;
      }
      if (rdna_op == 0x11F) {
        Id r = t.m.ExtInst(
            t.t_f, GLSLstd450Fma,
            {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
             t.SrcF(s1, inst.literal, neg & 2, abs & 2), t.VgF(vdst)});
        if (omod)
          r = t.FMul(r, t.F32(omod == 1 ? 2.f : omod == 2 ? 4.f : .5f));
        if (clamp)
          r = t.m.ExtInst(t.t_f, GLSLstd450FClamp, {r, t.F32(0.f), t.F32(1.f)});
        t.SetVgF(vdst, r);
        break;
      }
      if (rdna_op >= 0x100 && rdna_op < 0x140 && rdna_op != 0x101 &&
          !RdnaSharedVop2(rdna_op - 0x100)) {
        gpu::gcn::WarnUnsupported("vop3.alias.rdna", rdna_op, w, w1);
        break;
      }
      if (op == 0x141) {
        Id r = t.m.ExtInst(t.t_f, GLSLstd450Fma,
                           {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                            t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                            t.SrcF(s2, inst.literal, neg & 4, abs & 4)});
        if (omod)
          r = t.FMul(r, t.F32(omod == 1 ? 2.f : omod == 2 ? 4.f : .5f));
        if (clamp)
          r = t.m.ExtInst(t.t_f, GLSLstd450FClamp, {r, t.F32(0.f), t.F32(1.f)});
        t.SetVgF(vdst, r);
        break;
      }
      if ((op >= 0x161 && op <= 0x163) || op == 0x16b) {
        gpu::gcn::WarnUnsupported("vop3.opcode.rdna", op, w, w1);
        break;
      }
      const bool integer64_compare =
          (op >= 0xa0 && op <= 0xa7) || (op >= 0xb0 && op <= 0xb7) ||
          (op >= 0xe0 && op <= 0xe7) || (op >= 0xf0 && op <= 0xf7);
      if (integer64_compare) {
        if (neg || abs || clamp || omod || op_sel) {
          gpu::gcn::WarnUnsupported("vop3.integer-modifier", op, w, w1);
          break;
        }
        gpu::gcn::EmitVopc(t, op, t.SrcF(s0, inst.literal),
                           t.SrcF(s1, inst.literal), t.SrcRaw(s0, inst.literal),
                           t.SrcRaw(s1, inst.literal), vdst,
                           t.SrcRawHi(s0, inst.literal, true),
                           t.SrcRawHi(s1, inst.literal, true));
        break;
      }
      const bool supported_compare = op <= 0x1F || (op >= 0x80 && op <= 0x87) ||
                                     (op >= 0x90 && op <= 0x97) ||
                                     (op >= 0xC0 && op <= 0xC7) ||
                                     (op >= 0xD0 && op <= 0xD7);
      if (op < 0x100 && omod) {
        gpu::gcn::WarnUnsupported("vopc.omod", op, w, w1);
        break;
      }
      if (op < 0x100 && !supported_compare) {
        gpu::gcn::WarnUnsupported("vop3.compare.rdna", op, w, w1);
        break;
      }
      if (op < 0x100 && (op & 0x10)) {
        gpu::gcn::EmitVopc(t, op, t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                           t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                           t.SrcRaw(s0, inst.literal),
                           t.SrcRaw(s1, inst.literal), 126);
        break;
      }
      if (op == 0x178) {
        if (neg || abs || clamp || omod || op_sel) {
          gpu::gcn::WarnUnsupported("vop3.xor3-modifier", op, w, w1);
          break;
        }
        t.SetVg(vdst, t.Xor(t.Xor(t.SrcRaw(s0, inst.literal),
                                  t.SrcRaw(s1, inst.literal)),
                            t.SrcRaw(s2, inst.literal)));
        break;
      }
      if (op == 0x16f) {
        if (clamp || omod || op_sel) {
          gpu::gcn::WarnUnsupported("vop3.div-fmas-modifier", op, w, w1);
          break;
        }
        const Id fma =
            t.m.ExtInst(t.t_f, GLSLstd450Fma,
                        {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                         t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                         t.SrcF(s2, inst.literal, neg & 4, abs & 4)});
        t.SetVgF(vdst, t.SelectF(t.LaneActive(t.SgMask(106)),
                                 t.FMul(fma, t.F32(4294967296.0f)), fma));
        break;
      }
      Id source0 = t.SrcF(s0, inst.literal, neg & 1, abs & 1);
      if (op == 0x18b) {
        source0 = t.m.CompositeExtract(
            t.t_f,
            t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                        {RdnaF16Bits(t, s0, inst.literal)}),
            0);
        if (abs & 1)
          source0 = t.Ext1(GLSLstd450FAbs, source0);
        if (neg & 1)
          source0 = t.FNeg(source0);
      }
      u32 emit_op = op;
      if (op == 0x368)
        emit_op = 0x12d;
      else if (op == 0x369)
        emit_op = 0x12e;
      else if (op == 0x36a)
        emit_op = 0x130;
      else if (op == 0x36b)
        emit_op = 0x131;
      else if (op == 0x363)
        emit_op = 0x11e;
      else if (op == 0x364)
        emit_op = 0x122;
      else if (op == 0x365)
        emit_op = 0x123;
      else if (op == 0x366)
        emit_op = 0x124;
      gpu::gcn::EmitVop3(
          t, emit_op, vdst, source0, t.SrcRawHi(s0, inst.literal, op == 0x163),
          t.SrcF(s1, inst.literal, neg & 2, abs & 2),
          t.SrcF(s2, inst.literal, neg & 4, abs & 4),
          t.SrcRawHi(s2, inst.literal, op == 0x177), sdst, clamp, omod);
      break;
    }
    case Enc::kVopc: {
      const u32 op = inst.opcode;
      u32 vsrc1 = (w >> 9) & 0xFF;
      const u32 raw0 = w & 0x1FF;
      u32 src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        const SdwaMod sd = DecodeSdwa(inst, vsrc1, false);
        if (sd.src0_sel > 6 || sd.src1_sel > 6) {
          gpu::gcn::WarnUnsupported("vopc.sdwa-mod", op, w, w1);
          break;
        }
        const u32 dst = ((w1 >> 15) & 1) ? ((w1 >> 8) & 0x7F) : 106u;
        const Id a = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src0, 0), sd.src0_sel, sd.src0_sext),
            sd.src0_neg, sd.src0_abs);
        const Id b = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src1, 0), sd.src1_sel, sd.src1_sext),
            sd.src1_neg, sd.src1_abs);
        gpu::gcn::EmitVopc(t, op, t.m.Bitcast(t.t_f, a), t.m.Bitcast(t.t_f, b),
                           a, b, dst);
        break;
      }
      u32 vopc_src1 = 256 + vsrc1;
      if (raw0 == 249)
        vopc_src1 = DecodeSdwa(inst, vsrc1, false).src1;
      if ((op >= 0xC8 && op <= 0xCF) || (op >= 0xD8 && op <= 0xDF)) {
        auto f16 = [&](u32 f) {
          return t.m.CompositeExtract(
              t.t_f,
              t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                          {RdnaF16Bits(t, f, lit)}),
              0);
        };
        const u32 gcn_op = op >= 0xD8 ? op - 0xD8 + 0x10 : op - 0xC8;
        gpu::gcn::EmitVopc(t, gcn_op, f16(src0), f16(vopc_src1),
                           t.SrcRaw(src0, lit), t.SrcRaw(vopc_src1, lit),
                           gcn_op & 0x10 ? 126 : 106);
        break;
      }
      const bool supported = op <= 0x1F || (op >= 0x80 && op <= 0x87) ||
                             (op >= 0x90 && op <= 0x97) ||
                             (op >= 0xC0 && op <= 0xC7) ||
                             (op >= 0xD0 && op <= 0xD7);
      if (!supported) {
        gpu::gcn::WarnUnsupported("vopc.rdna", op, w, w1);
        break;
      }
      gpu::gcn::EmitVopc(t, op, t.SrcF(src0, lit), t.SrcF(vopc_src1, lit),
                         t.SrcRaw(src0, lit), t.SrcRaw(vopc_src1, lit),
                         (op & 0x10) ? 126 : 106);
      break;
    }
    case Enc::kVintrp:
      if (!sc.is_ps) {
        gpu::gcn::WarnUnsupported("vintrp.stage", inst.opcode, w, w1);
        break;
      }
      gpu::gcn::EmitVintrp(t, w, sc);
      break;
    case Enc::kExp:
      EmitExport(t, inst, sc);
      break;
    case Enc::kMtbuf:
    case Enc::kMubuf: {
      if (inst.enc == Enc::kMubuf && inst.opcode >= 0x08 &&
          inst.opcode <= 0x0f) {
        gpu::gcn::EmitGfxMubuf(t, inst, sc);
        break;
      }
      if (inst.enc == Enc::kMubuf
              ? RdnaMubufStore(inst.opcode) || RdnaMubufAtomicNoReturn(inst)
              : RdnaMtbufStore(inst.opcode)) {
        gpu::gcn::AuditNote(
            inst.enc == Enc::kMtbuf ? "mtbuf.store.rdna" : "mubuf.store.rdna",
            inst.opcode);
        if (ShDbg() && !g_warned_store) {
          g_warned_store = true;
          BASE_LOGI("gcnspv", "dropped buffer store op={:#x} @pc={:04x}",
                    inst.opcode, inst.pc);
        }
        break;
      }
      if (inst.opcode > 0x03) {
        gpu::gcn::WarnUnsupported(
            inst.enc == Enc::kMtbuf ? "mtbuf.rdna" : "mubuf.rdna", inst.opcode,
            w, w1);
        break;
      }
      const bool tfe = (w1 >> 23) & 1;
      const bool lds = inst.enc == Enc::kMubuf && ((w >> 16) & 1);
      if (tfe || lds) {
        gpu::gcn::WarnUnsupported("buffer.control.rdna", inst.opcode, w, w1);
        break;
      }
      if (auto sit = sc.vfetch_seed.find(inst.pc);
          sit != sc.vfetch_seed.end()) {
        const auto& vs = sit->second;
        const Id comp_ty = vs.num_comps == 1   ? t.t_f
                           : vs.num_comps == 2 ? t.t_v2
                           : vs.num_comps == 3 ? t.t_v3
                                               : t.t_v4;
        const Id val = t.m.Load(comp_ty, vs.in_var);
        for (u32 c = 0; c < vs.num_comps; c++)
          t.SetVgF(vs.dest_vgpr + c, vs.num_comps == 1
                                         ? val
                                         : t.m.CompositeExtract(t.t_f, val, c));
        break;
      }
      if (!RdnaEmitBufFormatLoad(t, inst, sc))
        gpu::gcn::WarnUnsupported(inst.enc == Enc::kMtbuf
                                      ? "mtbuf.format-conversion.rdna"
                                      : "mubuf.format-conversion.rdna",
                                  inst.opcode, w, w1);
      break;
    }
    case Enc::kMimg: {
      if (inst.opcode == 0x09 || inst.opcode == 0x21 || inst.opcode == 0x25) {
        gpu::gcn::WarnUnsupported("mimg.rdna-modifier", inst.opcode, w, w1);
        break;
      }
      const bool a16 = (w1 >> 30) & 1, d16 = (w1 >> 31) & 1;
      const bool tfe = (w >> 16) & 1, lwe = (w >> 17) & 1;
      const bool unrm = (w >> 12) & 1;
      if (a16 || d16 || tfe || lwe || unrm) {
        gpu::gcn::WarnUnsupported("mimg.control.rdna", inst.opcode, w, w1);
        break;
      }
      Inst lowered = inst;
      const u32 dim = (w >> 3) & 0x7;
      const bool arrayed_dim = dim == 3 || dim == 4 || dim == 5;
      if (dim > 5) {
        gpu::gcn::WarnUnsupported("mimg.dim", dim, w, w1);
        break;
      }
      if (((w >> 15) & 1) && arrayed_dim) {
        gpu::gcn::WarnUnsupported("mimg.r128-array.rdna", inst.opcode, w, w1);
        break;
      }
      lowered.raw[0] = (w & ~0x4000u) | (arrayed_dim ? 0x4000u : 0u);
      const u32 nsa = (w >> 1) & 0x3;
      const bool cube = dim == 3;
      if (nsa || cube) {
        base::Array<Id, 13> address{};
        const u32 vaddr = w1 & 0xFF;
        if (nsa) {
          address[0] = t.Vg(vaddr);
          for (u32 d = 0; d < nsa; d++)
            for (u32 c = 0; c < 4; c++)
              address[1 + d * 4 + c] =
                  t.Vg((inst.raw[2 + d] >> (c * 8)) & 0xFF);
        } else {
          for (u32 c = 0; c < address.size(); c++)
            address[c] = t.Vg(vaddr + c);
        }
        if (cube)
          for (u32 c = 0; c < 2; c++)
            address[c] = t.m.Bitcast(
                t.t_u, t.FSub(t.m.Bitcast(t.t_f, address[c]), t.F32(1.0f)));
        gpu::gcn::EmitMimg(t, lowered, sc, address.data());
        break;
      }
      gpu::gcn::EmitMimg(t, lowered, sc);
      break;
    }
    case Enc::kFlat:
      RdnaEmitFlat(t, inst, sc);
      break;
    case Enc::kDs:
      if (sc.is_cs || inst.opcode == 0x35) {
        gpu::gcn::EmitDs(t, inst, sc);
      } else if ((sc.lds_var || sc.lds_wave_base) &&
                 gpu::gcn::DsGraphicsSupported(inst.opcode)) {
        if (!sc.lds_wave_base && !sc.ds_own_lane.count(inst.pc))
          gpu::gcn::NoteApproximated("ds.private", inst.opcode);
        gpu::gcn::EmitDs(t, inst, sc);
      } else {
        gpu::gcn::WarnUnsupported("ds.graphics", inst.opcode, w, w1);
      }
      break;
    default:
      gpu::gcn::WarnUnsupported("rdna", inst.opcode, w, w1);
      break;
  }
}

int BranchKind(const Inst& inst) {
  if (inst.enc != Enc::kSopp)
    return 0;
  switch (inst.opcode) {
    case 0x01:
    case 0x1B:
    case 0x1E:
    case 0x1F:
      return 8;
    case 0x02:
      return 1;
    case 0x04:
      return 2;
    case 0x05:
      return 3;
    case 0x06:
      return 4;
    case 0x07:
      return 5;
    case 0x08:
      return 6;
    case 0x09:
      return 7;
    default:
      return 0;
  }
}

bool IsCall(const Inst& inst) {
  return inst.enc == Enc::kSopk && inst.opcode == 0x16;
}
bool IsReturn(const Inst& inst) {
  return inst.enc == Enc::kSop1 && inst.opcode == 0x20;
}

bool HasControlFlow(const Program& program) {
  return base::AnyOf(program.begin(), program.end(), [](const Inst& inst) {
    const int k = BranchKind(inst);
    return (k >= 1 && k <= 7) || IsCall(inst) || IsReturn(inst);
  });
}

Id BranchTaken(Translator& t, int kind) {
  const auto mask = [&](u32 sg) {
    return t.full_wave_masks ? t.Or(t.Sg(sg), t.Sg(sg + 1)) : t.SgMask(sg);
  };
  switch (kind) {
    case 2:
      return t.IsZero(t.Scc());
    case 3:
      return t.IsNonZero(t.Scc());
    case 4:
      return t.IsZero(mask(106));
    case 5:
      return t.IsNonZero(mask(106));
    case 6:
      return t.IsZero(mask(126));
    case 7:
      return t.IsNonZero(mask(126));
    default:
      return t.m.ConstBool(false);
  }
}

base::Vector<u32> BlockStarts(const Program& program, u32 max_pc) {
  base::Vector<u32> leaders{0};
  for (const Inst& inst : program) {
    const int k = BranchKind(inst);
    if (k == 0 && !IsCall(inst) && !IsReturn(inst))
      continue;
    leaders.push_back(inst.pc + inst.size);
    if (IsCall(inst) || (k >= 1 && k <= 7)) {
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
      leaders.push_back(static_cast<u32>(static_cast<i32>(inst.pc) +
                                         static_cast<i32>(inst.size) + simm));
    }
  }
  base::Sort(leaders.begin(), leaders.end());
  leaders.erase(base::Unique(leaders.begin(), leaders.end()), leaders.end());
  base::Vector<u32> starts;
  for (u32 l : leaders)
    if (l < max_pc)
      starts.push_back(l);
  return starts;
}

bool ForceCfg() {
  return kGpuSpirvCfg;
}

void EmitBody(Translator& t, const Program& program, StageContext& sc) {
  t.SeedExec();
  t.predicate_vector = true;
  t.uniform_here = sc.is_mesh;
  if (ForceCfg() || HasControlFlow(program)) {
    EmitCfg(t, program, sc);
    return;
  }
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSopp && (inst.opcode == 0x01 || inst.opcode == 0x1B ||
                                   inst.opcode == 0x1E || inst.opcode == 0x1F))
      break;
    RdnaEmitInst(t, inst, sc);
  }
}

struct FetchAttr {
  u32 semantic, num_comps, dest_vgpr, table_sgpr, dword_off;
  u32 pc = ~0u;
  u32 inst_format = 0;
  u32 inst_offset = 0;
};

base::Vector<FetchAttr> ParseFetchInsts(const Program& insts) {
  base::Vector<FetchAttr> out;
  u32 sem = 0;
  const auto chained_loads = MapTableChainedLoads(insts);
  for (const Inst& in : insts) {
    if (in.enc == Enc::kSop1 && in.opcode == 0x20)
      break;
    if (in.enc == Enc::kSopp && in.opcode == 1)
      break;
    const bool typed = in.enc == Enc::kMtbuf;
    if ((in.enc != Enc::kMubuf && !typed) || in.opcode > 0x03)
      continue;
    const u32 vdata = (in.raw[1] >> 8) & 0xFF;
    const u32 srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
    const u32 nc = (in.opcode & 3) + 1;
    const bool idxen = (in.raw[0] >> 13) & 1, offen = (in.raw[0] >> 12) & 1;
    const u32 inst_offset = in.raw[0] & 0xFFF;
    const u32 vaddr = in.raw[1] & 0xFF, soffset = (in.raw[1] >> 24) & 0xFF;
    const auto chain = chained_loads.find(in.pc);
    const bool chained = chain != chained_loads.end();
    const u32 table_sgpr = chained ? chain->second.first : srsrc;
    const u32 dword_off = chained ? chain->second.second * 4 : 0;
    const bool vtx = BufLoadIsVertexFetch(in, chained);
    if (ShDbg())
      BASE_LOGI("gcnspv",
                "buf_load nc={} vdst=v{} srsrc=s{} idxen={} offen={} "
                "vaddr=v{} soffset=s{} ioff={} -> {} (table_sgpr=s{} doff={})",
                nc, vdata, srsrc, idxen, offen, vaddr, soffset, inst_offset,
                vtx ? "vertex-attr" : "storage-buffer", table_sgpr, dword_off);
    if (vtx) {
      out.push_back({sem, nc, vdata, table_sgpr, dword_off, in.pc,
                     typed ? ((in.raw[0] >> 19) & 0x7F) : 0u, inst_offset});
      sem++;
    }
  }
  return out;
}

base::Vector<FetchAttr> ParseFetch(u64 fetch_addr) {
  constexpr u64 kMaxFetchBytes = 256 * sizeof(u32);
  if (!gpu::gcn::InGuest(fetch_addr) ||
      !gpu::IsReadableRangeCached(fetch_addr, kMaxFetchBytes))
    return {};
  const auto* code = reinterpret_cast<const u32*>(fetch_addr);
  base::Vector<FetchAttr> attrs = ParseFetchInsts(Decode(code, 256));
  for (FetchAttr& attr : attrs)
    attr.pc = ~0u;
  return attrs;
}

}  // namespace

void PlanCrossLane(const Program& program,
                   Translator& t,
                   StageContext& sc,
                   base::Vector<Id>& iface) {
  const bool addtid =
      base::AnyOf(program.begin(), program.end(), [](const Inst& i) {
        return i.enc == Enc::kDs && (i.opcode == 176 || i.opcode == 177);
      });
  if (!sc.subgroup_local_id &&
      (addtid || RdnaUsesDpp(program) ||
       gpu::gcn::UsesDsSwizzle(program, nullptr) || RdnaUsesLaneOps(program)))
    gpu::gcn::EnableDsSwizzle(t, sc, iface);
  if (sc.subgroup_local_id && !t.lane_id)
    t.lane_id = t.m.Load(t.t_u, sc.subgroup_local_id);
}

u64 RdnaTexDimMask(const Program& program,
                   const gpu::gcn::MimgBindingPlan& plan,
                   u32 dimensions) {
  u64 mask = 0;
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kMimg ||
        !(dimensions & (1u << ((inst.raw[0] >> 3) & 0x7))))
      continue;
    const auto it = plan.binding_by_pc.find(inst.pc);
    if (it != plan.binding_by_pc.end() &&
        it->second < StageContext::kMaxPsSamplers)
      mask |= 1ull << it->second;
  }
  return mask;
}

void PlanGraphicsLds(const Program& program, Translator& t, StageContext& sc) {
  sc.ds_own_lane = gpu::gcn::PlanDsOwnLane(program, nullptr);
  const u32 lds_dwords = gpu::gcn::GraphicsLdsDwords(program, nullptr);
  if (!lds_dwords)
    return;
  sc.lds_dwords = lds_dwords;
  if (sc.is_vs_shared_lds_capable) {
    sc.lds_storage = spv::StorageClass::StorageBuffer;
    t.EnsureLdsBuffer();
    t.RequireSubgroup(spv::Capability::GroupNonUniformBallot);
    const Id first = t.m.Emit(
        spv::Op::OpGroupNonUniformBroadcastFirst, t.t_u,
        {t.U32(static_cast<u32>(spv::Scope::Subgroup)), sc.vertex_index_value});
    u32 shift = 0;
    while ((1u << shift) < gpu::gcn::HostSubgroupSize())
      shift++;
    const Id wave =
        t.And(t.Shr(first, t.U32(shift)), t.U32(gpu::gcn::kLdsWaves - 1));
    sc.lds_wave_base = t.Mul(wave, t.U32(lds_dwords));
    if (kLdsMark) {
      const Id p = t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u);
      t.m.Store(
          t.m.AccessChain(p, t.lds_buf_var,
                          {t.U32(0), t.Add(sc.lds_wave_base, t.WaveLane())}),
          t.U32(0xDEADBEEF));
    }
    if (ShDbg())
      BASE_LOGI("gcnspv", "vs shared lds {} dwords ({} per wave)", lds_dwords,
                lds_dwords);
    return;
  }
  const Id lds_arr = t.m.TypeArray(t.t_u, lds_dwords);
  sc.lds_storage = spv::StorageClass::Private;
  sc.lds_var =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, lds_arr),
                   spv::StorageClass::Private, t.m.ConstNull(lds_arr));
  t.m.Name(sc.lds_var, "lds");
}

u64 FetchPlanHash(u64 fetch_addr) {
  const base::Vector<FetchAttr> attrs = ParseFetch(fetch_addr);
  if (attrs.empty())
    return 0;
  u64 h = 1469598103934665603ull;
  for (const FetchAttr& a : attrs)
    for (u32 field : {a.semantic, a.num_comps, a.dest_vgpr, a.table_sgpr,
                      a.dword_off, a.inst_format, a.inst_offset})
      h = (h ^ field) * 1099511628211ull;
  return h;
}

namespace {

thread_local u64 g_vs_addr = 0;

Id DeclareUserData(Translator& t) {
  const Id words = t.m.TypeArray(t.t_u, 16);
  t.m.Decorate(words, spv::Decoration::ArrayStride, {4});
  const bool code_base = gpu::gcn::PushCodeBase();
  const Id block = code_base ? t.m.TypeStruct({words, t.t_u, t.t_u})
                             : t.m.TypeStruct({words});
  t.m.Decorate(block, spv::Decoration::Block);
  t.m.MemberDecorate(block, 0, spv::Decoration::Offset,
                     {t.user_data_slot * 64});
  if (code_base) {
    t.m.MemberDecorate(block, 1, spv::Decoration::Offset,
                       {128 + t.user_data_slot * 8});
    t.m.MemberDecorate(block, 2, spv::Decoration::Offset,
                       {132 + t.user_data_slot * 8});
  }
  const Id v =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::PushConstant, block),
                   spv::StorageClass::PushConstant);
  if (code_base) {
    t.pc_base_var = v;
    t.pc_base_member = 1;
  }
  return v;
}

void SeedUserData(Translator& t, Id user_data, u32 sgpr_base, u32 count) {
  const Id p_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (u32 i = 0; i < base::Min(count, 32u); i++)
    t.SetSg(sgpr_base + i,
            t.indirect_cbufs
                ? t.DrawDataDword(gpu::gcn::kIndirectCbufBindings +
                                  t.user_data_slot * 32 + i)
                : t.m.Load(t.t_u, t.m.AccessChain(p_u, user_data,
                                                  {t.U32(0), t.U32(i)})));
  if (sgpr_base == 8 && count >= 2) {
    t.SetSg(0, t.Sg(8));
    t.SetSg(1, t.Sg(9));
  }
}

bool TranslateVs(const Program& program,
                 const u32* vs_user_data,
                 const base::HashSet<u32>& flat_attrs,
                 Recompiled& r,
                 Translator& t,
                 bool gl_clip_space,
                 u32 user_sgprs,
                 u32 tex_binding_base) {
  if (ShDbg())
    DumpProgram(program, "vs");
  const u64 fetch =
      user_sgprs >= 2
          ? (static_cast<u64>(vs_user_data[1] & 0xFFFF) << 32) | vs_user_data[0]
          : 0;
  base::Vector<FetchAttr> attrs = ParseFetch(fetch);
  r.direct_vs = r.direct && attrs.empty();
  base::HashSet<u32> direct_vertex_fetch;
  if (r.direct_vs)
    for (const FetchAttr& a : ParseFetchInsts(program))
      direct_vertex_fetch.insert(a.pc);
  if (attrs.empty() && !r.direct_vs)
    attrs = ParseFetchInsts(program);
  if (ShDbg())
    for (const FetchAttr& a : attrs)
      BASE_LOGI("gcnspv", "vs attr loc={} nc={} vgpr={} pc={:#x}", a.semantic,
                a.num_comps, a.dest_vgpr, a.pc);
  t.wave_masks = kWaveMasks && gpu::gcn::GraphicsLdsDwords(program, nullptr);
  t.movrel_m0 = PlanMovrelIndices(program);
  t.InitTypes(&program);

  base::Vector<Id> iface;
  const Id pos_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                   spv::StorageClass::Output);
  t.m.Decorate(pos_out, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::Position)});
  iface.push_back(pos_out);
  const Id point_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_f),
                   spv::StorageClass::Output);
  t.m.Decorate(point_out, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::PointSize)});
  iface.push_back(point_out);

  const Id user_data = DeclareUserData(t);
  StageContext direct_decl;
  if (r.direct_vs)
    gpu::gcn::DeclareDirectMemory(t, direct_decl);
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedUserData(t, user_data, 8, user_sgprs);
  t.m.Store(point_out, t.F32(1.f));
  if (kGpuPosunset)
    t.m.Store(pos_out,
              t.m.CompositeConstruct(t.t_v4, {t.F32(0.f), t.F32(0.f),
                                              t.F32(0.f), t.F32(-1234.f)}));

  t.SetSg(3, t.U32(kNggWaveInfo));

  base::HashMap<u32, StageContext::VfetchSeed> vfetch_seed;
  Id vertex_index = 0;
  {
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    const Id instance_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(vertex_index, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::VertexIndex)});
    t.m.Decorate(instance_index, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::InstanceIndex)});
    iface.push_back(vertex_index);
    iface.push_back(instance_index);
    const Id vertex = t.m.Load(t.t_u, vertex_index);
    t.SetVg(0, vertex);
    const Id instance = t.m.Load(t.t_u, instance_index);
    t.SetVg(1, instance);
    t.SetVg(3, instance);
    t.SetVg(5, vertex);
    t.SetVg(8, instance);
  }

  Id first_attr_var = 0;
  u32 first_attr_comps = 0;
  for (const FetchAttr& a : attrs) {
    const Id comp_ty = a.num_comps == 1   ? t.t_f
                       : a.num_comps == 2 ? t.t_v2
                       : a.num_comps == 3 ? t.t_v3
                                          : t.t_v4;
    const Id in_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, comp_ty),
                     spv::StorageClass::Input);
    t.m.Decorate(in_var, spv::Decoration::Location, {a.semantic});
    t.m.Name(in_var, "v_attr" + base::ToString(a.semantic));
    iface.push_back(in_var);
    const Id val = t.m.Load(comp_ty, in_var);
    if (a.pc == ~0u)
      for (u32 c = 0; c < a.num_comps; c++) {
        const Id comp =
            a.num_comps == 1 ? val : t.m.CompositeExtract(t.t_f, val, c);
        t.SetVgF(a.dest_vgpr + c, comp);
      }
    if (a.pc != ~0u)
      vfetch_seed[a.pc] = {in_var, a.dest_vgpr, a.num_comps};
    if (!first_attr_var) {
      first_attr_var = in_var;
      first_attr_comps = a.num_comps;
    }
    r.attrs.push_back({a.semantic, a.num_comps, a.table_sgpr, a.dword_off,
                       false, a.inst_format, a.pc, a.inst_offset});
  }

  StageContext sc;
  sc.r = &r;
  sc.iface = &iface;
  sc.main_fn = main_fn;
  sc.pos_out = pos_out;
  sc.flat_attrs = &flat_attrs;
  sc.vfetch_seed = base::move(vfetch_seed);
  sc.skip_launch_movs = LaunchExecMovPcs(program);
  if (gpu::gcn::GraphicsLdsDwords(program, nullptr)) {
    sc.is_vs_shared_lds_capable = true;
    sc.vertex_index_value = t.m.Load(t.t_u, vertex_index);
    r.shared_lds = true;
    t.SetVg(0, t.U32(0));
  }
  if (r.direct_vs) {
    PlanDirect(program, sc);
    sc.cs_guest_translate = direct_decl.cs_guest_translate;
    sc.direct_vertex_fetch = base::move(direct_vertex_fetch);
    sc.direct_vertex_index = t.m.Load(t.t_u, vertex_index);
  } else {
    if (!RdnaPlanCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind, sc.smem_cbuf_by_pc,
                       r.indirect_cbufs ? gpu::gcn::kIndirectCbufBindings
                                        : kMaxCbufBindings))
      return false;
    base::HashSet<u32> lifted;
    for (const FetchAttr& a : attrs)
      if (a.pc != ~0u)
        lifted.insert(a.pc);
    RdnaPlanGfxBuffers(program, 0, &lifted, r.vs_bufs, sc.gfx_buf_bind);
    NoteCbufWindows(r.vs_cbufs, sc);
  }
  const gpu::gcn::MimgBindingPlan vs_mimg_plan = RdnaPlanMimg(program);
  if (!vs_mimg_plan.binding_srsrc.empty()) {
    if (vs_mimg_plan.binding_srsrc.size() + tex_binding_base >
        StageContext::kMaxPsSamplers) {
      gpu::gcn::WarnUnsupported(
          "mimg.vs-binding-count",
          static_cast<u32>(vs_mimg_plan.binding_srsrc.size()));
      return false;
    }
    sc.mimg_plan = &vs_mimg_plan;
    sc.tex_binding_base = tex_binding_base;
    sc.tex_3d_mask = RdnaTexDimMask(program, vs_mimg_plan, 1u << 2);
    sc.tex_1d_mask =
        RdnaTexDimMask(program, vs_mimg_plan, (1u << 0) | (1u << 4));
    for (u32 i = 0; i < vs_mimg_plan.binding_srsrc.size(); i++)
      r.vs_texs.push_back({i + tex_binding_base, vs_mimg_plan.binding_srsrc[i],
                           vs_mimg_plan.binding_storage[i],
                           ((sc.tex_3d_mask >> i) & 1u) != 0});
  }
  PlanCrossLane(program, t, sc, iface);
  PlanGraphicsLds(program, t, sc);
  if (ShDbg())
    BASE_LOGI("gcnspv", "vs planned {} cbufs", r.vs_cbufs.size());
  static const u64 kDbgPosVs =
      kDbgPos ? std::strtoull(kDbgPos, nullptr, 0) : ~0ull;
  static const u32 kDbgPosOff = [] {
    const char* c = kDbgPos ? std::strchr(kDbgPos, ':') : nullptr;
    return c ? (u32)std::strtoul(c + 1, nullptr, 0) : 0u;
  }();
  static const int kDbgPosWorld = [] {
    const char* c = kDbgPos ? std::strchr(kDbgPos, ':') : nullptr;
    const char* c2 = c ? std::strchr(c + 1, ':') : nullptr;
    return c2 ? std::atoi(c2 + 1) : -1;
  }();
  if (kDbgPosVs == 0 || kDbgPosVs == g_vs_addr) {
    sc.dbg_pos_in = first_attr_var;
    sc.dbg_pos_comps = first_attr_comps;
    for (const auto& cb : r.vs_cbufs)
      if (cb.num_dwords >= 16) {
        sc.dbg_pos_cbuf = static_cast<int>(cb.binding);
        break;
      }
  }

  EmitBody(t, program, sc);
  r.num_params = sc.max_param;

  if (kLdsMark && t.lds_buf_var) {
    const Id p = t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u);
    const Id pos = t.m.Load(t.t_v4, pos_out);
    for (u32 i = 0; i < 4; i++)
      t.m.Store(t.m.AccessChain(
                    p, t.lds_buf_var,
                    {t.U32(0), t.U32(gpu::gcn::kLdsTraceBase + 0x3F0 + i)}),
                t.m.Bitcast(t.t_u, t.m.CompositeExtract(t.t_f, pos, i)));
  }

  if (kGpuPosunset) {
    const Id p_out_f = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id w = t.m.Load(t.t_f, t.m.AccessChain(p_out_f, pos_out, {t.U32(3)}));
    const Id unset =
        t.m.Emit(spv::Op::OpFOrdEqual, t.t_bool, {w, t.F32(-1234.f)});
    const Id vidx = sc.vertex_index_value ? sc.vertex_index_value : t.U32(0);
    const Id fx =
        t.SelectF(t.IsNonZero(t.And(vidx, t.U32(1))), t.F32(1.f), t.F32(-1.f));
    const Id fy =
        t.SelectF(t.IsNonZero(t.And(vidx, t.U32(2))), t.F32(1.f), t.F32(-1.f));
    const Id have = t.m.Load(t.t_v4, pos_out);
    const Id quad[4] = {fx, fy, t.F32(0.f), t.F32(1.f)};
    base::Vector<Id> comps;
    for (u32 i = 0; i < 4; i++)
      comps.push_back(
          t.SelectF(unset, quad[i], t.m.CompositeExtract(t.t_f, have, i)));
    t.m.Store(pos_out, t.m.CompositeConstruct(t.t_v4, comps));
  }

  if (kGpuForcequad) {
    Id vidx_var = vertex_index;
    if (!vidx_var) {
      vidx_var = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                              spv::StorageClass::Input);
      t.m.Decorate(vidx_var, spv::Decoration::BuiltIn,
                   {static_cast<u32>(spv::BuiltIn::VertexIndex)});
      iface.push_back(vidx_var);
    }
    const Id vidx = t.m.Load(t.t_u, vidx_var);
    const Id fx =
        t.SelectF(t.IsNonZero(t.And(vidx, t.U32(1))), t.F32(1.f), t.F32(-1.f));
    const Id fy =
        t.SelectF(t.IsNonZero(t.And(vidx, t.U32(2))), t.F32(1.f), t.F32(-1.f));
    t.m.Store(pos_out,
              t.m.CompositeConstruct(t.t_v4, {fx, fy, t.F32(0.f), t.F32(1.f)}));
  }

  if (kRawPos != 0.f && first_attr_var && first_attr_comps >= 2) {
    const Id comp_ty = first_attr_comps == 2   ? t.t_v2
                       : first_attr_comps == 3 ? t.t_v3
                                               : t.t_v4;
    const Id val = t.m.Load(comp_ty, first_attr_var);
    t.m.Store(pos_out,
              t.m.CompositeConstruct(
                  t.t_v4,
                  {t.FMul(t.m.CompositeExtract(t.t_f, val, 0), t.F32(kRawPos)),
                   t.FMul(t.m.CompositeExtract(t.t_f, val, 1), t.F32(kRawPos)),
                   t.F32(0.f), t.F32(1.f)}));
  }

  if (kGpuPosprobe) {
    const Id p_out_f0 = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id px =
        t.m.Load(t.t_f, t.m.AccessChain(p_out_f0, pos_out, {t.U32(0)}));
    const Id py =
        t.m.Load(t.t_f, t.m.AccessChain(p_out_f0, pos_out, {t.U32(1)}));
    const Id pw =
        t.m.Load(t.t_f, t.m.AccessChain(p_out_f0, pos_out, {t.U32(3)}));
    auto squash = [&](Id v) {
      const Id q = t.m.Emit(spv::Op::OpFDiv, t.t_f, {v, pw});
      const Id a = t.m.ExtInst(t.t_f, GLSLstd450FAbs, {q});
      return t.m.Emit(spv::Op::OpFDiv, t.t_f,
                      {q, t.m.Emit(spv::Op::OpFAdd, t.t_f, {t.F32(1.f), a})});
    };
    t.m.Store(pos_out,
              t.m.CompositeConstruct(
                  t.t_v4, {squash(px), squash(py), t.F32(0.f), t.F32(1.f)}));
  }

  if (gl_clip_space) {
    const Id p_out_f = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id z_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(2)});
    const Id w_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(3)});
    const Id z = t.m.Load(t.t_f, z_ptr), wv = t.m.Load(t.t_f, w_ptr);
    t.m.Store(z_ptr, t.FMul(t.FAdd(z, wv), t.F32(0.5f)));
  }

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Vertex, main_fn, "main", iface);
  return true;
}

bool TranslateMesh(Program es_program,
                   const Program& gs_program,
                   const u32* es_code,
                   const u32* user_data,
                   u32 user_sgprs,
                   const NggConfig& cfg,
                   const base::HashSet<u32>& flat_attrs,
                   u32 tex_binding_base,
                   bool gl_clip_space,
                   Recompiled& r,
                   Translator& t) {
  if (es_program.empty() || gs_program.empty() || !cfg.threads ||
      !cfg.input_primitives || !cfg.max_vertices || !cfg.max_primitives ||
      cfg.max_vertices > cfg.threads || cfg.max_primitives > cfg.threads ||
      !cfg.lds_dwords || (cfg.threads % 64) != 0)
    return false;
  if (cfg.separate_es) {
    const auto transfer_at = base::FindIf(
        es_program.begin(), es_program.end(), [](const Inst& inst) {
          return IsReturn(inst) && (inst.raw[0] & 0xff) == 6;
        });
    if (transfer_at == es_program.end())
      return false;
    es_program.erase(transfer_at + 1, es_program.end());
    Inst& transfer = es_program.back();
    transfer.enc = Enc::kSopp;
    transfer.opcode = 1;
    transfer.raw[0] = 0xbf810000;
  }
  t.rdna_sources = true;
  t.lane_masks = true;
  t.full_wave_masks = true;
  t.indirect_cbufs = r.indirect_cbufs = true;
  t.InitTypes(&es_program, &gs_program);
  t.xchg_lanes = cfg.threads;
  t.xchg_var =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Workgroup,
                                   t.m.TypeArray(t.t_u, 2 * cfg.threads + 1)),
                   spv::StorageClass::Workgroup);
  t.m.Capability(spv::Capability::MeshShadingEXT);
  t.m.Extension("SPV_EXT_mesh_shader");
  base::Vector<Id> iface;
  const Id uv3 = t.m.TypeVec(t.t_u, 3);
  const Id local =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                   spv::StorageClass::Input);
  t.m.Decorate(local, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::LocalInvocationIndex)});
  const Id group = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, uv3),
                                spv::StorageClass::Input);
  t.m.Decorate(group, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::WorkgroupId)});
  const Id words = t.m.TypeArray(t.t_u, 32);
  t.m.Decorate(words, spv::Decoration::ArrayStride, {4});
  const Id draw_type = t.m.TypeVec(t.t_u, 4);
  const Id push_type = t.m.TypeStruct({words, draw_type});
  t.m.Decorate(push_type, spv::Decoration::Block);
  t.m.MemberDecorate(push_type, 0, spv::Decoration::Offset, {0});
  t.m.MemberDecorate(push_type, 1, spv::Decoration::Offset, {144});
  const Id push =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::PushConstant, push_type),
                   spv::StorageClass::PushConstant);
  const Id arguments = t.m.Variable(
      t.m.TypePointer(spv::StorageClass::StorageBuffer, t.EnsureRawBlockType()),
      spv::StorageClass::StorageBuffer);
  t.m.Decorate(arguments, spv::Decoration::DescriptorSet, {3});
  t.m.Decorate(arguments, spv::Decoration::Binding, {0});
  t.m.Decorate(arguments, spv::Decoration::NonWritable);
  const Id lds =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Workgroup,
                                   t.m.TypeArray(t.t_u, cfg.lds_dwords)),
                   spv::StorageClass::Workgroup);
  const Id counts = t.m.Variable(
      t.m.TypePointer(spv::StorageClass::Workgroup, t.m.TypeArray(t.t_u, 2)),
      spv::StorageClass::Workgroup);
  const Id count_ptr = t.m.TypePointer(spv::StorageClass::Workgroup, t.t_u);
  const auto uses_pc = [](const Program& p) {
    return base::AnyOf(p.begin(), p.end(), [](const Inst& inst) {
      return inst.enc == Enc::kSop1 && inst.opcode == 0x1f;
    });
  };
  r.direct_vs = r.direct && !uses_pc(es_program) && !uses_pc(gs_program);
  StageContext direct_decl;
  if (r.direct_vs)
    gpu::gcn::DeclareDirectMemory(t, direct_decl);
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  const Id index = t.m.Load(t.t_u, local);
  t.xchg_index = index;
  const Id groups = t.m.Load(uv3, group);
  const Id base = t.Mul(t.m.CompositeExtract(t.t_u, groups, 0),
                        t.U32(cfg.input_primitives));
  const auto draw_param = [&](u32 i) {
    return t.m.Load(
        t.t_u,
        t.m.AccessChain(t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u),
                        push, {t.U32(1), t.U32(i)}));
  };
  const Id indirect = t.IsNonZero(draw_param(3));
  const auto argument = [&](u32 i) {
    return t.m.Load(
        t.t_u, t.m.AccessChain(
                   t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u),
                   arguments, {t.U32(0), t.Add(draw_param(2), t.U32(i))}));
  };
  const Id input_count = t.SelectB(indirect, argument(0), draw_param(0));
  const Id first_vertex = t.SelectB(indirect, argument(2), t.U32(0));
  const Id first_instance = t.SelectB(indirect, argument(3), t.U32(0));
  const Id group_count = t.UMin(
      t.SelectB(t.Ult(base, input_count), t.Sub(input_count, base), t.U32(0)),
      t.U32(cfg.input_primitives));
  const Id wave_base = t.And(index, t.U32(~63u));
  const Id wave_count =
      t.UMin(t.SelectB(t.Ult(wave_base, group_count),
                       t.Sub(group_count, wave_base), t.U32(0)),
             t.U32(64));
  const Id wave_info = t.Or(t.Or(wave_count, t.Shl(wave_count, t.U32(8))),
                            t.Shl(t.Shr(index, t.U32(6)), t.U32(24)));
  t.lane_id = t.mask_lane_id = t.And(index, t.U32(63));
  t.wave_base = wave_base;
  const Id is_first = t.Eq(index, t.U32(0));
  const Id init = t.m.NewBlock(), initialized = t.m.NewBlock();
  t.m.SelectionMerge(initialized);
  t.m.BranchConditional(is_first, init, initialized);
  t.m.OpenBlock(init);
  for (u32 i = 0; i < 2; ++i)
    t.m.Store(t.m.AccessChain(count_ptr, counts, {t.U32(i)}), t.U32(0));
  t.m.Branch(initialized);
  t.m.OpenBlock(initialized);
  t.Barrier();

  StageContext sc;
  Id subgroup_local_id = 0;
  for (u32 half = cfg.separate_es ? 0 : 1; half < 2; ++half) {
    sc = {};
    sc.subgroup_local_id = subgroup_local_id;
    sc.is_mesh = true;
    sc.r = &r;
    sc.iface = &iface;
    sc.main_fn = main_fn;
    sc.flat_attrs = &flat_attrs;
    sc.mesh_local_index = index;
    sc.mesh_counts = counts;
    sc.mesh_primitive =
        t.m.Variable(t.p_priv_u, spv::StorageClass::Private, t.U32(0x80000000));
    sc.pos_out =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, t.t_v4),
                     spv::StorageClass::Private, t.m.ConstNull(t.t_v4));
    sc.lds_storage = spv::StorageClass::Workgroup;
    sc.lds_var = lds;
    sc.lds_dwords = cfg.lds_dwords;
    const Program& program = half ? gs_program : es_program;
    if (r.direct_vs) {
      PlanDirect(program, sc);
      sc.cs_guest_translate = direct_decl.cs_guest_translate;
    } else {
      base::Vector<ShaderCbuf> cbufs;
      const u32 cb_base = static_cast<u32>(r.vs_cbufs.size());
      if (!RdnaPlanCbufs(program, cb_base, cbufs, sc.cbuf_bind,
                         sc.smem_cbuf_by_pc, gpu::gcn::kIndirectCbufBindings))
        return false;
      NoteCbufWindows(cbufs, sc);
      for (auto& cb : cbufs) {
        cb.from_gs = half != 0;
        r.vs_cbufs.push_back(cb);
      }
      base::Vector<gpu::gcn::ShaderBuffer> buffers;
      RdnaPlanGfxBuffers(program, static_cast<u32>(r.vs_bufs.size()), nullptr,
                         buffers, sc.gfx_buf_bind);
      for (auto& buf : buffers) {
        buf.from_gs = half != 0;
        r.vs_bufs.push_back(buf);
      }
    }
    const auto image_plan = RdnaPlanMimg(program);
    if (!half && !image_plan.binding_srsrc.empty())
      return false;
    if (image_plan.binding_srsrc.size() + tex_binding_base >
        StageContext::kMaxPsSamplers)
      return false;
    sc.mimg_plan = &image_plan;
    sc.tex_binding_base = tex_binding_base;
    sc.tex_3d_mask = RdnaTexDimMask(program, image_plan, 1u << 2);
    sc.tex_1d_mask = RdnaTexDimMask(program, image_plan, (1u << 0) | (1u << 4));
    for (u32 i = 0; i < image_plan.binding_srsrc.size(); ++i)
      r.vs_texs.push_back({i + tex_binding_base, image_plan.binding_srsrc[i],
                           image_plan.binding_storage[i],
                           ((sc.tex_3d_mask >> i) & 1u) != 0});
    g_stage_bufs = ResolveBuffers(half ? cfg.gs_code : es_code, user_data,
                                  user_sgprs, 8, 4096, cfg.gs_user_data_addr);
    t.predicate_vector = false;
    SeedUserData(t, push, 8, user_sgprs);
    const Id root_lo = t.DrawDataDword(gpu::gcn::kIndirectGsUserDataAddr);
    const Id root_hi = t.DrawDataDword(gpu::gcn::kIndirectGsUserDataAddr + 1);
    const Id has_root = t.IsNonZero(t.Or(root_lo, root_hi));
    t.SetSg(0, t.SelectB(has_root, root_lo, t.Sg(0)));
    t.SetSg(1, t.SelectB(has_root, root_hi, t.Sg(1)));
    t.SetSg(3, wave_info);
    t.SetVg(0, t.Mul(index, t.U32(4)));
    t.SetVg(5, t.Add(t.Add(base, index), first_vertex));
    t.SetVg(8, t.Add(t.m.CompositeExtract(t.t_u, groups, 1), first_instance));
    PlanCrossLane(program, t, sc, iface);
    subgroup_local_id = sc.subgroup_local_id;
    EmitBody(t, program, sc);
    if (sc.cs_unsupported)
      return false;
    t.Barrier();
  }

  const Id vertices =
      t.UMin(t.m.Load(t.t_u, t.m.AccessChain(count_ptr, counts, {t.U32(0)})),
             t.U32(cfg.max_vertices));
  const Id primitives =
      t.UMin(t.m.Load(t.t_u, t.m.AccessChain(count_ptr, counts, {t.U32(1)})),
             t.U32(cfg.max_primitives));
  t.m.EmitVoid(spv::Op::OpSetMeshOutputsEXT, {vertices, primitives});
  const auto output_array = [&](Id type, u32 size, spv::BuiltIn builtin) {
    const Id var = t.m.Variable(
        t.m.TypePointer(spv::StorageClass::Output, t.m.TypeArray(type, size)),
        spv::StorageClass::Output);
    t.m.Decorate(var, spv::Decoration::BuiltIn, {static_cast<u32>(builtin)});
    return var;
  };
  const Id positions =
      output_array(t.t_v4, cfg.max_vertices, spv::BuiltIn::Position);
  const Id triangles = output_array(uv3, cfg.max_primitives,
                                    spv::BuiltIn::PrimitiveTriangleIndicesEXT);
  const Id cull = output_array(t.t_bool, cfg.max_primitives,
                               spv::BuiltIn::CullPrimitiveEXT);
  t.m.Decorate(cull, spv::Decoration::PerPrimitiveEXT);
  const auto write_when = [&](Id condition, auto emit) {
    const Id body = t.m.NewBlock(), merge = t.m.NewBlock();
    t.m.SelectionMerge(merge);
    t.m.BranchConditional(condition, body, merge);
    t.m.OpenBlock(body);
    emit();
    t.m.Branch(merge);
    t.m.OpenBlock(merge);
  };
  write_when(t.Ult(index, vertices), [&] {
    Id position = t.m.Load(t.t_v4, sc.pos_out);
    if (gl_clip_space) {
      Id c[4];
      for (u32 i = 0; i < 4; ++i)
        c[i] = t.m.CompositeExtract(t.t_f, position, i);
      c[2] = t.FMul(t.FAdd(c[2], c[3]), t.F32(.5f));
      position = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
    }
    const Id p = t.m.TypePointer(spv::StorageClass::Output, t.t_v4);
    t.m.Store(t.m.AccessChain(p, positions, {index}), position);
    for (const auto& [slot, value] : sc.param_outs) {
      const Id out =
          t.m.Variable(t.m.TypePointer(spv::StorageClass::Output,
                                       t.m.TypeArray(t.t_v4, cfg.max_vertices)),
                       spv::StorageClass::Output);
      t.m.Decorate(out, spv::Decoration::Location, {slot});
      if (flat_attrs.count(slot))
        t.m.Decorate(out, spv::Decoration::Flat);
      t.m.Store(t.m.AccessChain(p, out, {index}), t.m.Load(t.t_v4, value));
    }
  });
  write_when(t.Ult(index, primitives), [&] {
    const Id packed = t.m.Load(t.t_u, sc.mesh_primitive);
    const Id p = t.m.TypePointer(spv::StorageClass::Output, uv3);
    const Id indices = t.m.CompositeConstruct(
        uv3, {t.And(packed, t.U32(0x3ff)),
              t.And(t.Shr(packed, t.U32(10)), t.U32(0x3ff)),
              t.And(t.Shr(packed, t.U32(20)), t.U32(0x3ff))});
    t.m.Store(t.m.AccessChain(p, triangles, {index}), indices);
    t.m.Store(
        t.m.AccessChain(t.m.TypePointer(spv::StorageClass::Output, t.t_bool),
                        cull, {index}),
        t.IsNonZero(t.And(packed, t.U32(0x80000000))));
  });
  r.num_params = sc.max_param;
  r.mesh_input_primitives = cfg.input_primitives;
  r.mesh_threads = cfg.threads;
  r.mesh_shared_bytes = (cfg.lds_dwords + 2 * cfg.threads + 3) * 4;
  r.mesh_vertices = cfg.max_vertices;
  r.mesh_primitives = cfg.max_primitives;
  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::MeshEXT, main_fn, "main", iface);
  t.m.ExecMode(main_fn, spv::ExecutionMode::LocalSize, {cfg.threads, 1, 1});
  t.m.ExecMode(main_fn, spv::ExecutionMode::OutputVertices, {cfg.max_vertices});
  t.m.ExecMode(main_fn, spv::ExecutionMode::OutputPrimitivesEXT,
               {cfg.max_primitives});
  t.m.ExecMode(main_fn, spv::ExecutionMode::OutputTrianglesEXT, {});
  return true;
}

bool TranslatePs(const Program& program,
                 const base::HashSet<u32>& flat_attrs,
                 u32 ps_input_ena,
                 Recompiled& r,
                 Translator& t,
                 u32 user_sgprs,
                 const u32* ps_in_cntl,
                 u32 ps_num_interp,
                 const base::Vector<u32>* vs_exported_params) {
  if (ShDbg())
    DumpProgram(program, "ps");
  base::Vector<Id> iface;
  StageContext sc;
  sc.is_ps = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  sc.ps_in_cntl = ps_in_cntl;
  sc.ps_num_interp = ps_num_interp;
  sc.vs_exported_params = vs_exported_params;
  sc.skip_launch_movs = LaunchExecMovPcs(program);
  sc.pervertex_attrs = gpu::gcn::PlanPerVertexAttrs(program, nullptr);
  if (kGpuPsVgpr && kGpuPsVgprBlock)
    sc.vgpr_snap_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, t.t_f),
                     spv::StorageClass::Private, t.m.ConstNull(t.t_f));
  if (kGpuPstex)
    t.last_texel_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Private, t.t_v4),
                     spv::StorageClass::Private, t.m.ConstNull(t.t_v4));
  if (r.direct) {
    PlanDirect(program, sc);
  } else {
    if (!RdnaPlanCbufs(program, static_cast<u32>(r.vs_cbufs.size()), r.ps_cbufs,
                       sc.cbuf_bind, sc.smem_cbuf_by_pc,
                       r.indirect_cbufs ? gpu::gcn::kIndirectCbufBindings
                                        : kMaxCbufBindings))
      return false;
    RdnaPlanGfxBuffers(program, static_cast<u32>(r.vs_bufs.size()), nullptr,
                       r.ps_bufs, sc.gfx_buf_bind);
    NoteCbufWindows(r.ps_cbufs, sc);
  }
  PlanGraphicsLds(program, t, sc);
  const gpu::gcn::MimgBindingPlan mimg_plan = RdnaPlanMimg(program);
  if (mimg_plan.binding_srsrc.size() > StageContext::kMaxPsSamplers) {
    if (ShDbg() || kDrawCensus)
      BASE_LOGI("gcnspv", "ps {:#x} rejected: {} textures", g_ps_addr,
                mimg_plan.binding_srsrc.size());
    return false;
  }
  sc.mimg_plan = &mimg_plan;
  sc.tex_3d_mask = RdnaTexDimMask(program, mimg_plan, 1u << 2);
  sc.tex_1d_mask = RdnaTexDimMask(program, mimg_plan, (1u << 0) | (1u << 4));
  for (u32 i = 0; i < mimg_plan.binding_srsrc.size(); i++)
    r.ps_texs.push_back({i, mimg_plan.binding_srsrc[i],
                         mimg_plan.binding_storage[i],
                         ((sc.tex_3d_mask >> i) & 1u) != 0});

  const Id user_data = DeclareUserData(t);
  if (sc.direct)
    gpu::gcn::DeclareDirectMemory(t, sc);
  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  PlanCrossLane(program, t, sc, iface);
  SeedUserData(t, user_data, 0, user_sgprs);
  gpu::gcn::SeedPsInputVgprs(t, ps_input_ena, iface);
  gpu::gcn::SeedPsBarycentrics(t, ps_input_ena, sc);

  const bool has_color_export =
      base::AnyOf(program.begin(), program.end(), [](const Inst& inst) {
        return inst.enc == Enc::kExp && ((inst.raw[0] >> 4) & 0x3F) <= 7 &&
               (inst.raw[0] & 0xF);
      });

  const bool cfg = ForceCfg() || HasControlFlow(program);
  if (cfg && has_color_export) {
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(0.f), t.F32(0.f), t.F32(0.f), t.F32(0.f)}));
    sc.color_written_var = t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                        t.m.ConstNull(t.t_u));
  }
  if (kGpuForcecolor && kGpuNokill)
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(0.f), t.F32(1.f), t.F32(0.f), t.F32(1.f)}));
  if (!kNullPs || g_ps_addr != kNullPs)
    EmitBody(t, program, sc);

  bool kills_lanes = false;
  for (const Inst& in : program) {
    if ((in.enc == Enc::kVopc && (in.opcode & 0x10)) ||
        (in.enc == Enc::kVop3 && in.opcode <= 0xFF && (in.opcode & 0x10)))
      kills_lanes = true;
  }
  if (sc.wrote_color && kills_lanes && !kGpuNokill) {
    const Id live = t.LaneActive(t.Exec());
    const Id kill_blk = t.m.NewBlock(), after_kill = t.m.NewBlock();
    t.m.SelectionMerge(after_kill);
    t.m.BranchConditional(live, after_kill, kill_blk);
    t.m.OpenBlock(kill_blk);
    t.m.Kill();
    t.m.OpenBlock(after_kill);
  }

  if (sc.color_written_var) {
    if (!kGpuNokill) {
      const Id wrote = t.IsNonZero(t.m.Load(t.t_u, sc.color_written_var));
      const Id kill_blk = t.m.NewBlock(), after_kill = t.m.NewBlock();
      t.m.SelectionMerge(after_kill);
      t.m.BranchConditional(wrote, after_kill, kill_blk);
      t.m.OpenBlock(kill_blk);
      t.m.Kill();
      t.m.OpenBlock(after_kill);
    }
  }

  if (kGpuPsuv) {
    const Id in0 = gpu::gcn::PsInputVar(t, sc, kGpuPsuvAttr);
    const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
    const Id u = t.m.Load(t.t_f, t.m.AccessChain(p_in_f, in0, {t.U32(0)}));
    const Id v = t.m.Load(t.t_f, t.m.AccessChain(p_in_f, in0, {t.U32(1)}));
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.CompositeConstruct(t.t_v4, {u, v, t.F32(0.f), t.F32(1.f)}));
  }

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, sc.main_fn, "main", iface);
  t.m.ExecMode(sc.main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

bool TranslateDepthOnlyPs(Translator& t) {
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, main_fn, "main", {});
  t.m.ExecMode(main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

}  // namespace

bool HasNggTransfer(const u32* code) {
  if (!code || !gpu::IsReadableRangeCached(reinterpret_cast<u64>(code), 1024))
    return false;
  const Program program = DecodeShader(code, 256);
  return base::AnyOf(program.begin(), program.end(), [](const Inst& inst) {
    return IsReturn(inst) && (inst.raw[0] & 0xff) == 6;
  });
}

bool HasNggPrimitiveExports(const u32* code) {
  if (!code || !gpu::IsReadableRangeCached(reinterpret_cast<u64>(code), 16384))
    return false;
  bool allocation = false, connectivity = false;
  for (const Inst& inst : *CachedReachableProgram(code, 4096)) {
    allocation |= inst.enc == Enc::kSopp && inst.opcode == 0x10 &&
                  (inst.raw[0] & 0xf) == 9;
    connectivity |= inst.enc == Enc::kExp && ((inst.raw[0] >> 4) & 0x3f) == 20;
    if (allocation && connectivity)
      return true;
  }
  return false;
}

void SetComputeAddress(u64 address) {
  g_cs_addr = address;
}

u32 LdsAccess(const Inst& inst) {
  if (inst.enc != Enc::kDs || ((inst.raw[0] >> 17) & 1))
    return 0;
  const u32 op = inst.opcode;
  if (op == 0x35 || op == 0xb2 || op == 0xb3 || (op >= 0x3d && op <= 0x3f))
    return 0;
  const bool read = (op >= 0x36 && op <= 0x3c) || (op >= 0x76 && op <= 0x78) ||
                    op == 0xfe || op == 0xff || (op >= 0xa1 && op <= 0xa6);
  const bool atomic_rtn =
      (op >= 0x20 && op <= 0x33) || (op >= 0x60 && op <= 0x73);
  return atomic_rtn ? 3 : read ? 1 : 2;
}

DELTA_OPTION(bool, kStructuredCfg, "DELTA_GPU_STRUCTURED_CFG", true);
DELTA_OPTION(bool, kCfgTrace, "DELTA_GPU_CFGTRACE", false);

namespace {
class Structurizer {
 public:
  Structurizer(Translator& t, const Program& program, StageContext& sc)
      : t_(t), program_(program), sc_(sc) {}

  const char* why() const { return why_; }

  bool Run() {
    if (!Build() || !Analyze())
      return false;
    dry_ = true;
    emitted_ = 0;
    ok_ = true;
    Emit(0, n_, Loop{});
    if (!ok_)
      return Fail(why_ ? why_ : "shape");
    if (emitted_ > 2 * program_.size() + 64)
      return Fail("duplication");
    dry_ = false;
    if (kCfgMaxIter)
      iter_var_ = t_.m.Variable(t_.p_priv_u, spv::StorageClass::Private,
                                t_.m.ConstNull(t_.t_u));
    Emit(0, n_, Loop{});
    return true;
  }

 private:
  struct Loop {
    u32 header = ~0u, exit = ~0u;
    Id cont = 0, brk = 0;
  };

  bool Build() {
    max_pc_ = program_.empty() ? 0 : program_.back().pc + program_.back().size;
    starts_ = BlockStarts(program_, max_pc_);
    n_ = static_cast<u32>(starts_.size());
    if (!n_ || n_ > 2048)
      return Fail("size");
    kind_.assign(n_, 0);
    term_.assign(n_, ~0u);
    succ_.assign(n_, {});
    first_.assign(n_, 0);
    last_.assign(n_, 0);
    u32 bi = 0;
    for (u32 i = 0; i < program_.size(); i++) {
      const Inst& inst = program_[i];
      while (bi + 1 < n_ && inst.pc >= starts_[bi + 1])
        bi++;
      if (IsCall(inst) || IsReturn(inst))
        return Fail("call");
      if (i == 0 || program_[i - 1].pc < starts_[bi])
        first_[bi] = i;
      last_[bi] = i + 1;
      if (BranchKind(inst) && term_[bi] == ~0u)
        term_[bi] = i;
    }
    for (u32 b = 0; b < n_; b++) {
      const u32 fall = b + 1 < n_ ? b + 1 : n_;
      if (term_[b] == ~0u) {
        succ_[b] = {fall};
        continue;
      }
      const Inst& inst = program_[term_[b]];
      kind_[b] = BranchKind(inst);
      if (kind_[b] == 8) {
        succ_[b] = {n_};
        continue;
      }
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
      const u32 target = BlockOf(static_cast<u32>(
          static_cast<i32>(inst.pc) + static_cast<i32>(inst.size) + simm));
      if (kind_[b] == 1 || target == fall)
        succ_[b] = {target};
      else
        succ_[b] = {target, fall};
    }
    return true;
  }

  u32 BlockOf(u32 pc) const {
    if (pc >= max_pc_)
      return n_;
    u32 b = 0;
    for (u32 i = 0; i < n_ && starts_[i] <= pc; i++)
      b = i;
    return b;
  }

  bool Analyze() {
    const u32 nodes = n_ + 1;
    base::Vector<base::Vector<u32> > pred(nodes);
    for (u32 b = 0; b < n_; b++)
      for (u32 s : succ_[b])
        pred[s].push_back(b);
    base::Vector<u32> order;
    base::Vector<u8> seen(nodes, 0);
    base::Vector<base::Pair<u32, u32> > stack{{0u, 0u}};
    seen[0] = 1;
    while (!stack.empty()) {
      auto& [b, i] = stack.back();
      const auto& out = b < n_ ? succ_[b] : base::Vector<u32>{};
      if (i < out.size()) {
        const u32 s = out[i++];
        if (!seen[s]) {
          seen[s] = 1;
          stack.push_back({s, 0u});
        }
        continue;
      }
      order.push_back(b);
      stack.pop_back();
    }
    if (!seen[n_])
      return Fail("no-exit");
    reachable_ = seen;
    base::Vector<u32> rpo(order.rbegin(), order.rend());
    base::Vector<u32> index(nodes, ~0u);
    for (u32 i = 0; i < rpo.size(); i++)
      index[rpo[i]] = i;
    idom_ = Dominators(rpo, index, pred, 0);
    base::Vector<u32> rorder;
    base::Vector<u8> rseen(nodes, 0);
    stack = {{n_, 0u}};
    rseen[n_] = 1;
    while (!stack.empty()) {
      auto& [b, i] = stack.back();
      if (i < pred[b].size()) {
        const u32 p = pred[b][i++];
        if (!rseen[p] && reachable_[p]) {
          rseen[p] = 1;
          stack.push_back({p, 0u});
        }
        continue;
      }
      rorder.push_back(b);
      stack.pop_back();
    }
    for (u32 b = 0; b < nodes; b++)
      if (reachable_[b] && !rseen[b])
        return Fail("dead-end");
    base::Vector<u32> prpo(rorder.rbegin(), rorder.rend());
    base::Vector<u32> pindex(nodes, ~0u);
    for (u32 i = 0; i < prpo.size(); i++)
      pindex[prpo[i]] = i;
    base::Vector<base::Vector<u32> > rsucc(nodes);
    for (u32 b = 0; b < n_; b++)
      if (reachable_[b])
        for (u32 s : succ_[b])
          rsucc[b].push_back(s);
    ipdom_ = Dominators(prpo, pindex, rsucc, n_);
    header_of_.assign(nodes, ~0u);
    exit_of_.assign(nodes, ~0u);
    in_loop_.assign(nodes, {});
    for (u32 b = 0; b < n_; b++) {
      if (!reachable_[b])
        continue;
      for (u32 s : succ_[b]) {
        if (s == n_ || index[s] > index[b])
          continue;
        if (!Dominates(s, b))
          return Fail("irreducible");
        if (in_loop_[s].empty())
          in_loop_[s].assign(nodes, 0);
        base::Vector<u32> work{b};
        in_loop_[s][s] = 1;
        while (!work.empty()) {
          const u32 x = work.back();
          work.pop_back();
          if (in_loop_[s][x])
            continue;
          in_loop_[s][x] = 1;
          for (u32 p : pred[x])
            if (reachable_[p])
              work.push_back(p);
        }
      }
    }
    for (u32 h = 0; h < n_; h++) {
      if (in_loop_[h].empty())
        continue;
      u32 exit = ~0u;
      for (u32 x = 0; x < n_; x++) {
        if (!in_loop_[h][x])
          continue;
        for (u32 s : succ_[x]) {
          if (s < nodes && in_loop_[h][s])
            continue;
          if (exit != ~0u && exit != s)
            return Fail("two-exits");
          exit = s;
        }
      }
      if (exit == ~0u)
        return Fail("no-loop-exit");
      exit_of_[h] = exit;
    }
    return true;
  }

  base::Vector<u32> Dominators(const base::Vector<u32>& rpo,
                               const base::Vector<u32>& index,
                               const base::Vector<base::Vector<u32> >& pred,
                               u32 root) {
    base::Vector<u32> dom(index.size(), ~0u);
    dom[root] = root;
    for (bool changed = true; changed;) {
      changed = false;
      for (u32 b : rpo) {
        if (b == root)
          continue;
        u32 d = ~0u;
        for (u32 p : pred[b]) {
          if (index[p] == ~0u || dom[p] == ~0u)
            continue;
          if (d == ~0u) {
            d = p;
            continue;
          }
          u32 x = p, y = d;
          while (x != y) {
            while (index[x] > index[y])
              x = dom[x];
            while (index[y] > index[x])
              y = dom[y];
          }
          d = x;
        }
        if (d != dom[b]) {
          dom[b] = d;
          changed = true;
        }
      }
    }
    return dom;
  }

  bool Dominates(u32 a, u32 b) const {
    for (u32 x = b;; x = idom_[x]) {
      if (x == a)
        return true;
      if (x == 0 || idom_[x] == ~0u || idom_[x] == x)
        return x == a;
    }
  }

  bool InLoop(const Loop& loop, u32 b) const {
    return loop.header == ~0u ||
           (b < in_loop_[loop.header].size() && in_loop_[loop.header][b]);
  }

  bool Emit(u32 b, u32 stop, const Loop& loop) {
    for (u32 guard = 0; ok_ && b != stop; guard++) {
      if (guard > 4 * n_ + 16) {
        ok_ = false;
        return false;
      }
      if (loop.header != ~0u && b == loop.header)
        return Leave(loop.cont);
      if (loop.header != ~0u && b == loop.exit)
        return Leave(loop.brk);
      if (b == n_) {
        ok_ = false;
        return false;
      }
      if (!in_loop_[b].empty() && b != loop.header) {
        b = EmitLoop(b, loop);
        continue;
      }
      b = EmitBlock(b, stop, loop);
      if (b == ~0u)
        return false;
    }
    return ok_;
  }

  bool Fail(const char* why) {
    if (!why_)
      why_ = why;
    return false;
  }

  bool Leave(Id target) {
    if (!dry_)
      t_.m.Branch(target);
    return false;
  }

  u32 EmitLoop(u32 h, const Loop& outer) {
    const u32 exit = exit_of_[h];
    if (outer.header != ~0u && exit != outer.exit && exit != outer.header &&
        !InLoop(outer, exit)) {
      ok_ = false;
      return n_;
    }
    Loop loop{h, exit, 0, 0};
    Id header = 0, body = 0;
    if (!dry_) {
      header = t_.m.NewBlock();
      body = t_.m.NewBlock();
      loop.cont = t_.m.NewBlock();
      loop.brk = t_.m.NewBlock();
      t_.m.Branch(header);
      t_.m.OpenBlock(header);
      t_.m.LoopMerge(loop.brk, loop.cont);
      t_.m.Branch(body);
      t_.m.OpenBlock(body);
    }
    const u32 next = EmitBlock(h, ~0u, loop);
    if (next != ~0u && ok_)
      Emit(next, ~0u, loop);
    if (!dry_) {
      t_.m.OpenBlock(loop.cont);
      if (iter_var_) {
        const Id it = t_.m.Load(t_.t_u, iter_var_);
        t_.m.Store(iter_var_,
                   t_.m.Emit(spv::Op::OpIAdd, t_.t_u, {it, t_.U32(1)}));
        t_.m.BranchConditional(t_.m.Emit(spv::Op::OpUGreaterThan, t_.t_bool,
                                         {it, t_.U32(kCfgMaxIter)}),
                               loop.brk, header);
      } else {
        t_.m.Branch(header);
      }
      t_.m.OpenBlock(loop.brk);
    }
    return exit;
  }

  u32 EmitBlock(u32 b, u32 stop, const Loop& loop) {
    const u32 end = term_[b] != ~0u ? term_[b] : last_[b];
    emitted_ += end - first_[b] + 1;
    u32 lds_seen = 3;
    if (!dry_)
      for (u32 i = first_[b]; i < end; i++) {
        const Inst& inst = program_[i];
        if (sc_.lds_sync) {
          const u32 access = LdsAccess(inst);
          if (access &&
              ((access & 1 && lds_seen & 2) || (access & 2 && lds_seen & 1))) {
            t_.Barrier();
            lds_seen = 0;
          }
          lds_seen |= access;
          if (inst.enc == Enc::kSopp && inst.opcode == 0x0a)
            lds_seen = 0;
        }
        RdnaEmitInst(t_, inst, sc_);
      }
    if (succ_[b].size() == 1)
      return succ_[b][0];
    const u32 target = succ_[b][0], fall = succ_[b][1];
    u32 merge = ipdom_[b];
    const bool joins = merge != ~0u && InLoop(loop, merge) &&
                       merge != loop.header &&
                       (loop.header == ~0u || merge != loop.exit);
    if (!joins)
      merge = stop;
    Id merge_label = 0, then_label = 0, else_label = 0;
    if (!dry_) {
      const Id cond = BranchTaken(t_, kind_[b]);
      merge_label = t_.m.NewBlock();
      then_label = t_.m.NewBlock();
      else_label = t_.m.NewBlock();
      t_.m.SelectionMerge(merge_label);
      t_.m.BranchConditional(cond, then_label, else_label);
      t_.m.OpenBlock(then_label);
    }
    const bool then_joins = Emit(target, merge, loop);
    if (then_joins && !dry_)
      t_.m.Branch(merge_label);
    if (!dry_)
      t_.m.OpenBlock(else_label);
    const bool else_joins = Emit(fall, merge, loop);
    if (else_joins && !dry_)
      t_.m.Branch(merge_label);
    if (!dry_)
      t_.m.OpenBlock(merge_label);
    if (!then_joins && !else_joins) {
      if (!dry_)
        t_.m.Unreachable();
      return ~0u;
    }
    return merge;
  }

  Translator& t_;
  const Program& program_;
  StageContext& sc_;
  u32 max_pc_ = 0, n_ = 0;
  base::Vector<u32> starts_, term_, first_, last_;
  base::Vector<int> kind_;
  base::Vector<base::Vector<u32> > succ_;
  base::Vector<u8> reachable_;
  base::Vector<u32> idom_, ipdom_, header_of_, exit_of_;
  base::Vector<base::Vector<u8> > in_loop_;
  bool dry_ = true, ok_ = true;
  const char* why_ = nullptr;
  u64 emitted_ = 0;
  Id iter_var_ = 0;
};
}  // namespace

void EmitCfg(Translator& t, const Program& program, StageContext& sc) {
  const u32 max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  base::Vector<u32> starts = BlockStarts(program, max_pc);
  const bool scheduled = sc.is_mesh || sc.wave_lockstep;
  const bool one_wave = scheduled && t.xchg_lanes && t.xchg_lanes <= 64;
  if ((!scheduled || one_wave) && kStructuredCfg) {
    t.uniform_here = one_wave || (sc.is_cs && starts.size() == 1);
    t.block_active = 0;
    Structurizer structurizer(t, program, sc);
    const bool done = structurizer.Run();
    if (kCfgTrace)
      BASE_LOGI("cfgtrace", "{} insts={} lanes={} -> {}{}",
                sc.is_mesh ? "mesh"
                : sc.is_cs ? "cs"
                : sc.is_ps ? "ps"
                           : "vs",
                program.size(), t.xchg_lanes,
                done ? "structured" : "state machine: ",
                done ? "" : (structurizer.why() ? structurizer.why() : "?"));
    if (done)
      return;
  } else if (kCfgTrace) {
    BASE_LOGI("cfgtrace", "{} insts={} lanes={} -> state machine: scheduled",
              sc.is_mesh ? "mesh"
              : sc.is_cs ? "cs"
              : sc.is_ps ? "ps"
                         : "vs",
              program.size(), t.xchg_lanes);
  }
  if (scheduled) {
    for (const Inst& inst : program) {
      if (inst.enc != Enc::kSopp || inst.opcode != 0x0a)
        continue;
      starts.push_back(inst.pc);
      if (inst.pc + inst.size < max_pc)
        starts.push_back(inst.pc + inst.size);
    }
    base::Sort(starts.begin(), starts.end());
    starts.erase(base::Unique(starts.begin(), starts.end()), starts.end());
  }
  const u32 num_blocks = static_cast<u32>(starts.size());
  const u32 kExit = num_blocks;
  t.uniform_here = scheduled || (sc.is_cs && num_blocks == 1);
  const auto block_of = [&](u32 pc) -> u32 {
    if (pc >= max_pc)
      return kExit;
    u32 b = 0;
    for (u32 i = 0; i < num_blocks; i++)
      if (starts[i] <= pc)
        b = i;
      else
        break;
    return b;
  };

  const Id header = t.m.NewBlock(), dispatch = t.m.NewBlock();
  const Id merge_sel = t.m.NewBlock();
  const Id cont = t.m.NewBlock(), merge = t.m.NewBlock();
  const Id exit_blk = t.m.NewBlock();
  base::Vector<Id> case_labels(num_blocks);
  for (Id& l : case_labels)
    l = t.m.NewBlock();

  const Id iter_var = kCfgMaxIter
                          ? t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                         t.m.ConstNull(t.t_u))
                          : 0;

  if (kLoopTrace) {
    u32 back_edges = 0;
    for (const Inst& inst : program) {
      const int k = BranchKind(inst);
      if ((k != 1 && k < 2) || IsCall(inst) || IsReturn(inst))
        continue;
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
      if (simm < 0)
        back_edges++;
    }
    if (back_edges)
      BASE_LOGI("looptrace", "{} vs={:#x} blocks={} back_edges={} insts={}",
                sc.is_cs ? "cs" : (sc.is_ps ? "ps" : "vs"),
                (unsigned long)(sc.is_cs ? 0 : g_vs_addr), num_blocks,
                back_edges, program.size());
  }

  t.SetState(0);
  t.m.Branch(header);
  t.m.OpenBlock(header);
  t.m.LoopMerge(merge, cont);
  t.m.Branch(dispatch);
  t.m.OpenBlock(dispatch);
  Id state = t.State();
  if (scheduled) {
    t.WavePublish(state);
    t.Barrier();
    const Id leader = t.m.NewBlock(), selected = t.m.NewBlock();
    const Id is_leader = t.IsZero(t.xchg_index);
    t.m.SelectionMerge(selected);
    t.m.BranchConditional(is_leader, leader, selected);
    t.m.OpenBlock(leader);
    const Id first = t.m.Load(t.t_u, t.XchgAt(t.U32(0)));
    Id all_equal = t.True(), pending = t.U32(kExit);
    for (u32 wave = 0; wave < t.xchg_lanes; wave += 64) {
      const Id pc = t.m.Load(t.t_u, t.XchgAt(t.U32(wave)));
      all_equal = t.m.Emit(spv::Op::OpLogicalAnd, t.t_bool,
                           {all_equal, t.Eq(pc, first)});
      Id runnable = t.Ult(pc, t.U32(kExit));
      for (const Inst& inst : program) {
        if (inst.enc == Enc::kSopp && inst.opcode == 0x0a)
          runnable =
              t.m.Emit(spv::Op::OpLogicalAnd, t.t_bool,
                       {runnable, t.m.Emit(spv::Op::OpINotEqual, t.t_bool,
                                           {pc, t.U32(block_of(inst.pc))})});
      }
      pending = t.SelectB(runnable, t.UMin(pending, pc), pending);
    }
    const Id choice = t.XchgAt(t.U32(2 * t.xchg_lanes));
    t.m.Store(choice, t.SelectB(all_equal, first, pending));
    t.m.Branch(selected);
    t.m.OpenBlock(selected);
    t.Barrier();
    state = t.m.Load(t.t_u, t.XchgAt(t.U32(2 * t.xchg_lanes)));
    t.Barrier();
    t.block_active = t.Eq(t.State(), state);
  }
  if (kCfgMaxIter) {
    const Id it = t.m.Load(t.t_u, iter_var);
    t.m.Store(iter_var, t.m.Emit(spv::Op::OpIAdd, t.t_u, {it, t.U32(1)}));
    const Id over =
        t.m.Emit(spv::Op::OpUGreaterThan, t.t_bool, {it, t.U32(kCfgMaxIter)});
    state = t.SelectB(over, t.U32(kExit), state);
  }
  t.m.SelectionMerge(merge_sel);
  base::Vector<base::Pair<u32, Id> > cases;
  for (u32 i = 0; i < num_blocks; i++)
    cases.push_back({i, case_labels[i]});
  t.m.Switch(state, exit_blk, cases);

  for (u32 bi = 0; bi < num_blocks; bi++) {
    t.m.OpenBlock(case_labels[bi]);
    if (sc.vgpr_snap_var && kGpuPsVgprBlock - 1 == bi)
      t.m.Store(sc.vgpr_snap_var, t.VgF(kGpuPsVgpr - 1));
    const u32 blk_start = starts[bi];
    const u32 blk_end = (bi + 1 < num_blocks) ? starts[bi + 1] : max_pc;
    bool terminated = false;
    u32 lds_seen = num_blocks > 1 ? 3 : 0;
    for (const Inst& inst : program) {
      if (inst.pc < blk_start || inst.pc >= blk_end)
        continue;
      const int k = BranchKind(inst);
      if (k == 0 && !IsCall(inst) && !IsReturn(inst)) {
        if (sc.lds_sync) {
          u32 access = LdsAccess(inst);
          if (access &&
              ((access & 1 && lds_seen & 2) || (access & 2 && lds_seen & 1))) {
            t.Barrier();
            lds_seen = 0;
          }
          lds_seen |= access;
          if (inst.enc == Enc::kSopp && inst.opcode == 0x0a)
            lds_seen = 0;
        }
        RdnaEmitInst(t, inst, sc);
        continue;
      }
      const u32 fall = (bi + 1 < num_blocks) ? bi + 1 : kExit;
      if (IsCall(inst)) {
        const u32 sdst = (inst.raw[0] >> 16) & 0x7F;
        const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
        t.SetSg(sdst, t.U32(fall));
        t.SetSg(sdst + 1, t.U32(0));
        t.SetState(block_of(static_cast<u32>(
            static_cast<i32>(inst.pc) + static_cast<i32>(inst.size) + simm)));
      } else if (IsReturn(inst)) {
        t.SetStateId(t.Sg(inst.raw[0] & 0xFF));
      } else if (k == 8) {
        t.SetState(kExit);
      } else if (k == 1) {
        const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
        t.SetState(block_of(static_cast<u32>(
            static_cast<i32>(inst.pc) + static_cast<i32>(inst.size) + simm)));
      } else {
        const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
        const u32 target = block_of(static_cast<u32>(
            static_cast<i32>(inst.pc) + static_cast<i32>(inst.size) + simm));
        t.SetStateId(t.SelectB(BranchTaken(t, k), t.U32(target), t.U32(fall)));
      }
      terminated = true;
      break;
    }
    if (!terminated)
      t.SetState((bi + 1 < num_blocks) ? bi + 1 : kExit);
    t.m.Branch(merge_sel);
  }
  t.m.OpenBlock(exit_blk);
  t.m.Branch(merge);
  t.m.OpenBlock(merge_sel);
  t.m.Branch(cont);
  t.m.OpenBlock(cont);
  t.m.Branch(header);
  t.m.OpenBlock(merge);
  t.block_active = 0;
}

Recompiled Recompile(const u32* vs_code,
                     const u32* ps_code,
                     const u32* vs_user_data,
                     const u32* ps_user_data,
                     u32 ps_input_ena,
                     bool gl_clip_space,
                     u32 vs_user_sgprs,
                     u32 ps_user_sgprs,
                     const u32* ps_in_cntl,
                     u32 ps_num_interp,
                     const NggConfig* ngg) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kShaderCache);
  Recompiled r;
  if (!vs_code || !vs_user_data || !ps_user_data)
    return r;

  constexpr u64 kMaxShaderBytes = 4096 * sizeof(u32);
  const u64 vs_address = reinterpret_cast<uintptr_t>(vs_code);
  const u64 ps_address = reinterpret_cast<uintptr_t>(ps_code);
  if (!gpu::IsReadableRange(vs_address, kMaxShaderBytes) ||
      (ps_code && !gpu::IsReadableRange(ps_address, kMaxShaderBytes)) ||
      (ngg && (!ngg->gs_code ||
               !gpu::IsReadableRange(reinterpret_cast<u64>(ngg->gs_code),
                                     kMaxShaderBytes))))
    return r;

  const Program vs_program = ReachableProgram(DecodeShader(vs_code, 4096));
  const Program gs_program =
      ngg && ngg->gs_code ? ReachableProgram(DecodeShader(ngg->gs_code, 4096))
                          : Program{};
  const Program ps_program =
      ps_code ? ReachableProgram(DecodeShader(ps_code, 4096)) : Program{};

  base::HashSet<u32> flat_attrs;
  for (const Inst& inst : ps_program)
    if (inst.enc == Enc::kVintrp && inst.opcode == 2 &&
        (inst.raw[0] & 0xFF) == 2)
      flat_attrs.insert((inst.raw[0] >> 10) & 0x3F);

  base::Vector<u32> vs_exported_params;
  for (const Inst& inst : ngg ? gs_program : vs_program)
    if (inst.enc == Enc::kExp) {
      const u32 tgt = (inst.raw[0] >> 4) & 0x3F;
      if (tgt >= 32 && tgt <= 63)
        vs_exported_params.push_back(tgt - 32);
    }
  base::Sort(vs_exported_params.begin(), vs_exported_params.end());
  vs_exported_params.erase(
      base::Unique(vs_exported_params.begin(), vs_exported_params.end()),
      vs_exported_params.end());

  r.indirect_cbufs = ngg || vs_user_sgprs > 16 || ps_user_sgprs > 16;
  r.direct = DirectGraphics();
  Translator tv;
  tv.rdna_sources = true;
  tv.indirect_cbufs = r.indirect_cbufs;
  gpu::gcn::ResetUnsupported();
  g_vs_addr = reinterpret_cast<uintptr_t>(vs_code);
  g_stage_bufs = ResolveBuffers(vs_code, vs_user_data, vs_user_sgprs, 8, 4096,
                                ngg ? ngg->gs_user_data_addr : 0);
  g_warned_store = false;
  const u32 ps_tex_count =
      ps_code ? static_cast<u32>(RdnaPlanMimg(ps_program).binding_srsrc.size())
              : 0;
  const bool vertex_ok =
      ngg ? TranslateMesh(vs_program, gs_program, vs_code, vs_user_data,
                          vs_user_sgprs, *ngg, flat_attrs, ps_tex_count,
                          gl_clip_space, r, tv)
          : TranslateVs(vs_program, vs_user_data, flat_attrs, r, tv,
                        gl_clip_space, vs_user_sgprs, ps_tex_count);
  if (!vertex_ok || gpu::gcn::HadUnsupported()) {
    if (ShDbg() || kDrawCensus)
      BASE_LOGI("gcnspv", "vs {:#x} rejected: {}",
                (unsigned long)reinterpret_cast<uintptr_t>(vs_code),
                gpu::gcn::UnsupportedOps().c_str());
    return r;
  }
  Translator tp;
  tp.rdna_sources = true;
  tp.indirect_cbufs = r.indirect_cbufs;
  tp.user_data_slot = 1;
  g_ps_addr = reinterpret_cast<uintptr_t>(ps_code);
  g_stage_bufs = ResolveBuffers(ps_code, ps_user_data, ps_user_sgprs, 0);
  g_warned_store = false;
  tp.movrel_m0 = PlanMovrelIndices(ps_program);
  tp.InitTypes(&ps_program);
  gpu::gcn::ResetUnsupported();
  if (ps_code ? !TranslatePs(ps_program, flat_attrs, ps_input_ena, r, tp,
                             ps_user_sgprs, ps_in_cntl, ps_num_interp,
                             &vs_exported_params)
              : !TranslateDepthOnlyPs(tp))
    return r;
  if (gpu::gcn::HadUnsupported()) {
    if (ShDbg() || kDrawCensus)
      BASE_LOGI("gcnspv", "ps {:#x} rejected: {}",
                (unsigned long)reinterpret_cast<uintptr_t>(ps_code),
                gpu::gcn::UnsupportedOps().c_str());
    return r;
  }

  const base::Vector<u32> vs = tv.m.Assemble();
  const base::Vector<u32> ps = tp.m.Assemble();
  const base::Vector<u32> gs =
      gpu::gcn::EmitRectListGeometry(r.num_params, flat_attrs);
  base::String err;
  if (!gpu::gcn::spirv::Validate(vs, &err)) {
    if (gpu::gcn::TraceEnabled() || kDrawCensus)
      BASE_LOGI("rdna", "VS invalid: {}", err.c_str());
    return r;
  }
  if (!gpu::gcn::spirv::Validate(ps, &err)) {
    if (gpu::gcn::TraceEnabled() || kDrawCensus)
      BASE_LOGI("rdna", "PS invalid: {}", err.c_str());
    return r;
  }
  if (const char* dir = kSpvDump) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/vs_%lx.spv", dir,
                  (unsigned long)reinterpret_cast<uintptr_t>(vs_code));
    if (FILE* f = std::fopen(path, "wb")) {
      std::fwrite(vs.data(), 4, vs.size(), f);
      std::fclose(f);
    }
    std::snprintf(path, sizeof(path), "%s/ps_%lx.spv", dir,
                  (unsigned long)reinterpret_cast<uintptr_t>(ps_code));
    if (FILE* f = std::fopen(path, "wb")) {
      std::fwrite(ps.data(), 4, ps.size(), f);
      std::fclose(f);
    }
  }
  if (ngg)
    r.mesh_spirv = vs;
  else
    r.vs_spirv = vs;
  r.fs_spirv = ps;
  base::String gs_err;
  if (!gs.empty() && gpu::gcn::spirv::Validate(gs, &gs_err))
    r.gs_spirv = gs;
  else if (gpu::gcn::TraceEnabled())
    BASE_LOGI("rdna", "RECTLIST GS invalid: {}", gs_err.c_str());
  r.ok = (!r.vs_spirv.empty() || !r.mesh_spirv.empty()) && !r.fs_spirv.empty();
  return r;
}

}  // namespace gpu::rdna

#endif
