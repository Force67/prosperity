/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * SPIR-V rewrites for constructs SPIRV-Cross cannot express in HLSL, done on
 * the binary before lowering:
 *   - gl_InvocationID of a geometry shader that runs one invocation becomes a
 *     private constant 0 (HLSL has no instance id without [instance]).
 *   - OpGroupNonUniformBallotFindLSB/MSB become per-component FindILsb /
 *     FindUMsb and selects.
 */

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "gpu/d3d12/d3d12_shader.h"

namespace gpu::d3d12 {

namespace {

enum : u32 {
  kOpExtInstImport = 11,
  kOpExtInst = 12,
  kOpMemoryModel = 14,
  kOpEntryPoint = 15,
  kOpExecutionMode = 16,
  kOpTypeBool = 20,
  kOpTypeInt = 21,
  kOpTypePointer = 32,
  kOpConstant = 43,
  kOpFunction = 54,
  kOpVariable = 59,
  kOpDecorate = 71,
  kOpCompositeExtract = 81,
  kOpIAdd = 128,
  kOpSelect = 169,
  kOpINotEqual = 171,
  kOpBallotFindLSB = 343,
  kOpBallotFindMSB = 344,
};
constexpr u32 kDecorationBuiltIn = 11;
constexpr u32 kBuiltInInvocationId = 8;
constexpr u32 kExecutionModeInvocations = 0;
constexpr u32 kStorageInput = 1;
constexpr u32 kStoragePrivate = 6;
constexpr u32 kFindILsb = 73;
constexpr u32 kFindUMsb = 75;

u32 Op(u32 opcode, u32 count) {
  return (count << 16) | opcode;
}

class Patcher {
 public:
  Patcher(const u32* words, size_t count) : in_(words, words + count) {}

  base::Vector<u32> Run() {
    if (in_.size() < 5)
      return in_;
    bound_ = in_[3];
    Scan();
    if (!invocation_var_ && !ballot_)
      return in_;
    base::Vector<u32> out(in_.begin(), in_.begin() + 5);
    bool globals_done = false;
    for (size_t at = 5; at < in_.size();) {
      const u32 op = in_[at] & 0xffff;
      const u32 n = in_[at] >> 16;
      if (!n)
        return in_;
      const u32* w = &in_[at];
      if (op == kOpMemoryModel && ballot_ && !glsl_) {
        glsl_ = bound_++;
        const char name[] = "GLSL.std.450";  // 12 chars + NUL: 4 words
        u32 packed[4] = {};
        for (size_t i = 0; i < sizeof(name); i++)
          packed[i / 4] |= u32(u8(name[i])) << (8 * (i % 4));
        out.insert(out.end(), {Op(kOpExtInstImport, 6), glsl_, packed[0],
                               packed[1], packed[2], packed[3]});
      }
      if (op == kOpFunction && !globals_done) {
        EmitGlobals(out);
        globals_done = true;
      }
      if (op == kOpDecorate && invocation_var_ && w[1] == invocation_var_ &&
          w[2] == kDecorationBuiltIn) {
        at += n;
        continue;
      }
      if (op == kOpVariable && invocation_var_ && w[2] == invocation_var_) {
        at += n;
        continue;
      }
      if (op == kOpEntryPoint && invocation_var_) {
        base::Vector<u32> ep(w, w + n);
        for (size_t i = 3; i < ep.size(); i++)
          if (ep[i] == invocation_var_) {
            ep.erase(ep.begin() + i);
            break;
          }
        ep[0] = Op(kOpEntryPoint, static_cast<u32>(ep.size()));
        out.insert(out.end(), ep.begin(), ep.end());
        at += n;
        continue;
      }
      if ((op == kOpBallotFindLSB || op == kOpBallotFindMSB) && n == 5) {
        EmitFind(out, op == kOpBallotFindLSB, w[1], w[2], w[4]);
        at += n;
        continue;
      }
      out.insert(out.end(), w, w + n);
      at += n;
    }
    out[3] = bound_;
    return out;
  }

 private:
  void Scan() {
    u32 invocations = 1;
    bool geometry = false;
    for (size_t at = 5; at < in_.size();) {
      const u32* w = &in_[at];
      const u32 op = w[0] & 0xffff, n = w[0] >> 16;
      if (!n || at + n > in_.size())
        break;
      switch (op) {
        case kOpExtInstImport:
          if (n >= 4 && !base::String(reinterpret_cast<const char*>(w + 2))
                             .compare(0, 12, "GLSL.std.450"))
            glsl_ = w[1];
          break;
        case kOpEntryPoint:
          geometry = w[1] == 3;  // ExecutionModelGeometry
          break;
        case kOpExecutionMode:
          if (w[2] == kExecutionModeInvocations && n >= 4)
            invocations = w[3];
          break;
        case kOpDecorate:
          if (n >= 4 && w[2] == kDecorationBuiltIn &&
              w[3] == kBuiltInInvocationId)
            invocation_var_ = w[1];
          break;
        case kOpTypeBool:
          bool_ = w[1];
          break;
        case kOpTypeInt:
          if (w[2] == 32 && w[3] == 0)
            uint_ = w[1];
          break;
        case kOpTypePointer:
          pointers_.push_back({w[1], w[2], w[3]});
          break;
        case kOpVariable:
          if (w[2] == invocation_var_ && w[3] == kStorageInput)
            invocation_ptr_ = w[1];
          break;
        case kOpBallotFindLSB:
        case kOpBallotFindMSB:
          ballot_ = true;
          break;
        default:
          break;
      }
      at += n;
    }
    if (!geometry || invocations != 1 || !invocation_ptr_)
      invocation_var_ = 0;
    for (const Pointer& p : pointers_)
      if (p.id == invocation_ptr_)
        invocation_type_ = p.pointee;
    if (!invocation_type_)
      invocation_var_ = 0;
  }

  void EmitGlobals(base::Vector<u32>& out) {
    if (invocation_var_) {
      const u32 ptr = bound_++, zero = bound_++;
      out.insert(out.end(), {Op(kOpTypePointer, 4), ptr, kStoragePrivate,
                             invocation_type_});
      out.insert(out.end(), {Op(kOpConstant, 4), invocation_type_, zero, 0});
      out.insert(out.end(), {Op(kOpVariable, 5), ptr, invocation_var_,
                             kStoragePrivate, zero});
    }
    if (ballot_) {
      if (!bool_) {
        bool_ = bound_++;
        out.insert(out.end(), {Op(kOpTypeBool, 2), bool_});
      }
      if (!uint_) {
        uint_ = bound_++;
        out.insert(out.end(), {Op(kOpTypeInt, 4), uint_, 32, 0});
      }
      for (u32 i = 0; i < 4; i++) {
        consts_[i] = bound_++;
        out.insert(out.end(), {Op(kOpConstant, 4), uint_, consts_[i], i * 32});
      }
    }
  }

  // result = the lowest (highest) set bit of a uvec4 ballot.
  void EmitFind(base::Vector<u32>& out,
                bool lsb,
                u32 type,
                u32 result,
                u32 value) {
    u32 comp[4], bit[4];
    for (u32 i = 0; i < 4; i++) {
      comp[i] = bound_++;
      out.insert(out.end(),
                 {Op(kOpCompositeExtract, 5), type, comp[i], value, i});
      const u32 raw = bound_++;
      out.insert(out.end(), {Op(kOpExtInst, 6), type, raw, glsl_,
                             lsb ? kFindILsb : kFindUMsb, comp[i]});
      bit[i] = raw;
      if (i) {
        bit[i] = bound_++;
        out.insert(out.end(), {Op(kOpIAdd, 5), type, bit[i], raw, consts_[i]});
      }
    }
    // LSB: x ? bx : y ? by : z ? bz : bw.  MSB: the same from w down.
    const u32 order[4] = {lsb ? 0u : 3u, lsb ? 1u : 2u, lsb ? 2u : 1u,
                          lsb ? 3u : 0u};
    u32 acc = bit[order[3]];
    for (int k = 2; k >= 0; k--) {
      const u32 c = order[k];
      const u32 nz = bound_++;
      out.insert(out.end(),
                 {Op(kOpINotEqual, 5), bool_, nz, comp[c], consts_[0]});
      const u32 sel = k == 0 ? result : bound_++;
      out.insert(out.end(), {Op(kOpSelect, 6), type, sel, nz, bit[c], acc});
      acc = sel;
    }
  }

  struct Pointer {
    u32 id, storage, pointee;
  };
  base::Vector<u32> in_;
  u32 bound_ = 0;
  u32 glsl_ = 0;
  u32 bool_ = 0;
  u32 uint_ = 0;
  u32 consts_[4] = {};
  bool ballot_ = false;
  u32 invocation_var_ = 0, invocation_ptr_ = 0, invocation_type_ = 0;
  base::Vector<Pointer> pointers_;
};

}  // namespace

base::Vector<u32> PatchSpirvForHlsl(const u32* words, size_t count) {
  return Patcher(words, count).Run();
}

bool DeclaresBuiltIn(const u32* words, size_t count, u32 builtin) {
  for (size_t at = 5; at < count;) {
    const u32 op = words[at] & 0xffff, n = words[at] >> 16;
    if (!n)
      break;
    if (op == kOpDecorate && n >= 4 && words[at + 2] == kDecorationBuiltIn &&
        words[at + 3] == builtin)
      return true;
    if (op == kOpFunction)
      break;
    at += n;
  }
  return false;
}

}  // namespace gpu::d3d12
