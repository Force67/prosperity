/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN compute-shader CPU interpreter. See gcn_interp.h. Executes one invocation
 * at a time and writes image_store results into guest memory so Doom64's atlas
 * builders populate the textures the 3D world samples.
 *
 * WIP: the per-thread VM, operand decode, MUBUF (idxen index*stride) load and
 * MIMG image_store -> guest-memory plumbing are in place; the ALU opcode table is
 * GFX6/7 (Liverpool) and filled for the ops Doom64's copy shaders use. Unknown
 * ops are logged once and treated as nops. Gated behind DELTA_GPU_CSRUN.
 */

#include "gcn_interp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "gcn_decode.h"
#include "gcn_detile.h"
#include "gcn_resource.h"

namespace gpu::gcn {
namespace {

inline bool guestOk(uint64_t a) {
  return a >= 0x1000000ull && a < 0x20000000000ull;
}

// Per-invocation register VM. EXEC is a single "this lane active" bit (we run one
// lane at a time), VCC/SCC are scalars. Memory writes are guarded on exec.
struct Lane {
  uint32_t s[256] = {};   // SGPRs (106=VCC_LO, 126=EXEC_LO)
  uint32_t v[256] = {};   // VGPRs
  uint32_t scc = 0;
  uint32_t exec = 1;
  bool unknownLogged = false;

  // Raw uint of a source operand field (mirrors gcn_spirv srcRaw).
  uint32_t rd(uint32_t f, uint32_t lit) const {
    if (f <= 127) return s[f];
    if (f == 128) return 0;
    if (f >= 129 && f <= 192) return f - 128;
    if (f >= 193 && f <= 208) return (uint32_t)(-(int)(f - 192));
    switch (f) {
      case 240: return 0x3f000000u; case 241: return 0xbf000000u;
      case 242: return 0x3f800000u; case 243: return 0xbf800000u;
      case 244: return 0x40000000u; case 245: return 0xc0000000u;
      case 246: return 0x40800000u; case 247: return 0xc0800000u;
    }
    if (f == 255) return lit;
    if (f >= 256) return v[f - 256];
    return 0;
  }
  void wrS(uint32_t d, uint32_t val) { if (d < 256) s[d] = val; }
  void wrV(uint32_t d, uint32_t val) { if (d < 256) v[d] = val; }
};

inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

void logUnknown(Lane &L, const char *enc, uint32_t op) {
  if (L.unknownLogged) return;
  if (std::getenv("DELTA_GPU_CSRUN_VERBOSE"))
    std::fprintf(stderr, "[csinterp] unhandled %s op=%#x\n", enc, op);
}

// VOP2 (GFX6/7, canonical [30:25] opcode numbers). Integer ops dominate compute
// address math.
void vop2(Lane &L, uint32_t op, uint32_t vdst, uint32_t s0, uint32_t s1) {
  switch (op) {
    case 0x00: { bool c = (L.s[106] & 1); L.wrV(vdst, c ? s1 : s0); break; }  // cndmask (VCC)
    case 0x03: L.wrV(vdst, f2u(u2f(s0) + u2f(s1))); break;            // add_f32
    case 0x04: L.wrV(vdst, f2u(u2f(s0) - u2f(s1))); break;            // sub_f32
    case 0x05: L.wrV(vdst, f2u(u2f(s1) - u2f(s0))); break;            // subrev_f32
    case 0x08: L.wrV(vdst, f2u(u2f(s0) * u2f(s1))); break;            // mul_f32
    case 0x09: L.wrV(vdst, (uint32_t)((int32_t)(s0 << 8) >> 8) * (int32_t)((s1 << 8) >> 8)); break; // mul_i32_i24
    case 0x0b: L.wrV(vdst, (s0 & 0xFFFFFF) * (s1 & 0xFFFFFF)); break; // mul_u32_u24
    case 0x0f: L.wrV(vdst, (uint32_t)std::min((int32_t)s0, (int32_t)s1)); break; // min_i32
    case 0x10: L.wrV(vdst, (uint32_t)std::max((int32_t)s0, (int32_t)s1)); break; // max_i32
    case 0x11: L.wrV(vdst, std::min(s0, s1)); break;                 // min_u32
    case 0x12: L.wrV(vdst, std::max(s0, s1)); break;                 // max_u32
    case 0x13: L.wrV(vdst, s0 >> (s1 & 31)); break;                  // lshr_b32
    case 0x14: L.wrV(vdst, s1 >> (s0 & 31)); break;                  // lshrrev_b32
    case 0x15: L.wrV(vdst, (uint32_t)((int32_t)s0 >> (s1 & 31))); break; // ashr_i32
    case 0x16: L.wrV(vdst, (uint32_t)((int32_t)s1 >> (s0 & 31))); break; // ashrrev_i32
    case 0x19: L.wrV(vdst, s0 << (s1 & 31)); break;                  // lshl_b32
    case 0x1a: L.wrV(vdst, s1 << (s0 & 31)); break;                  // lshlrev_b32
    case 0x1b: L.wrV(vdst, s0 & s1); break;                          // and_b32
    case 0x1c: L.wrV(vdst, s0 | s1); break;                          // or_b32
    case 0x1d: L.wrV(vdst, s0 ^ s1); break;                          // xor_b32
    case 0x1f: L.wrV(vdst, f2u(u2f(s0) * u2f(s1) + u2f(L.v[vdst]))); break;  // mac_f32
    case 0x22: L.wrV(vdst, (uint32_t)__builtin_popcount(s0) + s1); break;     // bcnt_u32_b32
    case 0x25: L.wrV(vdst, s0 + s1); break;                          // add_i32
    case 0x26: L.wrV(vdst, s0 - s1); break;                          // sub_i32
    case 0x27: L.wrV(vdst, s1 - s0); break;                          // subrev_i32
    case 0x28: { uint64_t r = (uint64_t)s0 + s1 + (L.s[106] & 1);    // addc_u32 (+VCC)
                 L.wrV(vdst, (uint32_t)r); L.s[106] = (r >> 32) & 1; break; }
    default: logUnknown(L, "vop2", op); break;
  }
}

// VOP1 (GFX6/7).
void vop1(Lane &L, uint32_t op, uint32_t vdst, uint32_t s0) {
  switch (op) {
    case 0x01: L.wrV(vdst, s0); break;                                          // mov_b32
    case 0x05: L.wrV(vdst, f2u((float)(int32_t)s0)); break;                     // cvt_f32_i32
    case 0x06: L.wrV(vdst, f2u((float)s0)); break;                              // cvt_f32_u32
    case 0x07: L.wrV(vdst, (uint32_t)(int32_t)std::lround(u2f(s0))); break;     // cvt_u32_f32
    case 0x08: L.wrV(vdst, (uint32_t)(int32_t)u2f(s0)); break;                  // cvt_i32_f32
    default: logUnknown(L, "vop1", op); break;
  }
}

void vopc(Lane &L, uint32_t op, uint32_t s0, uint32_t s1);  // fwd

// VOP3 (GFX6/7): op<0x100 are VOPC compares in VOP3 form (result -> sdst pair, not
// VCC). 0x100-0x13f alias VOP2, 0x180-0x1ff alias VOP1; 0x140+ are the 3-src ALU.
void vop3(Lane &L, uint32_t op, uint32_t vdst, uint32_t s0, uint32_t s1, uint32_t s2) {
  if (op < 0x100) {  // VOPC done as VOP3: compare, write bool to the sdst (=vdst).
    vopc(L, op, s0, s1);
    L.wrS(vdst, L.s[106]);  // vopc left the bool in VCC; copy to the real dest
    return;
  }
  if (op >= 0x100 && op < 0x140) { vop2(L, op - 0x100, vdst, s0, s1); return; }
  if (op >= 0x180 && op < 0x200) { vop1(L, op - 0x180, vdst, s0); return; }
  switch (op) {
    case 0x141: case 0x14b: L.wrV(vdst, f2u(u2f(s0) * u2f(s1) + u2f(s2))); break;  // mad/fma_f32
    case 0x142: L.wrV(vdst, (uint32_t)(((int32_t)(s0 << 8) >> 8) * ((int32_t)(s1 << 8) >> 8)) + s2); break; // mad_i32_i24
    case 0x143: L.wrV(vdst, (s0 & 0xFFFFFF) * (s1 & 0xFFFFFF) + s2); break;        // mad_u32_u24
    case 0x14a: L.wrV(vdst, (s2 >> (s0 & 31)) | (s1 << ((32 - s0) & 31))); break;  // alignbit (best-effort)
    case 0x15d: { uint32_t d = s0 > s1 ? s0 - s1 : s1 - s0; L.wrV(vdst, d + s2); break; }  // sad_u32 = |s0-s1|+s2
    case 0x169: case 0x16b: L.wrV(vdst, s0 * s1); break;                           // mul_lo_u32 / mul_lo_i32
    case 0x16a: case 0x16c: L.wrV(vdst, (uint32_t)(((uint64_t)s0 * s1) >> 32)); break; // mul_hi
    default: logUnknown(L, "vop3", op); break;
  }
}

// SOP2 (GFX6/7) scalar ALU.
void sop2(Lane &L, uint32_t op, uint32_t sdst, uint32_t s0, uint32_t s1) {
  switch (op) {
    case 0x00: { uint64_t r = (uint64_t)s0 + s1; L.wrS(sdst, (uint32_t)r); L.scc = r >> 32; break; } // add_u32
    case 0x02: { L.wrS(sdst, s0 - s1); L.scc = s0 < s1; break; }     // sub_u32
    case 0x04: L.wrS(sdst, s0 + s1); break;                          // add_i32
    case 0x0e: L.wrS(sdst, s0 & s1); L.scc = (s0 & s1) != 0; break;  // and_b32
    case 0x0f: { uint32_t r = s0 & s1; L.wrS(sdst, r);               // and_b64 (lane: AND exec/vcc)
                 if (sdst == 126) L.exec = r & 1; L.scc = r != 0; break; }
    case 0x1c: L.wrS(sdst, s0 & ~s1); break;                         // andn2
    case 0x1e: L.wrS(sdst, s0 << (s1 & 31)); break;                  // lshl_b32
    case 0x1f: L.wrS(sdst, s0 >> (s1 & 31)); break;                  // lshr_b32
    default: logUnknown(L, "sop2", op); break;
  }
}

// VOPC: vector compare -> VCC (s[106]) and, for this lane, EXEC gets ANDed by the
// following s_and. We store the bool in VCC; the bounds idiom then sets exec.
void vopc(Lane &L, uint32_t op, uint32_t s0, uint32_t s1) {
  uint32_t lo = op & 0xF; bool c = false;
  if (op >= 0x80 && op <= 0xBF) {  // i32
    int32_t a = (int32_t)s0, b = (int32_t)s1;
    switch (lo) { case 1: c=a<b; break; case 2: c=a==b; break; case 3: c=a<=b; break;
                  case 4: c=a>b; break; case 5: c=a!=b; break; case 6: c=a>=b; break; }
  } else if (op >= 0xC0) {  // u32
    switch (lo) { case 1: c=s0<s1; break; case 2: c=s0==s1; break; case 3: c=s0<=s1; break;
                  case 4: c=s0>s1; break; case 5: c=s0!=s1; break; case 6: c=s0>=s1; break; }
  } else {  // f32
    float a = u2f(s0), b = u2f(s1);
    switch (lo) { case 1: c=a<b; break; case 2: c=a==b; break; case 3: c=a<=b; break;
                  case 4: c=a>b; break; case 5: c=a!=b; break; case 6: c=a>=b; break; }
  }
  L.s[106] = c ? 1 : 0;
}

// Read a V#-described buffer element. For the copy shaders these are
// buffer_load_format_* (op 0..3) or buffer_load_ubyte/dword; we return the raw
// loaded bytes (the format-load/image-store round-trip is identity for a copy).
uint32_t mubufLoad(Lane &L, const Inst &in) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  uint32_t op = (w >> 18) & 0x7F, offset = w & 0xFFF;
  bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  uint32_t vaddr = w1 & 0xFF, srsrc = ((w1 >> 16) & 0x1F) * 4;
  uint32_t soff = L.rd((w1 >> 24) & 0xFF, in.literal);
  VBuffer vb = decodeVBuffer(&L.s[srsrc]);
  if (!guestOk(vb.base)) return 0;
  uint64_t addr = vb.base + soff + offset;
  if (idxen) addr += (uint64_t)L.v[vaddr] * (vb.stride ? vb.stride : 1);
  else if (offen) addr += L.v[vaddr];
  if (!guestOk(addr)) return 0;
  // Element size from the op: format_x / ubyte = 1, dword/format_xyzw = 4.
  uint32_t bytes = (op == 0x0c || op == 0x03) ? 4 : 1;
  uint32_t val = 0;
  std::memcpy(&val, reinterpret_cast<const void *>(addr), bytes);
  return val;
}

// Resolve a linear RGBA8 image texel. Array and mip coordinates are relative to
// the descriptor view; the returned pointer addresses the physical mip chain.
uint8_t *imagePixel(Lane &L, const Inst &in, TImage &t) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  bool mipOp = in.opcode == 0x01 || in.opcode == 0x09;
  uint32_t vaddr = w1 & 0xFF, srsrc = ((w1 >> 16) & 0x1F) * 4;
  t = decodeTImage(&L.s[srsrc]);
  uint64_t base = t.base;
  if (!t.valid || t.dfmt != 10 || (t.nfmt != 0 && t.nfmt != 4) ||
      !tilingIsLinear(t.tilingIdx) || !guestOk(base)) return nullptr;
  TextureLayout32 layout;
  if (!buildTextureLayout32(layout, t.width, t.height, t.pitch, t.layers,
                            t.mipLevels, t.tilingIdx, t.pow2Pad)) return nullptr;
  uint32_t viewMip = mipOp ? L.v[vaddr + ((w & 0x4000) ? 3 : 2)] : 0;
  viewMip = std::min(viewMip, t.viewMips - 1);
  uint32_t mip = t.baseMip + viewMip;
  const TextureMipLayout32 &level = layout.mips[mip];
  uint32_t x = L.v[vaddr], y = L.v[vaddr + 1], layer = 0;
  if (t.type == 13) {
    uint32_t viewLayer = (w & 0x4000) ? L.v[vaddr + 2] : 0;
    if (viewLayer >= t.viewLayers) return nullptr;
    layer = t.baseArray + viewLayer;
  }
  if (x >= level.width || y >= level.height || layer >= t.layers) return nullptr;
  uint64_t pixel = ((uint64_t)layer * level.storedHeight + y) * level.pitch + x;
  uint64_t addr = base + level.offset + pixel * 4;
  if (!guestOk(addr) || !guestOk(addr + 3)) return nullptr;
  return reinterpret_cast<uint8_t *>(addr);
}

uint32_t imageChannelLoad(uint8_t value, uint32_t nfmt) {
  if (nfmt == 4) return value;  // UINT
  float normalized = static_cast<float>(value) / 255.0f;
  uint32_t bits;
  std::memcpy(&bits, &normalized, sizeof(bits));
  return bits;
}

uint8_t imageChannelStore(uint32_t value, uint32_t nfmt) {
  if (nfmt == 4) return static_cast<uint8_t>(value & 0xFF);  // UINT
  float normalized;
  std::memcpy(&normalized, &value, sizeof(normalized));
  if (!std::isfinite(normalized)) normalized = 0.0f;
  normalized = std::clamp(normalized, 0.0f, 1.0f);
  return static_cast<uint8_t>(std::nearbyint(normalized * 255.0f));
}

void imageLoad(Lane &L, const Inst &in) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  uint32_t dmask = (w >> 8) & 0xF, vdata = (w1 >> 8) & 0xFF;
  TImage t;
  uint8_t *px = imagePixel(L, in, t);
  if (!px) return;
  uint32_t comp = 0;
  for (int i = 0; i < 4; i++)
    if (dmask & (1 << i)) L.wrV(vdata + comp++, imageChannelLoad(px[i], t.nfmt));
}

void imageStore(Lane &L, const Inst &in) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  uint32_t dmask = (w >> 8) & 0xF, vaddr = w1 & 0xFF;
  uint32_t vdata = (w1 >> 8) & 0xFF, srsrc = ((w1 >> 16) & 0x1F) * 4;
  TImage t;
  uint8_t *px = imagePixel(L, in, t);
  static int dbg = 0;
  if (std::getenv("DELTA_GPU_CSRUN_VERBOSE") && dbg < 4) {
    dbg++;
    uint32_t z = (w & 0x4000) ? L.v[vaddr + 2] : 0;
    std::fprintf(stderr, "[csimg] srsrc=%u base=%#lx %ux%ux%u pitch=%u dfmt=%u "
                         "nfmt=%u coord=[%u %u %u] valid=%d\n",
                 srsrc, (unsigned long)t.base, t.width, t.height, t.layers, t.pitch,
                 t.dfmt, t.nfmt, L.v[vaddr], L.v[vaddr + 1], z, px != nullptr);
  }
  if (!px) return;
  uint32_t comp = 0;
  for (int i = 0; i < 4; i++)
    if (dmask & (1 << i)) px[i] = imageChannelStore(L.v[vdata + comp++], t.nfmt);
}

void runLane(Lane &L, const std::vector<Inst> &insts) {
  for (size_t pc = 0; pc < insts.size(); pc++) {
    const Inst &in = insts[pc];
    uint32_t w = in.raw[0], w1 = in.raw[1];
    switch (in.enc) {
      case Enc::sop1: {
        uint32_t op = in.opcode, sdst = (w >> 16) & 0x7F, src0 = w & 0xFF;
        if (op == 0x03) L.wrS(sdst, L.rd(src0, in.literal));            // s_mov_b32
        else if (op == 0x24) {  // s_and_saveexec_b64: save exec, exec &= src
          uint32_t src = L.rd(src0, in.literal);
          L.wrS(sdst, L.exec); L.exec &= (src & 1); L.scc = L.exec;
        }
        break;
      }
      case Enc::sop2: {
        uint32_t sdst = (w >> 16) & 0x7F, s1 = L.rd((w >> 8) & 0xFF, in.literal),
                 s0 = L.rd(w & 0xFF, in.literal);
        sop2(L, in.opcode, sdst, s0, s1);
        break;
      }
      case Enc::smrd: {
        uint32_t op = in.opcode, sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
        bool imm = (w >> 8) & 1; uint32_t off = w & 0xFF;
        // s_load_dword* (0..4): load from the 2-dword pointer in s[sbase*2].
        // s_buffer_load* (8..): load from the V# in s[sbase*2].
        uint32_t n = op == 0x00 || op == 0x08 ? 1 : op == 0x01 || op == 0x09 ? 2
                   : op == 0x02 || op == 0x0a ? 4 : op == 0x03 || op == 0x0b ? 8 : 16;
        uint32_t base2 = sbase * 2;
        uint64_t addr = 0; uint32_t doff = imm ? off * 4 : L.rd(off, in.literal);
        if (op >= 0x08) {  // buffer load: V# base + offset
          VBuffer vb = decodeVBuffer(&L.s[base2]);
          addr = vb.base + doff;
        } else {           // plain load: 64-bit pointer
          addr = ((uint64_t)L.s[base2 + 1] << 32 | L.s[base2]) + doff;
        }
        if (guestOk(addr) && guestOk(addr + n * 4))
          for (uint32_t i = 0; i < n; i++)
            L.wrS(sdst + i, *reinterpret_cast<const uint32_t *>(addr + i * 4));
        break;
      }
      case Enc::vop1: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
        vop1(L, op, vdst, L.rd(src0, in.literal));
        break;
      }
      case Enc::vop2: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        vop2(L, op, vdst, L.rd(src0, in.literal), L.rd(256 + vsrc1, in.literal));
        break;
      }
      case Enc::vop3: {
        uint32_t op = in.opcode, vdst = w & 0xFF;
        uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF, s2 = (w1 >> 18) & 0x1FF;
        vop3(L, op, vdst, L.rd(s0, in.literal), L.rd(s1, in.literal), L.rd(s2, in.literal));
        break;
      }
      case Enc::vopc: {
        uint32_t op = in.opcode, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        vopc(L, op, L.rd(src0, in.literal), L.rd(256 + vsrc1, in.literal));
        break;
      }
      case Enc::mubuf: {
        uint32_t vdata = (w1 >> 8) & 0xFF;
        L.wrV(vdata, mubufLoad(L, in));
        break;
      }
      case Enc::mimg:
        if ((in.opcode == 0x00 || in.opcode == 0x01) && L.exec) imageLoad(L, in);
        else if ((in.opcode == 0x08 || in.opcode == 0x09) && L.exec) imageStore(L, in);
        break;
      case Enc::sopp:
        // s_cbranch_execz (op 0x08): if exec==0, jump past the (bounds-guarded)
        // store body. Our imageStore already bounds-checks, so a forward skip is a
        // safe no-op here; backward branches (loops) aren't used by these shaders.
        if (in.opcode == 0x01) return;  // s_endpgm
        break;
      default: break;
    }
  }
}

}  // namespace

uint64_t runComputeShader(uint64_t csAddr, uint32_t dimX, uint32_t dimY,
                          uint32_t dimZ, uint32_t tgX, uint32_t tgY, uint32_t tgZ,
                          uint32_t userSgpr, uint32_t tgidEnable,
                          const uint32_t *userData) {
  if (!guestOk(csAddr) || !tgX || !tgY) return 0;
  auto insts = decode(reinterpret_cast<const uint32_t *>(csAddr), 1024);
  if (insts.empty()) return 0;
  if (!dimZ) dimZ = 1; if (!tgZ) tgZ = 1;

  uint64_t stores = 0;
  for (uint32_t wz = 0; wz < dimZ; wz++)
  for (uint32_t wy = 0; wy < dimY; wy++)
  for (uint32_t wx = 0; wx < dimX; wx++)
    for (uint32_t lz = 0; lz < tgZ; lz++)
    for (uint32_t ly = 0; ly < tgY; ly++)
    for (uint32_t lx = 0; lx < tgX; lx++) {
      Lane L;
      for (uint32_t i = 0; i < userSgpr && i < 16; i++) L.s[i] = userData[i];
      // Workgroup id into the sgprs right after the user data, per tgid_enable.
      uint32_t sg = userSgpr;
      if (tgidEnable & 1) L.s[sg++] = wx;
      if (tgidEnable & 2) L.s[sg++] = wy;
      if (tgidEnable & 4) L.s[sg++] = wz;
      L.v[0] = lx; L.v[1] = ly; L.v[2] = lz;  // local invocation id (tidig)
      runLane(L, insts);
      stores++;
    }
  return stores;
}

}  // namespace gpu::gcn
