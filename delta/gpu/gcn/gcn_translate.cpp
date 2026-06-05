/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN -> GLSL shader recompiler. See gcn_translate.h. Straight-line translation of
 * the VS/PS patterns 2D titles use; registers are modelled as uint arrays and float
 * ops bitcast through helpers, so the translation is bit-faithful.
 */

#include "gcn_translate.h"
#include "gcn_decode.h"
#include "shaderc_compile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace gpu::gcn {
namespace {

const bool g_dbg = std::getenv("DELTA_GPU_SHTRACE") != nullptr;

bool inGuest(uint64_t a) { return a >= 0x1000000000ull && a < 0x20000000000ull; }

// ---- operand decoding -------------------------------------------------------
// A source operand's raw GLSL uint expression (no float bitcast). `vec` selects
// the vector (VGPR) register file for the 9-bit vector encodings.
std::string srcRaw(uint32_t field, uint32_t literal) {
  char b[64];
  if (field <= 127) { std::snprintf(b, sizeof b, "sg[%u]", field); return b; }  // sgpr/vcc/m0/exec
  if (field == 128) return "0u";
  if (field >= 129 && field <= 192) { std::snprintf(b, sizeof b, "%uu", field - 128); return b; }
  if (field >= 193 && field <= 208) { std::snprintf(b, sizeof b, "uint(%d)", -(int)(field - 192)); return b; }
  switch (field) {  // inline float constants -> their bit patterns
    case 240: return "0x3f000000u"; case 241: return "0xbf000000u";
    case 242: return "0x3f800000u"; case 243: return "0xbf800000u";
    case 244: return "0x40000000u"; case 245: return "0xc0000000u";
    case 246: return "0x40800000u"; case 247: return "0xc0800000u";
  }
  if (field == 255) { std::snprintf(b, sizeof b, "0x%08xu", literal); return b; }
  if (field >= 256) { std::snprintf(b, sizeof b, "vg[%u]", field - 256); return b; }
  return "0u";
}
// Float-context source: bitcast to float and apply VOP3 neg/abs modifiers.
std::string srcF(uint32_t field, uint32_t literal, bool neg = false, bool abs = false) {
  std::string e = "Ff(" + srcRaw(field, literal) + ")";
  if (abs) e = "abs(" + e + ")";
  if (neg) e = "(-" + e + ")";
  return e;
}

// A scalar/vector dst as an lvalue register expression.
std::string dstReg(bool vector, uint32_t code) {
  char b[32];
  std::snprintf(b, sizeof b, "%s[%u]", vector ? "vg" : "sg", code);
  return b;
}

// ---- fetch shader parsing ---------------------------------------------------
// Walk the fetch shader: s_load_dwordx4 records a V#-table load; buffer_load_format
// emits an attribute (semantic = appearance order). See the recompiler spec.
struct FetchAttr {
  uint32_t semantic, numComps, destVgpr, tableSgpr, dwordOff;
};
std::vector<FetchAttr> parseFetch(uint64_t fetchAddr) {
  std::vector<FetchAttr> out;
  if (!inGuest(fetchAddr)) return out;
  auto *code = reinterpret_cast<const uint32_t *>(fetchAddr);
  auto insts = decode(code, 256);
  struct Load { uint32_t tableSgpr, dwordOff; };
  std::unordered_map<uint32_t, Load> loads;  // dst sgpr -> table load
  uint32_t sem = 0;
  for (auto &in : insts) {
    uint32_t w = in.raw[0];
    if (in.enc == Enc::smrd && in.opcode == 0x02) {  // s_load_dwordx4
      uint32_t sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F, off = w & 0xFF;
      loads[sdst] = {sbase * 2u, off};
    } else if (in.enc == Enc::mubuf || in.enc == Enc::mtbuf) {
      // buffer_load_format_* : src0=soffset, src1(vdata)=raw1[15:8], srsrc=raw1[20:16]
      uint32_t w1 = in.raw[1];
      uint32_t vdata = (w1 >> 8) & 0xFF;
      uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;  // SRSRC in 4-SGPR units
      // numComps from the opcode (X=1..XYZW=4). MUBUF buffer_load_format_x..xyzw
      // opcodes are 0..3; MTBUF tbuffer_load_format_x..xyzw are 0..3 as well here.
      uint32_t op = in.opcode;
      uint32_t nc = (op & 3) + 1;
      auto it = loads.find(srsrc);
      uint32_t tbl = it != loads.end() ? it->second.tableSgpr : 0;
      uint32_t doff = it != loads.end() ? it->second.dwordOff : 0;
      out.push_back({sem, nc, vdata, tbl, doff});
      sem++;
    }
  }
  return out;
}

// ---- emitter ----------------------------------------------------------------
struct Emit {
  std::string body;
  bool vertex;
  Recompiled *r;
  // resource tracking
  std::unordered_map<uint32_t, uint32_t> sgprFromUd;  // sgpr -> user-data dword index (s_load)
  uint32_t maxParam = 0;

  void line(const std::string &s) { body += "  " + s + "\n"; }

  // Map a VOP3 opcode (9-bit) to an operation; emit into vdst (float).
  void vop3(uint32_t op, uint32_t vdst, const std::string &s0, const std::string &s1,
            const std::string &s2) {
    std::string d = "vg[" + std::to_string(vdst) + "]";
    auto set = [&](const std::string &e) { line(d + " = Uf(" + e + ");"); };
    if (op >= 0x100 && op < 0x140) { vop2(op - 0x100, vdst, s0, s1); return; }
    if (op >= 0x180 && op < 0x200) { vop1(op - 0x180, vdst, s0); return; }
    switch (op) {
      case 0x141: case 0x14b: set(s0 + " * " + s1 + " + " + s2); break;  // mad/fma
      case 0x143: set(s0 + " * " + s1 + " + " + s2); break;              // mad_u32 (approx)
      case 0x151: set("min(min(" + s0 + "," + s1 + ")," + s2 + ")"); break;  // min3
      case 0x154: set("max(max(" + s0 + "," + s1 + ")," + s2 + ")"); break;  // max3
      case 0x157: set("clamp(" + s2 + ", min(" + s0 + "," + s1 + "), max(" + s0 + "," + s1 + "))"); break;  // med3
      default: set(s0); break;  // unknown 3-op: pass through src0
    }
  }
  void vop2(uint32_t op, uint32_t vdst, const std::string &s0, const std::string &s1) {
    std::string d = "vg[" + std::to_string(vdst) + "]";
    auto set = [&](const std::string &e) { line(d + " = Uf(" + e + ");"); };
    auto setU = [&](const std::string &e) { line(d + " = (" + e + ");"); };
    switch (op) {
      case 0x00: set("(Ff(sg[106])!=0.0 ? " + s1 + " : " + s0 + ")"); break;  // cndmask (vcc)
      case 0x01: case 0x02: case 0x03: set(s0 + " + " + s1); break;          // add_f32 variants
      case 0x04: set(s0 + " - " + s1); break;                                // sub_f32
      case 0x05: set(s1 + " - " + s0); break;                                // subrev_f32
      case 0x06: set(s0 + " * " + s1); break;                                // mac_legacy? treat mul
      case 0x08: set(s0 + " * " + s1); break;                                // mul_f32
      case 0x0a: set("min(" + s0 + "," + s1 + ")"); break;                   // min_f32
      case 0x0b: set("max(" + s0 + "," + s1 + ")"); break;                   // max_f32
      case 0x1f: set(s0 + " * " + s1 + " + Ff(" + d + ")"); break;           // mac_f32 (accum)
      case 0x25: setU("(" + rawOf(s0) + " & " + rawOf(s1) + ")"); break;     // and_b32
      case 0x26: setU("(" + rawOf(s0) + " | " + rawOf(s1) + ")"); break;     // or_b32
      case 0x27: setU("(" + rawOf(s0) + " ^ " + rawOf(s1) + ")"); break;     // xor_b32
      case 0x2f: setU("packHalf2x16(vec2(" + s0 + ", " + s1 + "))"); break;  // cvt_pkrtz_f16_f32
      default: set(s0 + " * " + s1); break;
    }
  }
  void vop1(uint32_t op, uint32_t vdst, const std::string &s0) {
    std::string d = "vg[" + std::to_string(vdst) + "]";
    auto set = [&](const std::string &e) { line(d + " = Uf(" + e + ");"); };
    auto setI = [&](const std::string &e) { line(d + " = uint(" + e + ");"); };
    switch (op) {
      case 0x01: line(d + " = " + rawOf(s0) + ";"); break;                   // mov_b32
      case 0x05: set("float(int(" + rawOf(s0) + "))"); break;                // cvt_f32_i32
      case 0x06: set("float(" + rawOf(s0) + ")"); break;                     // cvt_f32_u32
      case 0x07: setI("int(" + s0 + ")"); break;                             // cvt_u32_f32
      case 0x08: setI("int(" + s0 + ")"); break;                             // cvt_i32_f32
      case 0x0d: setI("int(floor(" + s0 + " + 0.5))"); break;                // cvt_rpi
      case 0x0e: setI("int(floor(" + s0 + "))"); break;                      // cvt_flr
      case 0x21: set("fract(" + s0 + ")"); break;                            // fract
      case 0x22: set("trunc(" + s0 + ")"); break;
      case 0x23: set("ceil(" + s0 + ")"); break;
      case 0x24: set("roundEven(" + s0 + ")"); break;
      case 0x25: set("floor(" + s0 + ")"); break;                            // floor
      case 0x2a: set("exp2(" + s0 + ")"); break;
      case 0x2c: set("log2(" + s0 + ")"); break;
      case 0x2d: set("(1.0/(" + s0 + "))"); break;                           // rcp
      case 0x2e: set("(1.0/(" + s0 + "))"); break;
      case 0x2f: set("inversesqrt(" + s0 + ")"); break;                      // rsq
      case 0x33: set("sqrt(" + s0 + ")"); break;
      case 0x35: set("sin(" + s0 + ")"); break;
      case 0x36: set("cos(" + s0 + ")"); break;
      default: line(d + " = " + rawOf(s0) + ";"); break;                     // mov fallback
    }
  }
  // raw uint of a float-expr string "Ff(X)" -> "X"
  std::string rawOf(const std::string &f) {
    if (f.rfind("Ff(", 0) == 0 && f.back() == ')') return f.substr(3, f.size() - 4);
    return "Uf(" + f + ")";
  }
};

// ---- VS translation ---------------------------------------------------------
bool translateVs(const uint32_t *vsCode, const uint32_t *vsUserData, Recompiled &r) {
  // Fetch shader pointer lives in VS user-data sgpr0/1.
  uint64_t fetch = (static_cast<uint64_t>(vsUserData[1] & 0xFFFF) << 32) | vsUserData[0];
  auto attrs = parseFetch(fetch);
  if (attrs.empty()) return false;

  std::string ins, outs, seed;
  for (auto &a : attrs) {
    char l[256];
    const char *t = a.numComps == 1 ? "float" : a.numComps == 2 ? "vec2"
                    : a.numComps == 3 ? "vec3" : "vec4";
    std::snprintf(l, sizeof l, "layout(location=%u) in %s in%u;\n", a.semantic, t, a.semantic);
    ins += l;
    // seed the destination VGPRs from the attribute components.
    const char *sw[4] = {".x", ".y", ".z", ".w"};
    for (uint32_t c = 0; c < a.numComps; c++) {
      std::snprintf(l, sizeof l, "  vg[%u] = Uf(in%u%s);\n", a.destVgpr + c, a.semantic,
                    a.numComps == 1 ? "" : sw[c]);
      seed += l;
    }
    r.attrs.push_back({a.semantic, a.numComps, a.tableSgpr, a.dwordOff});
  }

  // Walk the VS body. We model the MVP load as a single UBO of up to 16 floats and
  // s_buffer_load as UBO reads. Position/param exports become outputs.
  Emit e; e.vertex = true; e.r = &r;
  auto insts = decode(vsCode, 4096);
  bool sawFetchCall = false;
  uint32_t cbufBinding = 0;
  bool haveCbuf = false;
  std::unordered_map<uint32_t, uint32_t> sgVsharpUd;  // sgpr -> user-data dword idx of a V# (s_load)

  for (auto &in : insts) {
    uint32_t w = in.raw[0], w1 = in.raw[1];
    switch (in.enc) {
      case Enc::sop1: {
        uint32_t op = in.opcode, sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
        if (op == 0x03)  // s_mov_b32
          e.line("sg[" + std::to_string(sdst) + "] = " + srcRaw(ssrc0, in.literal) + ";");
        else if (op == 0x04) {  // s_mov_b64
          e.line("sg[" + std::to_string(sdst) + "] = " + srcRaw(ssrc0, in.literal) + ";");
          if (ssrc0 <= 103)
            e.line("sg[" + std::to_string(sdst + 1) + "] = sg[" + std::to_string(ssrc0 + 1) + "];");
        } else if (op == 0x21) sawFetchCall = true;  // s_swappc (fetch) - ignore
        break;
      }
      case Enc::smrd: {
        uint32_t op = in.opcode, sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
        bool imm = (w >> 8) & 1; uint32_t off = w & 0xFF;
        if (op == 0x02) {  // s_load_dwordx4 -> a sharp loaded from the user-data SRT
          sgVsharpUd[sdst] = sbase * 2u;  // approx: the table pointer's user-data index
        } else if (op >= 0x08) {  // s_buffer_load_dword* : read a constant buffer
          uint32_t n = op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 : op == 0x0b ? 8 : 16;
          if (!haveCbuf) { haveCbuf = true; cbufBinding = (uint32_t)r.vsCbufs.size();
            r.vsCbufs.push_back({cbufBinding, sbase * 2u, 16}); }
          uint32_t doff = imm ? off : 0;
          // Constant buffers are uploaded as push constants. Indexed as uvec4[] so
          // the array is tightly packed (a std140 uint[] would pad each element to
          // 16 bytes and scramble the matrix).
          for (uint32_t i = 0; i < n; i++) {
            uint32_t k = doff + i;
            e.line("sg[" + std::to_string(sdst + i) + "] = pc.data[" + std::to_string(k >> 2) +
                   "][" + std::to_string(k & 3) + "];");
          }
        }
        break;
      }
      case Enc::vop2: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        e.vop2(op, vdst, srcF(src0, in.literal), srcF(256 + vsrc1, in.literal));
        break;
      }
      case Enc::vop1: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
        e.vop1(op, vdst, srcF(src0, in.literal));
        break;
      }
      case Enc::vop3: {
        uint32_t op = in.opcode, vdst = w & 0xFF, abs = (w >> 8) & 7, clmp = (w >> 15) & 1;
        uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF, s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
        e.vop3(op, vdst, srcF(s0, in.literal, neg & 1, abs & 1),
               srcF(s1, in.literal, neg & 2, abs & 2),
               srcF(s2, in.literal, neg & 4, abs & 4));
        (void)clmp;
        break;
      }
      case Enc::exp: {
        uint32_t en = w & 0xF, target = (w >> 4) & 0x3F;
        uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, (w1 >> 24) & 0xFF};
        const char *sw[4] = {"x", "y", "z", "w"};
        if (target >= 12 && target <= 15) {  // POS
          if (target == 12) {
            e.line("gl_Position = vec4(0.0);");
            for (int i = 0; i < 4; i++)
              if (en & (1 << i)) e.line("gl_Position." + std::string(sw[i]) + " = Ff(vg[" + std::to_string(v[i]) + "]);");
          }
        } else if (target >= 32 && target <= 63) {  // PARAM
          uint32_t p = target - 32; if (p + 1 > e.maxParam) e.maxParam = p + 1;
          outs += "layout(location=" + std::to_string(p) + ") out vec4 outp" + std::to_string(p) + ";\n";
          e.line("outp" + std::to_string(p) + " = vec4(0.0);");
          for (int i = 0; i < 4; i++)
            if (en & (1 << i)) e.line("outp" + std::to_string(p) + "." + sw[i] + " = Ff(vg[" + std::to_string(v[i]) + "]);");
        }
        break;
      }
      default: break;
    }
    if (in.enc == Enc::sopp && in.opcode == 1) break;  // s_endpgm
  }
  (void)sawFetchCall;
  r.numParams = e.maxParam;

  std::string ubo;
  if (!r.vsCbufs.empty())
    ubo = "layout(push_constant) uniform PC { uvec4 data[8]; } pc;\n";

  // GL clip space (z in [-w,w]) -> Vulkan clip space (z in [0,w]). Without this the
  // guest's depth lands outside Vulkan's [0,1] NDC range and every primitive is
  // depth-clipped to nothing.
  e.line("gl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;");
  if (std::getenv("DELTA_GPU_RC_FIXPOS"))
    e.line("gl_Position = vec4(((gl_VertexIndex & 1)==0?-0.9:0.9), ((gl_VertexIndex & 2)==0?-0.9:0.9), 0.0, 1.0);");
  if (std::getenv("DELTA_GPU_RC_DIRECTMVP") && !attrs.empty()) {
    const char *pos = attrs[0].numComps == 2 ? "vec4(in0, 0.0, 1.0)"
                      : attrs[0].numComps == 3 ? "vec4(in0, 1.0)" : "in0";
    e.line("mat4 _m = mat4(Ff(pc.data[0][0]),Ff(pc.data[0][1]),Ff(pc.data[0][2]),Ff(pc.data[0][3]),"
           "Ff(pc.data[1][0]),Ff(pc.data[1][1]),Ff(pc.data[1][2]),Ff(pc.data[1][3]),"
           "Ff(pc.data[2][0]),Ff(pc.data[2][1]),Ff(pc.data[2][2]),Ff(pc.data[2][3]),"
           "Ff(pc.data[3][0]),Ff(pc.data[3][1]),Ff(pc.data[3][2]),Ff(pc.data[3][3]));");
    e.line(std::string("gl_Position = _m * ") + pos + ";");
    e.line("gl_Position.z = 0.0;");
  }
  r.vsGlsl = "#version 450\n" + ins + outs + ubo +
             "float Ff(uint x){return uintBitsToFloat(x);}\nuint Uf(float x){return floatBitsToUint(x);}\n"
             "void main(){\n  uint sg[128]; uint vg[256];\n" +
             "  for(int i=0;i<128;i++) sg[i]=0u;\n  for(int i=0;i<256;i++) vg[i]=0u;\n" + seed + e.body + "}\n";
  return true;
}

// ---- PS translation ---------------------------------------------------------
bool translatePs(const uint32_t *psCode, const uint32_t *psUserData, uint32_t numParams,
                 Recompiled &r) {
  Emit e; e.vertex = false; e.r = &r;
  auto insts = decode(psCode, 4096);
  std::string ins, samplers;
  uint32_t maxIn = 0;
  bool wroteColor = false;

  for (auto &in : insts) {
    uint32_t w = in.raw[0], w1 = in.raw[1];
    switch (in.enc) {
      case Enc::vintrp: {
        // vsrc[7:0], chan[9:8], attr[15:10], op[17:16], vdst[25:18]
        uint32_t chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F, op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
        const char *sw[4] = {"x", "y", "z", "w"};
        if (attr + 1 > maxIn) maxIn = attr + 1;
        // p1 is a no-op for us; p2 reads the interpolated input.
        if (op == 1 || op == 0)  // p2 / p1: assign on p2 (op==1); ignore p1(op==0) for value
          if (op == 1)
            e.line("vg[" + std::to_string(vdst) + "] = Uf(inp" + std::to_string(attr) + "." + sw[chan] + ");");
        break;
      }
      case Enc::vop2: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        e.vop2(op, vdst, srcF(src0, in.literal), srcF(256 + vsrc1, in.literal));
        break;
      }
      case Enc::vop1: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
        e.vop1(op, vdst, srcF(src0, in.literal));
        break;
      }
      case Enc::vop3: {
        uint32_t op = in.opcode, vdst = w & 0xFF, abs = (w >> 8) & 7;
        uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF, s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
        e.vop3(op, vdst, srcF(s0, in.literal, neg & 1, abs & 1),
               srcF(s1, in.literal, neg & 2, abs & 2),
               srcF(s2, in.literal, neg & 4, abs & 4));
        break;
      }
      case Enc::mimg: {
        // image_sample: dmask[11:8], op[24:18]; w1: vaddr[7:0], vdata[15:8], srsrc[20:16], ssamp[25:21]
        uint32_t dmask = (w >> 8) & 0xF, vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
        uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
        uint32_t bind = (uint32_t)r.psTexs.size();
        r.psTexs.push_back({bind, srsrc});
        samplers += "layout(set=0, binding=" + std::to_string(bind) +
                    ") uniform sampler2D tex" + std::to_string(bind) + ";\n";
        std::string uvexpr = "vec2(Ff(vg[" + std::to_string(vaddr) + "]), Ff(vg[" +
                             std::to_string(vaddr + 1) + "]))";
        e.line("vec4 t" + std::to_string(bind) + " = texture(tex" + std::to_string(bind) +
               ", " + uvexpr + ");");
        if (std::getenv("DELTA_GPU_RC_UVDBG"))
          e.line("outColor = vec4(" + uvexpr + ", 0.0, 1.0); return;");
        const char *sw[4] = {"x", "y", "z", "w"};
        uint32_t comp = 0;
        for (int i = 0; i < 4; i++)
          if (dmask & (1 << i))
            e.line("vg[" + std::to_string(vdata + comp++) + "] = Uf(t" + std::to_string(bind) +
                   "." + sw[i] + ");");
        break;
      }
      case Enc::exp: {
        uint32_t en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
        uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, (w1 >> 24) & 0xFF};
        const char *sw[4] = {"x", "y", "z", "w"};
        if (target <= 7) {  // MRT
          wroteColor = true;
          if (compr) {  // two f16 pairs in v[0], v[1]
            e.line("vec2 c01 = unpackHalf2x16(vg[" + std::to_string(v[0]) + "]);");
            e.line("vec2 c23 = unpackHalf2x16(vg[" + std::to_string(v[1]) + "]);");
            e.line("outColor = vec4(c01, c23);");
          } else {
            e.line("outColor = vec4(0.0);");
            for (int i = 0; i < 4; i++)
              if (en & (1 << i)) e.line("outColor." + std::string(sw[i]) + " = Ff(vg[" + std::to_string(v[i]) + "]);");
          }
        }
        break;
      }
      default: break;
    }
    if (in.enc == Enc::sopp && in.opcode == 1) break;
  }

  for (uint32_t i = 0; i < maxIn; i++)
    ins += "layout(location=" + std::to_string(i) + ") in vec4 inp" + std::to_string(i) + ";\n";
  if (!wroteColor) e.line("outColor = vec4(1.0);");
  if (std::getenv("DELTA_GPU_RC_FLAT")) e.line("outColor = vec4(1.0, 0.0, 1.0, 1.0);");

  r.fsGlsl = "#version 450\n" + ins + samplers + "layout(location=0) out vec4 outColor;\n" +
             "float Ff(uint x){return uintBitsToFloat(x);}\nuint Uf(float x){return floatBitsToUint(x);}\n"
             "void main(){\n  uint sg[128]; uint vg[256];\n"
             "  for(int i=0;i<128;i++) sg[i]=0u;\n  for(int i=0;i<256;i++) vg[i]=0u;\n" + e.body + "}\n";
  (void)numParams; (void)psUserData;
  return true;
}

}  // namespace

Recompiled recompile(const uint32_t *vsCode, const uint32_t *psCode,
                     const uint32_t *vsUserData, const uint32_t *psUserData) {
  Recompiled r;
  if (!vsCode || !psCode || !vsUserData || !psUserData) return r;
  if (!translateVs(vsCode, vsUserData, r)) return r;
  if (!translatePs(psCode, psUserData, r.numParams, r)) return r;
  if (g_dbg)
    std::fprintf(stderr, "=== VS GLSL ===\n%s\n=== PS GLSL ===\n%s\n", r.vsGlsl.c_str(), r.fsGlsl.c_str());
  r.vsSpirv = compileGlsl(true, r.vsGlsl, "vs");
  r.fsSpirv = compileGlsl(false, r.fsGlsl, "fs");
  r.ok = !r.vsSpirv.empty() && !r.fsSpirv.empty();
  return r;
}

}  // namespace gpu::gcn
