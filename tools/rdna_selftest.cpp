/*
 * Standalone validation harness for the PS5 RDNA2 (gfx10.3) shader recompiler.
 * Hand-assembles minimal RDNA2 VS/PS programs, decodes them (rdna_decode),
 * recompiles them to SPIR-V (rdna_translate, reusing the shared gpu::gcn
 * backend), and validates the emitted binaries with SPIRV-Tools. Not part of the
 * emulator build; compiled directly (from the repo root), e.g.:
 *
 *   nix develop -c bash tools/build_rdna_selftest.sh
 *
 * The guest (Isaac) does not yet submit AGC DCBs, so this harness is the
 * regression check for the decoder + recompiler until submission is unblocked.
 */

#include <cstdint>
#include <cstdio>
#include <vector>

#include "ps5/rdna/rdna_decode.h"
#include "ps5/rdna/rdna_resource.h"
#include "ps5/rdna/rdna_translate.h"
#include "ps4/gcn/spirv/spv_post.h"

using gpu::gcn::Enc;

namespace {

int g_failures = 0;
void expect(bool cond, const char* what) {
  std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
  if (!cond) g_failures++;
}

// True if the SPIR-V word stream contains an instruction with the given opcode.
// Each instruction's first word packs wordCount[31:16] | opcode[15:0].
bool hasOpcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
  size_t i = 5;  // skip the 5-word module header
  while (i < spv.size()) {
    const uint32_t wc = spv[i] >> 16;
    if (wc == 0) break;
    if ((spv[i] & 0xFFFF) == opcode) return true;
    i += wc;
  }
  return false;
}

// True if the module decorates any id with BuiltIn `builtin` (OpDecorate == 71,
// Decoration BuiltIn == 11, then the BuiltIn enum). Used to prove a PS input
// VGPR was seeded from a Vulkan built-in (gl_FragCoord == 15).
bool hasBuiltin(const std::vector<uint32_t>& spv, uint32_t builtin) {
  size_t i = 5;
  while (i < spv.size()) {
    const uint32_t wc = spv[i] >> 16;
    if (wc == 0) break;
    if ((spv[i] & 0xFFFF) == 71 && wc >= 4 && spv[i + 2] == 11 &&
        spv[i + 3] == builtin)
      return true;
    i += wc;
  }
  return false;
}

// ---- RDNA2 instruction encoders (little bit-twiddling for readability) ------
// VOP1: [31:25]=0x3F, vdst[24:17], op[16:9], src0[8:0].
uint32_t vop1(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF);
}
// VOP2: [31]=0, op[30:25], vdst[24:17], vsrc1[16:9], src0[8:0].
uint32_t vop2(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t vsrc1) {
  return ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) |
         (src0 & 0x1FF);
}
// VOP3: word0 [31:26]=0x35, op[25:16], clamp[15], abs[10:8], vdst[7:0];
// word1: neg[31:29], src2[26:18], src1[17:9], src0[8:0].
void vop3(std::vector<uint32_t>& out, uint32_t op, uint32_t vdst, uint32_t s0,
          uint32_t s1, uint32_t s2) {
  out.push_back((0x35u << 26) | ((op & 0x3FF) << 16) | (vdst & 0xFF));
  out.push_back(((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF));
}
// VOPC: [31:25]=0x3E, op[24:17], vsrc1[16:9], src0[8:0] (writes VCC).
uint32_t vopc(uint32_t op, uint32_t src0, uint32_t vsrc1) {
  return (0x3Eu << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) |
         (src0 & 0x1FF);
}
// EXP: [31:26]=0x3E, done[11], compr[10], target[9:4], en[3:0]; word1 = 4 VGPRs.
void exp(std::vector<uint32_t>& out, uint32_t target, uint32_t en, bool done,
         uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  out.push_back((0x3Eu << 26) | ((done ? 1u : 0u) << 11) | ((target & 0x3F) << 4) |
                (en & 0xF));
  out.push_back((v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) |
                ((v3 & 0xFF) << 24));
}
// SOPP: [31:23]=0x17F, op[22:16], simm[15:0].
uint32_t sopp(uint32_t op, uint32_t simm) {
  return (0x17Fu << 23) | ((op & 0x7F) << 16) | (simm & 0xFFFF);
}
// SMEM: [31:26]=0x3D, op[25:18], sdst[12:6], sbase[5:0]; word1 imm[20:0].
void smem(std::vector<uint32_t>& out, uint32_t op, uint32_t sdst, uint32_t sbase,
          uint32_t imm) {
  out.push_back((0x3Du << 26) | ((op & 0xFF) << 18) | ((sdst & 0x7F) << 6) |
                (sbase & 0x3F));
  out.push_back(imm & 0x1FFFFF);
}

// MIMG (NSA=0, 64-bit): word0 [31:26]=0x3C, op[24:18], da[14], dmask[11:8],
// op[7] at bit 0; word1 ssamp[25:21], srsrc[20:16], vdata[15:8], vaddr[7:0].
// srsrc/ssamp are the SGPR index >> 2 (4-SGPR-aligned).
void mimg(std::vector<uint32_t>& out, uint32_t op, uint32_t dmask, uint32_t vdata,
          uint32_t vaddr, uint32_t srsrc, uint32_t ssamp) {
  out.push_back((0x3Cu << 26) | ((op & 0x7F) << 18) | ((dmask & 0xF) << 8) |
                ((op >> 7) & 1));
  out.push_back(((ssamp & 0x1F) << 21) | ((srsrc & 0x1F) << 16) |
                ((vdata & 0xFF) << 8) | (vaddr & 0xFF));
}

// VOP3P: word0 [31:26]=0x33, op[22:16], vdst[7:0]; word1 src2[26:18],
// src1[17:9], src0[8:0] (same source layout as VOP3).
void vop3p(std::vector<uint32_t>& out, uint32_t op, uint32_t vdst, uint32_t s0,
           uint32_t s1, uint32_t s2) {
  out.push_back((0x33u << 26) | ((op & 0x7F) << 16) | (vdst & 0xFF));
  out.push_back(((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF));
}

constexpr uint32_t kInline0 = 128;    // integer/float 0
constexpr uint32_t kInline1f = 242;   // float 1.0
constexpr uint32_t kEndpgm = 1;

}  // namespace

int main() {
  std::printf("== RDNA2 decoder ==\n");
  {
    // v_mov_b32 v0, 0.0 ; v_mov_b32 v3, 1.0 ; exp pos0 ; s_endpgm
    std::vector<uint32_t> code;
    code.push_back(vop1(0x01, 0, kInline0));
    code.push_back(vop1(0x01, 3, kInline1f));
    exp(code, /*POS0*/ 12, 0xF, true, 0, 0, 0, 3);
    code.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Program prog = gpu::rdna::Decode(code.data(),
                                                     (uint32_t)code.size());
    expect(prog.size() == 4, "decoded 4 instructions");
    expect(prog.size() >= 1 && prog[0].enc == Enc::kVop1 && prog[0].opcode == 0x01,
           "inst0 is VOP1 v_mov_b32");
    expect(prog.size() >= 3 && prog[2].enc == Enc::kExp && prog[2].size == 2,
           "inst2 is EXP (2 dwords)");
    expect(prog.size() >= 4 && prog[3].enc == Enc::kSopp && prog[3].opcode == 1,
           "inst3 is SOPP s_endpgm");
  }
  {
    // SMEM s_buffer_load_dwordx4 is a 2-dword instruction.
    std::vector<uint32_t> code;
    smem(code, /*s_buffer_load_dwordx4*/ 0x0A, /*sdst*/ 8, /*sbase*/ 2, /*imm*/ 0);
    code.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Program prog = gpu::rdna::Decode(code.data(),
                                                     (uint32_t)code.size());
    expect(prog.size() == 2 && prog[0].enc == Enc::kSmrd && prog[0].size == 2 &&
               prog[0].opcode == 0x0A,
           "SMEM s_buffer_load_dwordx4 decodes as 2-dword kSmrd");
  }

  std::printf("== RDNA2 -> SPIR-V recompile ==\n");
  {
    // Procedural VS: position = (0,0,0,1).
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));   // v0 = 0.0
    vs.push_back(vop1(0x01, 1, kInline0));   // v1 = 0.0
    vs.push_back(vop1(0x01, 2, kInline0));   // v2 = 0.0
    vs.push_back(vop1(0x01, 3, kInline1f));  // v3 = 1.0
    exp(vs, /*POS0*/ 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    // PS: color = (1,1,1,1) -> MRT0. (v_add_f32 exercises VOP2.)
    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline1f));         // v0 = 1.0
    ps.push_back(vop2(0x03, 1, kInline1f, 0));      // v1 = 1.0 + v0 = 2.0 (VOP2 add)
    ps.push_back(vop1(0x01, 2, kInline1f));         // v2 = 1.0
    ps.push_back(vop1(0x01, 3, kInline1f));         // v3 = 1.0
    exp(ps, /*MRT0*/ 0, 0xF, true, 0, 0, 2, 3);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};  // no fetch shader -> procedural path
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "VS+PS recompiled ok");
    expect(!r.vs_spirv.empty(), "VS SPIR-V non-empty");
    expect(!r.fs_spirv.empty(), "PS SPIR-V non-empty");
    expect(r.ps_mrt_mask & 1u, "PS exports MRT0");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.vs_spirv, &err), "VS SPIR-V validates");
    if (!err.empty()) std::printf("      vs: %s\n", err.c_str());
    err.clear();
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err), "PS SPIR-V validates");
    if (!err.empty()) std::printf("      ps: %s\n", err.c_str());
  }

  {
    // VS with a constant buffer + VOP3: load cbuffer[0..3] into s4.., move a
    // cbuf dword into a VGPR, v_mad_f32 (VOP3 0x141), export POS0. Exercises
    // RdnaPlanCbufs / RdnaEmitSmem and the VOP3 field decode.
    std::vector<uint32_t> vs;
    smem(vs, /*s_buffer_load_dwordx4*/ 0x0A, /*sdst s4*/ 4, /*sbase sgpr2*/ 1, 0);
    vs.push_back(vop1(0x01, 0, 4));           // v0 = s4 (cbuffer dword 0)
    vs.push_back(vop1(0x01, 1, kInline1f));   // v1 = 1.0
    vs.push_back(vop1(0x01, 2, kInline0));    // v2 = 0.0
    vs.push_back(vop1(0x01, 3, kInline1f));   // v3 = 1.0
    vop3(vs, /*v_mad_f32*/ 0x141, 0, 256, 257, 258);  // v0 = v0*v1 + v2
    exp(vs, /*POS0*/ 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline1f));
    exp(ps, /*MRT0*/ 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "VS(cbuf+VOP3)+PS recompiled ok");
    expect(r.vs_cbufs.size() == 1, "one VS constant buffer planned");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.vs_spirv, &err), "cbuf VS SPIR-V validates");
    if (!err.empty()) std::printf("      vs: %s\n", err.c_str());
  }

  {
    // PS exercising v_cndmask_b32 (VOP2 0x01) + v_add_nc_u32 (0x25): the shared
    // GFX7 emitter numbers these differently, so this guards the RDNA2 remap.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline0));       // v0 = 0.0
    ps.push_back(vop1(0x01, 1, kInline1f));      // v1 = 1.0
    ps.push_back(vopc(0x01, 256, 257));          // v_cmp_lt_f32 v0,v1 -> VCC
    ps.push_back(vop2(0x01, 2, 256, 257));       // v_cndmask v2 = VCC ? v1 : v0
    ps.push_back(vop2(0x25, 3, 256, 257));       // v_add_nc_u32 v3 = v0 + v1
    exp(ps, 0, 0xF, true, 2, 2, 2, 3);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "cndmask/add_nc PS recompiled ok");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err), "cndmask PS SPIR-V validates");
    if (!err.empty()) std::printf("      ps: %s\n", err.c_str());
  }

  {
    // RDNA2 VOP3-only integer ops (native add/sub_i32 + 3-input fused forms).
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline0));      // v0 = 0
    ps.push_back(vop1(0x01, 1, kInline1f));     // v1 = 1.0 bits
    vop3(ps, /*v_add_i32*/ 0x30F, 4, 256, 257, 0);      // v4 = v0 + v1
    vop3(ps, /*v_sub_i32*/ 0x310, 5, 256, 257, 0);      // v5 = v0 - v1
    vop3(ps, /*v_subrev_i32*/ 0x319, 6, 256, 257, 0);   // v6 = v1 - v0
    vop3(ps, /*v_add3_u32*/ 0x36D, 7, 256, 257, 260);   // v7 = v0+v1+v4
    vop3(ps, /*v_lshl_or_b32*/ 0x36F, 8, 256, 257, 260);   // v8 = (v0<<v1)|v4
    vop3(ps, /*v_lshl_add_u32*/ 0x346, 9, 256, 257, 260);  // v9 = (v0<<v1)+v4
    vop3(ps, /*v_add_lshl_u32*/ 0x347, 10, 256, 257, 260); // v10 = (v0+v1)<<v4
    vop3(ps, /*v_or3_b32*/ 0x372, 11, 256, 257, 260);   // v11 = v0|v1|v4
    vop3(ps, /*v_and_or_b32*/ 0x371, 12, 256, 257, 260);   // v12 = (v0&v1)|v4
    exp(ps, 0, 0xF, true, 7, 8, 11, 12);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "VOP3 int-ALU PS recompiled ok");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "VOP3 int-ALU PS SPIR-V validates");
    if (!err.empty()) std::printf("      ps: %s\n", err.c_str());
  }

  {
    // PS sampling a 2D texture: s_load T# (x8->s8) + S# (x4->s16), image_sample,
    // export the texel. Exercises RdnaPlanMimg + the shared EmitMimg.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    smem(ps, /*s_load_dwordx8 T#*/ 0x03, /*sdst s8*/ 8, /*sbase s0*/ 0, 0);
    smem(ps, /*s_load_dwordx4 S#*/ 0x02, /*sdst s16*/ 16, /*sbase s2*/ 1, 0);
    ps.push_back(vop1(0x01, 0, kInline0));   // v0 = u = 0.0
    ps.push_back(vop1(0x01, 1, kInline0));   // v1 = v = 0.0
    mimg(ps, /*image_sample*/ 0x20, /*dmask*/ 0xF, /*vdata v0*/ 0, /*vaddr v0*/ 0,
         /*srsrc s8>>2*/ 2, /*ssamp s16>>2*/ 4);
    exp(ps, /*MRT0*/ 0, 0xF, true, 0, 1, 2, 3);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "image_sample PS recompiled ok");
    // Prove EmitMimg actually emitted a texture sample (not the silent
    // unplanned-fallback): OpImageSampleImplicitLod == 87.
    expect(hasOpcode(r.fs_spirv, 87), "image_sample PS emits OpImageSampleImplicitLod");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "image_sample PS SPIR-V validates");
    if (!err.empty()) std::printf("      ps: %s\n", err.c_str());
  }

  {
    // VOP3P packed f16: v_pk_mul_f16, v_pk_add_f16, v_pk_fma_f16.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline1f));  // v0 = two f16 lanes (bit pattern)
    ps.push_back(vop1(0x01, 1, kInline1f));  // v1
    vop3p(ps, /*v_pk_mul_f16*/ 0x10, 4, 256, 257, 0);   // v4 = v0 .* v1
    vop3p(ps, /*v_pk_add_f16*/ 0x0F, 5, 256, 257, 0);   // v5 = v0 .+ v1
    vop3p(ps, /*v_pk_fma_f16*/ 0x0E, 6, 256, 257, 260); // v6 = v0.*v1 .+ v4
    exp(ps, 0, 0xF, true, 4, 5, 6, 4);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "VOP3P packed-f16 PS recompiled ok");
    // OpExtInst == 12: PackHalf2x16/UnpackHalf2x16 both go through it, so its
    // presence proves the packed path emitted rather than warned.
    expect(hasOpcode(r.fs_spirv, 12), "VOP3P PS emits pack/unpack ext-insts");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "VOP3P packed-f16 PS SPIR-V validates");
    if (!err.empty()) std::printf("      ps: %s\n", err.c_str());
  }

  {
    // RDNA2 f16 compare (v_cmp_lt_f16 = 0xC9), which GFX7 numbers as u32. Must
    // convert the low-half f16 operands (UnpackHalf2x16, OpExtInst) and run a
    // float predicate rather than an integer compare.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline0));    // v0
    ps.push_back(vop1(0x01, 1, kInline1f));   // v1
    ps.push_back(vopc(0xC9, 256, 257));       // v_cmp_lt_f16 v0,v1 -> VCC
    ps.push_back(vop2(0x01, 2, 256, 257));    // v_cndmask v2 = VCC ? v1 : v0
    exp(ps, 0, 0xF, true, 2, 2, 2, 2);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "f16-VOPC PS recompiled ok");
    expect(hasOpcode(r.fs_spirv, 12), "f16-VOPC converts operands (OpExtInst)");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "f16-VOPC PS SPIR-V validates");
    if (!err.empty()) std::printf("      ps: %s\n", err.c_str());
  }

  {
    // SPI_PS_INPUT_ENA VGPR seeding: a 2D-clip PS reads screen position from the
    // POS_X/POS_Y input VGPRs (not through v_interp). With PERSP_CENTER (bit 1)
    // + POS_X (bit 8) + POS_Y (bit 9) enabled they land in v2/v3; unseeded they
    // stay zero and a frag.x*a+frag.y*b<c clip discards every fragment. The seed
    // must load them from gl_FragCoord.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop2(0x03, 0, 256 + 2, 3));  // v0 = v2 + v3 (reads seeded frag x/y)
    exp(ps, 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r = gpu::rdna::Recompile(vs.data(), ps.data(), user_data,
                                                  user_data, /*ps_input_ena*/ 0x302);
    expect(r.ok, "frag-coord seed PS recompiled ok");
    expect(hasBuiltin(r.fs_spirv, 15), "POS_X/Y VGPRs seeded from gl_FragCoord");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "frag-coord seed PS SPIR-V validates");
    if (!err.empty()) std::printf("      ps: %s\n", err.c_str());
  }

  std::printf("== gfx10.3 T# decode ==\n");
  {
    // 256x128 2D texture at 0x800000000. width-1=255 -> d1[31:30]|d2[11:0];
    // height-1=127 -> d2[27:14]; type=9 (2D) -> d3[31:28].
    uint32_t d[8] = {0};
    d[0] = 0x08000000;                       // base_units[31:0] (base = <<8)
    d[1] = (255u & 0x3u) << 30 | (56u << 20);  // width-1 low 2 bits, fmt 8888UNORM
    d[2] = ((255u >> 2) & 0xFFF) | (127u << 14);
    d[3] = 9u << 28;                         // type = 2D
    gpu::gcn::TImage t = gpu::rdna::DecodeTImage(d);
    expect(t.base == 0x800000000ull, "T# base decodes (256-byte units << 8)");
    expect(t.width == 256 && t.height == 128, "T# 256x128 dimensions decode");
    expect(t.type == 9 && !t.arrayed, "T# type 2D, non-arrayed");
    expect(t.mip_levels == 1 && t.tiling_idx == 8, "T# single mip, linear");
    expect(t.dfmt == 10 && t.nfmt == 0, "T# fmt 56 -> 8_8_8_8 UNORM");
    expect(t.pitch == 256, "T# linear pitch 256B-row-aligned");
    d[1] = (255u & 0x3u) << 30 | (130u << 20);  // fmt 8_8_8_8_SRGB
    d[3] |= 25u << 20;                          // sw_mode 64KB_S_X (tiled)
    t = gpu::rdna::DecodeTImage(d);
    expect(t.dfmt == 10 && t.nfmt == 9, "T# fmt 130 -> 8_8_8_8 SRGB");
    expect(t.tiling_idx > 0x40, "T# gfx10 tiled mode maps out of GCN range");
  }

  std::printf(g_failures ? "\nFAILED (%d)\n" : "\nOK\n", g_failures);
  return g_failures ? 1 : 0;
}
