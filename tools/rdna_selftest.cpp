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
#include "ps5/rdna/rdna_translate.h"
#include "ps4/gcn/spirv/spv_post.h"

using gpu::gcn::Enc;

namespace {

int g_failures = 0;
void expect(bool cond, const char* what) {
  std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
  if (!cond) g_failures++;
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

  std::printf(g_failures ? "\nFAILED (%d)\n" : "\nOK\n", g_failures);
  return g_failures ? 1 : 0;
}
