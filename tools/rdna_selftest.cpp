/*
 * Standalone validation harness for the PS5 RDNA2 (gfx10.3) shader recompiler.
 * Hand-assembles minimal RDNA2 VS/PS programs, decodes them (rdna_decode),
 * recompiles them to SPIR-V (rdna_translate, reusing the shared gpu::gcn
 * backend), and validates the emitted binaries with SPIRV-Tools. Not part of
 * the emulator build; compiled directly (from the repo root), e.g.:
 *
 *   nix develop -c bash tools/build_rdna_selftest.sh
 *
 * The guest (Isaac) does not yet submit AGC DCBs, so this harness is the
 * regression check for the decoder + recompiler until submission is unblocked.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "guest_memory.h"
#include "ps4/gcn/spirv/spv_post.h"
#include "ps5/rdna/rdna_decode.h"
#include "ps5/rdna/rdna_resource.h"
#include "ps5/rdna/rdna_translate.h"

using gpu::gcn::Enc;

namespace {

int g_failures = 0;
void expect(bool cond, const char *what) {
  std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
  if (!cond)
    g_failures++;
}

// True if the SPIR-V word stream contains an instruction with the given opcode.
// Each instruction's first word packs wordCount[31:16] | opcode[15:0].
bool hasOpcode(const std::vector<uint32_t> &spv, uint32_t opcode) {
  size_t i = 5; // skip the 5-word module header
  while (i < spv.size()) {
    const uint32_t wc = spv[i] >> 16;
    if (wc == 0)
      break;
    if ((spv[i] & 0xFFFF) == opcode)
      return true;
    i += wc;
  }
  return false;
}

bool hasExtInst(const std::vector<uint32_t> &spv, uint32_t instruction) {
  size_t i = 5;
  while (i < spv.size()) {
    const uint32_t wc = spv[i] >> 16;
    if (wc == 0)
      break;
    // OpExtInst operands are result type, result id, set id, instruction.
    if ((spv[i] & 0xFFFF) == 12 && wc >= 5 && spv[i + 4] == instruction)
      return true;
    i += wc;
  }
  return false;
}

// True if the module decorates any id with BuiltIn `builtin` (OpDecorate == 71,
// Decoration BuiltIn == 11, then the BuiltIn enum). Used to prove a PS input
// VGPR was seeded from a Vulkan built-in (gl_FragCoord == 15).
bool hasBuiltin(const std::vector<uint32_t> &spv, uint32_t builtin) {
  size_t i = 5;
  while (i < spv.size()) {
    const uint32_t wc = spv[i] >> 16;
    if (wc == 0)
      break;
    if ((spv[i] & 0xFFFF) == 71 && wc >= 4 && spv[i + 2] == 11 &&
        spv[i + 3] == builtin)
      return true;
    i += wc;
  }
  return false;
}

bool hasVariableStorage(const std::vector<uint32_t> &spv, uint32_t storage) {
  size_t i = 5;
  while (i < spv.size()) {
    const uint32_t wc = spv[i] >> 16;
    if (wc == 0)
      break;
    if ((spv[i] & 0xFFFF) == 59 && wc >= 4 && spv[i + 3] == storage)
      return true;
    i += wc;
  }
  return false;
}

// ---- RDNA2 instruction encoders (little bit-twiddling for readability) ------
// VOP1: [31:25]=0x3F, vdst[24:17], op[16:9], src0[8:0].
uint32_t vop1(uint32_t op, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) |
         (src0 & 0x1FF);
}
// VOP2: [31]=0, op[30:25], vdst[24:17], vsrc1[16:9], src0[8:0].
uint32_t vop2(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t vsrc1) {
  return ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) |
         (src0 & 0x1FF);
}
// VOP3: word0 [31:26]=0x35, op[25:16], clamp[15], abs[10:8], vdst[7:0];
// word1: neg[31:29], src2[26:18], src1[17:9], src0[8:0].
void vop3(std::vector<uint32_t> &out, uint32_t op, uint32_t vdst, uint32_t s0,
          uint32_t s1, uint32_t s2) {
  out.push_back((0x35u << 26) | ((op & 0x3FF) << 16) | (vdst & 0xFF));
  out.push_back(((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF));
}

void vop3b(std::vector<uint32_t> &out, uint32_t op, uint32_t vdst,
           uint32_t sdst, uint32_t s0, uint32_t s1, uint32_t s2 = 0) {
  out.push_back((0x35u << 26) | ((op & 0x3FF) << 16) | ((sdst & 0x7F) << 8) |
                (vdst & 0xFF));
  out.push_back(((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF));
}
// VOPC: [31:25]=0x3E, op[24:17], vsrc1[16:9], src0[8:0] (writes VCC).
uint32_t vopc(uint32_t op, uint32_t src0, uint32_t vsrc1) {
  return (0x3Eu << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) |
         (src0 & 0x1FF);
}
// EXP: [31:26]=0x3E, done[11], compr[10], target[9:4], en[3:0]; word1 = 4
// VGPRs.
void exp(std::vector<uint32_t> &out, uint32_t target, uint32_t en, bool done,
         uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  out.push_back((0x3Eu << 26) | ((done ? 1u : 0u) << 11) |
                ((target & 0x3F) << 4) | (en & 0xF));
  out.push_back((v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) |
                ((v3 & 0xFF) << 24));
}
// SOPP: [31:23]=0x17F, op[22:16], simm[15:0].
uint32_t sopp(uint32_t op, uint32_t simm) {
  return (0x17Fu << 23) | ((op & 0x7F) << 16) | (simm & 0xFFFF);
}
// SMEM: [31:26]=0x3D, op[25:18], sdst[12:6], sbase[5:0]; word1 imm[20:0].
void smem(std::vector<uint32_t> &out, uint32_t op, uint32_t sdst,
          uint32_t sbase, uint32_t imm, uint32_t soffset = 125) {
  out.push_back((0x3Du << 26) | ((op & 0xFF) << 18) | ((sdst & 0x7F) << 6) |
                (sbase & 0x3F));
  out.push_back(((soffset & 0x7F) << 25) | (imm & 0x1FFFFF));
}

uint32_t sop1(uint32_t op, uint32_t sdst, uint32_t ssrc) {
  return (0x17Du << 23) | ((sdst & 0x7F) << 16) | ((op & 0xFF) << 8) |
         (ssrc & 0xFF);
}

void sop2(std::vector<uint32_t> &out, uint32_t op, uint32_t sdst,
          uint32_t ssrc0, uint32_t ssrc1, uint32_t literal = 0) {
  out.push_back((0x2u << 30) | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16) |
                ((ssrc1 & 0xFF) << 8) | (ssrc0 & 0xFF));
  if (ssrc0 == 255 || ssrc1 == 255)
    out.push_back(literal);
}

uint32_t sopc(uint32_t op, uint32_t ssrc0, uint32_t ssrc1) {
  return (0x17Eu << 23) | ((op & 0x7F) << 16) | ((ssrc1 & 0xFF) << 8) |
         (ssrc0 & 0xFF);
}

uint32_t sopk(uint32_t op, uint32_t sdst, uint32_t simm) {
  return (0xBu << 28) | ((op & 0x1F) << 23) | ((sdst & 0x7F) << 16) |
         (simm & 0xFFFF);
}

// MIMG (NSA=0, 64-bit): word0 [31:26]=0x3C, op[24:18], da[14], dmask[11:8],
// op[7] at bit 0; word1 ssamp[25:21], srsrc[20:16], vdata[15:8], vaddr[7:0].
// srsrc/ssamp are the SGPR index >> 2 (4-SGPR-aligned).
void mimg(std::vector<uint32_t> &out, uint32_t op, uint32_t dmask,
          uint32_t vdata, uint32_t vaddr, uint32_t srsrc, uint32_t ssamp,
          uint32_t dim = 1) {
  out.push_back((0x3Cu << 26) | ((op & 0x7F) << 18) | ((dmask & 0xF) << 8) |
                ((dim & 0x7) << 3) | ((op >> 7) & 1));
  out.push_back(((ssamp & 0x1F) << 21) | ((srsrc & 0x1F) << 16) |
                ((vdata & 0xFF) << 8) | (vaddr & 0xFF));
}

void mubuf(std::vector<uint32_t> &out, uint32_t op, uint32_t srsrc) {
  out.push_back((0x38u << 26) | ((op & 0x7F) << 18) | (1u << 13));
  out.push_back(((srsrc / 4) & 0x1F) << 16);
}

void mtbuf(std::vector<uint32_t> &out, uint32_t op, uint32_t format,
           uint32_t vdata, uint32_t vaddr, uint32_t srsrc) {
  out.push_back((0x3Au << 26) | ((format & 0x7F) << 19) | ((op & 0x07) << 16) |
                (1u << 13));
  out.push_back((((op >> 3) & 1) << 21) | (((srsrc / 4) & 0x1F) << 16) |
                ((vdata & 0xFF) << 8) | (vaddr & 0xFF));
}

// VOP3P with canonical componentwise selectors: low result uses low sources,
// high result uses high sources.
void vop3p(std::vector<uint32_t> &out, uint32_t op, uint32_t vdst, uint32_t s0,
           uint32_t s1, uint32_t s2) {
  out.push_back((0x33u << 26) | ((op & 0xFF) << 16) | (1u << 14) |
                (vdst & 0xFF));
  out.push_back((3u << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) |
                (s0 & 0x1FF));
}

constexpr uint32_t kInline0 = 128;  // integer/float 0
constexpr uint32_t kInline1f = 242; // float 1.0
constexpr uint32_t kEndpgm = 1;

} // namespace

int main() {
  uint32_t user_data[32] = {};
  const auto unmapped =
      gpu::rdna::Recompile(reinterpret_cast<const uint32_t *>(0x4000000000ull),
                           nullptr, user_data, user_data);
  expect(!unmapped.ok, "unmapped shader address is rejected");

  std::printf("== RDNA2 decoder ==\n");
  {
    // v_mov_b32 v0, 0.0 ; v_mov_b32 v3, 1.0 ; exp pos0 ; s_endpgm
    std::vector<uint32_t> code;
    code.push_back(vop1(0x01, 0, kInline0));
    code.push_back(vop1(0x01, 3, kInline1f));
    exp(code, /*POS0*/ 12, 0xF, true, 0, 0, 0, 3);
    code.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Program prog =
        gpu::rdna::Decode(code.data(), (uint32_t)code.size());
    expect(prog.size() == 4, "decoded 4 instructions");
    expect(prog.size() >= 1 && prog[0].enc == Enc::kVop1 &&
               prog[0].opcode == 0x01,
           "inst0 is VOP1 v_mov_b32");
    expect(prog.size() >= 3 && prog[2].enc == Enc::kExp && prog[2].size == 2,
           "inst2 is EXP (2 dwords)");
    expect(prog.size() >= 4 && prog[3].enc == Enc::kSopp && prog[3].opcode == 1,
           "inst3 is SOPP s_endpgm");
  }
  {
    // SMEM s_buffer_load_dwordx4 is a 2-dword instruction.
    std::vector<uint32_t> code;
    smem(code, /*s_buffer_load_dwordx4*/ 0x0A, /*sdst*/ 8, /*sbase*/ 2,
         /*imm*/ 0);
    code.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Program prog =
        gpu::rdna::Decode(code.data(), (uint32_t)code.size());
    expect(prog.size() == 2 && prog[0].enc == Enc::kSmrd && prog[0].size == 2 &&
               prog[0].opcode == 0x0A,
           "SMEM s_buffer_load_dwordx4 decodes as 2-dword kSmrd");
  }

  std::printf("== RDNA2 -> SPIR-V recompile ==\n");
  {
    // Procedural VS: position = (0,0,0,1).
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, 8));         // v0 = VS user SGPR s8
    vs.push_back(vop1(0x01, 1, kInline0));  // v1 = 0.0
    vs.push_back(vop1(0x01, 2, kInline0));  // v2 = 0.0
    vs.push_back(vop1(0x01, 3, kInline1f)); // v3 = 1.0
    exp(vs, /*POS0*/ 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    // PS: color = (1,1,1,1) -> MRT0. (v_add_f32 exercises VOP2.)
    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, 0));            // v0 = PS user SGPR s0
    ps.push_back(vop2(0x03, 1, kInline1f, 0)); // v1 = 1.0 + v0 = 2.0 (VOP2 add)
    ps.push_back(vop1(0x01, 2, kInline1f));    // v2 = 1.0
    ps.push_back(vop1(0x01, 3, kInline1f));    // v3 = 1.0
    exp(ps, /*MRT0*/ 0, 0xF, true, 0, 0, 2, 3);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0}; // no fetch shader -> procedural path
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "VS+PS recompiled ok");
    expect(!r.vs_spirv.empty(), "VS SPIR-V non-empty");
    expect(!r.fs_spirv.empty(), "PS SPIR-V non-empty");
    expect(r.ps_mrt_mask & 1u, "PS exports MRT0");
    expect(hasVariableStorage(r.vs_spirv, 9),
           "VS reads push-constant user data");
    expect(hasVariableStorage(r.fs_spirv, 9),
           "PS reads push-constant user data");
    const gpu::gcn::Recompiled no_user_data = gpu::rdna::Recompile(
        vs.data(), ps.data(), user_data, user_data, 0, false, 0, 0);
    expect(no_user_data.ok, "zero-user-data stages recompile");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.vs_spirv, &err), "VS SPIR-V validates");
    if (!err.empty())
      std::printf("      vs: %s\n", err.c_str());
    err.clear();
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err), "PS SPIR-V validates");
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());

    std::vector<uint32_t> fetch;
    smem(fetch, 0x02, 12, 4, 0);
    mubuf(fetch, 0x00, 12);
    fetch.push_back(sop1(0x20, 0, 0));
    const size_t fetch_bytes = fetch.size() * sizeof(uint32_t);
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void *fetch_memory =
        mmap(reinterpret_cast<void *>(0x1000100000ull), page_size,
             PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    expect(fetch_memory != MAP_FAILED, "mapped standalone fetch shader page");
    if (fetch_memory != MAP_FAILED) {
      std::memcpy(fetch_memory, fetch.data(), fetch_bytes);
      const uint64_t fetch_address = reinterpret_cast<uint64_t>(fetch_memory);
      user_data[0] = static_cast<uint32_t>(fetch_address);
      user_data[1] = static_cast<uint32_t>(fetch_address >> 32);
      const gpu::gcn::Recompiled fetched =
          gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
      expect(fetched.ok, "standalone fetch shader recompiles");
      expect(fetched.attrs.size() == 1,
             "standalone fetch shader plans one attribute");
      if (fetched.attrs.size() == 1) {
        expect(fetched.attrs[0].table_sgpr == 8,
               "standalone fetch shader retains its table root");
        expect(fetched.attrs[0].use_pc == ~0u,
               "standalone fetch shader retains static V# resolution");
      }
      const gpu::gcn::Recompiled undeclared_fetch = gpu::rdna::Recompile(
          vs.data(), ps.data(), user_data, user_data, 0, false, 0, 32);
      expect(undeclared_fetch.attrs.empty(),
             "undeclared GS user data cannot supply a fetch shader");

      std::vector<uint32_t> typed_vs;
      smem(typed_vs, 0x02, 12, 4, 0);
      mtbuf(typed_vs, 0x03, 56, 0, 0, 12);
      exp(typed_vs, 12, 0xF, true, 0, 1, 2, 3);
      typed_vs.push_back(sopp(kEndpgm, 0));
      user_data[0] = 0;
      user_data[1] = 0;
      const gpu::gcn::Recompiled typed = gpu::rdna::Recompile(
          typed_vs.data(), ps.data(), user_data, user_data);
      expect(typed.ok && typed.attrs.size() == 1 && typed.attrs[0].use_pc == 2,
             "table-loaded MTBUF remains an inline vertex attribute");
      munmap(fetch_memory, page_size);
    }
  }

  {
    // VS with a constant buffer + VOP3: load cbuffer[0..3] into s4.., move a
    // cbuf dword into a VGPR, v_mad_f32 (VOP3 0x141), export POS0. Exercises
    // RdnaPlanCbufs / RdnaEmitSmem and the VOP3 field decode.
    std::vector<uint32_t> vs;
    smem(vs, /*s_buffer_load_dwordx4*/ 0x0A, /*sdst s4*/ 4, /*sbase sgpr2*/ 1,
         0, /*dynamic soffset s20*/ 20);
    vs.push_back(vop1(0x01, 0, 4));         // v0 = s4 (cbuffer dword 0)
    vs.push_back(vop1(0x01, 1, kInline1f)); // v1 = 1.0
    vs.push_back(vop1(0x01, 2, kInline0));  // v2 = 0.0
    vs.push_back(vop1(0x01, 3, kInline1f)); // v3 = 1.0
    vop3(vs, /*v_mad_f32*/ 0x141, 0, 256, 257, 258); // v0 = v0*v1 + v2
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
    expect(r.vs_cbufs.size() == 1 &&
               r.vs_cbufs[0].num_dwords == gpu::gcn::kCbufDwords,
           "dynamic SMEM offsets retain the full cbuffer window");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.vs_spirv, &err),
           "cbuf VS SPIR-V validates");
    if (!err.empty())
      std::printf("      vs: %s\n", err.c_str());
  }

  {
    std::vector<uint32_t> vs;
    smem(vs, /*s_buffer_load_dword*/ 0x08, 20, /*s8*/ 4, 0);
    vs.push_back(sop1(/*s_mov_b32*/ 0x03, 8, 12));
    smem(vs, /*s_buffer_load_dword*/ 0x08, 21, /*s8*/ 4, 0);
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline1f));
    exp(ps, 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {};
    const gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "versioned-cbuffer VS recompiled ok");
    expect(r.vs_cbufs.size() == 2,
           "scalar writes split identical cbuffer operands into bindings");
  }

  {
    std::vector<uint32_t> vs;
    smem(vs, /*s_load_dword*/ 0x00, 12, /*s8*/ 4, 0);
    vs.push_back(sop1(/*s_mov_b32*/ 0x03, 12, 16));
    smem(vs, /*s_load_dword*/ 0x00, 20, /*s12*/ 6, 0);
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline1f));
    exp(ps, 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {};
    const gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "overwritten-cbuffer-chain VS recompiled ok");
    expect(r.vs_cbufs.size() == 2,
           "overwritten SMEM results remain cbuffer leaf reads");
  }

  {
    // PS exercising cndmask, CMPX, no-carry, and carry-in/out gfx10 semantics.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline0));  // v0 = 0.0
    ps.push_back(vop1(0x01, 1, kInline1f)); // v1 = 1.0
    ps.push_back(vopc(0x01, 256, 257));     // v_cmp_lt_f32 v0,v1 -> VCC
    ps.push_back(vop2(0x01, 2, 256, 257));  // v_cndmask v2 = VCC ? v1 : v0
    ps.push_back(vop2(0x25, 3, 256, 257));  // v_add_nc_u32 v3 = v0 + v1
    ps.push_back(vopc(0x11, 256, 257));     // v_cmpx_lt_f32 updates EXEC only
    ps.push_back(vop2(0x28, 4, 256, 257));  // v_add_co_ci_u32 masks VCC by EXEC
    exp(ps, 0, 0xF, true, 2, 2, 3, 4);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "cndmask/CMPX/carry PS recompiled ok");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "cndmask/CMPX/carry PS SPIR-V validates");
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());
  }

  {
    // RDNA2 VOP3-only integer ops (native no-carry add/sub + 3-input forms).
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline0));              // v0 = 0
    ps.push_back(vop1(0x01, 1, kInline1f));             // v1 = 1.0 bits
    vop3(ps, /*v_add_nc_i32*/ 0x37F, 4, 256, 257, 0);   // v4 = v0 + v1
    vop3(ps, /*v_sub_nc_i32*/ 0x376, 5, 256, 257, 0);   // v5 = v0 - v1
    vop3b(ps, /*v_add_co_u32*/ 0x30F, 6, 20, 256, 257); // v6, s20 = v0 + v1
    vop3b(ps, /*v_add_co_ci_u32*/ 0x128, 13, 22, 256, 257,
          106); // v13, s22 = v0 + v1 + VCC
    vop3(ps, /*v_add3_u32*/ 0x36D, 7, 256, 257, 260);      // v7 = v0+v1+v4
    vop3(ps, /*v_lshl_or_b32*/ 0x36F, 8, 256, 257, 260);   // v8 = (v0<<v1)|v4
    vop3(ps, /*v_lshl_add_u32*/ 0x346, 9, 256, 257, 260);  // v9 = (v0<<v1)+v4
    vop3(ps, /*v_add_lshl_u32*/ 0x347, 10, 256, 257, 260); // v10 = (v0+v1)<<v4
    vop3(ps, /*v_or3_b32*/ 0x372, 11, 256, 257, 260);      // v11 = v0|v1|v4
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
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());
  }

  {
    // Unsupported operations must reject the stage rather than return valid
    // SPIR-V containing the shared emitter's fallback behavior.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop2(/*unsupported v_add_f16*/ 0x32, 0, 256, 0));
    exp(ps, 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {};
    const gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(!r.ok, "unsupported RDNA instruction rejects recompilation");

    ps.clear();
    ps.push_back(sopp(/*s_branch*/ 0x02, 1));
    ps.push_back(vop2(/*unreachable v_add_f16*/ 0x32, 0, 256, 0));
    ps.push_back(vop1(0x01, 0, kInline1f));
    exp(ps, 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Recompiled reachable =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(reachable.ok, "unreachable unsupported instructions are ignored");

    for (const uint32_t op : {0x09u, 0x21u, 0x25u}) {
      ps.clear();
      mimg(ps, op, 0xF, 0, 0, 4, 8);
      ps.push_back(sopp(kEndpgm, 0));
      const gpu::gcn::Recompiled image =
          gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
      expect(!image.ok,
             "unsupported MIMG mip/clamp/bias semantics reject recompilation");
    }

    ps.clear();
    ps.push_back(vop2(0x03, 0, 249, 1));
    ps.push_back((6u << 8) | (1u << 13) | (6u << 16) | (6u << 24));
    ps.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Recompiled sdwa =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(!sdwa.ok, "unsupported SDWA clamp rejects recompilation");

    ps.clear();
    vop3p(ps, /*v_pk_add_f16*/ 0x0F, 0, 256, 257, 0);
    ps[0] |= 1u << 15; // CLAMP
    ps.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Recompiled vop3p_modifier =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(!vop3p_modifier.ok,
           "unsupported VOP3P modifiers reject recompilation");

    std::vector<uint32_t> prim_vs;
    prim_vs.push_back(vop1(0x01, 0, kInline0));
    exp(prim_vs, /*PRIM*/ 20, 1, true, 0, 0, 0, 0);
    prim_vs.push_back(sopp(kEndpgm, 0));
    ps.clear();
    ps.push_back(sopp(kEndpgm, 0));
    const gpu::gcn::Recompiled prim =
        gpu::rdna::Recompile(prim_vs.data(), ps.data(), user_data, user_data);
    expect(!prim.ok, "unsupported PRIM export rejects recompilation");
  }

  {
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    for (uint32_t i = 0; i < 17; i++) {
      mimg(ps, /*image_load*/ 0x00, 0xF, 0, 0, 4, 0);
      if (i != 16)
        ps.push_back(sop1(/*s_mov_b32*/ 0x03, 16, 20 + i));
    }
    exp(ps, 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[32] = {};
    const gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(!r.ok, "PS exceeding the texture binding limit is rejected");
  }

  {
    // PS sampling a 2D texture: s_load T# (x8->s8) + S# (x4->s16),
    // image_sample, export the texel. Exercises RdnaPlanMimg + the shared
    // EmitMimg.
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
    ps.push_back(vop1(0x01, 0, kInline0)); // v0 = u = 0.0
    ps.push_back(vop1(0x01, 1, kInline0)); // v1 = v = 0.0
    mimg(ps, /*image_sample*/ 0x20, /*dmask*/ 0xF, /*vdata v0*/ 0,
         /*vaddr v0*/ 0,
         /*srsrc s8>>2*/ 2, /*ssamp s16>>2*/ 4);
    exp(ps, /*MRT0*/ 0, 0xF, true, 0, 1, 2, 3);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "image_sample PS recompiled ok");
    // Prove EmitMimg actually emitted a texture sample (not the silent
    // unplanned-fallback): OpImageSampleImplicitLod == 87.
    expect(hasOpcode(r.fs_spirv, 87),
           "image_sample PS emits OpImageSampleImplicitLod");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "image_sample PS SPIR-V validates");
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());
  }

  {
    // Storage-image writes are lane side effects and must be guarded by EXEC.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(vop1(0x01, 0, kInline0));
    ps.push_back(vop1(0x01, 1, kInline0));
    ps.push_back(vop1(0x01, 2, kInline1f));
    mimg(ps, /*image_store*/ 0x08, /*dmask*/ 1, /*vdata v2*/ 2,
         /*vaddr v0*/ 0, /*srsrc s16>>2*/ 4, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {};
    const gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "image_store PS recompiled ok");
    expect(hasOpcode(r.fs_spirv, 99), "image_store emits OpImageWrite");
    expect(hasOpcode(r.fs_spirv, 247),
           "image_store side effect is guarded by EXEC");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "image_store PS SPIR-V validates");
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());
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
    ps.push_back(vop1(0x01, 0, kInline1f)); // v0 = two f16 lanes (bit pattern)
    ps.push_back(vop1(0x01, 1, kInline1f)); // v1
    vop3p(ps, /*v_pk_mul_f16*/ 0x10, 4, 256, 257, 0);   // v4 = v0 .* v1
    vop3p(ps, /*v_pk_add_f16*/ 0x0F, 5, 256, 257, 0);   // v5 = v0 .+ v1
    vop3p(ps, /*v_pk_fma_f16*/ 0x0E, 6, 256, 257, 260); // v6 = v0.*v1 .+ v4
    exp(ps, 0, 0xF, true, 4, 5, 6, 4);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r =
        gpu::rdna::Recompile(vs.data(), ps.data(), user_data, user_data);
    expect(r.ok, "VOP3P packed-f16 PS recompiled ok");
    expect(hasOpcode(r.fs_spirv, 12), "VOP3P PS emits pack/unpack ext-insts");
    expect(hasExtInst(r.fs_spirv, 50), "VOP3P PS emits fused GLSL.std.450 Fma");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "VOP3P packed-f16 PS SPIR-V validates");
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());
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
    ps.push_back(vop1(0x01, 0, kInline0));  // v0
    ps.push_back(vop1(0x01, 1, kInline1f)); // v1
    ps.push_back(vopc(0xC9, 256, 257));     // v_cmp_lt_f16 v0,v1 -> VCC
    ps.push_back(vop2(0x01, 2, 256, 257));  // v_cndmask v2 = VCC ? v1 : v0
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
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());
  }

  {
    // SPI_PS_INPUT_ENA VGPR seeding: a 2D-clip PS reads screen position from
    // the POS_X/POS_Y input VGPRs (not through v_interp). With PERSP_CENTER
    // (bit 1)
    // + POS_X (bit 8) + POS_Y (bit 9) enabled they land in v2/v3; unseeded they
    // stay zero and a frag.x*a+frag.y*b<c clip discards every fragment. The
    // seed must load them from gl_FragCoord.
    std::vector<uint32_t> vs;
    vs.push_back(vop1(0x01, 0, kInline0));
    vs.push_back(vop1(0x01, 1, kInline0));
    vs.push_back(vop1(0x01, 2, kInline0));
    vs.push_back(vop1(0x01, 3, kInline1f));
    exp(vs, 12, 0xF, true, 0, 1, 2, 3);
    vs.push_back(sopp(kEndpgm, 0));

    std::vector<uint32_t> ps;
    ps.push_back(
        vop2(0x03, 0, 256 + 2, 3)); // v0 = v2 + v3 (reads seeded frag x/y)
    exp(ps, 0, 0xF, true, 0, 0, 0, 0);
    ps.push_back(sopp(kEndpgm, 0));

    uint32_t user_data[16] = {0};
    gpu::gcn::Recompiled r = gpu::rdna::Recompile(
        vs.data(), ps.data(), user_data, user_data, /*ps_input_ena*/ 0x302);
    expect(r.ok, "frag-coord seed PS recompiled ok");
    expect(hasBuiltin(r.fs_spirv, 15),
           "POS_X/Y VGPRs seeded from gl_FragCoord");
    std::string err;
    expect(gpu::gcn::spirv::Validate(r.fs_spirv, &err),
           "frag-coord seed PS SPIR-V validates");
    if (!err.empty())
      std::printf("      ps: %s\n", err.c_str());
  }

  std::printf("== gfx10.3 T# decode ==\n");
  {
    // 256x128 2D texture at 0x800000000. width-1=255 -> d1[31:30]|d2[11:0];
    // height-1=127 -> d2[27:14]; type=9 (2D) -> d3[31:28].
    uint32_t d[8] = {0};
    d[0] = 0x08000000; // base_units[31:0] (base = <<8)
    d[1] =
        (255u & 0x3u) << 30 | (56u << 20); // width-1 low 2 bits, fmt 8888UNORM
    d[2] = ((255u >> 2) & 0xFFF) | (127u << 14);
    d[3] = 9u << 28; // type = 2D
    gpu::gcn::TImage t = gpu::rdna::DecodeTImage(d);
    expect(t.base == 0x800000000ull, "T# base decodes (256-byte units << 8)");
    expect(t.width == 256 && t.height == 128, "T# 256x128 dimensions decode");
    expect(t.type == 9 && !t.arrayed, "T# type 2D, non-arrayed");
    expect(t.mip_levels == 1 && t.tiling_idx == 8, "T# single mip, linear");
    expect(t.dfmt == 10 && t.nfmt == 0, "T# fmt 56 -> 8_8_8_8 UNORM");
    expect(t.pitch == 256, "T# linear pitch 256B-row-aligned");
    d[1] = (255u & 0x3u) << 30 | (130u << 20); // fmt 8_8_8_8_SRGB
    d[3] |= 25u << 20;                         // sw_mode 64KB_S_X (tiled)
    t = gpu::rdna::DecodeTImage(d);
    expect(t.dfmt == 10 && t.nfmt == 9, "T# fmt 130 -> 8_8_8_8 SRGB");
    expect(t.tiling_idx > 0x40, "T# gfx10 tiled mode maps out of GCN range");
  }

  std::printf("== RDNA2 resource resolution ==\n");
  {
    const long page_size = sysconf(_SC_PAGESIZE);
    void *memory =
        mmap(nullptr, static_cast<size_t>(page_size), PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(memory != MAP_FAILED, "mapped descriptor test page");
    if (memory != MAP_FAILED) {
      expect(
          gpu::IsReadableRange(reinterpret_cast<uint64_t>(memory), page_size),
          "readable guest mappings pass range validation");
      void *reserved = mmap(nullptr, static_cast<size_t>(page_size), PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      expect(reserved != MAP_FAILED &&
                 !gpu::IsReadableRange(reinterpret_cast<uint64_t>(reserved),
                                       page_size),
             "PROT_NONE guest reservations fail range validation");
      if (reserved != MAP_FAILED)
        munmap(reserved, static_cast<size_t>(page_size));
      auto *words = static_cast<uint32_t *>(memory);
      const uint64_t texture_base = reinterpret_cast<uint64_t>(memory);
      auto writeImage = [&](uint32_t *d) {
        std::memset(d, 0, 8 * sizeof(uint32_t));
        const uint64_t base_units = texture_base >> 8;
        d[0] = static_cast<uint32_t>(base_units);
        d[1] =
            static_cast<uint32_t>(base_units >> 32) | (3u << 30) | (56u << 20);
        d[2] = 3u << 14;
        d[3] = 9u << 28;
      };

      uint32_t *table = words + 64;
      writeImage(table + 4);
      expect(gpu::IsReadableRange(reinterpret_cast<uint64_t>(table + 4),
                                  8 * sizeof(uint32_t)),
             "descriptor table entry is readable");
      uint32_t user_data[32] = {};
      const uint64_t table_address = reinterpret_cast<uint64_t>(table);
      user_data[8] = static_cast<uint32_t>(table_address);
      user_data[9] = static_cast<uint32_t>(table_address >> 32);

      std::vector<uint32_t> dynamic;
      dynamic.push_back(sopk(0x00, 106, 16)); // s_movk_i32 s106, 16
      smem(dynamic, 0x03, 0, 4, 0, 106); // s_load_dwordx8 s[0:7], s[8:9], s106
      mimg(dynamic, 0x00, 0xF, 0, 0, 0, 0);
      dynamic.push_back(sopp(kEndpgm, 0));
      const auto dynamic_textures =
          gpu::rdna::TrackTextures(dynamic.data(), user_data, 16);
      expect(dynamic_textures.size() == 1 && dynamic_textures[0].valid &&
                 dynamic_textures[0].base == texture_base,
             "dynamic SMEM SOFFSET resolves the selected T#");

      writeImage(&user_data[16]);
      std::vector<uint32_t> inline_image;
      mimg(inline_image, 0x00, 0xF, 0, 0, 4, 0);
      inline_image.push_back(sopp(kEndpgm, 0));
      const auto declared24 =
          gpu::rdna::TrackTextures(inline_image.data(), user_data, 24);
      const auto declared16 =
          gpu::rdna::TrackTextures(inline_image.data(), user_data, 16);
      expect(declared24.size() == 1 && declared24[0].valid,
             "inline T# at s16 resolves inside the declared window");
      expect(declared16.size() == 1 && !declared16[0].valid,
             "inline T# at s16 is rejected outside the declared window");

      std::vector<uint32_t> dimensions;
      mimg(dimensions, 0x00, 0xF, 0, 0, 4, 0, 1);
      mimg(dimensions, 0x00, 0xF, 0, 0, 4, 0, 5);
      dimensions.push_back(sopp(kEndpgm, 0));
      const auto dim_program = gpu::rdna::Decode(
          dimensions.data(), static_cast<uint32_t>(dimensions.size()));
      const auto dim_plan = gpu::rdna::RdnaPlanMimg(dim_program);
      const auto dim_textures =
          gpu::rdna::TrackTextures(dimensions.data(), user_data, 24);
      expect(dim_plan.binding_srsrc.size() == 2,
             "MIMG DIM distinguishes 2D and 2D-array bindings");
      expect(dim_textures.size() == 2 && !dim_textures[0].arrayed &&
                 dim_textures[1].arrayed,
             "MIMG DIM controls array addressing");

      std::vector<uint32_t> reachable_image;
      reachable_image.push_back(sopp(/*s_branch*/ 0x02, 2));
      mimg(reachable_image, 0x00, 0xF, 0, 0, 0, 0); // unreachable T# at s0
      mimg(reachable_image, 0x00, 0xF, 0, 0, 4, 0); // live T# at s16
      reachable_image.push_back(sopp(kEndpgm, 0));
      const auto reachable_image_program = gpu::rdna::ReachableProgram(
          gpu::rdna::Decode(reachable_image.data(),
                            static_cast<uint32_t>(reachable_image.size())));
      const auto reachable_image_plan =
          gpu::rdna::RdnaPlanMimg(reachable_image_program);
      const auto reachable_textures =
          gpu::rdna::TrackTextures(reachable_image.data(), user_data, 24);
      expect(reachable_image_plan.binding_srsrc.size() == 1 &&
                 reachable_image_plan.binding_srsrc[0] == 16 &&
                 reachable_textures.size() == 1 && reachable_textures[0].valid,
             "unreachable MIMG instructions do not shift live bindings");

      std::vector<uint32_t> versioned_images;
      mimg(versioned_images, 0x00, 0xF, 0, 0, 4, 0);
      versioned_images.push_back(sopk(/*s_cmpk_eq_u32*/ 0x09, 16, 0));
      mimg(versioned_images, 0x00, 0xF, 0, 0, 4, 0);
      versioned_images.push_back(sop1(/*s_mov_b32*/ 0x03, 16, 20));
      mimg(versioned_images, 0x00, 0xF, 0, 0, 4, 0);
      versioned_images.push_back(sopp(kEndpgm, 0));
      const auto versioned_image_program =
          gpu::rdna::Decode(versioned_images.data(),
                            static_cast<uint32_t>(versioned_images.size()));
      const auto versioned_image_plan =
          gpu::rdna::RdnaPlanMimg(versioned_image_program);
      expect(versioned_image_plan.binding_srsrc.size() == 2,
             "texture versions ignore SOPK compares and track SGPR writes");

      uint32_t gs_user_data[32] = {};
      gs_user_data[0] = static_cast<uint32_t>(texture_base);
      gs_user_data[1] = static_cast<uint32_t>(texture_base >> 32);
      std::vector<uint32_t> gs_load;
      smem(gs_load, 0x00, 0, 4, 0);
      gs_load.push_back(sopp(kEndpgm, 0));
      const auto gs_cbufs =
          gpu::rdna::ResolveBuffers(gs_load.data(), gs_user_data, 2, 8);
      const auto unshifted_cbufs =
          gpu::rdna::ResolveBuffers(gs_load.data(), gs_user_data, 2, 0);
      expect(gs_cbufs.count(0) && gs_cbufs.at(0).base == texture_base,
             "GS user data starts at shader SGPR s8");
      expect(!unshifted_cbufs.count(0),
             "GS roots below s8 do not alias user data");

      uint32_t *vertex_table = words + 128;
      uint32_t *selected_vb = vertex_table + 4;
      const uint64_t vertex_base = reinterpret_cast<uint64_t>(words + 192);
      selected_vb[0] = static_cast<uint32_t>(vertex_base);
      selected_vb[1] = static_cast<uint32_t>(vertex_base >> 32) | (20u << 16);
      selected_vb[2] = 6;
      selected_vb[3] = 64u << 12;
      const uint64_t vertex_table_address =
          reinterpret_cast<uint64_t>(vertex_table);
      gs_user_data[0] = static_cast<uint32_t>(vertex_table_address);
      gs_user_data[1] = static_cast<uint32_t>(vertex_table_address >> 32);
      std::vector<uint32_t> vertex_fetch;
      sop2(vertex_fetch, 0x27, 106, 3, 255, 0x00080008);
      sop2(vertex_fetch, 0x1e, 106, 106, 132);
      smem(vertex_fetch, 0x02, 12, 4, 0, 106);
      vertex_fetch.push_back(sopc(0x00, 128, 128));
      sop2(vertex_fetch, 0x0a, 15, 15, 120);
      mubuf(vertex_fetch, 0x00, 12);
      vertex_fetch.push_back(sopp(kEndpgm, 0));
      const auto vertex_resources =
          gpu::rdna::ResolveBuffers(vertex_fetch.data(), gs_user_data, 2, 8);
      const auto vertex_resource = vertex_resources.find(7);
      expect(vertex_resource != vertex_resources.end() &&
                 vertex_resource->second.descriptor_valid &&
                 vertex_resource->second.base == vertex_base &&
                 vertex_resource->second.descriptor[2] == 6,
             "merged-wave scalar replay resolves the inline vertex V#");

      std::vector<uint32_t> wide_unknown;
      wide_unknown.push_back(sop1(/*s_mov_b64*/ 0x04, 12, 8));
      sop2(wide_unknown, /*s_and_b64*/ 0x0F, 12, 12, 120);
      smem(wide_unknown, 0x00, 0, /*s12*/ 6, 0);
      wide_unknown.push_back(sopp(kEndpgm, 0));
      const auto wide_unknown_resources =
          gpu::rdna::ResolveBuffers(wide_unknown.data(), gs_user_data, 2, 8);
      expect(wide_unknown_resources.empty(),
             "unknown wide writes clear both descriptor SGPRs");

      uint32_t stale_scc_user_data[8] = {};
      std::memcpy(stale_scc_user_data + 4, selected_vb, 4 * sizeof(uint32_t));
      std::vector<uint32_t> stale_scc;
      stale_scc.push_back(sopc(/*s_cmp_eq_u32*/ 0x00, 128, 128));
      sop2(stale_scc, /*s_add_u32*/ 0x00, 20, 128, 128);
      sop2(stale_scc, /*s_cselect_b32*/ 0x0A, 15, 15, 120);
      mubuf(stale_scc, 0x00, 12);
      stale_scc.push_back(sopp(kEndpgm, 0));
      const auto stale_scc_resources = gpu::rdna::ResolveBuffers(
          stale_scc.data(), stale_scc_user_data, 8, 8);
      expect(stale_scc_resources.empty(),
             "unmodeled arithmetic SCC invalidates conditional descriptors");

      std::vector<uint32_t> sopk_scc;
      sopk_scc.push_back(sopk(/*s_cmpk_eq_u32*/ 0x09, 14, 6));
      sop2(sopk_scc, /*s_cselect_b32*/ 0x0A, 15, 15, 120);
      mubuf(sopk_scc, 0x00, 12);
      sopk_scc.push_back(sopp(kEndpgm, 0));
      const auto sopk_scc_resources =
          gpu::rdna::ResolveBuffers(sopk_scc.data(), stale_scc_user_data, 8, 8);
      const auto sopk_scc_resource = sopk_scc_resources.find(2);
      expect(sopk_scc_resource != sopk_scc_resources.end() &&
                 sopk_scc_resource->second.descriptor_valid,
             "SOPK comparisons replay SCC without clobbering their source");

      uint32_t malformed[8] = {};
      malformed[0] = 1;
      malformed[1] = 56u << 20;
      malformed[3] = 9u << 28;
      expect(!gpu::rdna::DecodeTImage(malformed).valid,
             "low T# base is rejected");
      malformed[1] = 0xFFu | (56u << 20);
      expect(gpu::rdna::DecodeTImage(malformed).valid,
             "unmapped T# remains valid for live-target resolution");

      uint32_t malformed_view[8] = {};
      writeImage(malformed_view);
      malformed_view[3] = (12u << 28);
      malformed_view[4] = 1u | (3u << 16);
      expect(!gpu::rdna::DecodeTImage(malformed_view).valid,
             "out-of-range array view is rejected");
      writeImage(malformed_view);
      malformed_view[3] = (9u << 28) | (2u << 12) | (1u << 16);
      malformed_view[5] = 3u << 4;
      expect(!gpu::rdna::DecodeTImage(malformed_view).valid,
             "reversed mip view is rejected");
      writeImage(malformed_view);
      malformed_view[3] = (9u << 28) | (1u << 12) | (2u << 16);
      malformed_view[5] = 1u << 4;
      expect(!gpu::rdna::DecodeTImage(malformed_view).valid,
             "mip view beyond allocated levels is rejected");

      munmap(memory, static_cast<size_t>(page_size));
    }
  }

  std::printf(g_failures ? "\nFAILED (%d)\n" : "\nOK\n", g_failures);
  return g_failures ? 1 : 0;
}
