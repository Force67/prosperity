/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking. See gcn_resource.h.
 */

#include "gcn_resource.h"
#include "gcn_decode.h"

#include <cstdio>
#include <cstdlib>

namespace gpu::gcn {
namespace {
const bool g_trace = std::getenv("DELTA_GPU_TRACE") != nullptr;
}

VBuffer decodeVBuffer(const uint32_t *p) {
  // GCN V# (buffer resource descriptor), 4 dwords:
  //  [0]      base_address[31:0]
  //  [1] 0:43 base_address[47:32] in [11:0]; stride[13:0] in [29:16]
  //  [2]      num_records
  //  [3]      dst_sel/nfmt/dfmt/...: dfmt[18:15], nfmt[21:19]
  VBuffer v;
  v.base = (static_cast<uint64_t>(p[1] & 0xFFF) << 32) | p[0];
  v.stride = (p[1] >> 16) & 0x3FFF;
  v.numRecords = p[2];
  v.dfmt = (p[3] >> 15) & 0xF;
  v.nfmt = (p[3] >> 19) & 0x7;
  return v;
}

// SMRD operand fields (GFX6/7).
struct Smrd {
  uint32_t op, sdst, sbase, offset;
  bool imm;
};
Smrd decodeSmrd(uint32_t w) {
  Smrd s;
  s.op = (w >> 22) & 0x1F;
  s.sdst = (w >> 15) & 0x7F;
  s.sbase = (w >> 9) & 0x3F;  // SGPR pair index; actual base SGPR = sbase*2
  s.imm = (w >> 8) & 1;
  s.offset = w & 0xFF;
  return s;
}

std::vector<VBuffer> trackVertexBuffers(const uint32_t *fetchCode,
                                        uint32_t maxDwords,
                                        const uint32_t *vsUserData) {
  std::vector<VBuffer> result;
  if (!fetchCode || !vsUserData)
    return result;
  auto insts = decode(fetchCode, maxDwords);

  // The fetch shader loads each attribute's V# with an s_load_dwordx4 whose
  // SBASE is a user-SGPR pair holding the vertex-buffer-table pointer, at byte
  // offset (offset*4 for imm). Recover the table pointer from the user data and
  // read the V# there.
  for (const auto &in : insts) {
    if (in.enc != Enc::smrd)
      continue;
    Smrd s = decodeSmrd(in.raw[0]);
    if (s.op != 0x02)  // s_load_dwordx4 (a 4-dword V#)
      continue;
    uint32_t baseSgpr = s.sbase * 2;  // user_data index of the table pointer
    if (baseSgpr + 1 >= 16)
      continue;
    uint64_t table = (static_cast<uint64_t>(vsUserData[baseSgpr + 1] & 0xFFFF) << 32) |
                     vsUserData[baseSgpr];
    if (table < 0x1000000000ull || table >= 0x20000000000ull)
      continue;
    uint32_t byteOff = s.imm ? s.offset * 4 : 0;
    auto *vptr = reinterpret_cast<const uint32_t *>(table + byteOff);
    VBuffer v = decodeVBuffer(vptr);
    if (v.base >= 0x1000000000ull && v.base < 0x20000000000ull && v.stride &&
        v.stride <= 256 && v.numRecords && v.numRecords <= 0x100000) {
      if (g_trace)
        std::fprintf(stderr,
                     "[gcnres] VB sbase=sgpr%u table=%#lx off=%u -> base=%#lx "
                     "stride=%u nrec=%u dfmt=%u nfmt=%u\n",
                     baseSgpr, (unsigned long)table, byteOff,
                     (unsigned long)v.base, v.stride, v.numRecords, v.dfmt, v.nfmt);
      result.push_back(v);
    }
  }
  return result;
}

}  // namespace gpu::gcn
