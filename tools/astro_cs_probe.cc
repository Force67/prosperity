/*
 * Scratch probe: decode + resource-walk ONE Astro Bot compute shader against
 * the freshly dumped eboot, without booting the title.
 *
 *   nix develop -c bash tools/build_astro_cs_probe.sh <cs addr hex>
 */
#include "base/arch.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "guest_memory.h"
#include "gcn/gcn_detile.h"
#include "gcn/gcn_disasm.h"
#include "ps5/rdna/rdna_resource.h"
#include "ps5/rdna/rdna_compute.h"

namespace {

constexpr u64 kEbootBase = 0x201400000000ull;

// The dumped module's LOAD segments: (file offset, va, size).
struct LoadMap {
  u64 va;
  u64 off;
  u64 size;
};
std::vector<LoadMap> g_loads;

void ParseLoads(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::printf("no %s\n", path);
    std::exit(1);
  }
  u8 hdr[0x40];
  std::fread(hdr, 1, sizeof(hdr), f);
  u64 phoff;
  u16 phentsize, phnum;
  std::memcpy(&phoff, hdr + 0x20, 8);
  std::memcpy(&phentsize, hdr + 0x36, 2);
  std::memcpy(&phnum, hdr + 0x38, 2);
  for (u32 i = 0; i < phnum; i++) {
    u8 ph[0x38];
    std::fseek(f, (long)(phoff + (u64)i * phentsize), SEEK_SET);
    std::fread(ph, 1, sizeof(ph), f);
    u32 t;
    u64 off, va, fsz;
    std::memcpy(&t, ph, 4);
    std::memcpy(&off, ph + 8, 8);
    std::memcpy(&va, ph + 16, 8);
    std::memcpy(&fsz, ph + 32, 8);
    if (t == 1)
      g_loads.push_back({va, off, fsz});
  }
  std::fclose(f);
}

// Map the bytes of the guest VA window into host memory at that same address,
// so gpu::IsReadableRange (and the decoder) see the shader like the runtime
// does.
void MapVa(u64 va, u64 bytes) {
  const u64 module_va = va - kEbootBase;
  for (const LoadMap& l : g_loads) {
    if (module_va < l.va || module_va + bytes > l.va + l.size)
      continue;
    constexpr u64 kPage = 0x1000;
    const u64 file_off = module_va - l.va + l.off;
    const u64 off0 = file_off & ~(kPage - 1);
    const u64 va0 = va & ~(kPage - 1) - off0;
    const u64 span = bytes + (file_off - off0) + kPage;
    const int fd = open("/tmp/dumped_module.elf", O_RDONLY);
    if (fd < 0)
      std::exit(1);
    void* mem = mmap(reinterpret_cast<void*>(va0), span, PROT_READ,
                     MAP_PRIVATE | MAP_FIXED, fd, (off_t)off0);
    if (mem == MAP_FAILED) {
      perror("mmap");
      std::exit(1);
    }
    close(fd);
    return;
  }
  std::printf("VA %#llx (module va %#llx) not in any LOAD\n", (unsigned long long)va, (unsigned long long)(va - kEbootBase));
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: astro_cs_probe <cs addr>\n");
    return 1;
  }
  const u64 cs_addr = std::strtoull(argv[1], nullptr, 16);
  ParseLoads("/tmp/dumped_module.elf");
  const u64 module_va = cs_addr - kEbootBase;
  u64 file_off = 0;
  for (const LoadMap& l : g_loads) {
    if (module_va >= l.va && module_va + 0x100000 <= l.va + l.size) {
      file_off = module_va - l.va + l.off;
      break;
    }
  }
  std::vector<u32> code(0x40000);
  FILE* f = std::fopen("/tmp/dumped_module.elf", "rb");
  if (!f)
    return 1;
  std::fseek(f, (long)file_off, SEEK_SET);
  std::fread(code.data(), 4, code.size(), f);
  std::fclose(f);
  std::printf("first dwords: %08x %08x %08x %08x\n", code[0], code[1], code[2],
              code[3]);
  const u32* base = code.data();

  const gpu::rdna::Program prog =
      gpu::rdna::DecodeShader(reinterpret_cast<const u32*>(base), 0x40000);
  std::printf("program dwords=%zu\n", prog.size());
  std::printf("== disassembly ==\n");
  for (u32 i = 0; i < prog.size(); i++) {
    std::string line = gpu::gcn::DisasmLine(prog[i]);
    std::printf("  %4llu: %s\n", (unsigned long long)i, line.c_str());
  }

  const gpu::gcn::RecompiledCs rc = gpu::rdna::RecompileCompute(
      base, 64, 1, 1, 0, 0, 0);
  std::printf("recompiled ok=%d spirv=%zu resources=%zu gds=%d\n", (int)rc.ok,
              rc.spirv.size(), rc.resources.size(), (int)rc.gds_binding);
  for (const auto& r : rc.resources) {
    std::printf(
        "  res kind=%d binding=%d written=%d min_bytes=%#llx base_sgpr=%d "
        "use_pc=%u\n",
        (int)r.kind, (int)r.binding, (int)r.written,
        (unsigned long long)r.min_bytes, (int)r.base_sgpr, r.use_pc);
  }

  std::vector<u32> ud(16, 0);
  const auto resolved =
      gpu::rdna::ResolveBuffers(base, ud.data(),
                                16, 0);
  std::printf("\n== descriptors ResolveBuffers found at use PCs ==\n");
  for (const auto& [pc, br] : resolved) {
    if (!br.descriptor_valid) {
      std::printf("  pc=%u INVALID (base=%#llx)\n", pc,
                  (unsigned long long)br.base);
      continue;
    }
    std::printf("  pc=%u base=%#llx dwords=%u:", pc, (unsigned long long)br.base,
                br.descriptor_dwords);
    for (u32 i = 0; i < br.descriptor_dwords && i < 8; i++)
      std::printf(" %08x", br.descriptor[i]);
    std::printf("\n");
  }
  return 0;
}
