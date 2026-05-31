// Synthetic SELF -> ELF round trip for crypto::self2elf. Mirrors the layout the
// real fake-pkg eboots use, without needing a pkg fixture.
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <crypto/UnSELF.h>
#include <elf_types.h>
#include <sce_types.h>

TEST(UnSELF, RebuildsElfFromFakeSelf) {
  std::vector<uint8_t> buf(0x400, 0);

  auto *sh = reinterpret_cast<SELFHeader *>(buf.data());
  sh->magic = SELF_MAGIC;
  sh->numSegments = 1;

  // One block segment mapping to program header 0.
  auto *seg = reinterpret_cast<SELFSegmentTable *>(buf.data() + 0x20);
  seg->flags = SF_BFLG | (uint64_t(0) << 20);
  seg->offset = 0x200;
  seg->fileSize = 8;
  seg->memSize = 8;

  const size_t elfOff = 0x40; // 0x20 header + 1 * 0x20 segment table
  auto *eh = reinterpret_cast<ELFHeader *>(buf.data() + elfOff);
  eh->magic = ELF_MAGIC;
  eh->machine = ELF_MACHINE_X86_64;
  eh->type = ET_SCE_DYNEXEC;
  eh->phoff = 64;
  eh->ehsize = 64;
  eh->phentsize = sizeof(ELFPgHeader);
  eh->phnum = 1;

  auto *ph = reinterpret_cast<ELFPgHeader *>(buf.data() + elfOff + 64);
  ph->type = PT_LOAD;
  ph->flags = PF_R | PF_X;
  ph->offset = 0x100;
  ph->filesz = 8;
  ph->memsz = 8;

  const uint8_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  std::memcpy(buf.data() + 0x200, payload, sizeof(payload));

  auto elf = crypto::self2elf(buf.data(), buf.size());
  ASSERT_FALSE(elf.empty());
  ASSERT_EQ(elf.size(), 0x108u); // phdr offset 0x100 + filesz 8

  uint32_t magic = 0;
  std::memcpy(&magic, elf.data(), 4);
  EXPECT_EQ(magic, ELF_MAGIC);

  auto *outEh = reinterpret_cast<ELFHeader *>(elf.data());
  EXPECT_EQ(outEh->phnum, 1);
  EXPECT_EQ(outEh->phoff, 64u);

  // segment payload landed at the program header's file offset
  EXPECT_EQ(0, std::memcmp(elf.data() + 0x100, payload, sizeof(payload)));
}

TEST(UnSELF, RejectsNonSelf) {
  std::vector<uint8_t> buf(0x100, 0);
  uint32_t notMagic = 0xDEADBEEF;
  std::memcpy(buf.data(), &notMagic, 4);
  EXPECT_TRUE(crypto::self2elf(buf.data(), buf.size()).empty());
}
