// Synthetic SELF -> ELF round trip for formats::FselfToElf. Mirrors the layout
// the real fake-pkg eboots use, without needing a pkg fixture.
#include <cstring>
#include "base/arch.h"

#include <gtest/gtest.h>

#include "base/containers/vector.h"
#include "elf_types.h"
#include "formats/fself.h"
#include "sce_types.h"

TEST(Fself, RebuildsElfFromFakeSelf) {
  base::Vector<u8> buf(0x400, 0);

  auto* sh = reinterpret_cast<SELFHeader*>(buf.data());
  sh->magic = SELF_MAGIC;
  sh->numSegments = 1;

  // One block segment mapping to program header 0.
  auto* seg = reinterpret_cast<SELFSegmentTable*>(buf.data() + 0x20);
  seg->flags = SF_BFLG | (u64(0) << 20);
  seg->offset = 0x200;
  seg->fileSize = 8;
  seg->memSize = 8;

  const size_t elf_off = 0x40;  // 0x20 header + 1 * 0x20 segment table
  auto* eh = reinterpret_cast<ELFHeader*>(buf.data() + elf_off);
  eh->magic = ELF_MAGIC;
  eh->machine = ELF_MACHINE_X86_64;
  eh->type = ET_SCE_DYNEXEC;
  eh->phoff = 64;
  eh->ehsize = 64;
  eh->phentsize = sizeof(ELFPgHeader);
  eh->phnum = 1;

  auto* ph = reinterpret_cast<ELFPgHeader*>(buf.data() + elf_off + 64);
  ph->type = PT_LOAD;
  ph->flags = PF_R | PF_X;
  ph->offset = 0x100;
  ph->filesz = 8;
  ph->memsz = 8;

  const u8 payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  std::memcpy(buf.data() + 0x200, payload, sizeof(payload));

  auto elf = formats::FselfToElf(buf.data(), buf.size());
  ASSERT_FALSE(elf.empty());
  ASSERT_EQ(elf.size(), 0x108u);  // phdr offset 0x100 + filesz 8

  u32 magic = 0;
  std::memcpy(&magic, elf.data(), 4);
  EXPECT_EQ(magic, ELF_MAGIC);

  auto* out_eh = reinterpret_cast<ELFHeader*>(elf.data());
  EXPECT_EQ(out_eh->phnum, 1);
  EXPECT_EQ(out_eh->phoff, 64u);

  // segment payload landed at the program header's file offset
  EXPECT_EQ(0, std::memcmp(elf.data() + 0x100, payload, sizeof(payload)));
}

TEST(Fself, RejectsNonSelf) {
  base::Vector<u8> buf(0x100, 0);
  u32 not_magic = 0xDEADBEEF;
  std::memcpy(buf.data(), &not_magic, 4);
  EXPECT_TRUE(formats::FselfToElf(buf.data(), buf.size()).empty());
}
