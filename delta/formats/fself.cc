
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "formats/fself.h"
#include "base/arch.h"

#include <cstring>

#include "elf_types.h"
#include "sce_types.h"

namespace formats {
base::Vector<u8> FselfToElf(const u8* data, size_t size) {
  base::Vector<u8> out;

  if (size < sizeof(SELFHeader))
    return out;
  const auto* sh = reinterpret_cast<const SELFHeader*>(data);
  if (!isSelfMagic(sh->magic))
    return out;

  const size_t seg_table_off = sizeof(SELFHeader);
  const size_t elf_off = seg_table_off + static_cast<size_t>(sh->numSegments) *
                                             sizeof(SELFSegmentTable);
  if (elf_off + sizeof(ELFHeader) > size)
    return out;

  const auto* eh = reinterpret_cast<const ELFHeader*>(data + elf_off);
  if (eh->magic != ELF_MAGIC)
    return out;

  const size_t phoff = eh->phoff;
  const u16 phentsize = eh->phentsize;
  const u16 phnum = eh->phnum;
  const size_t hdr_end = phoff + static_cast<size_t>(phnum) * phentsize;
  if (elf_off + hdr_end > size)
    return out;

  auto phdr = [&](u32 i) {
    return reinterpret_cast<const ELFPgHeader*>(
        data + elf_off + phoff + static_cast<size_t>(i) * phentsize);
  };

  // The output layout is the union of the program-header extents and the
  // destination of every block segment.
  size_t total = hdr_end;
  for (u16 i = 0; i < phnum; ++i) {
    const size_t end = phdr(i)->offset + phdr(i)->filesz;
    if (end > total)
      total = end;
  }

  const auto* segs =
      reinterpret_cast<const SELFSegmentTable*>(data + seg_table_off);
  for (u16 i = 0; i < sh->numSegments; ++i) {
    const auto& s = segs[i];
    if (!(s.flags & SF_BFLG))
      continue;
    const u32 idx = static_cast<u32>(s.flags >> 20) & 0xFFF;
    if (idx >= phnum)
      continue;
    if (s.offset + s.fileSize > size)  // source out of range
      continue;
    const size_t end = phdr(idx)->offset + s.fileSize;
    if (end > total)
      total = end;
  }

  out.resize(total);  // value-initialized => zero filled
  std::memcpy(out.data(), data + elf_off, hdr_end);

  for (u16 i = 0; i < sh->numSegments; ++i) {
    const auto& s = segs[i];
    if (!(s.flags & SF_BFLG))
      continue;
    const u32 idx = static_cast<u32>(s.flags >> 20) & 0xFFF;
    if (idx >= phnum)
      continue;
    if (s.offset + s.fileSize > size)
      continue;
    std::memcpy(out.data() + phdr(idx)->offset, data + s.offset, s.fileSize);
  }

  return out;
}
}  // namespace formats
