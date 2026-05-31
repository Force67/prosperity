
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "UnSELF.h"

#include <cstring>

#include <elf_types.h>
#include <sce_types.h>

namespace crypto {
base::Vector<uint8_t> self2elf(const uint8_t *data, size_t size) {
  base::Vector<uint8_t> out;

  if (size < sizeof(SELFHeader))
    return out;
  const auto *sh = reinterpret_cast<const SELFHeader *>(data);
  if (sh->magic != SELF_MAGIC)
    return out;

  const size_t segTableOff = sizeof(SELFHeader);
  const size_t elfOff =
      segTableOff + static_cast<size_t>(sh->numSegments) * sizeof(SELFSegmentTable);
  if (elfOff + sizeof(ELFHeader) > size)
    return out;

  const auto *eh = reinterpret_cast<const ELFHeader *>(data + elfOff);
  if (eh->magic != ELF_MAGIC)
    return out;

  const size_t phoff = eh->phoff;
  const uint16_t phentsize = eh->phentsize;
  const uint16_t phnum = eh->phnum;
  const size_t hdrEnd = phoff + static_cast<size_t>(phnum) * phentsize;
  if (elfOff + hdrEnd > size)
    return out;

  auto phdr = [&](uint32_t i) {
    return reinterpret_cast<const ELFPgHeader *>(
        data + elfOff + phoff + static_cast<size_t>(i) * phentsize);
  };

  // The output layout is the union of the program-header extents and the
  // destination of every block segment.
  size_t total = hdrEnd;
  for (uint16_t i = 0; i < phnum; ++i) {
    const size_t end = phdr(i)->offset + phdr(i)->filesz;
    if (end > total)
      total = end;
  }

  const auto *segs =
      reinterpret_cast<const SELFSegmentTable *>(data + segTableOff);
  for (uint16_t i = 0; i < sh->numSegments; ++i) {
    const auto &s = segs[i];
    if (!(s.flags & SF_BFLG))
      continue;
    const uint32_t idx = static_cast<uint32_t>(s.flags >> 20) & 0xFFF;
    if (idx >= phnum)
      continue;
    if (s.offset + s.fileSize > size) // source out of range
      continue;
    const size_t end = phdr(idx)->offset + s.fileSize;
    if (end > total)
      total = end;
  }

  out.resize(total); // value-initialized => zero filled
  std::memcpy(out.data(), data + elfOff, hdrEnd);

  for (uint16_t i = 0; i < sh->numSegments; ++i) {
    const auto &s = segs[i];
    if (!(s.flags & SF_BFLG))
      continue;
    const uint32_t idx = static_cast<uint32_t>(s.flags >> 20) & 0xFFF;
    if (idx >= phnum)
      continue;
    if (s.offset + s.fileSize > size)
      continue;
    std::memcpy(out.data() + phdr(idx)->offset, data + s.offset, s.fileSize);
  }

  return out;
}
} // namespace crypto
