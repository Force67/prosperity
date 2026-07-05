
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// based off https://github.com/Zer0xFF/ps4-pup-unpacker/blob/master/PUP.cpp

#include "pup_object.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <zlib.h>

namespace vfs {
namespace {
struct fileNode {
  uint32_t id;
  const char *name;
};

// Well-known PUP segment ids -> human file names (the rest land as
// segment_<id>.bin). These are container images / firmware blobs, not modules.
const fileNode knownFileNames[] = {
    {3, "wlan_firmware.bin"}, {5, "secure_modules.bin"},
    {6, "system.img"},        {8, "eap.img"},
    {9, "recovery.img"},      {11, "preinst.img"},
    {12, "system_ex.img"},    {34, "torus2_firmware.bin"},
    {257, "eula.xml"},        {512, "orbis_swu.self"},
    {514, "orbis_swu.self"},  {3337, "cp_firmware.bin"}};

const char *knownName(uint32_t id) {
  for (const auto &n : knownFileNames)
    if (n.id == id)
      return n.name;
  return nullptr;
}

void appendLine(base::String &s, const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  s += buf;
}

// PS5 entry flag bits (decrypted PUP).
bool ps5Compressed(uint32_t flags) { return (flags & 0x8u) != 0; }
bool ps5Blocked(uint32_t flags) { return (flags & 0x800u) != 0; }
bool ps5IsTable(uint32_t flags) { return (flags & 0x1u) != 0; }
bool ps5IsSpecial(uint32_t flags) {
  uint32_t s = flags & 0xF0000000u;
  return s == 0xE0000000u || s == 0xF0000000u;
}
// Uncompressed block size = 1 << (((flags >> 12) & 0xF) + 12).
uint32_t ps5BlockSize(uint32_t flags) {
  return 1u << (((flags >> 12) & 0xFu) + 12u);
}

// One (offset,size) extent record from a block table.
struct blockExtent {
  uint32_t offset;
  uint32_t size;
};

// Inflate exactly one zlib stream into a buffer of the known output size.
bool inflateBlock(const uint8_t *in, size_t inLen, uint8_t *out, size_t outLen) {
  uLongf dst = static_cast<uLongf>(outLen);
  int r = uncompress(out, &dst, in, static_cast<uLong>(inLen));
  return r == Z_OK && dst == outLen;
}

// Pick a file extension from the leading bytes of a segment's plaintext.
const char *sniffExt(const uint8_t *p, size_t n) {
  auto has = [&](const char *sig, size_t len, size_t at = 0) {
    return n >= at + len && std::memcmp(p + at, sig, len) == 0;
  };
  if (has("\x7f"
          "ELF",
          4))
    return ".self";
  if (has("\x54\x14\xf5\xee", 4) || has("\x4f\x15\x3d\x1d", 4))
    return ".pup"; // nested PS5/PS4 update
  if (has("SLB2", 4))
    return ".slb2";
  if (has("PK\x03\x04", 4))
    return ".zip";
  if (has("<?xml", 5))
    return ".xml";
  if (has("EXFAT   ", 8, 3) || has("NTFS    ", 8, 3))
    return ".img"; // filesystem image
  if (n > 0 && (p[0] == '{' || p[0] == '['))
    return ".json";
  return ".bin";
}
} // namespace

pupReader::pupReader(const base::String &name) : file(name) {}

bool pupReader::load() {
  if (!file.IsOpen())
    return false;
  if (!file.Read(header))
    return false;
  // The PUP container header/entry table is plaintext even on retail firmware
  // (only the segment payloads are encrypted), so the magic is a reliable gate.
  if (header.magic == kPupMagicPS5)
    isPS5 = true;
  else if (header.magic != kPupMagicPS4)
    return false;

  for (int i = 0; i < header.numSegments; i++) {
    pup_entry e{};
    if (!file.Read(e))
      break;
    entries.emplace_back(e);
  }
  return entries.size() == static_cast<size_t>(header.numSegments);
}

bool pupReader::inflateEntry(const pup_entry &e, base::Vector<uint8_t> &in,
                             base::Vector<uint8_t> &out) {
  if (e.sizeUncompressed == 0 || e.sizeUncompressed > (1ull << 32))
    return false;
  out.resize(static_cast<size_t>(e.sizeUncompressed));
  uLongf dstLen = static_cast<uLongf>(e.sizeUncompressed);
  int r = uncompress(out.data(), &dstLen, in.data(),
                     static_cast<uLong>(in.size()));
  if (r != Z_OK)
    return false;
  out.resize(static_cast<size_t>(dstLen));
  return true;
}

base::String pupReader::extractAll(const base::String &outDir,
                                   bool &looksEncrypted) {
  base::String summary;
  looksEncrypted = false;
  if (!file.IsOpen()) {
    summary += "PUP not open\n";
    return summary;
  }

  // A decrypted PS5 PUP is plaintext; the block-compressed extractor handles it
  // in full (no encryption to defeat), so looksEncrypted stays false.
  if (isPS5)
    return extractAllPS5(outDir);

  appendLine(summary, "PUP container: %u segment(s)\n",
             static_cast<unsigned>(header.numSegments));

  int written = 0, failed = 0;
  for (size_t i = 0; i < entries.size(); i++) {
    const auto &e = entries[i];
    uint32_t special = e.flags & 0xF0000000u;
    if (special == 0xE0000000u || special == 0xF0000000u)
      continue; // signature / table blocks, not file segments
    uint32_t id = e.flags >> 20;
    bool compressed = (e.flags & 0x8u) != 0;

    base::Vector<uint8_t> raw;
    file.Seek(e.offset, utl::seekMode::seek_set);
    if (!file.Read(raw, static_cast<size_t>(e.sizeCompressed))) {
      failed++;
      appendLine(summary, "  [%u] read failed\n", id);
      continue;
    }

    const uint8_t *payload = raw.data();
    size_t payloadLen = raw.size();
    base::Vector<uint8_t> inflated;
    if (compressed) {
      if (inflateEntry(e, raw, inflated)) {
        payload = inflated.data();
        payloadLen = inflated.size();
      } else {
        // Encrypted payload won't inflate: keep the raw bytes, flag it.
        looksEncrypted = true;
      }
    }

    const char *kn = knownName(id);
    char fname[64];
    if (kn)
      std::snprintf(fname, sizeof(fname), "%s", kn);
    else
      std::snprintf(fname, sizeof(fname), "segment_%u.bin", id);

    base::String outPath = outDir;
    if (!outPath.empty() && outPath.back() != '/')
      outPath += "/";
    outPath += fname;
    utl::File out(outPath, utl::fileMode::write);
    if (!out.IsOpen()) {
      failed++;
      appendLine(summary, "  [%u] %s: cannot write\n", id, fname);
      continue;
    }
    out.Write(payload, payloadLen);
    written++;
    appendLine(summary, "  [%u] %s (%zu bytes%s)\n", id, fname, payloadLen,
               compressed ? (looksEncrypted ? ", raw" : ", inflated") : "");
  }

  appendLine(summary, "extracted %d segment(s), %d failed\n", written, failed);
  if (looksEncrypted)
    summary += "NOTE: segments did not decompress - this PUP is encrypted. "
               "Decrypted firmware modules (.sprx) cannot be recovered here; "
               "import a pre-extracted module set instead.\n";
  else
    summary += "NOTE: extracted the container images (system_ex.img etc.). The "
               "modules inside them are encrypted SELFs; import a pre-extracted "
               ".sprx module set to actually install firmware.\n";
  return summary;
}

// The block table for a data segment at index N is the entry flagged as a table
// (bit 0) whose id (flags >> 20) equals N. It always precedes the data entry.
int pupReader::tableForData(size_t dataIdx) const {
  for (size_t j = 0; j < entries.size(); j++) {
    uint32_t f = entries[j].flags;
    if (ps5IsTable(f) && (f >> 20) == dataIdx)
      return static_cast<int>(j);
  }
  return -1;
}

bool pupReader::extractPS5Segment(const pup_entry &e, size_t idx,
                                  const base::String &outDir,
                                  base::String &summary) {
  uint32_t id = e.flags >> 20;

  // Read the first plaintext chunk so we can sniff a file extension, then keep
  // streaming the rest. Everything below writes at most one block at a time.
  base::Vector<uint8_t> first; // decoded bytes of the first block/chunk
  base::Vector<blockExtent> exts;
  uint32_t blockSize = 0;
  bool blocked = ps5Compressed(e.flags) && ps5Blocked(e.flags);

  auto readAt = [&](uint64_t off, base::Vector<uint8_t> &buf, size_t n) {
    file.Seek(off, utl::seekMode::seek_set);
    return file.Read(buf, n);
  };

  if (blocked) {
    blockSize = ps5BlockSize(e.flags);
    uint64_t blockCount = (e.sizeUncompressed + blockSize - 1) / blockSize;
    int ti = tableForData(idx);
    if (ti < 0) {
      appendLine(summary, "  [%u] no block table\n", id);
      return false;
    }
    const auto &t = entries[ti];
    base::Vector<uint8_t> tbl;
    if (!readAt(t.offset, tbl, static_cast<size_t>(t.sizeCompressed)) ||
        tbl.size() < blockCount * 40) {
      appendLine(summary, "  [%u] bad block table\n", id);
      return false;
    }
    // Layout: blockCount digests (32 bytes) followed by blockCount extents.
    size_t extBase = static_cast<size_t>(blockCount) * 32;
    exts.resize(static_cast<size_t>(blockCount));
    for (uint64_t b = 0; b < blockCount; b++)
      std::memcpy(&exts[static_cast<size_t>(b)], tbl.data() + extBase + b * 8, 8);
  }

  // Produce the first chunk's plaintext to sniff the type.
  base::Vector<uint8_t> raw;
  if (!ps5Compressed(e.flags)) {
    // Stored plain (possibly block-hashed): the payload is the file itself.
    size_t peek = static_cast<size_t>(
        std::min<uint64_t>(e.sizeCompressed, 0x1000ull));
    if (!readAt(e.offset, first, peek))
      return false;
  } else if (!blocked) {
    if (!readAt(e.offset, raw, static_cast<size_t>(e.sizeCompressed)))
      return false;
    first.resize(static_cast<size_t>(e.sizeUncompressed));
    if (!inflateBlock(raw.data(), raw.size(), first.data(), first.size())) {
      appendLine(summary, "  [%u] inflate failed\n", id);
      return false;
    }
  } else {
    uint32_t ublk = static_cast<uint32_t>(
        std::min<uint64_t>(blockSize, e.sizeUncompressed));
    size_t stored = exts.size() > 1
                        ? exts[1].offset - exts[0].offset
                        : static_cast<size_t>(e.sizeCompressed) - exts[0].offset;
    if (!readAt(e.offset + exts[0].offset, raw, stored))
      return false;
    first.resize(ublk);
    if (exts[0].size >= ublk) // block stored raw
      std::memcpy(first.data(), raw.data(), ublk);
    else if (!inflateBlock(raw.data(), raw.size(), first.data(), ublk)) {
      appendLine(summary, "  [%u] block 0 inflate failed\n", id);
      return false;
    }
  }

  const char *ext = sniffExt(first.data(), first.size());
  char fname[64];
  std::snprintf(fname, sizeof(fname), "segment_%u%s", id, ext);
  base::String outPath = outDir;
  if (!outPath.empty() && outPath.back() != '/')
    outPath += "/";
  outPath += fname;
  utl::File out(outPath, utl::fileMode::write);
  if (!out.IsOpen()) {
    appendLine(summary, "  [%u] %s: cannot write\n", id, fname);
    return false;
  }
  out.Write(first.data(), first.size());
  uint64_t written = first.size();

  // Stream the remaining data.
  if (!ps5Compressed(e.flags)) {
    // Copy the rest of the stored payload in chunks.
    uint64_t remaining = e.sizeCompressed - first.size();
    uint64_t pos = e.offset + first.size();
    base::Vector<uint8_t> buf;
    while (remaining) {
      size_t n = static_cast<size_t>(std::min<uint64_t>(remaining, 1u << 20));
      if (!readAt(pos, buf, n))
        break;
      out.Write(buf.data(), n);
      pos += n;
      written += n;
      remaining -= n;
    }
  } else if (blocked) {
    for (size_t b = 1; b < exts.size(); b++) {
      uint32_t ublk = static_cast<uint32_t>(std::min<uint64_t>(
          blockSize, e.sizeUncompressed - static_cast<uint64_t>(b) * blockSize));
      size_t stored = b + 1 < exts.size()
                          ? exts[b + 1].offset - exts[b].offset
                          : static_cast<size_t>(e.sizeCompressed) -
                                exts[b].offset;
      if (!readAt(e.offset + exts[b].offset, raw, stored))
        break;
      base::Vector<uint8_t> dec(ublk);
      if (exts[b].size >= ublk)
        std::memcpy(dec.data(), raw.data(), ublk);
      else if (!inflateBlock(raw.data(), raw.size(), dec.data(), ublk)) {
        appendLine(summary, "  [%u] block %zu inflate failed\n", id, b);
        return false;
      }
      out.Write(dec.data(), ublk);
      written += ublk;
    }
  }

  appendLine(summary, "  [%u] %s (%llu bytes)\n", id, fname,
             static_cast<unsigned long long>(written));
  return written == e.sizeUncompressed;
}

base::String pupReader::extractAllPS5(const base::String &outDir) {
  base::String summary;
  appendLine(summary, "PS5 PUP container: %u segment(s)\n",
             static_cast<unsigned>(header.numSegments));

  int written = 0, failed = 0;
  for (size_t i = 0; i < entries.size(); i++) {
    const auto &e = entries[i];
    // Skip the special (0xE.../0xF...) ScePupMetadataEntry table - it holds the
    // per-segment AES key/IV/digest/HMAC, but a console-oracle *.PUP.dec has
    // already consumed and zeroed it - and the per-segment block tables. Only
    // the actual data segments become files.
    if (ps5IsSpecial(e.flags) || ps5IsTable(e.flags))
      continue;
    if (extractPS5Segment(e, i, outDir, summary))
      written++;
    else
      failed++;
  }

  appendLine(summary, "extracted %d segment(s), %d failed\n", written, failed);
  summary += "NOTE: this is a decrypted PS5 PUP; the container segments "
             "(filesystem images, SLB2 blobs, nested PUPs) are recovered in "
             "full. SELF modules inside those images are still encrypted.\n";
  return summary;
}
} // namespace vfs
