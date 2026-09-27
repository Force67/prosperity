
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// based off https://github.com/Zer0xFF/ps4-pup-unpacker/blob/master/PUP.cpp

#include "formats/pup_reader.h"
#include "base/arch.h"

#include <zlib.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/strings/xstring.h"

namespace vfs {
namespace {
struct FileNode {
  u32 id;
  const char* name;
};

// Well-known PUP segment ids -> human file names (the rest land as
// segment_<id>.bin). These are container images / firmware blobs, not modules.
const FileNode kKnownFileNames[] = {
    {3, "wlan_firmware.bin"}, {5, "secure_modules.bin"},
    {6, "system.img"},        {8, "eap.img"},
    {9, "recovery.img"},      {11, "preinst.img"},
    {12, "system_ex.img"},    {34, "torus2_firmware.bin"},
    {257, "eula.xml"},        {512, "orbis_swu.self"},
    {514, "orbis_swu.self"},  {3337, "cp_firmware.bin"}};

const char* KnownName(u32 id) {
  for (const auto& n : kKnownFileNames)
    if (n.id == id)
      return n.name;
  return nullptr;
}

void AppendLine(base::String& s, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  s += buf;
}

// PS5 entry flag bits (decrypted PUP).
bool Ps5Compressed(u32 flags) {
  return (flags & 0x8u) != 0;
}
bool Ps5Blocked(u32 flags) {
  return (flags & 0x800u) != 0;
}
bool Ps5IsTable(u32 flags) {
  return (flags & 0x1u) != 0;
}
bool Ps5IsSpecial(u32 flags) {
  u32 s = flags & 0xF0000000u;
  return s == 0xE0000000u || s == 0xF0000000u;
}
// Uncompressed block size = 1 << (((flags >> 12) & 0xF) + 12).
u32 Ps5BlockSize(u32 flags) {
  return 1u << (((flags >> 12) & 0xFu) + 12u);
}

// One (offset,size) extent record from a block table.
struct BlockExtent {
  u32 offset;
  u32 size;
};

// Inflate exactly one zlib stream into a buffer of the known output size.
bool InflateBlock(const u8* in, size_t in_len, u8* out, size_t out_len) {
  uLongf dst = static_cast<uLongf>(out_len);
  int r = uncompress(out, &dst, in, static_cast<uLong>(in_len));
  return r == Z_OK && dst == out_len;
}

// Pick a file extension from the leading bytes of a segment's plaintext.
const char* SniffExt(const u8* p, size_t n) {
  auto has = [&](const char* sig, size_t len, size_t at = 0) {
    return n >= at + len && std::memcmp(p + at, sig, len) == 0;
  };
  if (has("\x7f"
          "ELF",
          4))
    return ".self";
  if (has("\x54\x14\xf5\xee", 4) || has("\x4f\x15\x3d\x1d", 4))
    return ".pup";  // nested PS5/PS4 update
  if (has("SLB2", 4))
    return ".slb2";
  if (has("PK\x03\x04", 4))
    return ".zip";
  if (has("<?xml", 5))
    return ".xml";
  if (has("EXFAT   ", 8, 3) || has("NTFS    ", 8, 3))
    return ".img";  // filesystem image
  if (n > 0 && (p[0] == '{' || p[0] == '['))
    return ".json";
  return ".bin";
}
}  // namespace

PupReader::PupReader(const base::String& name) : file_(name) {}

bool PupReader::Load() {
  if (!file_.IsOpen())
    return false;
  if (!file_.Read(header_))
    return false;
  // The PUP container header/entry table is plaintext even on retail firmware
  // (only the segment payloads are encrypted), so the magic is a reliable gate.
  if (header_.magic == kPupMagicPS5)
    is_ps5_ = true;
  else if (header_.magic != kPupMagicPS4)
    return false;

  for (int i = 0; i < header_.num_segments; i++) {
    PupEntry e{};
    if (!file_.Read(e))
      break;
    entries_.emplace_back(e);
  }
  return entries_.size() == static_cast<size_t>(header_.num_segments);
}

bool PupReader::InflateEntry(const PupEntry& e,
                             base::Vector<u8>& in,
                             base::Vector<u8>& out) {
  if (e.size_uncompressed == 0 || e.size_uncompressed > (1ull << 32))
    return false;
  out.resize(static_cast<size_t>(e.size_uncompressed));
  uLongf dst_len = static_cast<uLongf>(e.size_uncompressed);
  int r = uncompress(out.data(), &dst_len, in.data(),
                     static_cast<uLong>(in.size()));
  if (r != Z_OK)
    return false;
  out.resize(static_cast<size_t>(dst_len));
  return true;
}

base::String PupReader::ExtractAll(const base::String& out_dir,
                                   bool& looks_encrypted) {
  base::String summary;
  looks_encrypted = false;
  if (!file_.IsOpen()) {
    summary += "PUP not open\n";
    return summary;
  }

  // A decrypted PS5 PUP is plaintext; the block-compressed extractor handles it
  // in full (no encryption to defeat), so looksEncrypted stays false.
  if (is_ps5_)
    return ExtractAllPS5(out_dir);

  AppendLine(summary, "PUP container: %u segment(s)\n",
             static_cast<unsigned>(header_.num_segments));

  int written = 0, failed = 0;
  for (size_t i = 0; i < entries_.size(); i++) {
    const auto& e = entries_[i];
    u32 special = e.flags & 0xF0000000u;
    if (special == 0xE0000000u || special == 0xF0000000u)
      continue;  // signature / table blocks, not file segments
    u32 id = e.flags >> 20;
    bool compressed = (e.flags & 0x8u) != 0;

    base::Vector<u8> raw;
    file_.Seek(e.offset, io::SeekMode::kSeekSet);
    if (!file_.Read(raw, static_cast<size_t>(e.size_compressed))) {
      failed++;
      AppendLine(summary, "  [%u] read failed\n", id);
      continue;
    }

    const u8* payload = raw.data();
    size_t payload_len = raw.size();
    base::Vector<u8> inflated;
    if (compressed) {
      if (InflateEntry(e, raw, inflated)) {
        payload = inflated.data();
        payload_len = inflated.size();
      } else {
        // Encrypted payload won't inflate: keep the raw bytes, flag it.
        looks_encrypted = true;
      }
    }

    const char* kn = KnownName(id);
    char fname[64];
    if (kn)
      std::snprintf(fname, sizeof(fname), "%s", kn);
    else
      std::snprintf(fname, sizeof(fname), "segment_%u.bin", id);

    base::String out_path = out_dir;
    if (!out_path.empty() && out_path.back() != '/')
      out_path += "/";
    out_path += fname;
    io::File out(out_path, io::FileMode::kWrite);
    if (!out.IsOpen()) {
      failed++;
      AppendLine(summary, "  [%u] %s: cannot write\n", id, fname);
      continue;
    }
    out.Write(payload, payload_len);
    written++;
    AppendLine(summary, "  [%u] %s (%zu bytes%s)\n", id, fname, payload_len,
               compressed ? (looks_encrypted ? ", raw" : ", inflated") : "");
  }

  AppendLine(summary, "extracted %d segment(s), %d failed\n", written, failed);
  if (looks_encrypted)
    summary +=
        "NOTE: segments did not decompress - this PUP is encrypted. "
        "Decrypted firmware modules (.sprx) cannot be recovered here; "
        "import a pre-extracted module set instead.\n";
  else
    summary +=
        "NOTE: extracted the container images (system_ex.img etc.). The "
        "modules inside them are encrypted SELFs; import a pre-extracted "
        ".sprx module set to actually install firmware.\n";
  return summary;
}

// The block table for a data segment at index N is the entry flagged as a table
// (bit 0) whose id (flags >> 20) equals N. It always precedes the data entry.
int PupReader::TableForData(size_t data_idx) const {
  for (size_t j = 0; j < entries_.size(); j++) {
    u32 f = entries_[j].flags;
    if (Ps5IsTable(f) && (f >> 20) == data_idx)
      return static_cast<int>(j);
  }
  return -1;
}

bool PupReader::ExtractPS5Segment(const PupEntry& e,
                                  size_t idx,
                                  const base::String& out_dir,
                                  base::String& summary) {
  u32 id = e.flags >> 20;

  // Read the first plaintext chunk so we can sniff a file extension, then keep
  // streaming the rest. Everything below writes at most one block at a time.
  base::Vector<u8> first;  // decoded bytes of the first block/chunk
  base::Vector<BlockExtent> exts;
  u32 block_size = 0;
  bool blocked = Ps5Compressed(e.flags) && Ps5Blocked(e.flags);

  auto read_at = [&](u64 off, base::Vector<u8>& buf, size_t n) {
    file_.Seek(off, io::SeekMode::kSeekSet);
    return file_.Read(buf, n);
  };

  if (blocked) {
    block_size = Ps5BlockSize(e.flags);
    u64 block_count = (e.size_uncompressed + block_size - 1) / block_size;
    int ti = TableForData(idx);
    if (ti < 0) {
      AppendLine(summary, "  [%u] no block table\n", id);
      return false;
    }
    const auto& t = entries_[ti];
    base::Vector<u8> tbl;
    if (!read_at(t.offset, tbl, static_cast<size_t>(t.size_compressed)) ||
        tbl.size() < block_count * 40) {
      AppendLine(summary, "  [%u] bad block table\n", id);
      return false;
    }
    // Layout: blockCount digests (32 bytes) followed by blockCount extents.
    size_t ext_base = static_cast<size_t>(block_count) * 32;
    exts.resize(static_cast<size_t>(block_count));
    for (u64 b = 0; b < block_count; b++)
      std::memcpy(&exts[static_cast<size_t>(b)], tbl.data() + ext_base + b * 8,
                  8);
  }

  // Produce the first chunk's plaintext to sniff the type.
  base::Vector<u8> raw;
  if (!Ps5Compressed(e.flags)) {
    // Stored plain (possibly block-hashed): the payload is the file itself.
    size_t peek =
        static_cast<size_t>(base::Min<u64>(e.size_compressed, 0x1000ull));
    if (!read_at(e.offset, first, peek))
      return false;
  } else if (!blocked) {
    if (!read_at(e.offset, raw, static_cast<size_t>(e.size_compressed)))
      return false;
    first.resize(static_cast<size_t>(e.size_uncompressed));
    if (!InflateBlock(raw.data(), raw.size(), first.data(), first.size())) {
      AppendLine(summary, "  [%u] inflate failed\n", id);
      return false;
    }
  } else {
    u32 ublk =
        static_cast<u32>(base::Min<u64>(block_size, e.size_uncompressed));
    size_t stored = exts.size() > 1 ? exts[1].offset - exts[0].offset
                                    : static_cast<size_t>(e.size_compressed) -
                                          exts[0].offset;
    if (!read_at(e.offset + exts[0].offset, raw, stored))
      return false;
    first.resize(ublk);
    if (exts[0].size >= ublk)  // block stored raw
      std::memcpy(first.data(), raw.data(), ublk);
    else if (!InflateBlock(raw.data(), raw.size(), first.data(), ublk)) {
      AppendLine(summary, "  [%u] block 0 inflate failed\n", id);
      return false;
    }
  }

  const char* ext = SniffExt(first.data(), first.size());
  char fname[64];
  std::snprintf(fname, sizeof(fname), "segment_%u%s", id, ext);
  base::String out_path = out_dir;
  if (!out_path.empty() && out_path.back() != '/')
    out_path += "/";
  out_path += fname;
  io::File out(out_path, io::FileMode::kWrite);
  if (!out.IsOpen()) {
    AppendLine(summary, "  [%u] %s: cannot write\n", id, fname);
    return false;
  }
  out.Write(first.data(), first.size());
  u64 written = first.size();

  // Stream the remaining data.
  if (!Ps5Compressed(e.flags)) {
    // Copy the rest of the stored payload in chunks.
    u64 remaining = e.size_compressed - first.size();
    u64 pos = e.offset + first.size();
    base::Vector<u8> buf;
    while (remaining) {
      size_t n = static_cast<size_t>(base::Min<u64>(remaining, 1u << 20));
      if (!read_at(pos, buf, n))
        break;
      out.Write(buf.data(), n);
      pos += n;
      written += n;
      remaining -= n;
    }
  } else if (blocked) {
    for (size_t b = 1; b < exts.size(); b++) {
      u32 ublk = static_cast<u32>(base::Min<u64>(
          block_size, e.size_uncompressed - static_cast<u64>(b) * block_size));
      size_t stored =
          b + 1 < exts.size()
              ? exts[b + 1].offset - exts[b].offset
              : static_cast<size_t>(e.size_compressed) - exts[b].offset;
      if (!read_at(e.offset + exts[b].offset, raw, stored))
        break;
      base::Vector<u8> dec(ublk);
      if (exts[b].size >= ublk)
        std::memcpy(dec.data(), raw.data(), ublk);
      else if (!InflateBlock(raw.data(), raw.size(), dec.data(), ublk)) {
        AppendLine(summary, "  [%u] block %zu inflate failed\n", id, b);
        return false;
      }
      out.Write(dec.data(), ublk);
      written += ublk;
    }
  }

  AppendLine(summary, "  [%u] %s (%llu bytes)\n", id, fname,
             static_cast<unsigned long long>(written));
  return written == e.size_uncompressed;
}

base::String PupReader::ExtractAllPS5(const base::String& out_dir) {
  base::String summary;
  AppendLine(summary, "PS5 PUP container: %u segment(s)\n",
             static_cast<unsigned>(header_.num_segments));

  int written = 0, failed = 0;
  for (size_t i = 0; i < entries_.size(); i++) {
    const auto& e = entries_[i];
    // Skip the special (0xE.../0xF...) ScePupMetadataEntry table - it holds the
    // per-segment AES key/IV/digest/HMAC, but a console-oracle *.PUP.dec has
    // already consumed and zeroed it - and the per-segment block tables. Only
    // the actual data segments become files.
    if (Ps5IsSpecial(e.flags) || Ps5IsTable(e.flags))
      continue;
    if (ExtractPS5Segment(e, i, out_dir, summary))
      written++;
    else
      failed++;
  }

  AppendLine(summary, "extracted %d segment(s), %d failed\n", written, failed);
  summary +=
      "NOTE: this is a decrypted PS5 PUP; the container segments "
      "(filesystem images, SLB2 blobs, nested PUPs) are recovered in "
      "full. SELF modules inside those images are still encrypted.\n";
  return summary;
}
}  // namespace vfs
