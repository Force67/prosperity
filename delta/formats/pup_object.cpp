
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// based off https://github.com/Zer0xFF/ps4-pup-unpacker/blob/master/PUP.cpp

#include "pup_object.h"

#include <cstdarg>
#include <cstdio>
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
} // namespace

pupReader::pupReader(const base::String &name) : file(name) {}

bool pupReader::load() {
  if (!file.IsOpen())
    return false;
  if (!file.Read(header))
    return false;
  // The PUP container header/entry table is plaintext even on retail firmware
  // (only the segment payloads are encrypted), so the magic is a reliable gate.
  if (header.magic != 0x1D3D154Fu)
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
} // namespace vfs
