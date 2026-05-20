
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// based off https://github.com/Zer0xFF/ps4-pup-unpacker/blob/master/PUP.cpp

#include "pup_object.h"
#include <zlib.h>

namespace vfs {
struct fileNode {
  uint32_t id;
  const char *name;
};

static const fileNode knownFileNames[] = {
    {3, "wlan_firmware.bin"}, {5, "secure_modules.bin"},
    {6, "system.img"},        {8, "eap.img"},
    {9, "recovery.img"},      {11, "preinst.img"},
    {12, "system_ex.img"},    {34, "torus2_firmware.bin"},
    {257, "eula.xml"},        {512, "orbis_swu.self"},
    {514, "orbis_swu.self"},  {3337, "cp_firmware.bin"}};

/*

uint32_t Id()
{
        return Flags >> 20;
}

bool IsCompressed()
{
        return (Flags & 8) != 0;
}
*/

pupReader::pupReader(const base::String &name) : file(name) {}

bool pupReader::load() {
  if (!file.IsOpen())
    return false;

  if (!file.Read(header))
    return false;

  if (header.magic != 0x1D3D154F && header.contentType == 4 /*4=PUP*/)
    return false;

  for (int i = 0; i < header.numSegments; i++) {
    pup_entry entry{};
    file.Read(entry);

    entries.emplace_back(entry);
  }

  return true;
}

utl::File pupReader::extractFile(uint16_t id) {
  if (file.IsOpen()) {

    for (int i = 0; i < header.numSegments; i++) {
      auto &e = entries[i];

      int special = e.flags & 0xF0000000;
      if (special == 0xE0000000 || special == 0xF0000000)
        continue;

      if (i == id) {
        base::Vector<uint8_t> data(e.sizeCompressed,
                                   base::VectorReservePolicy::kForData);

        file.Seek(e.offset, utl::seekMode::seek_set);
        file.Read(data);
        return utl::make_stream(std::move(data));
      }
    }
  }

  return utl::File();
}

void pupReader::extractAll() {}

void pupReader::unzipEntry(const pup_entry &entry, base::Vector<uint8_t> &out) {
  /*singular block*/
  if (entry.flags & 0x800) {
  }
}
} // namespace vfs