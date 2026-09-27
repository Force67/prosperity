
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "formats/slb2_reader.h"
#include "base/arch.h"
#include "base/containers/vector.h"

namespace formats {
/*


io::File file(converter.from_bytes(argv[1]));
if (file.IsOpen()) {
        PUPHeader pup{};
        file.Read(pup);

        if (pup.magic != 0x32424C53) {
                std::puts("Bad magic");
                return -2;
        }

        std::printf("Found %d file entries\n", pup.fileCount);

        base::Vector<PUPFile> entries(pup.fileCount);
        file.Read(entries);

        for (auto& e : entries) {
                file.Seek(e.offset, io::SeekMode::kSeekSet);

                base::Vector<u8> data(e.fileSize);
                file.Read(data);

                io::File out(converter.from_bytes(e.fileName),
io::FileMode::Write); if (out.IsOpen()) { out.Write(data);
                }
        }
}
*/

bool Slb2Object::Load(io::File&) {
  return true;
}
}  // namespace formats