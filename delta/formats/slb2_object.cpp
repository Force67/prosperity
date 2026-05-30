
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "slb2_object.h"

namespace formats {
/*


utl::File file(converter.from_bytes(argv[1]));
if (file.IsOpen()) {
        PUPHeader pup{};
        file.Read(pup);

        if (pup.magic != 0x32424C53) {
                std::puts("Bad magic");
                return -2;
        }

        std::printf("Found %d file entries\n", pup.fileCount);

        std::vector<PUPFile> entries(pup.fileCount);
        file.Read(entries);

        for (auto& e : entries) {
                file.Seek(e.offset, utl::seekMode::seek_set);

                std::vector<uint8_t> data(e.fileSize);
                file.Read(data);

                utl::File out(converter.from_bytes(e.fileName),
utl::fileMode::write); if (out.IsOpen()) { out.Write(data);
                }
        }
}
*/

bool slb2Object::load(utl::File &) { return true; }
} // namespace formats