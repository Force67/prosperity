#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstddef>
#include <cstdint>

#include <base/containers/vector.h>

namespace crypto {
// Rebuild a loadable ELF image from a fake-signed SELF (fSELF). A fake SELF
// wraps a plaintext ELF: there is no per-segment encryption, so this only
// reassembles the ELF header + program headers and copies each block segment to
// the file offset of the program header it references. Returns an empty vector
// if the input is not a SELF or is malformed.
base::Vector<uint8_t> self2elf(const uint8_t *data, size_t size);
} // namespace crypto
