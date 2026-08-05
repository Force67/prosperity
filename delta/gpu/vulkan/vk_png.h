/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// PNG output for the frame debugger. The rest of the module writes PPM, which
// is fine for a human with an image viewer and useless to an agent driving the
// emulator from a terminal: a PNG can be opened directly by every tool in that
// loop. Self-contained (a small fixed-Huffman deflate), so dumping a render
// target costs the build no new dependency.

#include <cstdint>

namespace gpu::vk {

// 8 bits per channel, 4 channels, tightly packed, top row first.
bool WritePngRgba8(const char* path,
                   const uint8_t* rgba,
                   uint32_t width,
                   uint32_t height);

// 16 bits per channel (host byte order in, big-endian in the file), 4 channels.
// The lossless form for an HDR target: no exposure choice is baked in.
bool WritePngRgba16(const char* path,
                    const uint16_t* rgba,
                    uint32_t width,
                    uint32_t height);

// Raw zlib stream (deflate + header + adler32), exposed for the PNG writers
// and for the unit test that round-trips it. Returns bytes written into `out`,
// which must hold at least DeflateBound(len).
uint64_t DeflateBound(uint64_t len);
uint64_t ZlibCompress(const uint8_t* data, uint64_t len, uint8_t* out);

}  // namespace gpu::vk
