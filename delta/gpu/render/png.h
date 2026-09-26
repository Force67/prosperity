/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// PNG output for the frame debugger. The rest of the module writes PPM, which
// is fine for a human with an image viewer and useless to an agent driving the
// emulator from a terminal: a PNG can be opened directly by every tool in that
// loop. Self-contained (a small fixed-Huffman deflate), so dumping a render
// target costs the build no new dependency.

#include "base/arch.h"

namespace gpu::render {

// 8 bits per channel, 4 channels, tightly packed, top row first.
bool WritePngRgba8(const char* path,
                   const u8* rgba,
                   u32 width,
                   u32 height);

// 16 bits per channel (host byte order in, big-endian in the file), 4 channels.
// The lossless form for an HDR target: no exposure choice is baked in.
bool WritePngRgba16(const char* path,
                    const u16* rgba,
                    u32 width,
                    u32 height);

// Raw zlib stream (deflate + header + adler32), exposed for the PNG writers
// and for the unit test that round-trips it. Returns bytes written into `out`,
// which must hold at least DeflateBound(len).
u64 DeflateBound(u64 len);
u64 ZlibCompress(const u8* data, u64 len, u8* out);

}  // namespace gpu::render
