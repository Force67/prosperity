/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_png.h"
#include "base/arch.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gpu::vk {
namespace {

// ---------------------------------------------------------------------------
// deflate (RFC 1951), fixed Huffman + a 32 KiB greedy LZ77 match finder.
//
// Fixed Huffman costs at most 9 bits per literal, so the worst case is 1.125x
// the input, but the images this writes (a mostly-black render target, a
// flat UI layer) are exactly the input LZ77 collapses, which is the difference
// between an 8 MB dump and a 40 KiB one. A full dynamic-Huffman encoder would
// win maybe another 20% and is not worth the code.
// ---------------------------------------------------------------------------

constexpr u16 kLenBase[29] = {3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                                   15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                                   67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr u8 kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                   2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr u16 kDistBase[30] = {
    1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
    33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr u8 kDistExtra[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                    4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                    9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

class BitWriter {
 public:
  explicit BitWriter(u8* out) : out_(out) {}

  // LSB-first, the packing order deflate uses for everything except a Huffman
  // code (see Code).
  void Bits(u32 value, u32 n) {
    acc_ |= static_cast<u64>(value & ((1u << n) - 1u)) << nbits_;
    nbits_ += n;
    while (nbits_ >= 8) {
      out_[pos_++] = static_cast<u8>(acc_);
      acc_ >>= 8;
      nbits_ -= 8;
    }
  }

  // A Huffman code is packed starting from its MOST significant bit.
  void Code(u32 code, u32 n) {
    u32 reversed = 0;
    for (u32 i = 0; i < n; i++)
      reversed |= ((code >> i) & 1u) << (n - 1 - i);
    Bits(reversed, n);
  }

  void Flush() {
    if (nbits_) {
      out_[pos_++] = static_cast<u8>(acc_);
      acc_ = 0;
      nbits_ = 0;
    }
  }

  u64 size() const { return pos_; }  // NOLINT: cheap accessor

 private:
  u8* out_;
  u64 pos_ = 0;
  u64 acc_ = 0;
  u32 nbits_ = 0;
};

void FixedLiteral(BitWriter& w, u32 symbol) {
  if (symbol <= 143)
    w.Code(0x30 + symbol, 8);
  else if (symbol <= 255)
    w.Code(0x190 + symbol - 144, 9);
  else if (symbol <= 279)
    w.Code(symbol - 256, 7);
  else
    w.Code(0xC0 + symbol - 280, 8);
}

u32 Adler32(const u8* data, u64 len) {
  u32 a = 1, b = 0;
  for (u64 i = 0; i < len; i++) {
    a = (a + data[i]) % 65521;
    b = (b + a) % 65521;
  }
  return (b << 16) | a;
}

u32 Crc32(const u8* data, u64 len, u32 crc = 0xFFFFFFFFu) {
  static u32 table[256];
  static bool ready = false;
  if (!ready) {
    for (u32 i = 0; i < 256; i++) {
      u32 c = i;
      for (u32 k = 0; k < 8; k++)
        c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    ready = true;
  }
  for (u64 i = 0; i < len; i++)
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return crc;
}

u64 Deflate(const u8* data, u64 len, u8* out) {
  constexpr u32 kHashSize = 1u << 15;
  constexpr u64 kWindow = 32768;
  constexpr u32 kMaxChain = 32;

  BitWriter w(out);
  w.Bits(1, 1);  // BFINAL
  w.Bits(1, 2);  // BTYPE = 01, fixed Huffman

  std::vector<i64> head(kHashSize, -1);
  std::vector<i64> prev(kWindow, -1);
  auto hash3 = [&](u64 p) {
    return static_cast<u32>(
               (static_cast<u32>(data[p]) * 2654435761u) ^
               (static_cast<u32>(data[p + 1]) * 2246822519u) ^
               (static_cast<u32>(data[p + 2]) * 3266489917u)) >>
           17;
  };
  auto insert = [&](u64 p) {
    if (p + 3 > len)
      return;
    const u32 h = hash3(p) & (kHashSize - 1);
    prev[p & (kWindow - 1)] = head[h];
    head[h] = static_cast<i64>(p);
  };

  u64 pos = 0;
  while (pos < len) {
    u32 best_len = 0;
    u64 best_dist = 0;
    if (pos + 3 <= len) {
      i64 candidate = head[hash3(pos) & (kHashSize - 1)];
      for (u32 chain = 0; candidate >= 0 && chain < kMaxChain; chain++) {
        const u64 c = static_cast<u64>(candidate);
        if (c >= pos || pos - c > kWindow)
          break;
        const u32 limit =
            static_cast<u32>(std::min<u64>(258, len - pos));
        u32 match = 0;
        while (match < limit && data[c + match] == data[pos + match])
          match++;
        if (match > best_len) {
          best_len = match;
          best_dist = pos - c;
          if (match >= 258)
            break;
        }
        candidate = prev[c & (kWindow - 1)];
      }
    }
    if (best_len >= 3) {
      u32 li = 28;
      while (li > 0 && kLenBase[li] > best_len)
        li--;
      FixedLiteral(w, 257 + li);
      if (kLenExtra[li])
        w.Bits(best_len - kLenBase[li], kLenExtra[li]);
      u32 di = 29;
      while (di > 0 && kDistBase[di] > best_dist)
        di--;
      w.Code(di, 5);
      if (kDistExtra[di])
        w.Bits(static_cast<u32>(best_dist - kDistBase[di]),
               kDistExtra[di]);
      for (u32 i = 0; i < best_len; i++)
        insert(pos + i);
      pos += best_len;
    } else {
      FixedLiteral(w, data[pos]);
      insert(pos);
      pos++;
    }
  }
  FixedLiteral(w, 256);  // end of block
  w.Flush();
  return w.size();
}

void PutBe32(std::vector<u8>& v, u32 value) {
  v.push_back(static_cast<u8>(value >> 24));
  v.push_back(static_cast<u8>(value >> 16));
  v.push_back(static_cast<u8>(value >> 8));
  v.push_back(static_cast<u8>(value));
}

void PutChunk(std::vector<u8>& v,
              const char tag[4],
              const u8* data,
              u64 len) {
  PutBe32(v, static_cast<u32>(len));
  const u64 start = v.size();
  v.insert(v.end(), tag, tag + 4);
  v.insert(v.end(), data, data + len);
  PutBe32(v, Crc32(v.data() + start, len + 4) ^ 0xFFFFFFFFu);
}

// Encode already-filtered scanlines (each row preceded by its filter byte).
bool WritePngRaw(const char* path,
                 const std::vector<u8>& scanlines,
                 u32 width,
                 u32 height,
                 u8 bit_depth) {
  std::vector<u8> z(DeflateBound(scanlines.size()));
  const u64 zn =
      ZlibCompress(scanlines.data(), scanlines.size(), z.data());

  std::vector<u8> png;
  const u8 signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  png.insert(png.end(), signature, signature + 8);
  u8 ihdr[13];
  ihdr[0] = static_cast<u8>(width >> 24);
  ihdr[1] = static_cast<u8>(width >> 16);
  ihdr[2] = static_cast<u8>(width >> 8);
  ihdr[3] = static_cast<u8>(width);
  ihdr[4] = static_cast<u8>(height >> 24);
  ihdr[5] = static_cast<u8>(height >> 16);
  ihdr[6] = static_cast<u8>(height >> 8);
  ihdr[7] = static_cast<u8>(height);
  ihdr[8] = bit_depth;
  ihdr[9] = 6;  // RGBA
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  PutChunk(png, "IHDR", ihdr, sizeof(ihdr));
  PutChunk(png, "IDAT", z.data(), zn);
  PutChunk(png, "IEND", nullptr, 0);

  std::FILE* f = std::fopen(path, "wb");
  if (!f)
    return false;
  const bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
  std::fclose(f);
  return ok;
}

}  // namespace

u64 DeflateBound(u64 len) {
  return len + len / 8 + 128;
}

u64 ZlibCompress(const u8* data, u64 len, u8* out) {
  out[0] = 0x78;  // CM = deflate, CINFO = 32 KiB window
  out[1] = 0x01;  // no preset dict, fastest; (0x7801 % 31) == 0
  const u64 n = Deflate(data, len, out + 2);
  const u32 adler = Adler32(data, len);
  out[2 + n + 0] = static_cast<u8>(adler >> 24);
  out[2 + n + 1] = static_cast<u8>(adler >> 16);
  out[2 + n + 2] = static_cast<u8>(adler >> 8);
  out[2 + n + 3] = static_cast<u8>(adler);
  return n + 6;
}

bool WritePngRgba8(const char* path,
                   const u8* rgba,
                   u32 width,
                   u32 height) {
  if (!width || !height)
    return false;
  const u64 row = static_cast<u64>(width) * 4;
  std::vector<u8> scanlines(static_cast<u64>(height) * (row + 1));
  for (u32 y = 0; y < height; y++) {
    u8* dst = scanlines.data() + static_cast<u64>(y) * (row + 1);
    dst[0] = 0;  // filter: none
    std::memcpy(dst + 1, rgba + static_cast<u64>(y) * row, row);
  }
  return WritePngRaw(path, scanlines, width, height, 8);
}

bool WritePngRgba16(const char* path,
                    const u16* rgba,
                    u32 width,
                    u32 height) {
  if (!width || !height)
    return false;
  const u64 row = static_cast<u64>(width) * 8;
  std::vector<u8> scanlines(static_cast<u64>(height) * (row + 1));
  for (u32 y = 0; y < height; y++) {
    u8* dst = scanlines.data() + static_cast<u64>(y) * (row + 1);
    dst[0] = 0;
    const u16* src = rgba + static_cast<u64>(y) * width * 4;
    for (u32 i = 0; i < width * 4u; i++) {
      dst[1 + i * 2 + 0] = static_cast<u8>(src[i] >> 8);
      dst[1 + i * 2 + 1] = static_cast<u8>(src[i]);
    }
  }
  return WritePngRaw(path, scanlines, width, height, 16);
}

}  // namespace gpu::vk
