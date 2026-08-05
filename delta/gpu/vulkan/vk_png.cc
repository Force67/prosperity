/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_png.h"

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
// the input -- but the images this writes (a mostly-black render target, a
// flat UI layer) are exactly the input LZ77 collapses, which is the difference
// between an 8 MB dump and a 40 KiB one. A full dynamic-Huffman encoder would
// win maybe another 20% and is not worth the code.
// ---------------------------------------------------------------------------

constexpr uint16_t kLenBase[29] = {3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                                   15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                                   67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                   2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr uint16_t kDistBase[30] = {
    1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
    33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr uint8_t kDistExtra[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                    4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                    9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

class BitWriter {
 public:
  explicit BitWriter(uint8_t* out) : out_(out) {}

  // LSB-first, the packing order deflate uses for everything except a Huffman
  // code (see Code).
  void Bits(uint32_t value, uint32_t n) {
    acc_ |= static_cast<uint64_t>(value & ((1u << n) - 1u)) << nbits_;
    nbits_ += n;
    while (nbits_ >= 8) {
      out_[pos_++] = static_cast<uint8_t>(acc_);
      acc_ >>= 8;
      nbits_ -= 8;
    }
  }

  // A Huffman code is packed starting from its MOST significant bit.
  void Code(uint32_t code, uint32_t n) {
    uint32_t reversed = 0;
    for (uint32_t i = 0; i < n; i++)
      reversed |= ((code >> i) & 1u) << (n - 1 - i);
    Bits(reversed, n);
  }

  void Flush() {
    if (nbits_) {
      out_[pos_++] = static_cast<uint8_t>(acc_);
      acc_ = 0;
      nbits_ = 0;
    }
  }

  uint64_t size() const { return pos_; }  // NOLINT: cheap accessor

 private:
  uint8_t* out_;
  uint64_t pos_ = 0;
  uint64_t acc_ = 0;
  uint32_t nbits_ = 0;
};

void FixedLiteral(BitWriter& w, uint32_t symbol) {
  if (symbol <= 143)
    w.Code(0x30 + symbol, 8);
  else if (symbol <= 255)
    w.Code(0x190 + symbol - 144, 9);
  else if (symbol <= 279)
    w.Code(symbol - 256, 7);
  else
    w.Code(0xC0 + symbol - 280, 8);
}

uint32_t Adler32(const uint8_t* data, uint64_t len) {
  uint32_t a = 1, b = 0;
  for (uint64_t i = 0; i < len; i++) {
    a = (a + data[i]) % 65521;
    b = (b + a) % 65521;
  }
  return (b << 16) | a;
}

uint32_t Crc32(const uint8_t* data, uint64_t len, uint32_t crc = 0xFFFFFFFFu) {
  static uint32_t table[256];
  static bool ready = false;
  if (!ready) {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (uint32_t k = 0; k < 8; k++)
        c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    ready = true;
  }
  for (uint64_t i = 0; i < len; i++)
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return crc;
}

uint64_t Deflate(const uint8_t* data, uint64_t len, uint8_t* out) {
  constexpr uint32_t kHashSize = 1u << 15;
  constexpr uint64_t kWindow = 32768;
  constexpr uint32_t kMaxChain = 32;

  BitWriter w(out);
  w.Bits(1, 1);  // BFINAL
  w.Bits(1, 2);  // BTYPE = 01, fixed Huffman

  std::vector<int64_t> head(kHashSize, -1);
  std::vector<int64_t> prev(kWindow, -1);
  auto hash3 = [&](uint64_t p) {
    return static_cast<uint32_t>(
               (static_cast<uint32_t>(data[p]) * 2654435761u) ^
               (static_cast<uint32_t>(data[p + 1]) * 2246822519u) ^
               (static_cast<uint32_t>(data[p + 2]) * 3266489917u)) >>
           17;
  };
  auto insert = [&](uint64_t p) {
    if (p + 3 > len)
      return;
    const uint32_t h = hash3(p) & (kHashSize - 1);
    prev[p & (kWindow - 1)] = head[h];
    head[h] = static_cast<int64_t>(p);
  };

  uint64_t pos = 0;
  while (pos < len) {
    uint32_t best_len = 0;
    uint64_t best_dist = 0;
    if (pos + 3 <= len) {
      int64_t candidate = head[hash3(pos) & (kHashSize - 1)];
      for (uint32_t chain = 0; candidate >= 0 && chain < kMaxChain; chain++) {
        const uint64_t c = static_cast<uint64_t>(candidate);
        if (c >= pos || pos - c > kWindow)
          break;
        const uint32_t limit =
            static_cast<uint32_t>(std::min<uint64_t>(258, len - pos));
        uint32_t match = 0;
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
      uint32_t li = 28;
      while (li > 0 && kLenBase[li] > best_len)
        li--;
      FixedLiteral(w, 257 + li);
      if (kLenExtra[li])
        w.Bits(best_len - kLenBase[li], kLenExtra[li]);
      uint32_t di = 29;
      while (di > 0 && kDistBase[di] > best_dist)
        di--;
      w.Code(di, 5);
      if (kDistExtra[di])
        w.Bits(static_cast<uint32_t>(best_dist - kDistBase[di]),
               kDistExtra[di]);
      for (uint32_t i = 0; i < best_len; i++)
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

void PutBe32(std::vector<uint8_t>& v, uint32_t value) {
  v.push_back(static_cast<uint8_t>(value >> 24));
  v.push_back(static_cast<uint8_t>(value >> 16));
  v.push_back(static_cast<uint8_t>(value >> 8));
  v.push_back(static_cast<uint8_t>(value));
}

void PutChunk(std::vector<uint8_t>& v,
              const char tag[4],
              const uint8_t* data,
              uint64_t len) {
  PutBe32(v, static_cast<uint32_t>(len));
  const uint64_t start = v.size();
  v.insert(v.end(), tag, tag + 4);
  v.insert(v.end(), data, data + len);
  PutBe32(v, Crc32(v.data() + start, len + 4) ^ 0xFFFFFFFFu);
}

// Encode already-filtered scanlines (each row preceded by its filter byte).
bool WritePngRaw(const char* path,
                 const std::vector<uint8_t>& scanlines,
                 uint32_t width,
                 uint32_t height,
                 uint8_t bit_depth) {
  std::vector<uint8_t> z(DeflateBound(scanlines.size()));
  const uint64_t zn =
      ZlibCompress(scanlines.data(), scanlines.size(), z.data());

  std::vector<uint8_t> png;
  const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  png.insert(png.end(), signature, signature + 8);
  uint8_t ihdr[13];
  ihdr[0] = static_cast<uint8_t>(width >> 24);
  ihdr[1] = static_cast<uint8_t>(width >> 16);
  ihdr[2] = static_cast<uint8_t>(width >> 8);
  ihdr[3] = static_cast<uint8_t>(width);
  ihdr[4] = static_cast<uint8_t>(height >> 24);
  ihdr[5] = static_cast<uint8_t>(height >> 16);
  ihdr[6] = static_cast<uint8_t>(height >> 8);
  ihdr[7] = static_cast<uint8_t>(height);
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

uint64_t DeflateBound(uint64_t len) {
  return len + len / 8 + 128;
}

uint64_t ZlibCompress(const uint8_t* data, uint64_t len, uint8_t* out) {
  out[0] = 0x78;  // CM = deflate, CINFO = 32 KiB window
  out[1] = 0x01;  // no preset dict, fastest; (0x7801 % 31) == 0
  const uint64_t n = Deflate(data, len, out + 2);
  const uint32_t adler = Adler32(data, len);
  out[2 + n + 0] = static_cast<uint8_t>(adler >> 24);
  out[2 + n + 1] = static_cast<uint8_t>(adler >> 16);
  out[2 + n + 2] = static_cast<uint8_t>(adler >> 8);
  out[2 + n + 3] = static_cast<uint8_t>(adler);
  return n + 6;
}

bool WritePngRgba8(const char* path,
                   const uint8_t* rgba,
                   uint32_t width,
                   uint32_t height) {
  if (!width || !height)
    return false;
  const uint64_t row = static_cast<uint64_t>(width) * 4;
  std::vector<uint8_t> scanlines(static_cast<uint64_t>(height) * (row + 1));
  for (uint32_t y = 0; y < height; y++) {
    uint8_t* dst = scanlines.data() + static_cast<uint64_t>(y) * (row + 1);
    dst[0] = 0;  // filter: none
    std::memcpy(dst + 1, rgba + static_cast<uint64_t>(y) * row, row);
  }
  return WritePngRaw(path, scanlines, width, height, 8);
}

bool WritePngRgba16(const char* path,
                    const uint16_t* rgba,
                    uint32_t width,
                    uint32_t height) {
  if (!width || !height)
    return false;
  const uint64_t row = static_cast<uint64_t>(width) * 8;
  std::vector<uint8_t> scanlines(static_cast<uint64_t>(height) * (row + 1));
  for (uint32_t y = 0; y < height; y++) {
    uint8_t* dst = scanlines.data() + static_cast<uint64_t>(y) * (row + 1);
    dst[0] = 0;
    const uint16_t* src = rgba + static_cast<uint64_t>(y) * width * 4;
    for (uint32_t i = 0; i < width * 4u; i++) {
      dst[1 + i * 2 + 0] = static_cast<uint8_t>(src[i] >> 8);
      dst[1 + i * 2 + 1] = static_cast<uint8_t>(src[i]);
    }
  }
  return WritePngRaw(path, scanlines, width, height, 16);
}

}  // namespace gpu::vk
