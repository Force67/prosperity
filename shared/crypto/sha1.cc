/*
 *  FIPS-180-1 compliant SHA-1 implementation
 *
 *  Copyright (C) 2006-2013, Brainspark B.V.
 *
 *  This file is part of PolarSSL (http://www.polarssl.org)
 *  Lead Maintainer: Paul Bakker <polarssl_maintainer at polarssl.org>
 *
 *  All rights reserved.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */
/*
 *  The SHA-1 standard was published by NIST in 1993.
 *
 *  http://www.itl.nist.gov/fipspubs/fip180-1.htm
 */

#include "crypto/sha1.h"
#include "base/arch.h"

/*
 * 32-bit integer manipulation macros (big endian)
 */
#ifndef GET_UINT32_BE
#define GET_UINT32_BE(n, b, i)                     \
                                                   \
  {                                                \
    (n) = (static_cast<u32>((b)[(i)]) << 24) |     \
          (static_cast<u32>((b)[(i) + 1]) << 16) | \
          (static_cast<u32>((b)[(i) + 2]) << 8) |  \
          (static_cast<u32>((b)[(i) + 3]));        \
  }
#endif

#ifndef PUT_UINT32_BE
#define PUT_UINT32_BE(n, b, i)                            \
                                                          \
  {                                                       \
    (b)[(i)] = static_cast<unsigned char>((n) >> 24);     \
    (b)[(i) + 1] = static_cast<unsigned char>((n) >> 16); \
    (b)[(i) + 2] = static_cast<unsigned char>((n) >> 8);  \
    (b)[(i) + 3] = static_cast<unsigned char>((n));       \
  }
#endif

/*
 * SHA-1 context setup
 */
void Sha1Starts(sha1_context* ctx) {
  ctx->total[0] = 0;
  ctx->total[1] = 0;

  ctx->state[0] = 0x67452301;
  ctx->state[1] = 0xEFCDAB89;
  ctx->state[2] = 0x98BADCFE;
  ctx->state[3] = 0x10325476;
  ctx->state[4] = 0xC3D2E1F0;
}

void Sha1Process(sha1_context* ctx, const unsigned char data[64]) {
  u32 temp, w[16], a, b, c, d, e;

  GET_UINT32_BE(w[0], data, 0);
  GET_UINT32_BE(w[1], data, 4);
  GET_UINT32_BE(w[2], data, 8);
  GET_UINT32_BE(w[3], data, 12);
  GET_UINT32_BE(w[4], data, 16);
  GET_UINT32_BE(w[5], data, 20);
  GET_UINT32_BE(w[6], data, 24);
  GET_UINT32_BE(w[7], data, 28);
  GET_UINT32_BE(w[8], data, 32);
  GET_UINT32_BE(w[9], data, 36);
  GET_UINT32_BE(w[10], data, 40);
  GET_UINT32_BE(w[11], data, 44);
  GET_UINT32_BE(w[12], data, 48);
  GET_UINT32_BE(w[13], data, 52);
  GET_UINT32_BE(w[14], data, 56);
  GET_UINT32_BE(w[15], data, 60);

#define S(x, n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))

#define R(t)                                                           \
                                                                       \
  (temp = w[(t - 3) & 0x0F] ^ w[(t - 8) & 0x0F] ^ w[(t - 14) & 0x0F] ^ \
          w[t & 0x0F],                                                 \
   (w[t & 0x0F] = S(temp, 1)))

#define P(a, b, c, d, e, x)            \
                                       \
  {                                    \
    e += S(a, 5) + F(b, c, d) + K + x; \
    b = S(b, 30);                      \
  }

  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];

#define F(x, y, z) (z ^ (x & (y ^ z)))
#define K 0x5A827999

  P(a, b, c, d, e, w[0]);
  P(e, a, b, c, d, w[1]);
  P(d, e, a, b, c, w[2]);
  P(c, d, e, a, b, w[3]);
  P(b, c, d, e, a, w[4]);
  P(a, b, c, d, e, w[5]);
  P(e, a, b, c, d, w[6]);
  P(d, e, a, b, c, w[7]);
  P(c, d, e, a, b, w[8]);
  P(b, c, d, e, a, w[9]);
  P(a, b, c, d, e, w[10]);
  P(e, a, b, c, d, w[11]);
  P(d, e, a, b, c, w[12]);
  P(c, d, e, a, b, w[13]);
  P(b, c, d, e, a, w[14]);
  P(a, b, c, d, e, w[15]);
  P(e, a, b, c, d, R(16));
  P(d, e, a, b, c, R(17));
  P(c, d, e, a, b, R(18));
  P(b, c, d, e, a, R(19));

#undef K
#undef F

#define F(x, y, z) (x ^ y ^ z)
#define K 0x6ED9EBA1

  P(a, b, c, d, e, R(20));
  P(e, a, b, c, d, R(21));
  P(d, e, a, b, c, R(22));
  P(c, d, e, a, b, R(23));
  P(b, c, d, e, a, R(24));
  P(a, b, c, d, e, R(25));
  P(e, a, b, c, d, R(26));
  P(d, e, a, b, c, R(27));
  P(c, d, e, a, b, R(28));
  P(b, c, d, e, a, R(29));
  P(a, b, c, d, e, R(30));
  P(e, a, b, c, d, R(31));
  P(d, e, a, b, c, R(32));
  P(c, d, e, a, b, R(33));
  P(b, c, d, e, a, R(34));
  P(a, b, c, d, e, R(35));
  P(e, a, b, c, d, R(36));
  P(d, e, a, b, c, R(37));
  P(c, d, e, a, b, R(38));
  P(b, c, d, e, a, R(39));

#undef K
#undef F

#define F(x, y, z) ((x & y) | (z & (x | y)))
#define K 0x8F1BBCDC

  P(a, b, c, d, e, R(40));
  P(e, a, b, c, d, R(41));
  P(d, e, a, b, c, R(42));
  P(c, d, e, a, b, R(43));
  P(b, c, d, e, a, R(44));
  P(a, b, c, d, e, R(45));
  P(e, a, b, c, d, R(46));
  P(d, e, a, b, c, R(47));
  P(c, d, e, a, b, R(48));
  P(b, c, d, e, a, R(49));
  P(a, b, c, d, e, R(50));
  P(e, a, b, c, d, R(51));
  P(d, e, a, b, c, R(52));
  P(c, d, e, a, b, R(53));
  P(b, c, d, e, a, R(54));
  P(a, b, c, d, e, R(55));
  P(e, a, b, c, d, R(56));
  P(d, e, a, b, c, R(57));
  P(c, d, e, a, b, R(58));
  P(b, c, d, e, a, R(59));

#undef K
#undef F

#define F(x, y, z) (x ^ y ^ z)
#define K 0xCA62C1D6

  P(a, b, c, d, e, R(60));
  P(e, a, b, c, d, R(61));
  P(d, e, a, b, c, R(62));
  P(c, d, e, a, b, R(63));
  P(b, c, d, e, a, R(64));
  P(a, b, c, d, e, R(65));
  P(e, a, b, c, d, R(66));
  P(d, e, a, b, c, R(67));
  P(c, d, e, a, b, R(68));
  P(b, c, d, e, a, R(69));
  P(a, b, c, d, e, R(70));
  P(e, a, b, c, d, R(71));
  P(d, e, a, b, c, R(72));
  P(c, d, e, a, b, R(73));
  P(b, c, d, e, a, R(74));
  P(a, b, c, d, e, R(75));
  P(e, a, b, c, d, R(76));
  P(d, e, a, b, c, R(77));
  P(c, d, e, a, b, R(78));
  P(b, c, d, e, a, R(79));

#undef K
#undef F

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
}

/*
 * SHA-1 process buffer
 */
void Sha1Update(sha1_context* ctx, const unsigned char* input, size_t ilen) {
  size_t fill;
  u32 left;

  if (ilen <= 0)
    return;

  left = ctx->total[0] & 0x3F;
  fill = 64 - left;

  ctx->total[0] += static_cast<u32>(ilen);
  ctx->total[0] &= 0xFFFFFFFF;

  if (ctx->total[0] < static_cast<u32>(ilen))
    ctx->total[1]++;

  if (left && ilen >= fill) {
    memcpy(ctx->buffer + left, input, fill);
    Sha1Process(ctx, ctx->buffer);
    input += fill;
    ilen -= fill;
    left = 0;
  }

  while (ilen >= 64) {
    Sha1Process(ctx, input);
    input += 64;
    ilen -= 64;
  }

  if (ilen > 0)
    memcpy(ctx->buffer + left, input, ilen);
}

static const unsigned char kSha1Padding[64] = {
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/*
 * SHA-1 final digest
 */
void Sha1Finish(sha1_context* ctx, unsigned char output[20]) {
  u32 last, padn;
  u32 high, low;
  unsigned char msglen[8];

  high = (ctx->total[0] >> 29) | (ctx->total[1] << 3);
  low = (ctx->total[0] << 3);

  PUT_UINT32_BE(high, msglen, 0);
  PUT_UINT32_BE(low, msglen, 4);

  last = ctx->total[0] & 0x3F;
  padn = (last < 56) ? (56 - last) : (120 - last);

  Sha1Update(ctx, kSha1Padding, padn);
  Sha1Update(ctx, msglen, 8);

  PUT_UINT32_BE(ctx->state[0], output, 0);
  PUT_UINT32_BE(ctx->state[1], output, 4);
  PUT_UINT32_BE(ctx->state[2], output, 8);
  PUT_UINT32_BE(ctx->state[3], output, 12);
  PUT_UINT32_BE(ctx->state[4], output, 16);
}

/*
 * output = SHA-1( input buffer )
 */
void Sha1(const unsigned char* input, size_t ilen, unsigned char output[20]) {
  sha1_context ctx;

  Sha1Starts(&ctx);
  Sha1Update(&ctx, input, ilen);
  Sha1Finish(&ctx, output);

  memset(&ctx, 0, sizeof(sha1_context));
}

/*
 * SHA-1 HMAC context setup
 */
void Sha1HmacStarts(sha1_context* ctx,
                    const unsigned char* key,
                    size_t keylen) {
  size_t i;
  unsigned char sum[20];

  if (keylen > 64) {
    Sha1(key, keylen, sum);
    keylen = 20;
    key = sum;
  }

  memset(ctx->ipad, 0x36, 64);
  memset(ctx->opad, 0x5C, 64);

  for (i = 0; i < keylen; i++) {
    ctx->ipad[i] ^= key[i];
    ctx->opad[i] ^= key[i];
  }

  Sha1Starts(ctx);
  Sha1Update(ctx, ctx->ipad, 64);

  memset(sum, 0, sizeof(sum));
}

/*
 * SHA-1 HMAC process buffer
 */
void Sha1HmacUpdate(sha1_context* ctx,
                    const unsigned char* input,
                    size_t ilen) {
  Sha1Update(ctx, input, ilen);
}

/*
 * SHA-1 HMAC final digest
 */
void Sha1HmacFinish(sha1_context* ctx, unsigned char output[20]) {
  unsigned char tmpbuf[20];

  Sha1Finish(ctx, tmpbuf);
  Sha1Starts(ctx);
  Sha1Update(ctx, ctx->opad, 64);
  Sha1Update(ctx, tmpbuf, 20);
  Sha1Finish(ctx, output);

  memset(tmpbuf, 0, sizeof(tmpbuf));
}

/*
 * SHA1 HMAC context reset
 */
void Sha1HmacReset(sha1_context* ctx) {
  Sha1Starts(ctx);
  Sha1Update(ctx, ctx->ipad, 64);
}

/*
 * output = HMAC-SHA-1( hmac key, input buffer )
 */
void Sha1Hmac(const unsigned char* key,
              size_t keylen,
              const unsigned char* input,
              size_t ilen,
              unsigned char output[20]) {
  sha1_context ctx;

  Sha1HmacStarts(&ctx, key, keylen);
  Sha1HmacUpdate(&ctx, input, ilen);
  Sha1HmacFinish(&ctx, output);

  memset(&ctx, 0, sizeof(sha1_context));
}