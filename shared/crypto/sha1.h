#pragma once

/**
 * \file sha1.h
 *
 * \brief SHA-1 cryptographic hash function
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
#include <string.h>
#include "base/arch.h"

#ifdef _MSC_VER
#include <basetsd.h>
typedef UINT32 u32;
#else
#include <inttypes.h>
#endif

#define POLARSSL_ERR_SHA1_FILE_IO_ERROR \
  -0x0076 /**< Read/write error in file. */

// Regular implementation
//

/**
 * \brief          SHA-1 context structure
 */
typedef struct {
  u32 total[2];             /*!< number of bytes processed  */
  u32 state[5];             /*!< intermediate digest state  */
  unsigned char buffer[64]; /*!< data block being processed */

  unsigned char ipad[64]; /*!< HMAC: inner padding        */
  unsigned char opad[64]; /*!< HMAC: outer padding        */
} sha1_context;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          SHA-1 context setup
 *
 * \param ctx      context to be initialized
 */
void Sha1Starts(sha1_context* ctx);

/**
 * \brief          SHA-1 process buffer
 *
 * \param ctx      SHA-1 context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
void Sha1Update(sha1_context* ctx, const unsigned char* input, size_t ilen);

/**
 * \brief          SHA-1 final digest
 *
 * \param ctx      SHA-1 context
 * \param output   SHA-1 checksum result
 */
void Sha1Finish(sha1_context* ctx, unsigned char output[20]);

/* Internal use */
void Sha1Process(sha1_context* ctx, const unsigned char data[64]);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          Output = SHA-1( input buffer )
 *
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   SHA-1 checksum result
 */
void Sha1(const unsigned char* input, size_t ilen, unsigned char output[20]);

/**
 * \brief          Output = SHA-1( file contents )
 *
 * \param path     input file name
 * \param output   SHA-1 checksum result
 *
 * \return         0 if successful, or POLARSSL_ERR_SHA1_FILE_IO_ERROR
 */
int Sha1File(const char* path, unsigned char output[20]);

/**
 * \brief          SHA-1 HMAC context setup
 *
 * \param ctx      HMAC context to be initialized
 * \param key      HMAC secret key
 * \param keylen   length of the HMAC key
 */
void Sha1HmacStarts(sha1_context* ctx, const unsigned char* key, size_t keylen);

/**
 * \brief          SHA-1 HMAC process buffer
 *
 * \param ctx      HMAC context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
void Sha1HmacUpdate(sha1_context* ctx, const unsigned char* input, size_t ilen);

/**
 * \brief          SHA-1 HMAC final digest
 *
 * \param ctx      HMAC context
 * \param output   SHA-1 HMAC checksum result
 */
void Sha1HmacFinish(sha1_context* ctx, unsigned char output[20]);

/**
 * \brief          SHA-1 HMAC context reset
 *
 * \param ctx      HMAC context to be reset
 */
void Sha1HmacReset(sha1_context* ctx);

/**
 * \brief          Output = HMAC-SHA-1( hmac key, input buffer )
 *
 * \param key      HMAC secret key
 * \param keylen   length of the HMAC key
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   HMAC-SHA-1 result
 */
void Sha1Hmac(const unsigned char* key,
              size_t keylen,
              const unsigned char* input,
              size_t ilen,
              unsigned char output[20]);

#ifdef __cplusplus
}
#endif