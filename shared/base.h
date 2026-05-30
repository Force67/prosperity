#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

/*fool intellisense*/
#if defined(__clang__) || defined(__GNUC__)

#define NAKED __attribute__((naked))
#define PACKED __attribute__((packed))
#define PS4ABI __attribute__((sysv_abi)) //, cdecl))
#define NORETURN __attribute__((noreturn))
#define F_INLINE __attribute__((always_inline))

#define bswap16 __builtin_bswap16
#define bswap32 __builtin_bswap32
#define bswap64 __builtin_bswap64

#define rotr16 __builtin_rotateright16
#define rotr32 __builtin_rotateright32
#define rotr64 __builtin_rotateright64

#define rotl16 __builtin_rotateleft16
#define rotl32 __builtin_rotateleft32
#define rotl64 __builtin_rotateleft64

#ifdef _WIN32
#define dbg_break() __debugbreak()
#else
#define dbg_break() __builtin_trap()

// MSVC intrinsics that the codebase relies on but POSIX/clang doesn't ship.
// Map them onto portable equivalents so we don't have to scatter ifdefs.
#ifndef __debugbreak
#define __debugbreak() __builtin_trap()
#endif

#ifndef _byteswap_ushort
#define _byteswap_ushort(x) __builtin_bswap16((x))
#endif
#ifndef _byteswap_ulong
#define _byteswap_ulong(x) __builtin_bswap32((x))
#endif
#ifndef _byteswap_uint64
#define _byteswap_uint64(x) __builtin_bswap64((x))
#endif

#ifndef _ReturnAddress
#define _ReturnAddress() __builtin_return_address(0)
#endif

// fopen_s shim. POSIX has no _s variants, so emulate the contract:
// returns 0 on success, errno on failure; *fp is set on success, nulled
// otherwise.
static inline int fopen_s(std::FILE** fp, const char* path, const char* mode) {
  if (!fp) return 22 /*EINVAL*/;
  *fp = std::fopen(path, mode);
  return *fp ? 0 : 1;
}
#endif

#else

#define NAKED naked
#define PACKED
#define PS4ABI
#define NORETURN
#define F_INLINE __forceinline

#define bswap16 _byteswap_ushort
#define bswap32 _byteswap_ulong
#define bswap64 _byteswap_uint64

#define dbg_break() DebugBreak()

#endif

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define POW2_MASK (align - static_cast<T>(1))

template <typename T> inline T align_up(const T addr, const T align) {
  return (addr + POW2_MASK) & ~POW2_MASK;
}
