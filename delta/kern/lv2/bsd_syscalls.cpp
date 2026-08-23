
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "kern/proc.h"
#include "kern/crash.h"
#include <utl/mem.h>
#include <cstdint>

namespace krnl {
void symbolize(uintptr_t addr, char *out, size_t n);
}
#include "base/arch.h"
#include "error_table.h"
#include <base.h>
#include <base/logging.h>
#include <base/strings/format.h>
#include <logger/logger.h>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <string>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kQuietGuest, "DELTA_QUIET_GUEST", false);
}  // namespace

namespace krnl {
int PS4ABI sys_exit() {
  __debugbreak();
  return 0;
}

int PS4ABI sys_rfork() {
  __debugbreak();
  return 0;
}

int PS4ABI sys_execve() {
  __debugbreak();
  return 0;
}

// We deliver no signals, so the mask is inert; still, a caller that saves the
// old mask here to restore it later must not read uninitialised memory. The
// FreeBSD sigset_t is 16 bytes (4x uint32). Report an empty old mask.
int PS4ABI sys_sigprocmask(int how, const int *set, int *oset) {
  (void)how;
  (void)set;
  if (oset)
    std::memset(oset, 0, 16);
  return 0;
}

// Likewise report "no previous handler" rather than leaving the caller's oldact
// buffer uninitialised. struct sigaction on amd64 is 32 bytes (handler pointer +
// flags + 16-byte sa_mask, padded).
int PS4ABI sys_sigaction(int sig, const void *act, void *oact) {
  (void)sig;
  (void)act;
  if (oact)
    std::memset(oact, 0, 32);
  return 0;
}

// sys_namedobj_create (557): registers a name -> data-pointer pair in the
// process's id table. The kernel allocates a 24-byte object {char* name@0;
// void* data@8; uint32 flags@16}, copies the name (max 32 chars), and stores
// the caller's data pointer. The flags field is ORed with 0x1000 (the
// namedobj type tag for the id table). Returns the allocated id in rax.
// We return 0 (success, id 0): the only known caller (debug instrumentation)
// ignores the id.
int PS4ABI sys_namedobj_create(const char *name, void *arg2, u32 arg3) {
  (void)name;
  (void)arg2;
  (void)arg3;
  return 0;
}

int PS4ABI sys_namedobj_delete() { return 0; }

int PS4ABI sys_sysarch(int num, void *args) {
  // amd64 machine-dependent syscall. The four base ops are the only ones libc
  // and libthr issue; everything else is genuinely unsupported (EINVAL), which
  // is what the real kernel returns for an unknown number.
  enum { AMD64_GET_FSBASE = 128, AMD64_SET_FSBASE, AMD64_GET_GSBASE, AMD64_SET_GSBASE };
  auto &env = proc::getActive()->getEnv();
  if (!args)
    return -SysError::eFAULT;
  switch (num) {
  case AMD64_GET_FSBASE:
    *static_cast<void **>(args) = env.fsBase;
    return 0;
  case AMD64_SET_FSBASE: {
    auto fsbase = *static_cast<void **>(args);
    env.fsBase = fsbase;
    setThreadFsBase(reinterpret_cast<u64>(fsbase));
    return 0;
  }
  case AMD64_GET_GSBASE:
    // We don't track a separate gs base (amd64 TLS uses fs); report none set.
    *static_cast<void **>(args) = nullptr;
    return 0;
  case AMD64_SET_GSBASE:
    // Accept silently: no guest we run relies on a distinct gs base.
    return 0;
  default:
    return -SysError::eINVAL;
  }
}

struct nonsys_int {
  union {
    u64 encoded_id;
    struct {
      u8 data[4];
      u8 table;
      u8 index;
      u16 checksum;
    } encoded_id_parts;
  };
  u32 unknown;
  u32 value;
};

struct nonsys_bin {
  u64 encoded_id;
  u64 unknown;
  u64 size;
  u8 data[];
};

/*TODO: clearly does not belong here*/
int PS4ABI sys_regmgr_call(u32 op, u32 id, void *result, void *value,
                           u64 type) {
  if (op == 0x19)  // non-system get int
  {
    auto int_value = static_cast<nonsys_int *>(value);

    if (int_value->encoded_id == 0x0CAE671ADF3AEB34ull ||
        int_value->encoded_id == 0x338660835BDE7CB1ull) {
      int_value->value = 0;
      return 0;
    }

    // The remaining keys are Sony's obfuscated (checksummed) registry ids whose
    // plaintext we can't recover, so we can't know the correct value to return.
    // The guest tolerates the "key not available" error and uses its defaults,
    // which is safer than inventing a value for an unidentified setting. Clear
    // the output anyway: a caller that reads it despite the error would
    // otherwise get stack garbage.
    int_value->value = 0;
    BASE_LOGI("regmgr", "get-int unknown encoded_id={:#x}",
              (unsigned long long)int_value->encoded_id);
    // Name the guest code that asked, once. A key a title reads ONCE at boot
    // and one it polls every frame want different answers, and only the caller
    // says which this is.
    static std::mutex seenMtx;
    static u64 seen[24]{};
    static int seenN = 0;
    bool fresh = false;
    {
      std::lock_guard<std::mutex> lk(seenMtx);
      fresh = true;
      for (int i = 0; i < seenN; i++)
        if (seen[i] == int_value->encoded_id)
          fresh = false;
      if (fresh && seenN < 24)
        seen[seenN++] = int_value->encoded_id;
      else
        fresh = false;
    }
    if (fresh)
      guestStackTrace("regmgr", 8);
    return 0x800D0203;
  }

  // The non-system registry family, read out of libSceRegMgr (every export is a
  // thin wrapper around syscall 532 with the op in rdi):
  //   0x18/0x19 Set/GetInt   0x1a/0x1b Set/GetStr   0x1c/0x1d Set/GetBin
  //   0x1e takes a bare u32
  // Get-str and get-bin share one struct and differ only in how the caller
  // reads the payload back, so they are served together here.
  if (op == 0x1b || op == 0x1d)  // non-system get str / get bin
  {
    // {u64 encoded_id, u32 index, u32 pad, u64 size, u8 data[size]}, with the
    // payload returned in place -- `type` is the whole struct, 0x18 + size.
    auto *bin = static_cast<nonsys_bin *>(value);
    if (type < sizeof(nonsys_bin) || bin->size != type - sizeof(nonsys_bin))
      return 0x800D0203;

    // Every key here is one of Sony's obfuscated ids. Unlike get-int, failing
    // is not an option for get-str: libSceNpCommon reads id
    // 0x6b976df7f847ea43 (a 17-byte per-console blob) during NpAsm
    // resource-context setup and treats any error as fatal, which aborts the
    // whole NP bring-up. An all-zero payload is what an unprovisioned console
    // has, and NP accepts it. Get-bin is more forgiving (libkernel's accessor
    // for key 0x44746d9c58675bee defaults to 0 on any error and never reads
    // the buffer), but answering it the same way keeps the reply honest rather
    // than leaving the caller's blob at whatever it held.
    std::memset(bin->data, 0, bin->size);
    // The wrapper returns this int32 to its caller when the syscall succeeds.
    if (result && utl::isMemoryRangeMapped(result, sizeof(u32)))
      *static_cast<u32 *>(result) = 0;
    return 0;
  }

  // SCOUT: soft-fail unknown regmgr ops with the same "not available" error the
  // op-25 unknown-key path returns (the guest copes with it) instead of trapping.
  BASE_LOGI("regmgr", "UNHANDLED op={} id={:#x} type={:#x} result={:p} value={:p}",
            op, id, (unsigned long long)type, result, value);
  // Name the guest module and offset that asked, so the op can be read out of
  // that library rather than guessed from the op number.
  {
    char sym[256];
    symbolize(reinterpret_cast<uintptr_t>(_ReturnAddress()), sym, sizeof(sym));
    BASE_LOGI("regmgr", "  called from {}", sym);
    if (value && utl::isMemoryRangeMapped(value, type < 64 ? type : 64)) {
      base::String words;
      const auto *w = static_cast<const u32 *>(value);
      for (u64 i = 0; i * 4 < type && i < 16; i++)
        base::FormatTo(words, " {:08x}", w[i]);
      BASE_LOGI("regmgr", "  value[]:{}", words.c_str());
    }
  }
  // Same reasoning as the op-25 unknown-key path: a caller that reads the
  // result despite the error should see zero rather than stack garbage.
  if (result && utl::isMemoryRangeMapped(result, sizeof(u32)))
    *static_cast<u32 *>(result) = 0;
  return 0x800D0203;
}

// sys_randomized_path (602): libkernel's sceKernelGetRandomizedPath. Args are a
// struct {char* set_path@0; char* out@8; size_t* out_len@16}. If set_path is
// non-null the kernel stores it as the new randomized prefix (requires priv
// 0x2AF); the current prefix (up to 256 bytes) is always copied to out/out_len.
// The prefix is the per-title randomized sandbox component used under
// /system_data. We have no such mapping, so report an empty path (len 0) with
// success; the guest treats that as "no randomized prefix" and falls through to
// the plain sandbox path.
int PS4ABI sys_randomized_path(const char *set_path, char *out,
                               size_t *out_len) {
  (void)set_path;
  if (out && out_len) {
    if (*out_len >= 1)
      out[0] = '\0';
    *out_len = 0;
  }
  return 0;
}

// sys_uuidgen (392): fill `store` with `count` version-4 UUIDs (FreeBSD's
// struct uuid is 16 bytes). Demon's Souls generates one while bringing up its
// resource system; the old stub returned success without writing the buffer,
// and the engine's I/O layer crashed on the uninitialised id.
int PS4ABI sys_uuidgen(u8 *store, int count) {
  if (!store)
    return -SysError::eFAULT;
  if (count < 1 || count > 2048)
    return -SysError::eINVAL;
  static std::atomic<u64> seq{1};
  for (int i = 0; i < count; i++) {
    u8 *p = store + i * 16;
    u64 a = seq.fetch_add(1) * 0x9E3779B97F4A7C15ull;
    u64 b = a * 0xBF58476D1CE4E5B9ull ^ (a >> 31);
    std::memcpy(p, &a, 8);
    std::memcpy(p + 8, &b, 8);
    p[7] = (p[7] & 0x0F) | 0x40;  // version 4
    p[8] = (p[8] & 0x3F) | 0x80;  // variant 10x
  }
  return 0;
}

// sys_workaround8849 (605): a restricted registry-int getter. The kernel reads
// a uint32 regmgr key from the args struct, validates it against a whitelist of
// four fixed keys (0x19780100, 0x78026300, 0x78028300, 0x78028A00), calls
// sceRegMgrGetInt, and returns the value. Anything else is EINVAL. We have no
// registry, so return 0 (the value an unprovisioned console would have).
int PS4ABI sys_workaround8849() { return 0; }

// sys_blockpool_open (653): allocates a "block pool" used by the flexible-memory
// allocator and returns a descriptor. We don't model block pools yet; the only
// caller during boot does not feed the result into blockpool_map/mmap, so hand
// back a fixed positive descriptor (a non-zero, non-stdio handle) to signal
// success without colliding with a real object-table fd.
int PS4ABI sys_blockpool_open() { return 0x4000; }

// sys_dynlib_do_copy_relocations (596): processes R_X86_64_COPY relocations for
// the main executable. Our loader already resolves data relocations when it maps
// each module, so there is nothing extra to copy here; return success.
int PS4ABI sys_dynlib_do_copy_relocations() { return 0; }

int PS4ABI sys_getpid() { return 0x1337; }

int PS4ABI sys_write(u32 fd, const void *buf, size_t nbytes) {
  if (fd == 1 || fd == 2) // stdout, stderr
  {
    // DELTA_QUIET_GUEST: the game's debug prints (per-frame message-pump chatter)
    // flood stdout char-by-char and corrupt our diagnostic logs via interleaving.
    // Suppress guest fd1/2 output while diagnosing the host-side render path.
    if (kQuietGuest)
      return static_cast<int>(nbytes);
    // Through the logger a line at a time, like sys_writev: host stdout is
    // usually a redirected file, so a plain fwrite sits in a host stdio buffer
    // that a killed run never flushes -- a title that stops printing and one
    // whose last 4 KiB were lost look identical. Titles print this fd a
    // character at a time, hence the accumulator.
    static std::mutex mtx;
    static std::string line;
    std::lock_guard<std::mutex> lk(mtx);
    line.append(static_cast<const char *>(buf), nbytes);
    for (size_t nl; (nl = line.find('\n')) != std::string::npos;) {
      std::string one = line.substr(0, nl);
      line.erase(0, nl + 1);
      while (!one.empty() && one.back() == '\r')
        one.pop_back();
      if (!one.empty())
        BASE_LOGI("guest", "{}", one.c_str());
        if (const char *m = std::getenv("DELTA_GUEST_LOG_STACK");
            m && one.find(m) != base::String::npos)
          guestStackTrace("guestlog", 12);
    }
    return static_cast<int>(nbytes);
  }

  // A device-backed fd (console/tty): let it handle the write. Otherwise just
  // accept the bytes; libkernel writes debug output to fds we don't model, and
  // trapping there kills the boot.
  if (auto *proc = proc::getActive()) {
    auto *obj = proc->getObjTable().get(fd);
    if (obj && obj->type() == kObject::oType::device) {
      i64 r = static_cast<device *>(obj)->write(buf, nbytes);
      if (r >= 0)
        return static_cast<int>(r);
    }
  }
  return static_cast<int>(nbytes);
}
} // namespace krnl