
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "../../proc.h"
#include "error_table.h"
#include <base.h>
#include <logger/logger.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
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

/*does not belong here*/
int PS4ABI sys_namedobj_create(const char *name, void *arg2, uint32_t arg3) {
  // Registers a debug name for a kernel object. Success is 0 -- returning the
  // old fake counter read back as an errno (EPERM on the 2nd call) and aborted
  // libc init. The actual id is reported through other channels we don't model.
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
    setThreadFsBase(reinterpret_cast<uint64_t>(fsbase));
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
    uint64_t encoded_id;
    struct {
      uint8_t data[4];
      uint8_t table;
      uint8_t index;
      uint16_t checksum;
    } encoded_id_parts;
  };
  uint32_t unknown;
  uint32_t value;
};

struct nonsys_bin {
  uint64_t encoded_id;
  uint64_t unknown;
  uint64_t size;
  uint8_t data[];
};

/*TODO: clearly does not belong here*/
int PS4ABI sys_regmgr_call(uint32_t op, uint32_t id, void *result, void *value,
                           uint64_t type) {
  if (op == 25) // non-system get int
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
    printf("[regmgr] op25 get-int unknown encoded_id=%#llx\n",
           (unsigned long long)int_value->encoded_id);
    return 0x800D0203;
  }

  if (op == 27) // non-system get bin
  {
    // {u64 encoded_id, u64 unknown, u64 size, u8 data[size]}, with the blob
    // returned in place -- `type` is the whole struct, 0x18 + size.
    auto *bin = static_cast<nonsys_bin *>(value);
    if (type < sizeof(nonsys_bin) || bin->size != type - sizeof(nonsys_bin))
      return 0x800D0203;

    // Every key here is one of Sony's obfuscated ids. Unlike get-int, failing
    // is not an option: libSceNpCommon reads id 0x6b976df7f847ea43 (a 17-byte
    // per-console blob) during NpAsm resource-context setup and treats any
    // error as fatal, which aborts the whole NP bring-up. An all-zero blob is
    // what an unprovisioned console has, and NP accepts it.
    std::memset(bin->data, 0, bin->size);
    return 0;
  }

  // SCOUT: soft-fail unknown regmgr ops with the same "not available" error the
  // op-25 unknown-key path returns (the guest copes with it) instead of trapping.
  printf("[regmgr] UNHANDLED op=%u id=%#x type=%#llx result=%p value=%p\n", op,
         id, (unsigned long long)type, result, value);
  return 0x800D0203;
}

// sys_randomized_path (602): libkernel's sceKernelGetRandomizedPath. arg0 is an
// optional path to *set* (NULL = query the current one); arg1 is the output
// buffer and arg2 an in/out length. The kernel hands back the per-title
// randomized sandbox component used under /system_data. We have no such mapping,
// so report an empty path (len 0) with success; the guest treats that as "no
// randomized prefix" and falls through to the plain sandbox path.
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

// sys_workaround8849 (605): a Sony libkernel-internal no-op kept for an old
// firmware errata (CXX runtime guard). The real kernel just validates its args
// and returns success; nothing observable comes back.
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

int PS4ABI sys_write(uint32_t fd, const void *buf, size_t nbytes) {
  if (fd == 1 || fd == 2) // stdout, stderr
  {
    // DELTA_QUIET_GUEST: the game's debug prints (per-frame message-pump chatter)
    // flood stdout char-by-char and corrupt our diagnostic logs via interleaving.
    // Suppress guest fd1/2 output while diagnosing the host-side render path.
    if (kQuietGuest)
      return static_cast<int>(nbytes);
    fwrite(buf, 1, nbytes, stdout);
    return static_cast<int>(nbytes);
  }

  // A device-backed fd (console/tty): let it handle the write. Otherwise just
  // accept the bytes; libkernel writes debug output to fds we don't model, and
  // trapping there kills the boot.
  if (auto *proc = proc::getActive()) {
    auto *obj = proc->getObjTable().get(fd);
    if (obj && obj->type() == kObject::oType::device) {
      int64_t r = static_cast<device *>(obj)->write(buf, nbytes);
      if (r >= 0)
        return static_cast<int>(r);
    }
  }
  return static_cast<int>(nbytes);
}
} // namespace krnl