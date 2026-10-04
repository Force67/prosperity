// The SCE registry manager syscall: typed get/set of system settings by id.

#include "kern/lv2/sys_regmgr.h"

#include <cstring>

#include "base/logging.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "host_memory/host_memory.h"
#include "kern/crash.h"
#include "kern/lv2/error_table.h"
#include "kern/system_settings.h"

namespace kern {

struct NonsysInt {
  union {
    u64 encoded_id;
    struct {
      u8 data[4];
      u8 table;
      u8 index;
      u16 checksum;
    } encoded_id_parts;
  };
  u32 index;
  u32 value;
};

struct NonsysBin {
  u64 encoded_id;
  u64 unknown;
  u64 size;
  u8 data[];
};

int PS4ABI
sys_regmgr_call(u32 op, u32 id, void* result, void* value, u64 type) {
  if (!host_memory::IsMemoryRangeMapped(result, sizeof(u32)))
    return -static_cast<i32>(SysError::eFAULT);
  if (op == 0x18 || op == 0x19) {
    if (type != sizeof(NonsysInt))
      return -static_cast<i32>(SysError::eINVAL);
    if (!host_memory::IsMemoryRangeMapped(value, sizeof(NonsysInt)))
      return -static_cast<i32>(SysError::eFAULT);
    auto* int_value = static_cast<NonsysInt*>(value);
    i32 setting;
    const bool known =
        int_value->index == 0 &&
        (op == 0x18 ? SetRegistryInt(int_value->encoded_id, int_value->value)
                    : GetRegistryInt(int_value->encoded_id, setting));
    if (known) {
      if (op == 0x19)
        int_value->value = setting;
      *static_cast<u32*>(result) = 0;
      return 0;
    }
    if (op == 0x18) {
      *static_cast<u32*>(result) = 0x800d0203;
      return 0;
    }
    // For unsupported keys, let the guest use defaults. Clear the output so
    // readers that ignore the error do not consume garbage.
    int_value->value = 0;
    BASE_LOGI("regmgr", "get-int unknown encoded_id={:#x}",
              (unsigned long long)int_value->encoded_id);
    // Name the guest code that asked, once. A key a title reads ONCE at boot
    // and one it polls every frame want different answers, and only the caller
    // says which this is.
    static base::Mutex seen_mtx;
    static u64 seen[24]{};
    static int seen_n = 0;
    bool fresh = false;
    {
      base::LockGuard<base::Mutex> lk(seen_mtx);
      fresh = true;
      for (int i = 0; i < seen_n; i++)
        if (seen[i] == int_value->encoded_id)
          fresh = false;
      if (fresh && seen_n < 24)
        seen[seen_n++] = int_value->encoded_id;
      else
        fresh = false;
    }
    if (fresh)
      GuestStackTrace("regmgr", 8);
    *static_cast<u32*>(result) = 0x800d0203;
    return 0;
  }

  // The non-system registry family (libSceRegMgr wrappers around syscall 532,
  // op in rdi): 0x18/19 Set/GetInt, 0x1a/1b Set/GetStr, 0x1c/1d Set/GetBin,
  // 0x1e bare u32. Get-str and get-bin share one struct, differing only in how
  // the payload is read.
  if (op == 0x1b || op == 0x1d)  // non-system get str / get bin
  {
    // {u64 encoded_id, u32 index, u32 pad, u64 size, u8 data[size]}, with the
    // payload returned in place; `type` is the whole struct, 0x18 + size.
    if (type < sizeof(NonsysBin))
      return -static_cast<i32>(SysError::eINVAL);
    if (!host_memory::IsMemoryRangeMapped(value, sizeof(NonsysBin)))
      return -static_cast<i32>(SysError::eFAULT);
    auto* bin = static_cast<NonsysBin*>(value);
    if (bin->size != type - sizeof(NonsysBin))
      return -static_cast<i32>(SysError::eINVAL);
    if (!host_memory::IsMemoryRangeMapped(value, type))
      return -static_cast<i32>(SysError::eFAULT);

    // Every key here is an obfuscated id, and get-str may NOT fail:
    // libSceNpCommon reads 0x6b976df7f847ea43 (a 17-byte per-console blob)
    // during NpAsm setup and treats any error as fatal, aborting NP bring-up.
    // All-zero is what an unprovisioned console has, and NP accepts it.
    // Get-bin's accessor defaults to 0 on error anyway; answering the same
    // keeps the reply honest.
    std::memset(bin->data, 0, bin->size);
    // The wrapper returns this int32 to its caller when the syscall succeeds.
    if (result && host_memory::IsMemoryRangeMapped(result, sizeof(u32)))
      *static_cast<u32*>(result) = 0;
    return 0;
  }

  // SCOUT: soft-fail unknown regmgr ops with the same "not available" error the
  // op-25 unknown-key path returns (the guest copes with it) instead of
  // trapping.
  BASE_LOGI("regmgr",
            "UNHANDLED op={} id={:#x} type={:#x} result={:p} value={:p}", op,
            id, (unsigned long long)type, result, value);
  // Name the guest module and offset that asked, so the op can be read out of
  // that library rather than guessed from the op number.
  {
    char sym[256];
    Symbolize(reinterpret_cast<uintptr_t>(__builtin_return_address(0)), sym,
              sizeof(sym));
    BASE_LOGI("regmgr", "  called from {}", sym);
    if (value &&
        host_memory::IsMemoryRangeMapped(value, type < 64 ? type : 64)) {
      base::String words;
      const auto* w = static_cast<const u32*>(value);
      for (u64 i = 0; i * 4 < type && i < 16; i++)
        base::FormatTo(words, " {:08x}", w[i]);
      BASE_LOGI("regmgr", "  value[]:{}", words.c_str());
    }
  }
  // Registry failures use the method result, separate from transport status.
  *static_cast<u32*>(result) = 0x800d0203;
  return 0;
}

}  // namespace kern
