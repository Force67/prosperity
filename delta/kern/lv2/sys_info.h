#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {
int PS4ABI sys_is_in_sandbox();
int PS4ABI sys_cpuset_getaffinity(int level,
                                  int which,
                                  i64 id,
                                  size_t cpusetsize,
                                  void* mask);
int PS4ABI sys_get_authinfo(int pid, void*);
int PS4ABI sys_sysctl(int* name,
                      u32 namelen,
                      void* oldp,
                      size_t* oldlenp,
                      const void* newp,
                      size_t newlen);
int PS4ABI sys_get_proc_type_info(void* oinfo);

int PS4ABI sys_uuidgen(u8* store, int count);

}  // namespace kern