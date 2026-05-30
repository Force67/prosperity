#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

namespace krnl {
int PS4ABI sys_is_in_sandbox();
int PS4ABI sys_cpuset_getaffinity();
int PS4ABI sys_get_authinfo(int pid, void *);
int PS4ABI sys_sysctl(int *name, uint32_t namelen, void *oldp, size_t *oldlenp,
                      const void *newp, size_t newlen);
int PS4ABI sys_get_proc_type_info(void *oinfo);
}