#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <guest_abi.h>
#include "base/arch.h"

namespace krnl {
struct thread_prio {
  u16 type;
  u16 prio;
};

struct thr_param;
int PS4ABI sys_thr_new(thr_param *p, int size);
int PS4ABI sys_thr_self(i64 *tid);
int PS4ABI sys_rtprio_thread(int, u64, thread_prio *);
int PS4ABI sys_umtx_op(void *, int, u64, void *, void *);
}
