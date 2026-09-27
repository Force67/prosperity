#pragma once

// Copyright (C) Force67 2019

#include <guest_abi.h>
#include "base/arch.h"

namespace krnl {
struct sce_timespec {
  i64 tv_sec;
  i64 tv_nsec;
};

int PS4ABI sys_clock_gettime(u32 clock_id, sce_timespec *tp);
int PS4ABI sys_nanosleep(const sce_timespec *rqtp, sce_timespec *rmtp);
}  // namespace krnl
