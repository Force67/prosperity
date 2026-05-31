#pragma once

// Copyright (C) Force67 2019

#include <base.h>

namespace krnl {
struct sce_timespec {
  int64_t tv_sec;
  int64_t tv_nsec;
};

int PS4ABI sys_clock_gettime(uint32_t clock_id, sce_timespec *tp);
}  // namespace krnl
