#pragma once

// Copyright (C) Force67 2019

#include "base/arch.h"
#include "guest_abi.h"

namespace krnl {
struct sce_timespec {  // NOLINT(readability-identifier-naming): guest ABI name
  i64 tv_sec;
  i64 tv_nsec;
};

int PS4ABI sys_clock_gettime(u32 clock_id, sce_timespec* tp);
int PS4ABI sys_nanosleep(const sce_timespec* rqtp, sce_timespec* rmtp);
}  // namespace krnl
