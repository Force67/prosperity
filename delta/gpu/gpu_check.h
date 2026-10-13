/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include <cstdlib>
#include "base/logging.h"

#define GPU_BUGCHECK(expression, ...)                                          \
  do {                                                                         \
    if (!(expression)) {                                                       \
      BASE_LOGI("gpu", "BUGCHECK {}:{}: {}", __FILE__, __LINE__, #expression); \
      __VA_OPT__(BASE_LOGI("gpu", "  " __VA_ARGS__);)                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
