#pragma once

#include "base/atomic.h"
#include "base/logging.h"

namespace kern {

// Log a message the first time a given handler runs. Used for the stubs whose
// fakery would silently break if a title actually exercised the subsystem.
inline void LogOnce(base::Atomic<bool>& flag, const char* msg) {
  if (!flag.exchange(true))
    BASE_LOGI("sce", "{}", msg);
}

}  // namespace kern
