#pragma once

#include <cstddef>

#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/threading/mutex.h"

namespace cpu::fex {

// Never unmapped mid-session: guest code keeps pointers into retired stacks.
class StackPool {
 public:
  void* Take(size_t size);
  void Put(void* stack, size_t size);
  void Release();

 private:
  base::Mutex mutex_;
  base::Vector<base::Pair<void*, size_t>> stacks_;
};

}  // namespace cpu::fex
