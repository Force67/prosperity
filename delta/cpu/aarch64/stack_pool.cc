#include "cpu/aarch64/stack_pool.h"

#include "base/threading/lock_guard.h"
#include "cpu/aarch64/cpu_map.h"
#include "guest/session.h"

namespace cpu::fex {

void* StackPool::Take(size_t size) {
  base::LockGuard<base::Mutex> lk(mutex_);
  for (size_t i = 0; i < stacks_.size(); i++) {
    if (stacks_[i].second == size) {
      void* p = stacks_[i].first;
      stacks_.erase(stacks_.begin() + i);
      return p;
    }
  }
  return nullptr;
}

void StackPool::Put(void* stack, size_t size) {
  base::LockGuard<base::Mutex> lk(mutex_);
  stacks_.push_back({stack, size});
}

void StackPool::Release() {
  for (const auto& [stack, size] : stacks_)
    CpuUnmap(stack, size);
  guest::ResetResource(stacks_);
}

}  // namespace cpu::fex
