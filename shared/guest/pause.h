#pragma once

#include <stdint.h>

#include "base/arch.h"

namespace guest {

bool Paused();
void SetPaused(bool paused);
void PollPause();
void PausePoint();
bool OnGuestThread();
const i32* PauseFlagAddress();
void RegisterCode(uintptr_t address, mem_size size);
// True when `pc` lies in guest code (or in this thread's JIT, under FEX).
bool IsGuestCode(uintptr_t pc);
void SetThreadExitHandler(void (*exit)());
void ResetCodeRanges();

class ThreadRegistration {
 public:
  explicit ThreadRegistration(bool (*is_guest_pc)(uintptr_t) = nullptr);
  ~ThreadRegistration();
  ThreadRegistration(const ThreadRegistration&) = delete;
  ThreadRegistration& operator=(const ThreadRegistration&) = delete;

 private:
  bool owner_ = false;
};

}  // namespace guest
