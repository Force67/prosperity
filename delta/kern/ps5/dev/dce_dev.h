#pragma once

#include "kern/ps4/dev/device.h"

namespace kern {
bool DcePs5Active();
u64 DcePs5ScanoutBuffer();
bool DcePs5IsDisplayBuffer(u64 address);
bool DcePs5BeginEopFlip(u64 context, u64* scanout);
void DcePs5CompleteEopFlip(u64 context);

class DceDevicePs5 : public Device {
 public:
  explicit DceDevicePs5(ObjectTable&);
  bool Init(const char*, u32, u32) override;
  i32 Ioctl(u32 command, void* args) override;
  u8* Map(void*, size_t, u32, u32, size_t) override;

 private:
  u8* pool_ = nullptr;
};
}  // namespace kern
