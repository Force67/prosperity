#pragma once

#include "kern/ps4/dev/device.h"

namespace kern {
bool DcePs5Active();
u64 DcePs5ScanoutBuffer();
bool DcePs5IsDisplayBuffer(u64 address);

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
