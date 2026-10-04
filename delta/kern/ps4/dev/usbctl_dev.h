#pragma once

#include <memory>

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
// USB directory enumeration for the virtual controller.
class UsbctlDevice : public Device {
 public:
  UsbctlDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};

struct UgenState;
class UgenDevice : public Device {
 public:
  UgenDevice(ObjectTable& objects);
  ~UgenDevice();

  i32 Ioctl(u32 command, void* args) override;
  u16 Poll(u16 events) override;
  int Fstat(void* stat) override;

 private:
  std::unique_ptr<UgenState> state_;
  bool EnsureController();
  i32 Request(u8 type,
              u8 request,
              u16 value,
              u16 index,
              void* data,
              u16 length,
              u16& actual);
  i32 TransferIoctl(u32 command, void* args);
};
}  // namespace kern
