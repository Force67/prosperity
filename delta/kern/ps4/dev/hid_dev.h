#pragma once

#include <mutex>
#include <unordered_set>

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/hid: handles and reports used by the guest controller client. Legacy evdev
// reads and uinput writes are enabled with DELTA_HID_PASSTHROUGH=1.
class HidDevice : public Device {
 public:
  HidDevice(ObjectTable& objects);
  ~HidDevice() override;

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;

 private:
  i32 ReadControllerReports(u32 handle,
                            void* reports,
                            u32 count,
                            u32* connection,
                            u32* extended,
                            u32 stride);
  std::mutex handles_mutex_;
  std::unordered_set<u32> controller_handles_;
};
}  // namespace kern
