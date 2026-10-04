#pragma once

#include <memory>

#include "kern/ps4/dev/device.h"

namespace kern {
struct PipeState;

class PipeDevice : public Device {
 public:
  PipeDevice(ObjectTable& objects,
             std::shared_ptr<PipeState> state,
             bool reader);
  ~PipeDevice();

  i64 Read(void* buffer, size_t length) override;
  i64 Write(const void* buffer, size_t length) override;
  u16 Poll(u16 events) override;
  int Fstat(void* stat) override;
  u32 GetFlags() const;
  void SetFlags(u32 flags);

 private:
  std::shared_ptr<PipeState> state_;
  bool reader_;
  bool nonblocking_ = false;
};

void CreatePipe(ObjectTable& objects, u32& reader, u32& writer);
}  // namespace kern
