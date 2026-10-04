#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "kern/ps4/dev/ajm_decoder.h"
#include "kern/ps4/dev/device.h"

#include <memory>
#include <unordered_map>

namespace kern {

// /dev/ajm: the AJM (audio job manager) device. The guest audio job client
// routes FMOD AT9 decoding through it. Opening the device initializes the
// context: the fd is the context id. The client then drives
// everything with 40-byte control blocks (ioctl cmd encodes the size):
//
//   0xC02889C8 initialize   0x900 finalize        0x901 memory register
//   0x902 memory unregister 0x903 module register 0x904 module unregister
//   0x905 instance create   0x906 instance destroy 0x907 batch start
//   0x908 batch wait        0x909 batch cancel    0x90a instance extend
//   0x90b instance switch   0x90c instance available 0x90d instance get size
//   0x92c batch get log     0x965 get failed instance
//
// Batch start hands the device a command buffer recorded by the client
// (batch-job helpers): a stream of jobs, each a type-0 header followed by
// typed commands (input/output data, sidebands, control/run). We parse them
// and execute synchronously, decoding AT9 through LibAtrac9, so the
// sideband results and PCM the client reads back are complete by the time
// batch wait returns (which we always let succeed).
class AjmDevice : public Device {
 public:
  explicit AjmDevice(ObjectTable& objects);
  i32 Ioctl(u32 command, void* args) override;

 private:
  struct Job;
  struct InstanceState;

  bool ExecuteBatch(void* pBuffer, size_t offset);
  InstanceState* FindInstance(u32 id);

  std::mutex mutex_;
  u32 next_instance_ = 1;
  u32 next_batch_ = 1;
  // Codecs registered in this context (module register is per context).
  u32 registered_ = 0;

  // Per-instance codec state. The decoder exists only for AT9; other codecs
  // create fine and their jobs report a codec error (nothing on PS4/PS5
  // ships its music through AJM in anything else).
  struct InstanceState {
    u32 codec = 0;
    u64 flags = 0;
    ajm::GaplessState gapless;
    std::unique_ptr<ajm::Decoder> decoder;
  };

  std::unordered_map<u32, InstanceState> instances_;
};
}  // namespace kern
