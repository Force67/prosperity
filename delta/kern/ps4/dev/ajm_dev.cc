/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "kern/ps4/dev/ajm_dev.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "kern/ps4/dev/ajm_decoder.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kAjmTrace, "DELTA_AJM_TRACE", false);

// Control-block field offsets (all blocks are 40 bytes, size in the ioctl cmd).
constexpr size_t kResultOff = 0x00;  // i32, out
constexpr size_t kIdOff = 0x14;      // u32, out (instance create)
constexpr size_t kAOff = 0x08;       // u64 in/out, command specific
constexpr size_t kBOff = 0x10;       // u64 in/out
constexpr size_t kCOff = 0x18;       // u64 in/out
constexpr size_t kBatchIdOff = 0x20; // u32, out (batch start)

u32 Rd32(const void* data, size_t off) {
  u32 v;
  std::memcpy(&v, static_cast<const u8*>(data) + off, sizeof(v));
  return v;
}
void Wr32(void* data, size_t off, u32 v) {
  std::memcpy(static_cast<u8*>(data) + off, &v, sizeof(v));
}
u64 Rd64(const void* data, size_t off) {
  u64 v;
  std::memcpy(&v, static_cast<const u8*>(data) + off, sizeof(v));
  return v;
}

// AJM errors (ajm/error.h) the ioctls report through the result field.
constexpr u32 kErrInvalidInstance = 0x80930003;
constexpr u32 kErrInvalidParameter = 0x80930005;
constexpr u32 kErrCodecNotSupported = 0x80930008;
constexpr u32 kErrMalformedBatch = 0x80930011;

// Codec identifiers. AT9 is the only one we decode;
// the rest register cleanly and report a codec error on decode jobs.
enum Codec : u32 {
  kDecMp3 = 0,
  kDecAt9 = 1,
  kDecM4aac = 2,
  kDecCelp8 = 3,
  kEncCelp8 = 4,
  kDecCelp = 12,
  kEncCelp = 13,
  kDecHevag = 22,
  kDecLpcm = 23,
  kDecOpus = 24,
};

bool CodecKnown(u32 codec) {
  switch (codec) {
    case kDecMp3:
    case kDecAt9:
    case kDecM4aac:
    case kDecCelp8:
    case kEncCelp8:
    case kDecCelp:
    case kEncCelp:
    case kDecHevag:
    case kDecLpcm:
    case kDecOpus:
      return true;
    default:
      return false;
  }
}

const char* CodecName(u32 codec) {
  switch (codec) {
    case kDecMp3: return "MP3 decoder";
    case kDecAt9: return "ATRAC9 decoder";
    case kDecM4aac: return "MPEG4-AAC decoder";
    case kDecCelp8: return "CELP8 decoder";
    case kEncCelp8: return "CELP8 encoder";
    case kDecCelp: return "CELP decoder";
    case kEncCelp: return "CELP encoder";
    case kDecHevag: return "HE-VAG decoder";
    case kDecLpcm: return "LPCM decoder";
    case kDecOpus: return "Opus decoder";
    default: return "unknown";
  }
}

void TraceHex(const char* tag, const void* data, size_t size) {
  base::String line;
  base::FormatTo(line, "ajm: {} ({} bytes):", tag, size);
  const u8* b = static_cast<const u8*>(data);
  for (size_t i = 0; i < size && i < 128; i++)
    base::FormatTo(line, " {:02x}", b[i]);
  BASE_LOGI("ajm", "{}", line.c_str());
}
}  // namespace

namespace kern {

struct AjmDevice::Job {
  const void* input = nullptr;
  size_t input_size = 0;
  void* output = nullptr;
  size_t output_size = 0;
  const void* sideband_in = nullptr;
  size_t sideband_in_size = 0;
  void* sideband_out = nullptr;
  size_t sideband_out_size = 0;
  u64 flags = 0;
  bool has_flags = false;
  bool is_control = false;
};

AjmDevice::AjmDevice(ObjectTable& objects) : Device(objects) {}

// One job = a type-0 header (instance id, total size) followed by typed
// commands. Command words: bits 0-3 type, bit 4 marks output buffers,
// bits 6+ carry the instance (headers) or flags[32:48] (control/run).
bool AjmDevice::ExecuteBatch(void* pBuffer, size_t offset) {
  auto* p = static_cast<u8*>(pBuffer);
  u8* const end = p + offset;

  Job job{};
  InstanceState* inst = nullptr;
  auto make_result = [&](InstanceState* state) {
    ajm::DecodeResult r{};
    if (state && state->decoder) {
      r.format = state->decoder->GetFormat();
      r.total_decoded_samples = state->decoder->total_decoded();
    } else if (state) {
      r.format.channels = (u32)(state->flags & ajm::kFlagMaxChannelMask);
      r.format.encoding =
          (u32)((state->flags >> ajm::kFlagFormatShift) & ajm::kFlagFormatMask);
      r.format.sample_rate = 48000;
      r.format.channel_mask = ajm::ChannelMask(r.format.channels);
    }
    return r;
  };
  auto flush = [&]() {
    if (!job.has_flags && !job.input && !job.output)
      return;

    if (!inst) {
      // Nobody to run the job against: report through the sideband result.
      if (job.sideband_out && job.sideband_out_size >= 8)
        Wr32(job.sideband_out, 0, kErrInvalidInstance);
      job = {};
      return;
    }

    ajm::DecodeResult r = make_result(inst);
    if (job.is_control) {
      auto& gapless = inst->gapless;
      if (job.flags & ajm::kFlagControlReset) {
        gapless.Reset();
        if (inst->decoder)
          inst->decoder->Reset();
      }
      if ((job.flags & ajm::kFlagSidebandGapless) && job.sideband_in &&
          job.sideband_in_size >= sizeof(ajm::SidebandGaplessDecode)) {
        ajm::SidebandGaplessDecode params;
        std::memcpy(&params, job.sideband_in, sizeof(params));
        gapless.Set(params, (job.flags & ajm::kFlagControlReset) != 0);
      }
      if (job.flags & ajm::kFlagControlInitialize) {
        gapless.Reset();
        if (inst->decoder)
          r = inst->decoder->Initialize(job.sideband_in, job.sideband_in_size);
        else
          r.result = ajm::kResultInvalidParameter;
      }
    } else {
      if (job.input && job.input_size) {
        if (inst->decoder) {
          r = inst->decoder->Decode(job.input, job.input_size, job.output, job.output_size,
                                    (job.flags & ajm::kFlagRunMultipleFrames) != 0,
                                    &inst->gapless);
        } else {
          r.result = ajm::kResultCodecError;
        }
      }
    }

    // Sideband output: result first, then every structure a requested flag
    // maps to, in device order.
    if (job.sideband_out && job.sideband_out_size) {
      u8* sb = static_cast<u8*>(job.sideband_out);
      const size_t sb_size = job.sideband_out_size;
      size_t off = 0;
      auto put = [&](const void* data, size_t size) {
        if (off + size > sb_size)
          return false;
        std::memcpy(sb + off, data, size);
        off += size;
        return true;
      };

      const u64 flags = job.flags;
      ajm::SidebandResult head{r.result, r.internal_result};
      if (!put(&head, sizeof(head)))
        return;

      if (flags & ajm::kFlagRunGetCodecInfo) {
        if (!inst->decoder)
          return;
        if (off + inst->decoder->CodecInfoSize() > sb_size)
          return;
        inst->decoder->WriteCodecInfo(sb + off, sb_size - off);
        off += inst->decoder->CodecInfoSize();
      }
      if (flags & ajm::kFlagSidebandResampleInfo) {
        struct ResampleInfo {
          float ratio;
          int32_t num_samples;
          u32 reserved[8];
        } info{1.0f, 0, {}};
        if (!put(&info, sizeof(info)))
          return;
      }
      if (flags & ajm::kFlagSidebandGapless) {
        if (!put(&inst->gapless.current, sizeof(inst->gapless.current)))
          return;
      }
      if (flags & ajm::kFlagSidebandFormat) {
        if (!put(&r.format, sizeof(r.format)))
          return;
      }
      if (flags & ajm::kFlagSidebandStream) {
        struct Stream {
          int32_t consumed;
          int32_t produced;
          u64 total;
        } stream{(int32_t)r.input_consumed, (int32_t)r.output_written,
                 r.total_decoded_samples};
        if (!put(&stream, sizeof(stream)))
          return;
      }
      if (flags & ajm::kFlagRunMultipleFrames) {
        struct MFrame {
          u32 num_frames;
          u32 reserved;
        } mframe{r.frames, 0};
        if (!put(&mframe, sizeof(mframe)))
          return;
      }
    }

    job = {};
  };

  while (p + 8 <= end) {
    const u32 word0 = Rd32(p, 0);
    const u32 word1 = Rd32(p, 4);
    const u32 type = word0 & 0x1f;
    switch (type) {
      case 0: {  // job header: instance in bits 6-25, size in word1
        flush();
        const u32 instance = (word0 >> 6) & 0xfffff;
        const u64 job_size = 8ull + word1;
        if (job_size > (u64)(end - p))
          return false;
        auto it = instances_.find(instance);
        inst = it != instances_.end() ? &it->second : nullptr;
        p += 8;
        break;
      }
      case 1:    // data buffer
      case 2:    // sideband buffer
      case 0x11: // output data buffer
      case 0x12: {  // output sideband buffer
        if (p + 16 > end)
          return false;
        const u32 size = word1;
        void* const ptr = reinterpret_cast<void*>(Rd64(p, 8));
        switch (type) {
          case 1:
            job.input = ptr;
            job.input_size = size;
            break;
          case 2:
            job.sideband_in = ptr;
            job.sideband_in_size = size;
            break;
          case 0x11:
            job.output = ptr;
            job.output_size = size;
            break;
          case 0x12:
            job.sideband_out = ptr;
            job.sideband_out_size = size;
            break;
        }
        p += 16;
        break;
      }
      case 6:  // return address: {type, 0, u64 addr}; batches finish inline
        if (p + 16 > end)
          return false;
        p += 16;
        break;
      case 3:  // control
      case 4:  // run module
        job.flags = ((u64)((word0 >> 6) & 0xffff) << 32) | word1;
        job.has_flags = true;
        job.is_control = type == 3;
        p += 8;
        break;
      default:
        return false;
    }
  }
  flush();
  return true;
}

AjmDevice::InstanceState* AjmDevice::FindInstance(u32 id) {
  auto it = instances_.find(id);
  return it != instances_.end() ? &it->second : nullptr;
}

i32 AjmDevice::Ioctl(u32 cmd, void* data) {
  if (!data)
    return -1;

  // FMOD drives several decode threads through one context; the control
  // plane (instance table, ids) and per-instance decoder state are not
  // race-free without this. Batches complete synchronously, so nothing
  // ever blocks while the lock is held.
  std::lock_guard lock(mutex_);

  const u32 result = [&]() -> u32 {
    switch (cmd) {
      case 0xC02889C8:  // initialize (PS5): {fd, initializeFlag}
        return 0;
      case 0xC0288900:  // finalize
        return 0;
      case 0xC0288901:  // memory register {pRegion, pages}
        return 0;
      case 0xC0288902:  // memory unregister {pRegion}
        return 0;
      case 0xC0288903: {  // module register {codec}
        const u32 codec = Rd32(data, kAOff);
        if (!CodecKnown(codec)) {
          BASE_LOGI("ajm", "module register: unknown codec {}", codec);
          return kErrInvalidParameter;
        }
        registered_ |= 1u << codec;
        BASE_LOGI("ajm", "module register: {} ({})", codec, CodecName(codec));
        return 0;
      }
      case 0xC0288904:  // module unregister {codec}
        registered_ &= ~(1u << Rd32(data, kAOff));
        return 0;
      case 0xC0288905: {  // instance create {flags, codec} -> {result, id}
        const u64 flags = Rd64(data, kAOff);
        const u32 codec = Rd32(data, kBOff);
        if (!CodecKnown(codec))
          return kErrInvalidParameter;
        if (!(registered_ & (1u << codec)))
          BASE_LOGI("ajm", "instance create: codec {} not registered, allowing anyway",
                    codec);

        const u32 slot = next_instance_++ & 0x3fff;
        const u32 id = (codec << 14) | slot;
        auto& state = instances_[id];
        state.codec = codec;
        state.flags = flags;
        if (codec == kDecAt9) {
          const auto encoding = (ajm::SampleEncoding)((flags >> ajm::kFlagFormatShift) &
                                                      ajm::kFlagFormatMask);
          state.decoder = std::make_unique<ajm::Decoder>(
              (u32)(flags & ajm::kFlagMaxChannelMask), encoding, flags);
        }
        Wr32(data, kIdOff, id);
        BASE_LOGI("ajm", "instance create: {} ({}) flags={:#x} -> id={:#x}", codec,
                  CodecName(codec), flags, id);
        return 0;
      }
      case 0xC0288906: {  // instance destroy {instance}
        instances_.erase(Rd32(data, kAOff));
        return 0;
      }
      case 0xC0288907: {  // batch start {pBuffer, size, priority, pBatchError}
        void* buffer = reinterpret_cast<void*>(Rd64(data, kAOff));
        const size_t offset = Rd32(data, kBOff);
        if (!buffer || !offset)
          return kErrInvalidParameter;
        if (kAjmTrace)
          TraceHex("batch in", buffer, std::min<size_t>(offset, 256));
        if (!ExecuteBatch(buffer, offset))
          return kErrMalformedBatch;
        Wr32(data, kBatchIdOff, next_batch_++ & 0x3fff);
        return 0;
      }
      case 0xC0288908:  // batch wait {batch, timeout, pBatchError}
        return 0;  // batches run synchronously, always done
      case 0xC0288909:  // batch cancel {batch}
        return 0;
      case 0xC028890A:  // instance extend {flags, codec, instance}
        return 0;
      case 0xC028890B: {  // instance switch {flags, codec, instance}
        auto* inst = FindInstance(Rd32(data, kCOff));
        if (!inst)
          return kErrInvalidInstance;
        inst->codec = Rd32(data, kBOff);
        inst->flags = Rd64(data, kAOff);
        inst->decoder.reset();
        if (inst->codec == kDecAt9) {
          const u64 flags = inst->flags;
          const auto encoding = (ajm::SampleEncoding)((flags >> ajm::kFlagFormatShift) &
                                                      ajm::kFlagFormatMask);
          inst->decoder = std::make_unique<ajm::Decoder>(
              (u32)(flags & ajm::kFlagMaxChannelMask), encoding, flags);
        }
        return 0;
      }
      case 0xC028890C:  // instance available {flags, codec}
        return CodecKnown(Rd32(data, kBOff)) ? 0 : kErrCodecNotSupported;
      case 0xC028890D:  // instance get size {flags, codec} -> size
        return CodecKnown(Rd32(data, kBOff)) ? 0x8000 : 0;
      case 0xC028892C:  // batch get log {maxEntries, pLog, pSize}
        return 0;
      case 0xC0288965:  // get failed instance {pInstanceId}
        return 0;
      case 0xC0288990:  // unknown diagnostic: {u8 out, u64 in}
        Wr32(data, kAOff, 0);
        return 0;
      case 0xC0288991:  // unknown diagnostic
        return 0;
      default:
        BASE_LOGI("ajm", "unhandled ioctl {:#x}", cmd);
        return kErrInvalidParameter;
    }
  }();

  Wr32(data, kResultOff, result);

  if (kAjmTrace) {
    base::String line;
    base::FormatTo(line, "ajm: ioctl {:#x} -> {:#x}:", cmd, result);
    for (size_t i = 0; i < 40; i += 4)
      base::FormatTo(line, " {:08x}", Rd32(data, i));
    BASE_LOGI("ajm", "{}", line.c_str());
  }
  return 0;
}
}  // namespace kern
