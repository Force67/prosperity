#pragma once

/*
 * PS4Delta : PS4 emulation project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// AJM codec layer for the /dev/ajm device: the per-instance decoder state and
// the job-result semantics (gapless tracking, superframe handling, sideband
// codecs). ATRAC9 decoding uses LibAtrac9 (vendor/libatrac9).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {
#include <libatrac9.h>
}

namespace kern::ajm {

// Result codes the codecs report per job (ajm/error.h).
constexpr int32_t kResultOk = 0;
constexpr int32_t kResultNotInitialized = 0x00000001;
constexpr int32_t kResultInvalidData = 0x00000002;
constexpr int32_t kResultInvalidParameter = 0x00000004;
constexpr int32_t kResultPartialInput = 0x00000008;
constexpr int32_t kResultNotEnoughRoom = 0x00000010;
constexpr int32_t kResultCodecError = 0x40000000;
constexpr int32_t kResultFatal = static_cast<int32_t>(0x80000000u);

// Instance flag layout (ajm/sideband.h).
constexpr uint64_t kFlagMaxChannelMask = 0x7f;
constexpr uint64_t kFlagFormatShift = 7;
constexpr uint64_t kFlagFormatMask = 0x7;
constexpr uint64_t kFlagGaplessLoop = 1ull << 10;
// Codec-specific flags start at bit 32; bit 32 is AT9's parse-RIFF request.
constexpr uint64_t kInstanceFlagCodecOffset = 32;

// Run/control command flags (ajm/sideband.h, bits 11+).
constexpr uint64_t kFlagRunGetCodecInfo = 1ull << 11;
constexpr uint64_t kFlagRunMultipleFrames = 1ull << 12;
constexpr uint64_t kFlagControlReset = 1ull << 13;
constexpr uint64_t kFlagControlInitialize = 1ull << 14;
constexpr uint64_t kFlagSidebandResampleInfo = 1ull << 44;
constexpr uint64_t kFlagSidebandGapless = 1ull << 45;
constexpr uint64_t kFlagSidebandFormat = 1ull << 46;
constexpr uint64_t kFlagSidebandStream = 1ull << 47;

enum class SampleEncoding : uint32_t {
  S16 = 0,
  S32 = 1,
  Float = 2,
};

inline size_t BytesPerSample(SampleEncoding encoding) {
  switch (encoding) {
    case SampleEncoding::S16:
      return sizeof(int16_t);
    case SampleEncoding::S32:
      return sizeof(int32_t);
    case SampleEncoding::Float:
      return sizeof(float);
  }
  return sizeof(int16_t);
}

inline uint32_t ChannelMask(uint32_t channels) {
  switch (channels) {
    case 1:
      return 0x4;
    case 2:
      return 0x3;
    case 3:
      return 0x7;
    case 4:
      return 0x33;
    case 5:
      return 0x607;
    case 6:
      return 0x60f;
    case 7:
      return 0x70f;
    case 8:
      return 0x63f;
    default:
      return 0;
  }
}

// Sideband format.
struct SidebandFormat {
  uint32_t channels;
  uint32_t channel_mask;
  uint32_t sample_rate;
  uint32_t encoding;
  uint32_t bitrate;
  uint32_t reserved;
};
static_assert(sizeof(SidebandFormat) == 24);

// Sideband result: the header every sideband output starts with.
struct SidebandResult {
  int32_t result;
  int32_t internal_result;
};
static_assert(sizeof(SidebandResult) == 8);

// Gapless decode sideband.
struct SidebandGaplessDecode {
  uint32_t total_samples;
  uint16_t skip_samples;
  uint16_t skipped_samples;
};
static_assert(sizeof(SidebandGaplessDecode) == 8);

struct GaplessState {
  SidebandGaplessDecode init{};
  SidebandGaplessDecode current{};

  bool HasSampleLimit() const {
    return init.total_samples != 0 && init.total_samples != 0xffffffffu;
  }

  bool IsEnd() const { return HasSampleLimit() && current.total_samples == 0; }

  void Reset() {
    current = init;
    current.skipped_samples = 0;
  }

  void Set(const SidebandGaplessDecode& params, bool reset) {
    init.total_samples = params.total_samples;
    init.skip_samples = params.skip_samples;
    if (reset || (current.total_samples == 0 && current.skip_samples == 0 &&
                  current.skipped_samples == 0)) {
      Reset();
    }
  }
};

struct DecodeResult {
  int32_t result = kResultOk;
  int32_t internal_result = kResultOk;
  size_t input_consumed = 0;
  size_t output_written = 0;
  uint64_t total_decoded_samples = 0;
  uint32_t frames = 0;
  SidebandFormat format{};
};

// ATRAC9 codec-info result payload.
struct At9CodecInfo {
  uint32_t super_frame_size;
  uint32_t frames_in_super_frame;
  uint32_t next_frame_size;
  uint32_t frame_samples;
};

class Decoder {
 public:
  Decoder(uint32_t channels, SampleEncoding encoding, uint64_t flags)
      : max_channels_(channels ? channels : 2),
        encoding_(encoding),
        parse_riff_((flags & (1ull << kInstanceFlagCodecOffset)) != 0),
        handle_(Atrac9GetHandle()) {}

  ~Decoder() {
    if (handle_)
      Atrac9ReleaseHandle(handle_);
  }

  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  // ATRAC9 initialization parameters: 4 bytes of config data (+ reserved).
  DecodeResult Initialize(const void* params, size_t size) {
    DecodeResult result = MakeResult();
    if (!params || size < 4) {
      result.result = kResultInvalidParameter;
      return result;
    }
    if (!InitializeConfig(static_cast<const uint8_t*>(params), &result))
      return result;
    return MakeResult();
  }

  void Reset() {
    if (handle_) {
      Atrac9ReleaseHandle(handle_);
      handle_ = Atrac9GetHandle();
    }
    num_frames_ = 0;
    total_decoded_ = 0;
    superframe_remain_ = initialized_ ? (uint32_t)codec_info_.superframeSize : 0;
    if (has_config_ && handle_) {
      DecodeResult result = MakeResult();
      (void)InitializeConfig(config_data_, &result);
    }
  }

  DecodeResult Decode(const void* input, size_t input_size, void* output,
                      size_t output_size, bool multiple_frames, GaplessState* gapless) {
    DecodeResult result = MakeResult();
    if (!input || input_size == 0) {
      result.result = kResultPartialInput;
      return result;
    }

    const uint8_t* in = static_cast<const uint8_t*>(input);
    uint8_t* out = static_cast<uint8_t*>(output);
    size_t in_offset = 0;

    if (parse_riff_ && input_size >= 12 && FourCc(in, 'R', 'I', 'F', 'F')) {
      if (!ParseRiffHeader(in, input_size, &in_offset, gapless, &result))
        return result;
      if (in_offset >= input_size) {
        result.input_consumed = in_offset;
        result.result = kResultPartialInput;
        return result;
      }
    }

    if (!initialized_) {
      result.result = kResultNotInitialized;
      return result;
    }

    for (;;) {
      if (in_offset >= input_size)
        break;
      if (gapless && gapless->IsEnd())
        break;

      const size_t remaining_input = input_size - in_offset;
      if (remaining_input < MinimumInputSize()) {
        if (result.frames == 0)
          result.result = kResultPartialInput;
        break;
      }

      const size_t frame_bytes = FrameOutputBytes(gapless);
      if (frame_bytes != 0 && !out) {
        result.result |= kResultNotEnoughRoom;
        break;
      }
      if (output_size - result.output_written < frame_bytes) {
        result.result |= kResultNotEnoughRoom;
        break;
      }

      int bytes_used = 0;
      const int ret = DecodeFrame(in + in_offset, &bytes_used);
      if (ret != 0) {
        result.result = kResultCodecError | kResultFatal;
        result.internal_result = ret;
        break;
      }
      if (bytes_used <= 0 || (size_t)bytes_used > remaining_input ||
          (uint32_t)bytes_used > superframe_remain_) {
        result.result = kResultCodecError | kResultInvalidData;
        break;
      }

      const uint64_t written = CopyFrame(out, output_size, &result.output_written, gapless);

      in_offset += (size_t)bytes_used;
      result.input_consumed = in_offset;
      result.frames++;
      total_decoded_ += written;

      AdvanceSuperframe((uint32_t)bytes_used, input_size, &in_offset, &result);
      if (result.result != kResultOk)
        break;
      if (gapless && gapless->IsEnd())
        DrainSuperframe(in, input_size, &in_offset, &result);
      if (result.result != kResultOk || !multiple_frames)
        break;
    }

    result.total_decoded_samples = total_decoded_;
    result.format = GetFormat();
    return result;
  }

  void WriteCodecInfo(void* output, size_t output_size) const {
    if (!output || output_size < sizeof(At9CodecInfo))
      return;
    auto* info = static_cast<At9CodecInfo*>(output);
    info->super_frame_size = initialized_ ? (uint32_t)codec_info_.superframeSize : 0;
    info->frames_in_super_frame = initialized_ ? (uint32_t)codec_info_.framesInSuperframe : 0;
    info->next_frame_size = (uint32_t)MinimumInputSize();
    info->frame_samples = initialized_ ? (uint32_t)codec_info_.frameSamples : 0;
  }

  size_t CodecInfoSize() const { return sizeof(At9CodecInfo); }

  SidebandFormat GetFormat() const {
    SidebandFormat f{};
    f.channels = initialized_ ? (uint32_t)codec_info_.channels : max_channels_;
    f.channel_mask = ChannelMask(f.channels);
    f.sample_rate = initialized_ ? (uint32_t)codec_info_.samplingRate : 48000;
    f.encoding = (uint32_t)encoding_;
    f.bitrate = 0;
    if (initialized_ && codec_info_.framesInSuperframe > 0 && codec_info_.frameSamples > 0) {
      f.bitrate = (uint32_t)(((uint64_t)codec_info_.samplingRate *
                              (uint64_t)codec_info_.superframeSize * 8ull) /
                             ((uint64_t)codec_info_.framesInSuperframe *
                              (uint64_t)codec_info_.frameSamples));
    }
    return f;
  }

  uint64_t total_decoded() const { return total_decoded_; }

 private:
  DecodeResult MakeResult() const {
    DecodeResult result{};
    result.total_decoded_samples = total_decoded_;
    result.format = GetFormat();
    return result;
  }

  static bool FourCc(const uint8_t* data, char a, char b, char c, char d) {
    return data[0] == (uint8_t)a && data[1] == (uint8_t)b && data[2] == (uint8_t)c &&
           data[3] == (uint8_t)d;
  }

  static uint32_t ReadLe32(const uint8_t* data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) |
           ((uint32_t)data[3] << 24);
  }

  uint32_t FrameOutputSamples(const GaplessState* gapless) const {
    if (!initialized_ || codec_info_.channels <= 0 || codec_info_.frameSamples <= 0)
      return 0;
    uint32_t samples = (uint32_t)codec_info_.frameSamples;
    if (gapless) {
      const uint32_t skip = std::min<uint32_t>(samples, gapless->current.skip_samples);
      samples -= skip;
      if (gapless->HasSampleLimit())
        samples = std::min<uint32_t>(samples, gapless->current.total_samples);
    }
    return samples;
  }

  size_t FrameOutputBytes(const GaplessState* gapless = nullptr) const {
    if (!initialized_ || codec_info_.channels <= 0 || codec_info_.frameSamples <= 0)
      return 0;
    return (size_t)codec_info_.channels * FrameOutputSamples(gapless) *
           BytesPerSample(encoding_);
  }

  size_t MinimumInputSize() const { return initialized_ ? superframe_remain_ : 0; }

  bool InitializeConfig(const uint8_t* config_data, DecodeResult* result) {
    if (!config_data || !result)
      return false;
    if (!handle_) {
      result->result = kResultCodecError | kResultFatal;
      return false;
    }

    std::memcpy(config_data_, config_data, 4);
    has_config_ = true;

    const int init_ret = Atrac9InitDecoder(handle_, config_data_);
    if (init_ret != 0) {
      initialized_ = false;
      result->result = kResultCodecError | kResultInvalidData;
      result->internal_result = init_ret;
      return false;
    }

    const int info_ret = Atrac9GetCodecInfo(handle_, &codec_info_);
    if (info_ret != 0 || codec_info_.channels <= 0 || codec_info_.frameSamples <= 0 ||
        codec_info_.superframeSize <= 0 || codec_info_.framesInSuperframe <= 0) {
      initialized_ = false;
      result->result = kResultCodecError | kResultInvalidData;
      result->internal_result = info_ret;
      return false;
    }

    initialized_ = true;
    num_frames_ = 0;
    total_decoded_ = 0;
    superframe_remain_ = (uint32_t)codec_info_.superframeSize;
    return true;
  }

  int DecodeFrame(const uint8_t* input, int* bytes_used) {
    // LibAtrac9 decodes to s16; widen for the other encodings.
    const int ret =
        Atrac9Decode(handle_, input, reinterpret_cast<short*>(pcm_.data()), bytes_used);
    if (ret != 0)
      return ret;

    const int channels = codec_info_.channels;
    const int samples = codec_info_.frameSamples;
    const size_t frame_bytes = (size_t)channels * samples * BytesPerSample(encoding_);
    if (frame_bytes > pcm_.size())
      return -1;
    if (encoding_ == SampleEncoding::S16)
      return 0;

    const short* src = reinterpret_cast<const short*>(pcm_.data());
    if (encoding_ == SampleEncoding::Float) {
      float* dst = reinterpret_cast<float*>(pcm_.data());
      // Widen in place from the back so the src window is intact.
      for (int i = channels * samples - 1; i >= 0; i--)
        dst[i] = (float)src[i];
    } else {
      int32_t* dst = reinterpret_cast<int32_t*>(pcm_.data());
      for (int i = channels * samples - 1; i >= 0; i--)
        dst[i] = (int32_t)src[i];
    }
    (void)frame_bytes;
    return 0;
  }

  uint64_t CopyFrame(uint8_t* output, size_t output_size, size_t* output_written,
                     GaplessState* gapless) {
    if (!initialized_ || codec_info_.channels <= 0 || codec_info_.frameSamples <= 0)
      return 0;

    const uint32_t channels = (uint32_t)codec_info_.channels;
    const uint32_t frame_samples = (uint32_t)codec_info_.frameSamples;
    const size_t bytes_per_frame = channels * BytesPerSample(encoding_);
    uint32_t skip_samples = 0;
    uint32_t samples_to_write = frame_samples;

    if (gapless) {
      skip_samples = std::min<uint32_t>(frame_samples, gapless->current.skip_samples);
      gapless->current.skip_samples =
          (uint16_t)(gapless->current.skip_samples - skip_samples);
      samples_to_write -= skip_samples;
      if (gapless->HasSampleLimit())
        samples_to_write = std::min<uint32_t>(samples_to_write, gapless->current.total_samples);
    }

    const size_t copy_bytes = (size_t)samples_to_write * bytes_per_frame;
    if (copy_bytes != 0) {
      if (!output || *output_written > output_size || copy_bytes > output_size - *output_written)
        return 0;
      std::memcpy(output + *output_written,
                  pcm_.data() + (size_t)skip_samples * bytes_per_frame, copy_bytes);
      *output_written += copy_bytes;
    }

    if (gapless) {
      if (gapless->HasSampleLimit()) {
        gapless->current.total_samples =
            (gapless->current.total_samples > samples_to_write
                 ? gapless->current.total_samples - samples_to_write
                 : 0);
      }
      const uint32_t skipped = frame_samples - samples_to_write;
      const uint32_t skipped_total = std::min<uint32_t>(
          0xffffu, (uint32_t)gapless->current.skipped_samples + skipped);
      gapless->current.skipped_samples = (uint16_t)skipped_total;
    }

    return samples_to_write;
  }

  void AdvanceSuperframe(uint32_t bytes_used, size_t input_size, size_t* input_offset,
                         DecodeResult* result) {
    superframe_remain_ -= bytes_used;
    num_frames_++;

    if (num_frames_ < (uint32_t)codec_info_.framesInSuperframe)
      return;

    if (superframe_remain_ != 0) {
      const size_t remaining_input = (*input_offset < input_size ? input_size - *input_offset : 0);
      if (remaining_input < superframe_remain_) {
        result->result = kResultPartialInput;
        return;
      }
      *input_offset += superframe_remain_;
      result->input_consumed = *input_offset;
    }

    superframe_remain_ = (uint32_t)codec_info_.superframeSize;
    num_frames_ = 0;
  }

  void DrainSuperframe(const uint8_t* input, size_t input_size, size_t* input_offset,
                       DecodeResult* result) {
    while (num_frames_ != 0) {
      const size_t remaining_input = (*input_offset < input_size ? input_size - *input_offset : 0);
      if (remaining_input < MinimumInputSize()) {
        result->result = kResultPartialInput;
        return;
      }

      int bytes_used = 0;
      const int ret = DecodeFrame(input + *input_offset, &bytes_used);
      if (ret != 0) {
        result->result = kResultCodecError | kResultFatal;
        result->internal_result = ret;
        return;
      }
      if (bytes_used <= 0 || (size_t)bytes_used > remaining_input ||
          (uint32_t)bytes_used > superframe_remain_) {
        result->result = kResultCodecError | kResultInvalidData;
        return;
      }

      *input_offset += (size_t)bytes_used;
      result->input_consumed = *input_offset;
      result->frames++;
      AdvanceSuperframe((uint32_t)bytes_used, input_size, input_offset, result);
      if (result->result != kResultOk)
        return;
    }
  }

  bool ParseRiffHeader(const uint8_t* input, size_t input_size, size_t* data_offset,
                       GaplessState* gapless, DecodeResult* result) {
    *data_offset = 0;
    if (input_size < 12 || !FourCc(input, 'R', 'I', 'F', 'F') ||
        !FourCc(input + 8, 'W', 'A', 'V', 'E')) {
      result->result = kResultInvalidData;
      return false;
    }

    size_t offset = 12;
    while (offset + 8 <= input_size) {
      const uint8_t* chunk = input + offset;
      const size_t chunk_size = ReadLe32(chunk + 4);
      const size_t payload = offset + 8;

      if (FourCc(chunk, 'f', 'm', 't', ' ')) {
        if (chunk_size < 48 || payload + 48 > input_size) {
          result->result = kResultInvalidData;
          return false;
        }
        const uint8_t* fmt = input + payload;
        static constexpr uint8_t kAt9Guid[16] = {0xd2, 0x42, 0xe1, 0x47, 0xba, 0x36,
                                                 0x8d, 0x4d, 0x88, 0xfc, 0x61, 0x65,
                                                 0x4f, 0x8c, 0x83, 0x6c};
        if ((ReadLe32(fmt) & 0xffff) != 0xfffe || std::memcmp(fmt + 24, kAt9Guid, 16) != 0) {
          result->result = kResultInvalidData;
          return false;
        }
        if (!InitializeConfig(fmt + 44, result))
          return false;
      } else if (FourCc(chunk, 'f', 'a', 'c', 't')) {
        if (chunk_size >= 12 && payload + 8 <= input_size && gapless) {
          SidebandGaplessDecode gapless_decode{};
          gapless_decode.total_samples = ReadLe32(input + payload);
          gapless_decode.skip_samples =
              (uint16_t)std::min<uint32_t>(ReadLe32(input + payload + 4), 0xffffu);
          gapless->Set(gapless_decode, true);
        }
      } else if (FourCc(chunk, 'd', 'a', 't', 'a')) {
        // The data chunk is normally much larger than one decode call; only
        // its start offset must be present.
        if (payload > input_size) {
          result->result = kResultPartialInput;
          return false;
        }
        *data_offset = payload;
        result->input_consumed = payload;
        result->format = GetFormat();
        result->total_decoded_samples = total_decoded_;
        return true;
      }

      const size_t padded_size = chunk_size + (chunk_size & 1u);
      offset += 8 + padded_size;
    }

    result->result = kResultPartialInput;
    return false;
  }

  uint32_t max_channels_ = 2;
  SampleEncoding encoding_ = SampleEncoding::S16;
  bool parse_riff_ = false;
  bool initialized_ = false;
  bool has_config_ = false;
  void* handle_ = nullptr;
  uint8_t config_data_[4]{};
  uint32_t superframe_remain_ = 0;
  uint32_t num_frames_ = 0;
  uint64_t total_decoded_ = 0;
  Atrac9CodecInfo codec_info_{};
  // A superframe is at most 8 frames of 256 samples in 8 channels; 16 KiB of
  // s16 covers every config, and s32/float widening happens in place.
  std::vector<uint8_t> pcm_ = std::vector<uint8_t>(8 * 256 * 4, 0);
};

}  // namespace kern::ajm
