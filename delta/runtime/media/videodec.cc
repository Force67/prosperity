#ifdef DELTA_HAVE_AVCODEC
#include "runtime/media/videodec.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <base/logging.h>
#include <utl/mem.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
}

namespace runtime::video {
namespace {
constexpr i32 kSize = i32(0x811d0101), kPointer = i32(0x811d0102);
constexpr i32 kHandle = i32(0x811d0103), kConfig = i32(0x811d0104);
constexpr i32 kFrame = i32(0x811d0109), kDecode = i32(0x811d0200);
bool Readable(const void* p, u64 bytes) {
  return p && bytes && bytes <= std::numeric_limits<size_t>::max() &&
         utl::isMemoryRangeMapped(p, bytes);
}
template<class T> bool Sized(const T* p) { return Readable(p, sizeof(T)) && p->size == sizeof(T); }
bool ValidConfig(const Config* c) {
  return Readable(c, sizeof(*c)) && (c->size == 0x48 || c->size == 0x50) &&
      c->codec == 1 && c->width > 0 && c->width <= 8192 && c->height > 0 && c->height <= 8192;
}
u32 Align(u32 n, u32 alignment) { return (n + alignment - 1) & ~(alignment - 1); }
struct Stamp { u64 pts, dts, attached; };
struct Decoder {
  AVCodecContext* context = nullptr;
  AVFrame* frame = av_frame_alloc();
  bool draining = false, pending = false;
  u32 outputs = 0;
  ~Decoder() { av_frame_free(&frame); avcodec_free_context(&context); }
};
std::mutex lock;
std::unordered_map<void*, std::unique_ptr<Decoder>> decoders;
std::unordered_map<void*, std::array<u8, 0x78>> pictures;

template<class T> void Put(std::array<u8, 0x78>& b, size_t offset, T value) {
  std::memcpy(b.data() + offset, &value, sizeof(value));
}

i32 OutputArgs(FrameBuffer* target, Output* out) {
  if (!Sized(target) || !Readable(out, 8)) return kPointer;
  if (out->size != 0x30 && out->size != 0x38) return kSize;
  if (!Readable(out, out->size)) return kPointer;
  target->accepted = false;
  const u64 size = out->size;
  std::memset(out, 0, size);
  out->size = size;
  return 0;
}

// Receive exactly one presentation-order picture. EAGAIN is normal decoder
// latency; Flush drains delayed B pictures rather than inventing completion.
i32 Receive(Decoder& d, FrameBuffer& target, Output& out) {
  if (!d.pending) {
    const int status = avcodec_receive_frame(d.context, d.frame);
    if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) return 0;
    if (status < 0) return kDecode;
    d.pending = true;
  }
  const AVFrame& f = *d.frame;
  if (f.format != AV_PIX_FMT_YUV420P && f.format != AV_PIX_FMT_NV12) return kDecode;
  const u32 width = f.width, height = f.height;
  if (!width || !height || width > 8192 || height > 8192 || (width & 1) || (height & 1)) return kDecode;
  const u32 pitch = Align(width, 64), rows = Align(height, 16);
  const u64 bytes = u64(pitch) * rows * 3 / 2;
  if (target.bytes < bytes || !Readable(target.data, bytes)) return kFrame;
  auto* dst = static_cast<u8*>(target.data);
  std::memset(dst, 16, u64(pitch) * rows);
  std::memset(dst + u64(pitch) * rows, 128, u64(pitch) * rows / 2);
  for (u32 y = 0; y < height; ++y)
    std::memcpy(dst + u64(y) * pitch, f.data[0] + ptrdiff_t(y) * f.linesize[0], width);
  auto* uv = dst + u64(pitch) * rows;
  for (u32 y = 0; y < height / 2; ++y) {
    if (f.format == AV_PIX_FMT_NV12) {
      std::memcpy(uv + u64(y) * pitch, f.data[1] + ptrdiff_t(y) * f.linesize[1], width);
    } else {
      for (u32 x = 0; x < width / 2; ++x) {
        uv[u64(y) * pitch + x * 2] = f.data[1][ptrdiff_t(y) * f.linesize[1] + x];
        uv[u64(y) * pitch + x * 2 + 1] = f.data[2][ptrdiff_t(y) * f.linesize[2] + x];
      }
    }
  }
  out.valid = 1;
  out.pictures = 1;
  out.codec = 1;
  out.width = Align(width, 16);
  out.pitch = pitch;
  out.height = rows;
  out.data = target.data;
  out.bytes = bytes;
  if (out.size == sizeof(Output)) { out.format = 0; out.pitch_bytes = pitch; }
  target.accepted = true;
  std::array<u8, 0x78> info{};
  Put(info, 0, u64(info.size()));
  info[8] = 1;
  Stamp stamp{u64(f.pts), u64(f.pkt_dts), 0};
  if (f.opaque_ref && f.opaque_ref->size == sizeof(Stamp))
    std::memcpy(&stamp, f.opaque_ref->data, sizeof(stamp));
  Put(info, 0x10, stamp.pts); Put(info, 0x18, stamp.dts); Put(info, 0x20, stamp.attached);
  info[0x28] = (f.flags & AV_FRAME_FLAG_KEY) != 0;
  info[0x29] = d.context->profile;
  info[0x2a] = d.context->level;
  Put(info, 0x2c, (width + 15) / 16 - 1);
  Put(info, 0x30, (height + 15) / 16 - 1);
  info[0x34] = 1; // progressive frame
  info[0x35] = out.width != width || rows != height;
  Put(info, 0x3c, (out.width - width) / 2);
  Put(info, 0x44, (rows - height) / 2);
  info[0x48] = 1; info[0x49] = 1; // square pixels
  Put(info, 0x4a, u16(1)); Put(info, 0x4c, u16(1));
  info[0x4e] = 1; info[0x4f] = 5;
  info[0x50] = f.color_range == AVCOL_RANGE_JPEG;
  info[0x51] = 1; info[0x52] = f.color_primaries;
  info[0x53] = f.color_trc; info[0x54] = f.colorspace;
  if (d.context->framerate.num && d.context->framerate.den) {
    info[0x55] = 1;
    Put(info, 0x58, u32(d.context->framerate.den));
    Put(info, 0x5c, u32(d.context->framerate.num * 2));
    info[0x60] = 1;
  }
  if (pictures.size() >= 512 && !pictures.count(target.data)) pictures.erase(pictures.begin());
  pictures[target.data] = info;
  ++d.outputs;
  if (d.outputs == 1 || d.outputs % 120 == 0)
    BASE_LOGI("videodec", "CPU H.264 frame {} {}x{} pts={} buffer={:p}",
              d.outputs, width, height, stamp.pts, target.data);
  av_frame_unref(d.frame);
  d.pending = false;
  return 0;
}
}

i32 PS4ABI QueryCompute(ComputeMemory* m) {
  if (!Sized(m)) return kSize;
  m->bytes = 4096; m->data = nullptr;
  return 0;
}
i32 PS4ABI AllocateQueue(const ComputeConfig* c, const ComputeMemory* m, void** result) {
  if (!Sized(c) || !Sized(m) || !Readable(result, sizeof(*result))) return kPointer;
  if (!m->data || m->bytes < 4096 || c->pipe > 4 || c->queue > 7) return kConfig;
  *result = m->data;
  return 0;
}
i32 PS4ABI ReleaseQueue(void*) { return 0; }
i32 PS4ABI QueryMemory(const Config* c, Memory* m) {
  if (!ValidConfig(c) || !Sized(m)) return kConfig;
  m->cpu_bytes = m->gpu_bytes = m->shared_bytes = 65536;
  m->cpu = m->gpu = m->shared = nullptr;
  m->frame_bytes = u64(Align(c->width, 64)) * Align(c->height, 16) * 3 / 2 + 4096;
  m->alignment = 256; m->reserved = 0;
  return 0;
}
i32 PS4ABI Create(const Config* c, const Memory* m, void** result) {
  if (!ValidConfig(c) || !Sized(m) || !Readable(result, sizeof(*result))) return kConfig;
  *result = nullptr;
  auto d = std::make_unique<Decoder>();
  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (!codec || !d->frame) return kDecode;
  d->context = avcodec_alloc_context3(codec);
  if (!d->context) return kDecode;
  d->context->width = c->width; d->context->height = c->height;
  d->context->thread_count = 2;
  d->context->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
  if (avcodec_open2(d->context, codec, nullptr) < 0) return kDecode;
  std::lock_guard guard(lock);
  *result = d.get();
  decoders.emplace(d.get(), std::move(d));
  BASE_LOGI("videodec", "created CPU H.264 decoder {}x{} config={:#x}", c->width, c->height, c->size);
  return 0;
}
i32 PS4ABI Delete(void* handle) {
  std::lock_guard guard(lock);
  return decoders.erase(handle) ? 0 : kHandle;
}
i32 PS4ABI Decode(void* handle, const Input* input, FrameBuffer* target, Output* out) {
  if (!Sized(input) || !Readable(input->data, input->bytes) || input->bytes > 64 * 1024 * 1024) return kPointer;
  if (const i32 error = OutputArgs(target, out)) return error;
  std::lock_guard guard(lock);
  const auto it = decoders.find(handle);
  if (it == decoders.end()) return kHandle;
  Decoder& d = *it->second;
  if (d.draining || d.pending) return kDecode;
  AVPacket* packet = av_packet_alloc();
  if (!packet) return kDecode;
  // Own and pad the packet: libavcodec may retain it across B-frame reordering.
  int status = av_new_packet(packet, input->bytes);
  if (status >= 0) {
    std::memcpy(packet->data, input->data, input->bytes);
    packet->pts = input->pts; packet->dts = input->dts;
    packet->opaque_ref = av_buffer_alloc(sizeof(Stamp));
    if (!packet->opaque_ref) status = AVERROR(ENOMEM);
    else {
      const Stamp stamp{input->pts, input->dts, input->attached};
      std::memcpy(packet->opaque_ref->data, &stamp, sizeof(stamp));
      status = avcodec_send_packet(d.context, packet);
    }
  }
  av_packet_free(&packet);
  if (status < 0) {
    BASE_LOGI("videodec", "H.264 packet rejected: {} ({} bytes)", status, input->bytes);
    return kDecode;
  }
  return Receive(d, *target, *out);
}
i32 PS4ABI Flush(void* handle, FrameBuffer* target, Output* out) {
  if (const i32 error = OutputArgs(target, out)) return error;
  std::lock_guard guard(lock);
  const auto it = decoders.find(handle);
  if (it == decoders.end()) return kHandle;
  Decoder& d = *it->second;
  if (!d.draining) {
    const int status = avcodec_send_packet(d.context, nullptr);
    if (status < 0 && status != AVERROR_EOF) return kDecode;
    d.draining = true;
  }
  return Receive(d, *target, *out);
}
i32 PS4ABI Reset(void* handle) {
  std::lock_guard guard(lock);
  const auto it = decoders.find(handle);
  if (it == decoders.end()) return kHandle;
  Decoder& d = *it->second;
  avcodec_flush_buffers(d.context); av_frame_unref(d.frame);
  d.draining = d.pending = false;
  return 0;
}
i32 PS4ABI Picture(const Output* out, void* first, void* second) {
  if (!Readable(out, 0x30) || (out->size != 0x30 && out->size != 0x38)) return kPointer;
  std::lock_guard guard(lock);
  for (void* dest : {first, second}) {
    if (!dest) continue;
    if (!Readable(dest, 8)) return kPointer;
    u64 size; std::memcpy(&size, dest, 8);
    if ((size != 0x68 && size != 0x78) || !Readable(dest, size)) return kSize;
    std::memset(dest, 0, size);
    if (dest == first && out->valid && out->pictures) {
      const auto it = pictures.find(out->data);
      if (it == pictures.end()) return kFrame;
      std::memcpy(dest, it->second.data(), size);
    }
    std::memcpy(dest, &size, 8);
  }
  return 0;
}
}
#endif
