/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_trace.h"

#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/guest_memory.h"
#include "gpu/vulkan/vk_capture.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_png.h"
#include "gpu/vulkan/vk_render_target.h"

#include <sys/stat.h>
#include <utl/options.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
// Arming. Any one of these turns the debugger on; everything else follows.
DELTA_OPTION(int, kCaptureFrame, "DELTA_GPU_CAPTURE", 0);
DELTA_OPTION(float, kCaptureAfter, "DELTA_GPU_CAPTURE_AFTER", 0.f);
DELTA_OPTION(int, kCaptureBusy, "DELTA_GPU_CAPTURE_BUSY", 0);
DELTA_OPTION(int, kCaptureCount, "DELTA_GPU_CAPTURE_COUNT", 1);
DELTA_OPTION(const char*, kCaptureDir, "DELTA_GPU_CAPTURE_DIR", nullptr);
// What to read back and write as PNG when the frame closes: any of
// rt, depth, tex, all, none (comma separated).
DELTA_OPTION(const char*, kCaptureDump, "DELTA_GPU_CAPTURE_DUMP", "rt,depth");
// Mid-frame snapshots: a comma list of draw indices, "every:N", or "all".
DELTA_OPTION(const char*, kCaptureAt, "DELTA_GPU_CAPTURE_AT", nullptr);
DELTA_OPTION(float, kCaptureExposure, "DELTA_GPU_CAPTURE_EXPOSURE", 1.f);
DELTA_OPTION(float, kCaptureGamma, "DELTA_GPU_CAPTURE_GAMMA", 2.2f);
// Bytes of each constant buffer written into the capture (0 = the whole
// binding).
DELTA_OPTION(int, kCaptureCbufBytes, "DELTA_GPU_CAPTURE_CBUF_BYTES", 256);
// Also write the untouched readback bytes next to each PNG.
DELTA_OPTION(bool, kCaptureRaw, "DELTA_GPU_CAPTURE_RAW", false);
DELTA_OPTION(bool, kValidate, "DELTA_GPU_VALIDATE", false);
DELTA_OPTION(bool, kExitAfter, "DELTA_GPU_CAPTURE_EXIT", false);
}  // namespace

namespace gpu::vk::trace {

bool g_recording = false;

namespace {

// --- state -----------------------------------------------------------------

std::FILE* g_file = nullptr;
std::string g_dir;
std::string g_prefix;  // "<dir>/frame_<n>"
int g_frame_num = 0;
uint64_t g_seq = 0;
uint32_t g_draw_seq = 0;
int g_frames_left = 0;
int g_armed_frame = 0;  // resolved frame number once a trigger fires
bool g_finished = false;
uint64_t g_start_ns = 0;
VkDebugUtilsMessengerEXT g_messenger = VK_NULL_HANDLE;
uint32_t g_validation_messages = 0;

// Mid-frame readbacks recorded into the frame's own command buffer; drained
// once the queue is idle at FrameEnd.
struct Snapshot {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* map = nullptr;
  uint64_t bytes = 0;
  uint32_t w = 0, h = 0;
  VkFormat fmt = VK_FORMAT_UNDEFINED;
  uint64_t base = 0;
  bool depth = false;
  uint32_t at_draw = 0;
};
std::vector<Snapshot> g_snapshots;

// Every distinct guest texture descriptor the frame bound, for the end-of-
// frame dump.
struct TexKey {
  uint64_t base = 0;
  uint32_t w = 0, h = 0, dfmt = 0, nfmt = 0, tiling = 0, pitch = 0;
  uint32_t layers = 0, mips = 0;
  bool pow2_pad = false;
  bool operator==(const TexKey&) const = default;
};
struct TexKeyHash {
  size_t operator()(const TexKey& k) const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix(k.base);
    mix((uint64_t(k.w) << 32) | k.h);
    mix((uint64_t(k.dfmt) << 32) | k.nfmt);
    mix((uint64_t(k.tiling) << 32) | k.pitch);
    mix((uint64_t(k.layers) << 32) | k.mips);
    return size_t(h);
  }
};
std::unordered_set<TexKey, TexKeyHash> g_frame_texs;

std::unordered_map<uint64_t, std::string>& NameTable() {
  static std::unordered_map<uint64_t, std::string> table;
  return table;
}

uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// --- arming ----------------------------------------------------------------

bool Armed() {
  static const bool armed = kCaptureFrame.get() > 0 ||
                            kCaptureAfter.get() > 0.f || kCaptureBusy.get() > 0;
  return armed;
}

// A comma list of draw indices, "every:N", or "all".
struct SnapshotPlan {
  bool all = false;
  uint32_t every = 0;
  std::vector<uint32_t> at;
};
const SnapshotPlan& Plan() {
  static const SnapshotPlan plan = [] {
    SnapshotPlan p;
    const char* spec = kCaptureAt;
    if (!spec || !*spec)
      return p;
    if (!std::strcmp(spec, "all")) {
      p.all = true;
      return p;
    }
    if (!std::strncmp(spec, "every:", 6)) {
      p.every = static_cast<uint32_t>(std::strtoul(spec + 6, nullptr, 0));
      return p;
    }
    for (const char* c = spec; *c;) {
      char* end = nullptr;
      const unsigned long v = std::strtoul(c, &end, 0);
      if (end == c)
        break;
      p.at.push_back(static_cast<uint32_t>(v));
      c = *end ? end + 1 : end;
    }
    return p;
  }();
  return plan;
}

bool SnapshotWanted(uint32_t draw_index) {
  const SnapshotPlan& p = Plan();
  if (p.all)
    return true;
  if (p.every)
    return draw_index % p.every == 0;
  return std::find(p.at.begin(), p.at.end(), draw_index) != p.at.end();
}

bool DumpWanted(const char* what) {
  static const std::string spec = kCaptureDump.get() ? kCaptureDump.get() : "";
  if (spec.find("none") != std::string::npos)
    return false;
  return spec.find("all") != std::string::npos ||
         spec.find(what) != std::string::npos;
}

// --- JSON ------------------------------------------------------------------

// A capture line. JSON, one object per line, so a query tool can stream a
// frame of any size and grep can find a draw without a parser.
class Line {
 public:
  explicit Line(const char* type) {
    s_ = "{\"t\":\"";
    s_ += type;
    s_ += '"';
  }

  Line& Int(const char* key, long long v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lld", v);
    return Raw(key, buf);
  }
  Line& U(const char* key, unsigned long long v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llu", v);
    return Raw(key, buf);
  }
  // Guest addresses and register words are hex STRINGS: they survive every
  // JSON reader unchanged, and they grep against the rest of the module's
  // logs by eye.
  Line& Hex(const char* key, uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "\"0x%llx\"", (unsigned long long)v);
    return Raw(key, buf);
  }
  Line& Num(const char* key, double v) { return Raw(key, NumText(v)); }
  Line& Bool(const char* key, bool v) { return Raw(key, v ? "true" : "false"); }
  Line& Str(const char* key, const char* v) {
    std::string q = "\"";
    for (const char* c = v; c && *c; c++) {
      if (*c == '"' || *c == '\\')
        q += '\\';
      if (static_cast<unsigned char>(*c) < 0x20) {
        q += ' ';
        continue;
      }
      q += *c;
    }
    q += '"';
    return Raw(key, q);
  }
  Line& Raw(const char* key, const std::string& json) {
    s_ += ",\"";
    s_ += key;
    s_ += "\":";
    s_ += json;
    return *this;
  }

  // Non-finite floats are not JSON, but every JSON reader in this loop
  // (Python's included) accepts the bare literals -- and a NaN in a viewport
  // or a constant is exactly what a capture exists to show, so it must not be
  // silently rewritten to null.
  static std::string NumText(double v) {
    if (std::isnan(v))
      return "NaN";
    if (std::isinf(v))
      return v > 0 ? "Infinity" : "-Infinity";
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    return buf;
  }
  static std::string HexText(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "\"0x%llx\"", (unsigned long long)v);
    return buf;
  }
  static std::string IntText(long long v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lld", v);
    return buf;
  }

  void Emit();

 private:
  std::string s_;
};

// An array/object under construction inside a Line.
class Obj {
 public:
  Obj() : s_("{") {}
  Obj& Int(const char* key, long long v) { return Raw(key, Line::IntText(v)); }
  Obj& U(const char* key, unsigned long long v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llu", v);
    return Raw(key, buf);
  }
  Obj& Hex(const char* key, uint64_t v) { return Raw(key, Line::HexText(v)); }
  Obj& Num(const char* key, double v) { return Raw(key, Line::NumText(v)); }
  Obj& Bool(const char* key, bool v) { return Raw(key, v ? "true" : "false"); }
  Obj& Str(const char* key, const char* v) {
    std::string q = "\"";
    for (const char* c = v; c && *c; c++) {
      if (*c == '"' || *c == '\\')
        q += '\\';
      q += *c;
    }
    q += '"';
    return Raw(key, q);
  }
  Obj& Raw(const char* key, const std::string& json) {
    if (s_.size() > 1)
      s_ += ',';
    s_ += '"';
    s_ += key;
    s_ += "\":";
    s_ += json;
    return *this;
  }
  std::string Done() const { return s_ + "}"; }

 private:
  std::string s_;
};

class Arr {
 public:
  Arr() : s_("[") {}
  void Add(const std::string& json) {
    if (s_.size() > 1)
      s_ += ',';
    s_ += json;
  }
  void Add(const Obj& o) { Add(o.Done()); }
  std::string Done() const { return s_ + "]"; }
  bool empty() const { return s_.size() == 1; }  // NOLINT: cheap accessor

 private:
  std::string s_;
};

void Line::Emit() {
  if (!g_file)
    return;
  s_ += "}\n";
  std::fwrite(s_.data(), 1, s_.size(), g_file);
}

// --- guest memory ----------------------------------------------------------

struct MemStat {
  bool readable = false;
  uint64_t hash = 0;
  uint64_t nonzero = 0;
  uint64_t sampled = 0;
};

MemStat StatGuest(uint64_t base, uint64_t bytes) {
  MemStat s;
  if (!base || !bytes)
    return s;
  const uint64_t probe = std::min<uint64_t>(bytes, 1u << 20);
  if (!gpu::IsReadableRange(base, probe))
    return s;
  s.readable = true;
  const auto* p = reinterpret_cast<const uint8_t*>(base);
  uint64_t h = 1469598103934665603ull;
  const uint64_t step = probe > 65536 ? probe / 65536 : 1;
  for (uint64_t i = 0; i < probe; i += step) {
    h = (h ^ p[i]) * 1099511628211ull;
    s.nonzero += p[i] != 0;
    s.sampled++;
  }
  s.hash = h;
  return s;
}

std::string GuestObj(uint64_t base, uint64_t bytes) {
  const MemStat s = StatGuest(base, bytes);
  Obj o;
  o.Bool("readable", s.readable);
  if (s.readable) {
    o.Hex("hash", s.hash);
    o.U("nonzero", s.nonzero);
    o.U("sampled", s.sampled);
  }
  return o.Done();
}

std::string HexBytes(uint64_t base, uint32_t bytes) {
  if (!base || !bytes || !gpu::IsReadableRange(base, bytes))
    return "\"\"";
  const auto* p = reinterpret_cast<const uint8_t*>(base);
  std::string s = "\"";
  s.reserve(bytes * 2 + 2);
  static const char* kHex = "0123456789abcdef";
  for (uint32_t i = 0; i < bytes; i++) {
    s += kHex[p[i] >> 4];
    s += kHex[p[i] & 0xF];
  }
  s += '"';
  return s;
}

// A stable fingerprint of a guest shader: the code from its entry point up to
// s_endpgm. Guest shader ADDRESSES move between runs (titles stream shader
// code), so the address alone cannot identify a program across captures.
uint64_t GuestCodeHash(uint64_t addr, uint32_t* out_dwords) {
  if (out_dwords)
    *out_dwords = 0;
  if (!addr || !gpu::IsReadableRange(addr, 8))
    return 0;
  const auto* code = reinterpret_cast<const uint32_t*>(addr);
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = 0; i < 8192; i++) {
    if (!gpu::IsReadableRange(addr + i * 4ull, 4))
      break;
    h = (h ^ code[i]) * 1099511628211ull;
    if (out_dwords)
      *out_dwords = i + 1;
    if (code[i] == 0xBF810000u)  // s_endpgm
      break;
  }
  return h;
}

uint64_t SpirvHash(const std::vector<uint32_t>& spirv) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t w : spirv)
    h = (h ^ w) * 1099511628211ull;
  return h;
}

// --- format names / decode -------------------------------------------------

const char* FormatName(VkFormat fmt) {
  switch (fmt) {
    case VK_FORMAT_UNDEFINED:
      return "UNDEFINED";
    case VK_FORMAT_R8_UNORM:
      return "R8_UNORM";
    case VK_FORMAT_R8_SNORM:
      return "R8_SNORM";
    case VK_FORMAT_R8_UINT:
      return "R8_UINT";
    case VK_FORMAT_R8G8_UNORM:
      return "R8G8_UNORM";
    case VK_FORMAT_R8G8_UINT:
      return "R8G8_UINT";
    case VK_FORMAT_R8G8B8A8_UNORM:
      return "R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SNORM:
      return "R8G8B8A8_SNORM";
    case VK_FORMAT_R8G8B8A8_UINT:
      return "R8G8B8A8_UINT";
    case VK_FORMAT_R8G8B8A8_SRGB:
      return "R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_UNORM:
      return "B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB:
      return "B8G8R8A8_SRGB";
    case VK_FORMAT_R16_UNORM:
      return "R16_UNORM";
    case VK_FORMAT_R16_UINT:
      return "R16_UINT";
    case VK_FORMAT_R16_SFLOAT:
      return "R16_SFLOAT";
    case VK_FORMAT_R16G16_UNORM:
      return "R16G16_UNORM";
    case VK_FORMAT_R16G16_UINT:
      return "R16G16_UINT";
    case VK_FORMAT_R16G16_SFLOAT:
      return "R16G16_SFLOAT";
    case VK_FORMAT_R16G16B16A16_UNORM:
      return "R16G16B16A16_UNORM";
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return "R16G16B16A16_SFLOAT";
    case VK_FORMAT_R32_UINT:
      return "R32_UINT";
    case VK_FORMAT_R32_SFLOAT:
      return "R32_SFLOAT";
    case VK_FORMAT_R32G32_UINT:
      return "R32G32_UINT";
    case VK_FORMAT_R32G32_SFLOAT:
      return "R32G32_SFLOAT";
    case VK_FORMAT_R32G32B32_SFLOAT:
      return "R32G32B32_SFLOAT";
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return "R32G32B32A32_SFLOAT";
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      return "A2R10G10B10_UNORM_PACK32";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return "A2B10G10R10_UNORM_PACK32";
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return "B10G11R11_UFLOAT_PACK32";
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return "D32_SFLOAT_S8_UINT";
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
      return "BC1_RGBA_UNORM_BLOCK";
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      return "BC1_RGBA_SRGB_BLOCK";
    case VK_FORMAT_BC2_UNORM_BLOCK:
      return "BC2_UNORM_BLOCK";
    case VK_FORMAT_BC2_SRGB_BLOCK:
      return "BC2_SRGB_BLOCK";
    case VK_FORMAT_BC3_UNORM_BLOCK:
      return "BC3_UNORM_BLOCK";
    case VK_FORMAT_BC3_SRGB_BLOCK:
      return "BC3_SRGB_BLOCK";
    case VK_FORMAT_BC4_UNORM_BLOCK:
      return "BC4_UNORM_BLOCK";
    case VK_FORMAT_BC4_SNORM_BLOCK:
      return "BC4_SNORM_BLOCK";
    case VK_FORMAT_BC5_UNORM_BLOCK:
      return "BC5_UNORM_BLOCK";
    case VK_FORMAT_BC5_SNORM_BLOCK:
      return "BC5_SNORM_BLOCK";
    case VK_FORMAT_BC7_UNORM_BLOCK:
      return "BC7_UNORM_BLOCK";
    case VK_FORMAT_BC7_SRGB_BLOCK:
      return "BC7_SRGB_BLOCK";
    default:
      return "OTHER";
  }
}

const char* LayoutName(VkImageLayout layout) {
  switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return "UNDEFINED";
    case VK_IMAGE_LAYOUT_GENERAL:
      return "GENERAL";
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return "COLOR_ATTACHMENT";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return "DS_ATTACHMENT";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      return "DS_READ_ONLY";
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return "SHADER_READ_ONLY";
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return "TRANSFER_SRC";
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return "TRANSFER_DST";
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
      return "DEPTH_ATTACHMENT";
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
      return "DEPTH_READ_ONLY";
    case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
      return "STENCIL_ATTACHMENT";
    case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
      return "STENCIL_READ_ONLY";
    default:
      return "OTHER";
  }
}

float HalfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1F;
  uint32_t mantissa = value & 0x3FF;
  uint32_t bits;
  if (!exponent) {
    if (!mantissa) {
      bits = sign;
    } else {
      int unbiased = -14;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        unbiased--;
      }
      bits = sign | (static_cast<uint32_t>(unbiased + 127) << 23) |
             ((mantissa & 0x3FF) << 13);
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof out);
  return out;
}

float PackedUfloat(uint32_t value, uint32_t mantissa_bits) {
  const uint32_t mantissa = value & ((1u << mantissa_bits) - 1);
  const uint32_t exponent = value >> mantissa_bits;
  if (!exponent)
    return std::ldexp(static_cast<float>(mantissa),
                      -14 - static_cast<int>(mantissa_bits));
  if (exponent == 0x1F)
    return mantissa ? NAN : INFINITY;
  return std::ldexp(1.0f + static_cast<float>(mantissa) / (1u << mantissa_bits),
                    static_cast<int>(exponent) - 15);
}

// True when the texel is a floating-point (HDR) encoding, i.e. the exposure
// knob applies. Fills rgba with the raw channel values.
bool DecodeTexel(const uint8_t* src,
                 VkFormat fmt,
                 float rgba[4],
                 bool* is_hdr) {
  uint32_t packed;
  *is_hdr = false;
  switch (fmt) {
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
      const auto* h = reinterpret_cast<const uint16_t*>(src);
      for (int i = 0; i < 4; i++)
        rgba[i] = HalfToFloat(h[i]);
      *is_hdr = true;
      return true;
    }
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      std::memcpy(rgba, src, 16);
      *is_hdr = true;
      return true;
    case VK_FORMAT_R32G32B32_SFLOAT:
      std::memcpy(rgba, src, 12);
      rgba[3] = 1.0f;
      *is_hdr = true;
      return true;
    case VK_FORMAT_R32G32_SFLOAT:
      std::memcpy(rgba, src, 8);
      rgba[2] = 0.0f;
      rgba[3] = 1.0f;
      *is_hdr = true;
      return true;
    case VK_FORMAT_R32_SFLOAT:
      std::memcpy(rgba, src, 4);
      rgba[1] = rgba[2] = rgba[0];
      rgba[3] = 1.0f;
      *is_hdr = true;
      return true;
    case VK_FORMAT_R16G16_SFLOAT: {
      const auto* h = reinterpret_cast<const uint16_t*>(src);
      rgba[0] = HalfToFloat(h[0]);
      rgba[1] = HalfToFloat(h[1]);
      rgba[2] = 0.0f;
      rgba[3] = 1.0f;
      *is_hdr = true;
      return true;
    }
    case VK_FORMAT_R16_SFLOAT: {
      const auto* h = reinterpret_cast<const uint16_t*>(src);
      rgba[0] = rgba[1] = rgba[2] = HalfToFloat(h[0]);
      rgba[3] = 1.0f;
      *is_hdr = true;
      return true;
    }
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      std::memcpy(&packed, src, 4);
      rgba[0] = PackedUfloat(packed & 0x7FF, 6);
      rgba[1] = PackedUfloat((packed >> 11) & 0x7FF, 6);
      rgba[2] = PackedUfloat((packed >> 22) & 0x3FF, 5);
      rgba[3] = 1.0f;
      *is_hdr = true;
      return true;
    case VK_FORMAT_R16G16B16A16_UNORM: {
      const auto* u = reinterpret_cast<const uint16_t*>(src);
      for (int i = 0; i < 4; i++)
        rgba[i] = u[i] / 65535.0f;
      return true;
    }
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
      rgba[0] = src[2] / 255.0f;
      rgba[1] = src[1] / 255.0f;
      rgba[2] = src[0] / 255.0f;
      rgba[3] = src[3] / 255.0f;
      return true;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_UINT:
      for (int i = 0; i < 4; i++)
        rgba[i] = src[i] / 255.0f;
      return true;
    default:
      return false;
  }
}

uint8_t ToByte(float v, bool hdr) {
  if (!std::isfinite(v))
    return hdr ? 255 : 0;  // a NaN in an HDR target must be visible, not black
  if (hdr) {
    v *= kCaptureExposure.get();
    const float gamma = kCaptureGamma.get();
    if (gamma > 0.0f && gamma != 1.0f)
      v = std::pow(std::clamp(v, 0.0f, 1.0f), 1.0f / gamma);
  }
  return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
}

// --- image readback --------------------------------------------------------

// One-shot copy of an image to host memory: a private command buffer submitted
// on the device's diagnostic fence, exactly like the RTSTAT readback. Only a
// captured frame runs it, so the stall it costs buys a complete answer.
bool ReadImage(VkImage image,
               VkImageAspectFlags aspect,
               uint32_t w,
               uint32_t h,
               uint32_t texel_bytes,
               VkImageLayout layout,
               VkImageLayout* new_layout,
               std::vector<uint8_t>& out) {
  if (!image || !w || !h)
    return false;
  const VkDeviceSize bytes = VkDeviceSize(w) * h * texel_bytes;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = bytes;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer buffer = VK_NULL_HANDLE;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &buffer) != VK_SUCCESS)
    return false;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, buffer, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = FindMemoryType(mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &memory) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, buffer, nullptr);
    return false;
  }
  vkBindBufferMemory(g_dev.device, buffer, memory, 0);

  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g_dev.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  bool ok = vkAllocateCommandBuffers(g_dev.device, &ca, &cmd) == VK_SUCCESS;
  if (ok) {
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ok = vkBeginCommandBuffer(cmd, &cbi) == VK_SUCCESS;
  }
  if (ok) {
    if (aspect == VK_IMAGE_ASPECT_COLOR_BIT)
      ImageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   ColorImageAccess(layout), VK_ACCESS_TRANSFER_READ_BIT);
    else
      DepthBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {aspect, 0, 0, 1};
    copy.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buffer, 1, &copy);
    ok = vkEndCommandBuffer(cmd) == VK_SUCCESS;
  }
  if (ok) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ok = vkResetFences(g_dev.device, 1, &g_dev.fence) == VK_SUCCESS &&
         vkQueueSubmit(g_dev.queue, 1, &si, g_dev.fence) == VK_SUCCESS &&
         vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX) ==
             VK_SUCCESS;
  }
  if (ok) {
    if (new_layout)
      *new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    void* map = nullptr;
    if (vkMapMemory(g_dev.device, memory, 0, bytes, 0, &map) == VK_SUCCESS) {
      out.resize(bytes);
      std::memcpy(out.data(), map, bytes);
      vkUnmapMemory(g_dev.device, memory);
    } else {
      ok = false;
    }
  }
  if (cmd)
    vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &cmd);
  vkDestroyBuffer(g_dev.device, buffer, nullptr);
  vkFreeMemory(g_dev.device, memory, nullptr);
  return ok;
}

// Convert readback bytes to RGBA8 and write the PNG, returning the per-channel
// statistics the capture records next to it (a dump nobody looks at still has
// to say whether the target was black).
struct PixelStats {
  double min[4] = {1e30, 1e30, 1e30, 1e30};
  double max[4] = {-1e30, -1e30, -1e30, -1e30};
  double mean[4] = {};
  uint64_t nonzero = 0;
  uint64_t nan = 0;
  uint64_t count = 0;
};

std::string StatsObj(const PixelStats& s) {
  Obj o;
  Arr mn, mx, mean;
  for (int i = 0; i < 4; i++) {
    mn.Add(Line::NumText(s.count ? s.min[i] : 0.0));
    mx.Add(Line::NumText(s.count ? s.max[i] : 0.0));
    mean.Add(Line::NumText(s.count ? s.mean[i] / double(s.count) : 0.0));
  }
  o.Raw("min", mn.Done());
  o.Raw("max", mx.Done());
  o.Raw("mean", mean.Done());
  o.U("nonzero", s.nonzero);
  o.U("nan", s.nan);
  o.U("texels", s.count);
  return o.Done();
}

bool WriteImagePng(const std::string& path,
                   const uint8_t* src,
                   uint32_t w,
                   uint32_t h,
                   VkFormat fmt,
                   PixelStats* stats) {
  const uint32_t texel = FormatBytes(fmt);
  std::vector<uint8_t> rgba(size_t(w) * h * 4);
  for (uint64_t i = 0; i < uint64_t(w) * h; i++) {
    float v[4] = {0, 0, 0, 1};
    bool hdr = false;
    if (!DecodeTexel(src + i * texel, fmt, v, &hdr)) {
      uint8_t bgra[4];
      ReadbackPixelBgra(src + i * texel, fmt, bgra);
      v[0] = bgra[2] / 255.0f;
      v[1] = bgra[1] / 255.0f;
      v[2] = bgra[0] / 255.0f;
      v[3] = bgra[3] / 255.0f;
    }
    if (stats) {
      stats->count++;
      bool any = false;
      for (int c = 0; c < 4; c++) {
        if (std::isnan(v[c])) {
          stats->nan++;
          continue;
        }
        stats->min[c] = std::min<double>(stats->min[c], v[c]);
        stats->max[c] = std::max<double>(stats->max[c], v[c]);
        stats->mean[c] += v[c];
        any |= c < 3 && v[c] != 0.0f;
      }
      stats->nonzero += any;
    }
    uint8_t* dst = rgba.data() + i * 4;
    for (int c = 0; c < 4; c++)
      dst[c] = ToByte(v[c], hdr && c < 3);
    dst[3] = ToByte(v[3], false);
  }
  return WritePngRgba8(path.c_str(), rgba.data(), w, h);
}

// Depth is a single float per texel with no meaningful absolute range (a
// reversed-Z target lives in [0.996, 1]), so it is normalised against its own
// extent and the extent is recorded.
bool WriteDepthPng(const std::string& path,
                   const uint8_t* src,
                   uint32_t w,
                   uint32_t h,
                   PixelStats* stats) {
  const auto* d = reinterpret_cast<const float*>(src);
  const uint64_t n = uint64_t(w) * h;
  float lo = 1e30f, hi = -1e30f;
  for (uint64_t i = 0; i < n; i++) {
    if (!std::isfinite(d[i])) {
      if (stats)
        stats->nan++;
      continue;
    }
    lo = std::min(lo, d[i]);
    hi = std::max(hi, d[i]);
  }
  if (stats) {
    stats->count = n;
    for (int c = 0; c < 4; c++) {
      stats->min[c] = lo;
      stats->max[c] = hi;
    }
    for (uint64_t i = 0; i < n; i++) {
      if (std::isfinite(d[i])) {
        stats->mean[0] += d[i];
        stats->nonzero += d[i] != 0.0f;
      }
    }
  }
  const float span = (hi > lo) ? (hi - lo) : 1.0f;
  std::vector<uint8_t> rgba(size_t(n) * 4);
  for (uint64_t i = 0; i < n; i++) {
    const float t = std::isfinite(d[i]) ? (d[i] - lo) / span : 0.0f;
    const uint8_t g =
        static_cast<uint8_t>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
    rgba[i * 4 + 0] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = g;
    rgba[i * 4 + 3] = 255;
  }
  return WritePngRgba8(path.c_str(), rgba.data(), w, h);
}

void WriteRawSidecar(const std::string& path,
                     const uint8_t* src,
                     uint64_t bytes) {
  if (!kCaptureRaw)
    return;
  if (std::FILE* f = std::fopen((path + ".raw").c_str(), "wb")) {
    std::fwrite(src, 1, bytes, f);
    std::fclose(f);
  }
}

// --- guest texture decode --------------------------------------------------

void Bc1Colors(const uint8_t* block, uint8_t out[4][4], bool punchthrough) {
  uint16_t c0 = uint16_t(block[0] | (block[1] << 8));
  uint16_t c1 = uint16_t(block[2] | (block[3] << 8));
  auto expand = [](uint16_t c, uint8_t* p) {
    p[0] = uint8_t((((c >> 11) & 0x1F) * 255 + 15) / 31);
    p[1] = uint8_t((((c >> 5) & 0x3F) * 255 + 31) / 63);
    p[2] = uint8_t(((c & 0x1F) * 255 + 15) / 31);
    p[3] = 255;
  };
  expand(c0, out[0]);
  expand(c1, out[1]);
  if (!punchthrough || c0 > c1) {
    for (int i = 0; i < 3; i++) {
      out[2][i] = uint8_t((2 * out[0][i] + out[1][i]) / 3);
      out[3][i] = uint8_t((out[0][i] + 2 * out[1][i]) / 3);
    }
    out[2][3] = out[3][3] = 255;
  } else {
    for (int i = 0; i < 3; i++) {
      out[2][i] = uint8_t((out[0][i] + out[1][i]) / 2);
      out[3][i] = 0;
    }
    out[2][3] = 255;
    out[3][3] = 0;
  }
}

void Bc4Alpha(const uint8_t* block, uint8_t out[16]) {
  uint8_t a[8];
  a[0] = block[0];
  a[1] = block[1];
  if (a[0] > a[1]) {
    for (int i = 1; i < 7; i++)
      a[i + 1] = uint8_t(((7 - i) * a[0] + i * a[1]) / 7);
  } else {
    for (int i = 1; i < 5; i++)
      a[i + 1] = uint8_t(((5 - i) * a[0] + i * a[1]) / 5);
    a[6] = 0;
    a[7] = 255;
  }
  uint64_t bits = 0;
  for (int i = 0; i < 6; i++)
    bits |= uint64_t(block[2 + i]) << (8 * i);
  for (int i = 0; i < 16; i++)
    out[i] = a[(bits >> (3 * i)) & 7];
}

// Decode one 4x4 block into rgba (row-major, 16 texels). Returns false for a
// format this does not decode (recorded as a skip reason, never guessed at).
bool DecodeBlock(uint32_t dfmt, const uint8_t* block, uint8_t rgba[16][4]) {
  switch (dfmt) {
    case 35: {  // BC1
      uint8_t palette[4][4];
      Bc1Colors(block, palette, true);
      uint32_t idx;
      std::memcpy(&idx, block + 4, 4);
      for (int i = 0; i < 16; i++)
        std::memcpy(rgba[i], palette[(idx >> (2 * i)) & 3], 4);
      return true;
    }
    case 36: {  // BC2: 4-bit explicit alpha + BC1 colour
      uint8_t palette[4][4];
      Bc1Colors(block + 8, palette, false);
      uint32_t idx;
      std::memcpy(&idx, block + 12, 4);
      for (int i = 0; i < 16; i++) {
        std::memcpy(rgba[i], palette[(idx >> (2 * i)) & 3], 4);
        const uint8_t nibble = (block[i / 2] >> ((i & 1) * 4)) & 0xF;
        rgba[i][3] = uint8_t(nibble * 17);
      }
      return true;
    }
    case 37: {  // BC3: BC4 alpha + BC1 colour
      uint8_t alpha[16];
      Bc4Alpha(block, alpha);
      uint8_t palette[4][4];
      Bc1Colors(block + 8, palette, false);
      uint32_t idx;
      std::memcpy(&idx, block + 12, 4);
      for (int i = 0; i < 16; i++) {
        std::memcpy(rgba[i], palette[(idx >> (2 * i)) & 3], 4);
        rgba[i][3] = alpha[i];
      }
      return true;
    }
    case 38: {  // BC4: single channel
      uint8_t red[16];
      Bc4Alpha(block, red);
      for (int i = 0; i < 16; i++) {
        rgba[i][0] = rgba[i][1] = rgba[i][2] = red[i];
        rgba[i][3] = 255;
      }
      return true;
    }
    case 39: {  // BC5: two BC4 channels
      uint8_t red[16], green[16];
      Bc4Alpha(block, red);
      Bc4Alpha(block + 8, green);
      for (int i = 0; i < 16; i++) {
        rgba[i][0] = red[i];
        rgba[i][1] = green[i];
        rgba[i][2] = 0;
        rgba[i][3] = 255;
      }
      return true;
    }
    default:
      return false;
  }
}

// One linear (already de-tiled) element -> RGBA8.
bool DecodeGuestTexel(uint32_t dfmt,
                      uint32_t nfmt,
                      const uint8_t* src,
                      uint8_t out[4]) {
  uint32_t packed;
  switch (dfmt) {
    case 1:  // 8
      out[0] = out[1] = out[2] = src[0];
      out[3] = 255;
      return true;
    case 2:  // 16
      if (nfmt == 7) {
        out[0] = out[1] = out[2] =
            ToByte(HalfToFloat(uint16_t(src[0] | (src[1] << 8))), true);
      } else {
        out[0] = out[1] = out[2] = src[1];
      }
      out[3] = 255;
      return true;
    case 3:  // 8_8
      out[0] = src[0];
      out[1] = src[1];
      out[2] = 0;
      out[3] = 255;
      return true;
    case 10:  // 8_8_8_8
      out[0] = src[0];
      out[1] = src[1];
      out[2] = src[2];
      out[3] = src[3];
      return true;
    case 6:  // 10_11_11 / 11_11_10 float
      std::memcpy(&packed, src, 4);
      out[0] = ToByte(PackedUfloat(packed & 0x7FF, 6), true);
      out[1] = ToByte(PackedUfloat((packed >> 11) & 0x7FF, 6), true);
      out[2] = ToByte(PackedUfloat((packed >> 22) & 0x3FF, 5), true);
      out[3] = 255;
      return true;
    case 8:  // 2_10_10_10 (ARGB)
    case 9:  // 2_10_10_10 (ABGR)
      std::memcpy(&packed, src, 4);
      out[dfmt == 8 ? 2 : 0] = uint8_t(((packed & 0x3FF) * 255) / 1023);
      out[1] = uint8_t((((packed >> 10) & 0x3FF) * 255) / 1023);
      out[dfmt == 8 ? 0 : 2] = uint8_t((((packed >> 20) & 0x3FF) * 255) / 1023);
      out[3] = uint8_t(((packed >> 30) & 3) * 85);
      return true;
    case 5:  // 16_16
      if (nfmt == 7) {
        const auto* h = reinterpret_cast<const uint16_t*>(src);
        out[0] = ToByte(HalfToFloat(h[0]), true);
        out[1] = ToByte(HalfToFloat(h[1]), true);
      } else {
        out[0] = src[1];
        out[1] = src[3];
      }
      out[2] = 0;
      out[3] = 255;
      return true;
    case 12: {  // 16_16_16_16
      const auto* h = reinterpret_cast<const uint16_t*>(src);
      for (int i = 0; i < 4; i++)
        out[i] =
            nfmt == 7 ? ToByte(HalfToFloat(h[i]), i < 3) : uint8_t(h[i] >> 8);
      return true;
    }
    case 14: {  // 32_32_32_32 float
      const auto* f = reinterpret_cast<const float*>(src);
      for (int i = 0; i < 4; i++)
        out[i] = ToByte(f[i], i < 3);
      return true;
    }
    default:
      return false;
  }
}

// Read a guest texture out of guest memory, de-tile it (the layout the GPU
// stores, not the one a linear read sees) and write mip 0 layer 0 as a PNG.
// `reason` receives why it could not be written.
bool DumpGuestTexture(const TexKey& t,
                      const std::string& path,
                      const char** reason) {
  *reason = "";
  const bool compressed = GuestFormatBlockCompressed(t.dfmt);
  const uint32_t elem = GuestFormatElemBytes(t.dfmt);
  const uint32_t ew = compressed ? (t.w + 3) / 4 : t.w;
  const uint32_t eh = compressed ? (t.h + 3) / 4 : t.h;
  const uint32_t epitch = compressed ? ((t.pitch ? t.pitch : t.w) + 3) / 4
                                     : (t.pitch ? t.pitch : t.w);
  if (!t.w || !t.h) {
    *reason = "empty";
    return false;
  }
  gcn::TextureLayout32 layout;
  if (!gcn::BuildTextureLayout32(layout, ew, eh, epitch, std::max(1u, t.layers),
                                 std::max(1u, t.mips), t.tiling, t.pow2_pad,
                                 elem)) {
    *reason = "layout";
    return false;
  }
  if (!gpu::IsReadableRange(t.base, layout.mips[0].size)) {
    *reason = "unreadable";
    return false;
  }
  std::vector<uint8_t> linear(uint64_t(ew) * eh * elem);
  if (!gcn::DetileTextureMip32(reinterpret_cast<const void*>(t.base),
                               linear.data(), layout, 0, 0)) {
    *reason = "detile";
    return false;
  }
  std::vector<uint8_t> rgba(uint64_t(t.w) * t.h * 4, 0);
  if (compressed) {
    for (uint32_t by = 0; by < eh; by++) {
      for (uint32_t bx = 0; bx < ew; bx++) {
        uint8_t block[16][4];
        if (!DecodeBlock(t.dfmt,
                         linear.data() + (uint64_t(by) * ew + bx) * elem,
                         block)) {
          *reason = "format";
          return false;
        }
        for (uint32_t y = 0; y < 4; y++) {
          for (uint32_t x = 0; x < 4; x++) {
            const uint32_t px = bx * 4 + x, py = by * 4 + y;
            if (px >= t.w || py >= t.h)
              continue;
            std::memcpy(&rgba[(uint64_t(py) * t.w + px) * 4], block[y * 4 + x],
                        4);
          }
        }
      }
    }
  } else {
    for (uint64_t i = 0; i < uint64_t(t.w) * t.h; i++) {
      const uint32_t x = uint32_t(i % t.w), y = uint32_t(i / t.w);
      if (!DecodeGuestTexel(t.dfmt, t.nfmt,
                            linear.data() + (uint64_t(y) * ew + x) * elem,
                            &rgba[i * 4])) {
        *reason = "format";
        return false;
      }
    }
  }
  return WritePngRgba8(path.c_str(), rgba.data(), t.w, t.h);
}

// --- draw serialization ----------------------------------------------------

std::string ShaderObj(uint64_t addr, const std::vector<uint32_t>* spirv) {
  Obj o;
  o.Hex("addr", addr);
  uint32_t dwords = 0;
  o.Hex("guest_hash", GuestCodeHash(addr, &dwords));
  o.U("guest_dwords", dwords);
  if (spirv && !spirv->empty()) {
    o.Hex("spirv_hash", SpirvHash(*spirv));
    o.U("spirv_words", spirv->size());
  }
  return o.Done();
}

std::string TexObj(uint32_t index,
                   const rhi::DrawInfo::DrawTex& t,
                   const DrawBindings* b) {
  Obj o;
  o.U("i", index);
  o.Hex("base", t.base);
  // The address the descriptor itself was s_loaded from: a binding that
  // resolves to zeros is only diagnosable from the memory behind it.
  o.Hex("src", t.src);
  o.U("w", t.w);
  o.U("h", t.h);
  o.U("depth", t.depth);
  o.U("dfmt", t.dfmt);
  o.U("nfmt", t.nfmt);
  o.Str("format", FormatName(GuestTextureFormat(t.dfmt, t.nfmt)));
  o.U("tiling", t.tiling);
  o.U("pitch", t.pitch);
  o.U("layers", t.layers);
  o.U("base_array", t.base_array);
  o.U("view_layers", t.view_layers);
  o.U("mip_levels", t.mip_levels);
  o.U("base_mip", t.base_mip);
  o.U("view_mips", t.view_mips);
  o.U("min_lod", t.min_lod);
  o.Hex("swizzle", t.swizzle);
  o.Bool("pow2_pad", t.pow2_pad);
  o.Bool("arrayed", t.arrayed);
  o.Bool("is_3d", t.is_3d);
  o.Bool("is_1d", t.is_1d);
  o.Bool("storage", t.storage);
  o.Bool("depth_compare", t.depth_compare);
  o.Bool("force_lod_zero", t.force_lod_zero);
  o.Bool("null_descriptor", t.null_descriptor);
  o.Bool("sampler_valid", t.sampler_valid);
  Arr sampler;
  for (uint32_t i = 0; i < 4; i++)
    sampler.Add(Line::HexText(t.sampler[i]));
  o.Raw("sampler", sampler.Done());
  // How the binding actually resolved -- the whole point of a capture.
  const char* how = "unknown";
  uint64_t resolved = 0;
  if (b && index < b->tex_count) {
    if (b->tex_storage && b->tex_storage[index]) {
      how = "storage";
      resolved = b->tex_storage[index];
    } else if (b->tex_feedback && b->tex_feedback[index]) {
      how = "feedback";
      resolved = b->tex_feedback[index];
    } else if (b->tex_color && b->tex_color[index]) {
      how = "rt";
      resolved = b->tex_color[index];
    } else if (b->tex_depth && b->tex_depth[index]) {
      how = "depth";
      resolved = b->tex_depth[index];
    } else if (b->tex_guest && b->tex_guest[index]) {
      how = "guest";
    } else {
      how = "default";  // the 1x1 white/zero fallback: nothing resolved
    }
  }
  o.Str("resolved", how);
  o.Hex("resolved_base", resolved);
  const uint32_t elem = GuestFormatElemBytes(t.dfmt);
  const bool compressed = GuestFormatBlockCompressed(t.dfmt);
  const uint64_t stride = t.pitch ? t.pitch : t.w;
  const uint64_t bytes = compressed
                             ? ((stride + 3) / 4) * ((t.h + 3ull) / 4) * elem
                             : stride * t.h * elem;
  o.Raw("guest", GuestObj(t.base, bytes ? bytes : 4));
  o.Raw("descriptor_src", GuestObj(t.src, 32));
  return o.Done();
}

void NoteTexture(const rhi::DrawInfo::DrawTex& t) {
  if (!t.base || !t.w || !t.h)
    return;
  TexKey k;
  k.base = t.base;
  k.w = t.w;
  k.h = t.h;
  k.dfmt = t.dfmt;
  k.nfmt = t.nfmt;
  k.tiling = t.tiling;
  k.pitch = t.pitch;
  k.layers = t.layers;
  k.mips = t.mip_levels;
  k.pow2_pad = t.pow2_pad;
  g_frame_texs.insert(k);
}

// --- mid-frame snapshots ---------------------------------------------------

void QueueSnapshot(VkImage image,
                   VkImageAspectFlags aspect,
                   uint32_t w,
                   uint32_t h,
                   VkFormat fmt,
                   uint64_t base,
                   bool depth,
                   uint32_t at_draw,
                   VkImageLayout layout,
                   VkImageLayout* layout_out) {
  const uint32_t texel = depth ? 4 : FormatBytes(fmt);
  if (!image || !w || !h || !texel)
    return;
  Snapshot s;
  s.bytes = uint64_t(w) * h * texel;
  s.w = w;
  s.h = h;
  s.fmt = fmt;
  s.base = base;
  s.depth = depth;
  s.at_draw = at_draw;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = s.bytes;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &s.buffer) != VK_SUCCESS)
    return;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, s.buffer, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = FindMemoryType(mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &s.memory) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, s.buffer, nullptr);
    return;
  }
  vkBindBufferMemory(g_dev.device, s.buffer, s.memory, 0);
  vkMapMemory(g_dev.device, s.memory, 0, s.bytes, 0, &s.map);

  if (aspect == VK_IMAGE_ASPECT_COLOR_BIT)
    ImageBarrier(g_frame.cmd, image, layout,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ColorImageAccess(layout),
                 VK_ACCESS_TRANSFER_READ_BIT);
  else
    DepthBarrier(g_frame.cmd, image, layout,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT);
  if (layout_out)
    *layout_out = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkBufferImageCopy copy{};
  copy.imageSubresource = {aspect, 0, 0, 1};
  copy.imageExtent = {w, h, 1};
  vkCmdCopyImageToBuffer(g_frame.cmd, image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buffer, 1,
                         &copy);
  g_snapshots.push_back(s);
}

// Snapshot every attachment of the region that is open right now. A copy
// cannot be recorded inside dynamic rendering, so the region is closed first;
// the next draw reopens it with loadOp LOAD, which is what it would have done
// anyway for an already-rendered target.
void SnapshotOpenRegion(uint32_t draw_index) {
  if (!g_region.open)
    return;
  uint64_t mrt[8];
  const uint32_t n = std::min(g_region.cur_mrt_count, 8u);
  for (uint32_t i = 0; i < n; i++)
    mrt[i] = g_region.cur_mrt[i];
  const uint64_t depth_base = g_region.cur_depth;
  EndRegion();
  for (uint32_t i = 0; i < n; i++) {
    auto it = g_rts.find(mrt[i]);
    if (it == g_rts.end())
      continue;
    RTarget& rt = it->second;
    QueueSnapshot(rt.image, VK_IMAGE_ASPECT_COLOR_BIT, rt.w, rt.h, rt.fmt,
                  mrt[i], false, draw_index, rt.layout, &rt.layout);
  }
  auto dit = g_depths.find(depth_base);
  if (dit != g_depths.end()) {
    DepthTarget& dt = dit->second;
    QueueSnapshot(dt.image, VK_IMAGE_ASPECT_DEPTH_BIT, dt.w, dt.h, kDepthFormat,
                  depth_base, true, draw_index, dt.layout, &dt.layout);
  }
}

void DrainSnapshots() {
  if (g_snapshots.empty())
    return;
  vkQueueWaitIdle(g_dev.queue);
  for (Snapshot& s : g_snapshots) {
    char name[256];
    std::snprintf(name, sizeof name, "%s_d%04u_%s_%#llx_%ux%u.png",
                  g_prefix.c_str(), s.at_draw, s.depth ? "depth" : "rt",
                  (unsigned long long)s.base, s.w, s.h);
    PixelStats stats;
    const auto* bytes = static_cast<const uint8_t*>(s.map);
    const bool ok = s.depth
                        ? WriteDepthPng(name, bytes, s.w, s.h, &stats)
                        : WriteImagePng(name, bytes, s.w, s.h, s.fmt, &stats);
    WriteRawSidecar(name, bytes, s.bytes);
    Line l("dump");
    l.U("seq", g_seq++)
        .Str("kind", s.depth ? "depth" : "rt")
        .Str("when", "mid-frame")
        .Int("after_draw", s.at_draw)
        .Hex("base", s.base)
        .U("w", s.w)
        .U("h", s.h)
        .Str("format", FormatName(s.depth ? kDepthFormat : s.fmt))
        .Str("file", ok ? name : "")
        .Raw("stats", StatsObj(stats));
    l.Emit();
    vkUnmapMemory(g_dev.device, s.memory);
    vkDestroyBuffer(g_dev.device, s.buffer, nullptr);
    vkFreeMemory(g_dev.device, s.memory, nullptr);
  }
  g_snapshots.clear();
}

// --- end-of-frame dumps ----------------------------------------------------

void DumpFrameResources() {
  if (DumpWanted("rt")) {
    for (auto& kv : g_rts) {
      RTarget& rt = kv.second;
      if (!rt.used_this_frame && !rt.ever_rendered)
        continue;
      std::vector<uint8_t> bytes;
      if (!ReadImage(rt.image, VK_IMAGE_ASPECT_COLOR_BIT, rt.w, rt.h,
                     FormatBytes(rt.fmt), rt.layout, &rt.layout, bytes))
        continue;
      // The frame's submission has already stamped submitted_layout; this
      // readback executes after it, so the anchor moves with it.
      rt.submitted_layout = rt.layout;
      char name[256];
      std::snprintf(name, sizeof name, "%s_end_rt_%#llx_%ux%u.png",
                    g_prefix.c_str(), (unsigned long long)kv.first, rt.w, rt.h);
      PixelStats stats;
      const bool ok =
          WriteImagePng(name, bytes.data(), rt.w, rt.h, rt.fmt, &stats);
      WriteRawSidecar(name, bytes.data(), bytes.size());
      Line l("dump");
      l.U("seq", g_seq++)
          .Str("kind", "rt")
          .Str("when", "frame-end")
          .Hex("base", kv.first)
          .U("w", rt.w)
          .U("h", rt.h)
          .Str("format", FormatName(rt.fmt))
          .U("draws", rt.draws)
          .Bool("used_this_frame", rt.used_this_frame)
          .Hex("last_vs", rt.last_vs)
          .Hex("last_ps", rt.last_ps)
          .Hex("last_cbuf_mask", rt.last_cbuf_mask)
          .Hex("last_rawbuf_mask", rt.last_rawbuf_mask)
          .Str("file", ok ? name : "")
          .Raw("stats", StatsObj(stats));
      l.Emit();
    }
  }
  if (DumpWanted("depth")) {
    for (auto& kv : g_depths) {
      DepthTarget& dt = kv.second;
      if (!dt.used_this_frame)
        continue;
      std::vector<uint8_t> bytes;
      if (!ReadImage(dt.image, VK_IMAGE_ASPECT_DEPTH_BIT, dt.w, dt.h, 4,
                     dt.layout, &dt.layout, bytes))
        continue;
      dt.submitted_layout = dt.layout;
      char name[256];
      std::snprintf(name, sizeof name, "%s_end_depth_%#llx_%ux%u.png",
                    g_prefix.c_str(), (unsigned long long)kv.first, dt.w, dt.h);
      PixelStats stats;
      const bool ok = WriteDepthPng(name, bytes.data(), dt.w, dt.h, &stats);
      WriteRawSidecar(name, bytes.data(), bytes.size());
      Line l("dump");
      l.U("seq", g_seq++)
          .Str("kind", "depth")
          .Str("when", "frame-end")
          .Hex("base", kv.first)
          .U("w", dt.w)
          .U("h", dt.h)
          .Str("format", FormatName(kDepthFormat))
          .Str("file", ok ? name : "")
          .Raw("stats", StatsObj(stats));
      l.Emit();
    }
  }
  if (DumpWanted("tex")) {
    for (const TexKey& t : g_frame_texs) {
      char name[256];
      std::snprintf(name, sizeof name, "%s_tex_%#llx_%ux%u_d%u_n%u.png",
                    g_prefix.c_str(), (unsigned long long)t.base, t.w, t.h,
                    t.dfmt, t.nfmt);
      const char* reason = "";
      const bool ok = DumpGuestTexture(t, name, &reason);
      Line l("dump");
      l.U("seq", g_seq++)
          .Str("kind", "tex")
          .Str("when", "frame-end")
          .Hex("base", t.base)
          .U("w", t.w)
          .U("h", t.h)
          .U("dfmt", t.dfmt)
          .U("nfmt", t.nfmt)
          .U("tiling", t.tiling)
          .Str("format", FormatName(GuestTextureFormat(t.dfmt, t.nfmt)))
          .Str("file", ok ? name : "")
          .Str("skipped", ok ? "" : reason);
      l.Emit();
    }
  }
}

// --- validation ------------------------------------------------------------

VKAPI_ATTR VkBool32 VKAPI_CALL
ValidationCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                   VkDebugUtilsMessageTypeFlagsEXT,
                   const VkDebugUtilsMessengerCallbackDataEXT* data,
                   void*) {
  const char* level =
      severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT     ? "error"
      : severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "warning"
      : severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT    ? "info"
                                                                   : "verbose";
  g_validation_messages++;
  // The active label stack is what names the guest draw: vk_debug opens
  // "frame N" / "region rt=..." / "recomp vs=... ps=..." around every command.
  std::string labels;
  for (uint32_t i = 0; i < data->cmdBufLabelCount; i++) {
    if (!labels.empty())
      labels += " > ";
    labels += data->pCmdBufLabels[i].pLabelName;
  }
  std::fprintf(stderr, "[vkval] %s f%d draw#%u [%s] %s: %s\n", level,
               g_frame.num, g_frame.draws, labels.c_str(),
               data->pMessageIdName ? data->pMessageIdName : "?",
               data->pMessage ? data->pMessage : "");
  if (g_recording) {
    Line l("validation");
    l.U("seq", g_seq++)
        .Str("severity", level)
        .Int("draw", int(g_draw_seq))
        .Int("frame_draw", int(g_frame.draws))
        .Str("id", data->pMessageIdName ? data->pMessageIdName : "")
        .Str("labels", labels.c_str())
        .Str("message", data->pMessage ? data->pMessage : "");
    l.Emit();
  }
  return VK_FALSE;
}

}  // namespace

// --- public ----------------------------------------------------------------

bool WantValidation() {
  static const bool want = kValidate.get();
  return want;
}

const char* ValidationLayerName() {
  return "VK_LAYER_KHRONOS_validation";
}

void InstallValidationMessenger(VkInstance instance) {
  if (!WantValidation() || !instance || g_messenger)
    return;
  auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  if (!create) {
    std::fprintf(stderr,
                 "[vkval] validation requested but the layer is not loaded "
                 "(is VK_LAYER_PATH set?)\n");
    return;
  }
  VkDebugUtilsMessengerCreateInfoEXT ci{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  ci.pfnUserCallback = ValidationCallback;
  create(instance, &ci, nullptr, &g_messenger);
  std::fprintf(stderr, "[vkval] validation layer active\n");
}

void DestroyValidationMessenger(VkInstance instance) {
  if (!g_messenger || !instance)
    return;
  auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  if (destroy)
    destroy(instance, g_messenger, nullptr);
  g_messenger = VK_NULL_HANDLE;
}

bool NamesWanted() {
  static const bool want = Armed() || WantValidation();
  return want;
}

void RegisterObjectName(VkObjectType, uint64_t handle, const char* name) {
  if (!handle || !name)
    return;
  NameTable()[handle] = name;
}

const char* ObjectName(uint64_t handle) {
  auto it = NameTable().find(handle);
  return it == NameTable().end() ? "" : it->second.c_str();
}

void FrameBegin(int frame_num) {
  if (!Armed() || g_finished)
    return;
  if (!g_start_ns)
    g_start_ns = NowNs();
  if (!g_frames_left) {
    // Called before the frame counters reset, so g_frame.draws still holds the
    // PREVIOUS frame's count -- which is what the busy trigger needs (a frame's
    // own draw count is not known until it ends).
    bool trigger = false;
    if (kCaptureFrame.get() > 0 && frame_num == kCaptureFrame.get())
      trigger = true;
    if (kCaptureAfter.get() > 0.f &&
        double(NowNs() - g_start_ns) / 1e9 >= double(kCaptureAfter.get()))
      trigger = true;
    if (kCaptureBusy.get() > 0 && g_frame.draws >= uint32_t(kCaptureBusy.get()))
      trigger = true;
    if (!trigger)
      return;
    g_frames_left = std::max(1, kCaptureCount.get());
    g_armed_frame = frame_num;
    const char* dir = kCaptureDir;
    g_dir = (dir && *dir) ? dir : (std::string(DumpDir()) + "/gpucap");
    ::mkdir(g_dir.c_str(), 0755);
  }

  // A frame that died before EndFrame (a device fault) leaves the file open;
  // do not leak it into the next one.
  if (g_file) {
    std::fclose(g_file);
    g_file = nullptr;
  }
  char prefix[256];
  std::snprintf(prefix, sizeof prefix, "%s/frame_%d", g_dir.c_str(), frame_num);
  g_prefix = prefix;
  const std::string path = g_prefix + ".jsonl";
  g_file = std::fopen(path.c_str(), "wb");
  if (!g_file) {
    std::fprintf(stderr, "[gpucap] cannot open %s\n", path.c_str());
    g_frames_left = 0;
    return;
  }
  g_recording = true;
  g_frame_num = frame_num;
  g_seq = 0;
  g_draw_seq = 0;
  g_frame_texs.clear();
  std::fprintf(stderr, "[gpucap] capturing frame %d -> %s\n", frame_num,
               path.c_str());
  Line l("capture");
  l.Int("version", 1)
      .Int("frame", frame_num)
      .Int("armed_frame", g_armed_frame)
      .Str("dir", g_dir.c_str())
      .Num("exposure", double(kCaptureExposure.get()))
      .Num("gamma", double(kCaptureGamma.get()))
      .Bool("validation", WantValidation())
      .Str("dump", kCaptureDump.get() ? kCaptureDump.get() : "");
  l.Emit();
  Line f("frame_begin");
  f.U("seq", g_seq++).Int("frame", frame_num);
  f.Emit();
}

void FrameEnd(uint64_t scanout_base) {
  if (!g_recording)
    return;
  DrainSnapshots();
  DumpFrameResources();
  Line l("frame_end");
  l.U("seq", g_seq++)
      .Int("frame", g_frame_num)
      .U("draws", g_draw_seq)
      .U("backend_draws", g_frame.draws)
      .U("heuristic_draws", g_frame.heuristic)
      .U("validation_messages", g_validation_messages)
      .Hex("scanout", scanout_base);
  l.Emit();
  std::fclose(g_file);
  g_file = nullptr;
  g_recording = false;
  if (--g_frames_left <= 0) {
    g_finished = true;
    std::fprintf(stderr, "[gpucap] capture complete: %s*.jsonl\n",
                 g_dir.c_str());
    if (kExitAfter) {
      std::fflush(nullptr);
      std::_Exit(0);
    }
  }
}

void RegionBegin(const RegionInfo& region) {
  if (!g_recording)
    return;
  Arr colors;
  for (uint32_t i = 0; i < region.mrt_count && i < 8; i++) {
    Obj o;
    o.U("i", i);
    o.Hex("base", region.mrt_base[i]);
    o.Hex("info", region.mrt_info[i]);
    o.Str("format", FormatName(ColorTargetFormat(region.mrt_info[i])));
    o.Str("load_op", (region.color_clear_mask & (1u << i)) ? "CLEAR" : "LOAD");
    auto it = g_rts.find(region.mrt_base[i]);
    if (it != g_rts.end()) {
      o.U("image_w", it->second.w);
      o.U("image_h", it->second.h);
      o.Str("layout", LayoutName(it->second.layout));
      o.U("draws_so_far", it->second.draws);
    }
    colors.Add(o);
  }
  Line l("region_begin");
  l.U("seq", g_seq++)
      .Int("after_draw", int(g_draw_seq))
      .U("w", region.width)
      .U("h", region.height)
      .Raw("color", colors.Done())
      .Hex("depth", region.depth_base)
      .Hex("stencil", region.stencil_base)
      .Str("depth_load_op", region.depth_clear ? "CLEAR" : "LOAD")
      .Num("depth_clear_value", region.depth_clear_value);
  l.Emit();
}

void RegionEnd() {
  if (!g_recording)
    return;
  Line l("region_end");
  l.U("seq", g_seq++).Int("after_draw", int(g_draw_seq));
  l.Emit();
}

void RecordDraw(const rhi::DrawInfo& d,
                const char* path,
                const DrawBindings* b) {
  if (!g_recording)
    return;
  const uint32_t index = g_draw_seq++;

  Arr rts;
  for (uint32_t i = 0; i < d.mrt_count && i < 8; i++) {
    Obj o;
    o.U("i", i);
    o.Hex("base", d.mrt_base[i]);
    o.Hex("info", d.mrt_info[i]);
    o.Str("format", FormatName(ColorTargetFormat(d.mrt_info[i])));
    o.Hex("blend_control", d.mrt_blend[i]);
    o.Bool("blend_enable", (d.mrt_blend_mask >> i) & 1);
    Arr clear;
    clear.Add(Line::HexText(d.mrt_clear_word[i][0]));
    clear.Add(Line::HexText(d.mrt_clear_word[i][1]));
    o.Raw("clear_word", clear.Done());
    rts.Add(o);
  }

  Arr vbufs;
  for (uint32_t i = 0; i < d.num_vbufs && i < 8; i++) {
    Obj o;
    o.U("i", i);
    o.Hex("base", reinterpret_cast<uint64_t>(d.vbufs[i].data));
    o.U("stride", d.vbufs[i].stride);
    o.U("records", d.vbufs[i].num_records);
    o.Raw("guest",
          GuestObj(reinterpret_cast<uint64_t>(d.vbufs[i].data),
                   uint64_t(d.vbufs[i].stride) * d.vbufs[i].num_records));
    vbufs.Add(o);
  }
  Arr vattrs;
  for (uint32_t i = 0; i < d.num_vattrs && i < 8; i++) {
    Obj o;
    o.U("location", d.vattrs[i].location);
    o.U("binding", d.vattrs[i].binding);
    o.U("offset", d.vattrs[i].offset);
    o.U("num_comps", d.vattrs[i].num_comps);
    o.U("dfmt", d.vattrs[i].dfmt);
    o.U("nfmt", d.vattrs[i].nfmt);
    o.Str("format",
          FormatName(VertexFormat(d.vattrs[i].dfmt, d.vattrs[i].nfmt)));
    vattrs.Add(o);
  }

  Arr texs;
  const uint32_t ntex = std::min<uint32_t>(d.num_texs, 24);
  for (uint32_t i = 0; i < ntex; i++) {
    texs.Add(TexObj(i, d.texs[i], b));
    NoteTexture(d.texs[i]);
  }

  Arr cbufs;
  const uint32_t cbuf_cap =
      kCaptureCbufBytes.get() < 0 ? 0 : uint32_t(kCaptureCbufBytes.get());
  for (uint32_t i = 0; i < d.num_cbufs && i < 16; i++) {
    if (!d.cbufs[i].base && !d.cbufs[i].size)
      continue;
    Obj o;
    o.U("i", i);
    o.Hex("base", d.cbufs[i].base);
    o.U("size", d.cbufs[i].size);
    o.Bool("staged", b ? ((b->cbuf_mask >> i) & 1) != 0 : false);
    o.Raw("guest", GuestObj(d.cbufs[i].base, d.cbufs[i].size));
    const uint32_t bytes =
        cbuf_cap ? std::min(cbuf_cap, d.cbufs[i].size) : d.cbufs[i].size;
    o.Raw("data", HexBytes(d.cbufs[i].base, bytes));
    cbufs.Add(o);
  }
  Arr bufs;
  for (uint32_t i = 0; i < d.num_bufs && i < rhi::DrawInfo::kMaxBuffers; i++) {
    if (!d.bufs[i].base && !d.bufs[i].size)
      continue;
    Obj o;
    o.U("i", i);
    o.Hex("base", d.bufs[i].base);
    o.U("size", d.bufs[i].size);
    o.Bool("staged", b ? ((b->rawbuf_mask >> i) & 1) != 0 : false);
    o.Raw("guest", GuestObj(d.bufs[i].base, d.bufs[i].size));
    bufs.Add(o);
  }

  Obj depth;
  depth.Hex("base", d.depth_base);
  depth.Bool("valid", d.depth_valid);
  depth.Bool("test", d.depth_test_enable);
  depth.Bool("write", d.depth_write_enable);
  depth.U("func", d.depth_func);
  depth.Num("clear", d.depth_clear);
  depth.Hex("control", d.depth_control);
  Obj stencil;
  stencil.Hex("base", d.stencil_base);
  stencil.Bool("enable", d.stencil_enable);
  stencil.Bool("backface", d.stencil_backface_enable);
  stencil.Hex("control", d.stencil_control);
  stencil.Hex("refmask", d.stencil_refmask);
  stencil.Hex("refmask_bf", d.stencil_refmask_bf);
  stencil.U("clear", d.stencil_clear);

  Obj viewport;
  viewport.Num("x_scale", d.viewport_x_scale);
  viewport.Num("x_offset", d.viewport_x_offset);
  viewport.Num("y_scale", d.viewport_y_scale);
  viewport.Num("y_offset", d.viewport_y_offset);
  // The rectangle the backend actually sets (a negative height is the y-up
  // guest raster, not a bug).
  viewport.Num("vk_x", d.viewport_x_offset - d.viewport_x_scale);
  viewport.Num("vk_y", d.viewport_y_offset - d.viewport_y_scale);
  viewport.Num("vk_w", d.viewport_x_scale * 2.0f);
  viewport.Num("vk_h", d.viewport_y_scale * 2.0f);

  Arr vs_ud, ps_ud;
  for (uint32_t i = 0; i < 16; i++) {
    vs_ud.Add(Line::HexText(d.vs_user_data[i]));
    ps_ud.Add(Line::HexText(d.ps_user_data[i]));
  }

  Line l("draw");
  l.U("seq", g_seq++)
      .Int("frame", g_frame_num)
      .U("draw", index)
      .U("frame_draw", g_frame.draws)
      .Str("path", path)
      .Hex("rt", d.rt_base)
      .U("rt_w", d.rt_w)
      .U("rt_h", d.rt_h)
      .Raw("color_targets", rts.Done())
      .Hex("target_mask", d.target_mask)
      .Hex("shader_mask", d.shader_mask)
      .Hex("color_control", d.color_control)
      .Bool("blend_enable", d.blend_enable)
      .Hex("blend_control", d.blend_control)
      .Raw("depth", depth.Done())
      .Raw("stencil", stencil.Done())
      .Raw("viewport", viewport.Done())
      .U("cull_mode", d.cull_mode)
      .Bool("front_ccw", d.front_ccw)
      .U("prim_type", d.prim_type)
      .Bool("clear_rect", d.is_clear_rect)
      .Bool("indexed", d.index_data != nullptr)
      .Hex("index_base", reinterpret_cast<uint64_t>(d.index_data))
      .U("index_count", d.index_count)
      .U("index_type", d.index_type)
      .U("vertex_count", d.vertex_count)
      .U("instance_count", d.instance_count)
      .Raw("vbufs", vbufs.Done())
      .Raw("vattrs", vattrs.Done())
      .Raw("textures", texs.Done())
      .Raw("cbufs", cbufs.Done())
      .Raw("bufs", bufs.Done())
      .Raw("vs", ShaderObj(d.vs_addr, d.recomp ? &d.recomp->vs_spirv : nullptr))
      .Raw("ps", ShaderObj(d.ps_addr, d.recomp ? &d.recomp->fs_spirv : nullptr))
      .Bool("neo", d.ps4_neo)
      .Raw("vs_user_data", vs_ud.Done())
      .Raw("ps_user_data", ps_ud.Done());
  l.Emit();

  if (SnapshotWanted(index))
    SnapshotOpenRegion(index);
}

void RecordDecline(const char* reason) {
  if (!g_recording)
    return;
  Line l("decline");
  l.U("seq", g_seq++).Int("after_draw", int(g_draw_seq)).Str("reason", reason);
  l.Emit();
}

void RecordDispatch(const rhi::ComputeInfo& ci) {
  if (!g_recording)
    return;
  Arr res;
  for (uint32_t i = 0; i < ci.num_res && i < rhi::ComputeInfo::kMaxResources;
       i++) {
    const auto& r = ci.res[i];
    Obj o;
    o.U("binding", r.binding);
    o.Hex("base", r.base);
    o.U("size", r.size);
    o.U("guest_size", r.guest_size);
    o.Bool("shader_writes", r.shader_writes);
    o.Bool("written", r.written);
    o.Bool("zero_fill", r.zero_fill);
    o.Bool("image_staging", r.image_staging);
    o.U("width", r.width);
    o.U("height", r.height);
    o.U("pitch", r.pitch);
    o.U("layers", r.layers);
    o.U("mip_levels", r.mip_levels);
    o.U("tiling_idx", r.tiling_idx);
    o.U("elem_bytes", r.elem_bytes);
    o.U("dfmt", r.dfmt);
    o.Raw("guest", GuestObj(r.base, r.guest_size ? r.guest_size : r.size));
    res.Add(o);
  }
  Arr ud;
  for (uint32_t i = 0; i < 16; i++)
    ud.Add(Line::HexText(ci.user_data[i]));
  Arr groups;
  for (uint32_t i = 0; i < 3; i++)
    groups.Add(Line::IntText(ci.groups[i]));
  Line l("dispatch");
  l.U("seq", g_seq++)
      .Int("frame", g_frame_num)
      .Int("after_draw", int(g_draw_seq))
      .Raw("cs", ShaderObj(ci.cs_addr, ci.recomp ? &ci.recomp->spirv : nullptr))
      .Raw("groups", groups.Done())
      .Raw("resources", res.Done())
      .Raw("user_data", ud.Done());
  l.Emit();
}

void RecordBarrier(const char* aspect,
                   VkImage image,
                   VkImageLayout from,
                   VkImageLayout to,
                   VkAccessFlags src_access,
                   VkAccessFlags dst_access) {
  if (!g_recording)
    return;
  // Name the image after the guest resource it holds. The RT and depth caches
  // are the authority; anything else falls back to the debug-utils name.
  std::string name = ObjectName(reinterpret_cast<uint64_t>(image));
  if (name.empty()) {
    for (const auto& kv : g_rts)
      if (kv.second.image == image) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "rt %#llx",
                      (unsigned long long)kv.first);
        name = buf;
        break;
      }
  }
  Line l("barrier");
  l.U("seq", g_seq++)
      .Int("after_draw", int(g_draw_seq))
      .Str("aspect", aspect)
      .Hex("image", reinterpret_cast<uint64_t>(image))
      .Str("resource", name.c_str())
      .Str("from", LayoutName(from))
      .Str("to", LayoutName(to))
      .Hex("src_access", src_access)
      .Hex("dst_access", dst_access);
  l.Emit();
}

void RecordMemoryFill(uint64_t base, uint64_t bytes, uint32_t value) {
  if (!g_recording)
    return;
  Line l("memory_fill");
  l.U("seq", g_seq++)
      .Int("after_draw", int(g_draw_seq))
      .Hex("base", base)
      .U("bytes", bytes)
      .Hex("value", value);
  l.Emit();
}

}  // namespace gpu::vk::trace
