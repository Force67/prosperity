/*
 * PS4Delta : PS4 emulation and research project
 *
 * Headless Vulkan renderer. See vk_render.h. Renders the decoded PM4 draws as
 * MVP-transformed quads into an offscreen render target that mirrors the guest
 * scanout, then reads it back (presented to a window when a display exists).
 */

#include "vk_render.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gfx/gfx.h"
#include "gcn/gcn_translate.h"
#include "gcn/gcn_detile.h"
#include "shaders/quad_vert_spv.h"
#include "shaders/quad_frag_spv.h"
#include "shaders/tex_vert_spv.h"
#include "shaders/tex_frag_spv.h"

namespace gpu::vk {
namespace {

#define VKOK(x)                                                                \
  do {                                                                         \
    VkResult _r = (x);                                                         \
    if (_r != VK_SUCCESS) {                                                    \
      std::fprintf(stderr, "[gpuvk] %s failed: %d\n", #x, (int)_r);            \
      return false;                                                            \
    }                                                                          \
  } while (0)

constexpr VkDeviceSize kVbRing = 16ull * 1024 * 1024;  // per-frame vertex ring
constexpr VkDeviceSize kIbRing = 8ull * 1024 * 1024;   // per-frame index ring (32-bit)
constexpr VkDeviceSize kUboRing = 64ull * 1024 * 1024; // per-frame recomp cbuffer ring
constexpr uint32_t kCbufWindow = 1024;                 // bytes per draw (uvec4 data[64])

struct State {
  bool ready = false;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t qfam = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  VkFormat rtFormat = VK_FORMAT_B8G8R8A8_UNORM;

  // Shared readback buffer (sized for the largest RT presented).
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readbackMem = VK_NULL_HANDLE;
  void *readbackMap = nullptr;
  VkDeviceSize readbackSize = 0;

  uint64_t curRt = 0;   // primary RT (MRT0) of the open region (0 = none)
  uint64_t curMrt[8] = {0};   // all color targets bound in the open region
  uint32_t curMrtCount = 0;   // how many of curMrt[] are bound
  uint64_t curDepth = 0;      // depth target bound in the open region (0 = none)
  uint64_t lastRt = 0;  // last RT rendered to (present fallback)
  uint64_t busiestRt = 0;
  uint32_t busiestRtDraws = 0;

  // Pipeline (textured/colored quad).
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  // Per-frame vertex ring (host-visible). Holds interleaved pos+uv (vec4) per vert.
  VkBuffer vb = VK_NULL_HANDLE;
  VkDeviceMemory vbMem = VK_NULL_HANDLE;
  uint8_t *vbMap = nullptr;
  VkDeviceSize vbOffset = 0;

  // Per-frame index ring (host-visible). 32-bit indices (16-bit guest indices are
  // widened on upload) for indexed triangle-list draws.
  VkBuffer ib = VK_NULL_HANDLE;
  VkDeviceMemory ibMem = VK_NULL_HANDLE;
  uint8_t *ibMap = nullptr;
  VkDeviceSize ibOffset = 0;

  // Textured pipeline + texture cache.
  VkPipeline texPipeline = VK_NULL_HANDLE;
  VkPipelineLayout texLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
  VkDescriptorPool dsPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> dsPools;
  VkSampler sampler = VK_NULL_HANDLE;

  // Multi-texture (recomp PS sampling >1 texture, e.g. Doom64 3D): an 8-binding
  // set-0 layout + a pool for the N-sampler sets, plus a 1x1 white default for any
  // binding the PS samples that we couldn't resolve (so diffuse*lightmap with a
  // missing map shows the diffuse instead of going black). The single-texture path
  // (Isaac/Undertale/composites) is unchanged.
  static constexpr uint32_t kMaxTex = 8;
  VkDescriptorSetLayout texArrayLayout = VK_NULL_HANDLE;
  VkDescriptorPool mtexPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> mtexPools;
  VkImage whiteImg = VK_NULL_HANDLE;
  VkDeviceMemory whiteMem = VK_NULL_HANDLE;
  VkImageView whiteView = VK_NULL_HANDLE;
  VkImageView whiteArrayView = VK_NULL_HANDLE;
  VkDescriptorSet whiteSet = VK_NULL_HANDLE;
  VkDescriptorSet whiteArraySet = VK_NULL_HANDLE;

  // Recomp cbuffer ring: per-draw VS/PS constant buffers live at set 1 bindings
  // 0..7. Textures stay at set 0. Each binding uses a dynamic offset into this
  // host-visible ring; emptyLayout fills set 0 for untextured recomp draws.
  VkDescriptorSetLayout uboLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
  VkDescriptorPool uboPool = VK_NULL_HANDLE;
  VkDescriptorSet uboSet = VK_NULL_HANDLE;
  VkBuffer uboBuf = VK_NULL_HANDLE;
  VkDeviceMemory uboMem = VK_NULL_HANDLE;
  uint8_t *uboMap = nullptr;
  VkDeviceSize uboOffset = 0;
  uint32_t uboAlign = 256;

  // Pipelines keyed by blend state (textured<<0, enable<<1, blendControl<<2) so
  // each draw uses the guest's CB_BLEND0_CONTROL blend instead of one hardcoded mode.
  std::unordered_map<uint64_t, VkPipeline> pipeCache;

  uint32_t frameDraws = 0;
  uint32_t frameHeuristic = 0;  // draws this frame that fell back to the heuristic quad path
  uint32_t frameMaxIdx = 0;  // largest indexCount of any draw this frame (3D geometry detector)
  int frameNum = 0;
  bool recording = false;
  bool samplerAnisotropy = false;
  bool samplerMirrorClamp = false;
  bool frameHadRoom = false;  // this frame sampled a room-sized (~832w) RT
  bool frameRoomBake = false; // this frame RENDERED into a room-sized (~832w) RT
} g;

struct TexImageKey {
  uint64_t base = 0;
  uint32_t w = 0, h = 0, tiling = 8, pitch = 0, layers = 1;
  uint32_t mipLevels = 1;
  bool pow2Pad = false;
  bool operator==(const TexImageKey &o) const {
    return base == o.base && w == o.w && h == o.h && tiling == o.tiling &&
           pitch == o.pitch && layers == o.layers && mipLevels == o.mipLevels &&
           pow2Pad == o.pow2Pad;
  }
};
uint64_t hashWord(uint64_t h, uint64_t v) {
  return (h ^ v) * 1099511628211ull;
}
struct TexImageKeyHash {
  size_t operator()(const TexImageKey &k) const {
    uint64_t h = 1469598103934665603ull;
    h = hashWord(h, k.base); h = hashWord(h, k.w); h = hashWord(h, k.h);
    h = hashWord(h, k.tiling); h = hashWord(h, k.pitch); h = hashWord(h, k.layers);
    h = hashWord(h, k.mipLevels); h = hashWord(h, k.pow2Pad);
    return static_cast<size_t>(h);
  }
};

struct SamplerKey {
  uint32_t raw[4] = {};
  uint32_t imageMinLod = 0;
  bool valid = false;
  bool operator==(const SamplerKey &o) const {
    return valid == o.valid && imageMinLod == o.imageMinLod &&
           std::memcmp(raw, o.raw, sizeof(raw)) == 0;
  }
};
struct SamplerKeyHash {
  size_t operator()(const SamplerKey &k) const {
    uint64_t h = hashWord(1469598103934665603ull, k.valid);
    for (uint32_t word : k.raw) h = hashWord(h, word);
    h = hashWord(h, k.imageMinLod);
    return static_cast<size_t>(h);
  }
};

struct TexKey {
  TexImageKey image;
  uint32_t baseArray = 0, viewLayers = 1;
  uint32_t baseMip = 0, viewMips = 1;
  SamplerKey sampler;
  bool arrayed = false;
  bool operator==(const TexKey &o) const {
    return image == o.image && baseArray == o.baseArray &&
           viewLayers == o.viewLayers && baseMip == o.baseMip &&
           viewMips == o.viewMips && sampler == o.sampler &&
           arrayed == o.arrayed;
  }
};
struct TexKeyHash {
  size_t operator()(const TexKey &k) const {
    uint64_t h = TexImageKeyHash{}(k.image);
    h = hashWord(h, k.baseArray); h = hashWord(h, k.viewLayers);
    h = hashWord(h, k.baseMip); h = hashWord(h, k.viewMips);
    h = hashWord(h, SamplerKeyHash{}(k.sampler));
    return static_cast<size_t>(hashWord(h, k.arrayed));
  }
};

struct TexViewKey {
  TexImageKey image;
  uint32_t baseArray = 0, viewLayers = 1;
  uint32_t baseMip = 0, viewMips = 1;
  bool arrayed = false;
  bool operator==(const TexViewKey &o) const {
    return image == o.image && baseArray == o.baseArray &&
           viewLayers == o.viewLayers && baseMip == o.baseMip &&
           viewMips == o.viewMips && arrayed == o.arrayed;
  }
};
struct TexViewKeyHash {
  size_t operator()(const TexViewKey &k) const {
    uint64_t h = TexImageKeyHash{}(k.image);
    h = hashWord(h, k.baseArray); h = hashWord(h, k.viewLayers);
    h = hashWord(h, k.baseMip); h = hashWord(h, k.viewMips);
    return static_cast<size_t>(hashWord(h, k.arrayed));
  }
};

struct TexImageEntry {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  uint64_t footprint = 0;
  uint64_t hash = 0;
  int lastCheckedFrame = -1;
};
struct TexViewEntry { VkImageView view = VK_NULL_HANDLE; };
struct TexEntry {
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
};
std::unordered_map<TexImageKey, TexImageEntry, TexImageKeyHash> g_texImages;
std::unordered_map<TexViewKey, TexViewEntry, TexViewKeyHash> g_texViews;
std::unordered_map<TexKey, TexEntry, TexKeyHash> g_texCache;
std::unordered_map<SamplerKey, VkSampler, SamplerKeyHash> g_samplerCache;
std::vector<TexImageEntry> g_retiredTexImages;
std::vector<TexViewEntry> g_retiredTexViews;
std::vector<TexEntry> g_retiredTexSets;

// Content fingerprint of every guest dword. This is checked at most once per
// frame unless a compute write explicitly invalidates the resource.
uint64_t texHash(uint64_t base, uint64_t bytes) {
  const uint32_t *p = reinterpret_cast<const uint32_t *>(base);
  uint64_t count = bytes / 4;
  uint64_t hsh = 1469598103934665603ull;
  for (uint64_t i = 0; i < count; i++)
    hsh = (hsh ^ p[i]) * 1099511628211ull;
  return hsh ^ (count << 1);
}

TexKey textureKey(uint64_t base, uint32_t w, uint32_t h, uint32_t tiling,
                  uint32_t pitch, uint32_t layers, uint32_t baseArray,
                   uint32_t viewLayers, uint32_t mipLevels, uint32_t baseMip,
                   uint32_t viewMips, uint32_t minLod, bool pow2Pad,
                   const uint32_t *sampler, bool samplerValid, bool arrayed) {
  if (layers && baseArray < layers)
    viewLayers = std::min(viewLayers, layers - baseArray);
  if (!arrayed) viewLayers = 1;
  if (mipLevels && baseMip < mipLevels)
    viewMips = std::min(viewMips, mipLevels - baseMip);
  TexKey key;
  key.image = {base, w, h, tiling, pitch, layers, mipLevels, pow2Pad};
  key.baseArray = baseArray; key.viewLayers = viewLayers;
  key.baseMip = baseMip; key.viewMips = viewMips;
  key.sampler.valid = samplerValid && sampler;
  if (key.sampler.valid) std::memcpy(key.sampler.raw, sampler, sizeof(key.sampler.raw));
  key.sampler.imageMinLod = minLod;
  key.arrayed = arrayed;
  return key;
}

TexViewKey textureViewKey(const TexKey &key) {
  return {key.image, key.baseArray, key.viewLayers, key.baseMip, key.viewMips,
          key.arrayed};
}

uint32_t textureTiling(uint32_t tiling) {
  static const int forced = [] {
    const char *e = std::getenv("DELTA_GPU_FORCETILE");
    return e ? std::atoi(e) : -1;
  }();
  return forced >= 0 ? static_cast<uint32_t>(forced) : tiling;
}

// An image in the resource cache: a render target keyed by its guest base address,
// that also doubles as a sampleable texture (render-to-texture). This is the unit
// of the resource model -- RT-bind and texture-sample both resolve to
// the same Image via the address page table, so render-to-texture/MRT "just work".
struct RTarget {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;  // for sampling this RT as a texture
  uint32_t w = 0, h = 0;
  VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;  // identity: addr alone doesn't pin format
  bool isDepth = false;                      // depth/stencil attachment (MRT/depth task)
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool usedThisFrame = false;
  uint32_t draws = 0;     // draws into this RT this frame
  int lastFrame = -1000;  // frame number this RT was last rendered into
  bool clearPending = false;  // a fullscreen black clear was requested; applied lazily
                              // (as loadOp=CLEAR) only when later content redraws this RT
                              // in the same frame.
  VkClearColorValue clearValue{{0.0f, 0.0f, 0.0f, 0.0f}};
  bool everRendered = false;  // false until first real render (then loadOp can LOAD)
};
std::unordered_map<uint64_t, RTarget> g_rts;

// Depth/stencil attachment, keyed by its guest DB_Z_WRITE_BASE. Allocated on demand
// when a 3D draw binds a Z buffer. Internal format is always D32_SFLOAT (we never
// read depth back to guest memory, so only a valid depth format is needed).
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
struct DepthTarget {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  uint32_t w = 0, h = 0;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  int lastFrame = -1000;
  bool usedThisFrame = false;
  bool clearPending = false;
  float clearValue = 1.0f;
};
std::unordered_map<uint64_t, DepthTarget> g_depths;

// Address -> image page table (the resource model's core). Maps a 64 KiB guest page
// to the RT bases whose memory footprint covers it, so a sampled address resolves to
// every overlapping live image in O(pages) instead of scanning the whole cache. A
// page can be touched by several overlapping/aliased RTs (double-buffer pairs, a
// pool of cycled scene buffers), so each page holds a list.
constexpr uint32_t kRtPageShift = 16;  // 64 KiB
std::unordered_map<uint64_t, std::vector<uint64_t>> g_rtPages;

uint64_t rtByteSizeWH(uint32_t w, uint32_t h) { return (uint64_t)w * h * 4; }

// Register an RT's footprint pages so the page table can find it by overlap.
void registerRtPages(uint64_t base, uint32_t w, uint32_t h) {
  uint64_t lo = base >> kRtPageShift, hi = (base + rtByteSizeWH(w, h) - 1) >> kRtPageShift;
  for (uint64_t p = lo; p <= hi; p++) {
    auto &v = g_rtPages[p];
    bool seen = false;
    for (uint64_t b : v) if (b == base) { seen = true; break; }
    if (!seen) v.push_back(base);
  }
}

const bool g_dump = std::getenv("DELTA_GPU_DUMP") != nullptr;
int g_dumpedFrames = 0;

// Perf profiling accumulators (ns), reset each FPS window. Reveals where the
// per-frame wall time goes: our GPU code (draw + endFrame, incl. the readback
// stall and synchronous texture uploads) vs the guest/FEX time outside it.
uint64_t g_nsDraw = 0, g_nsEnd = 0, g_nsReadback = 0, g_nsTexUp = 0;
uint32_t g_texUps = 0;
inline uint64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct ScopeNs {
  uint64_t t0; uint64_t *acc;
  explicit ScopeNs(uint64_t *a) : t0(nowNs()), acc(a) {}
  ~ScopeNs() { *acc += nowNs() - t0; }
};

// Directory frame dumps go to. Defaults to /tmp; Android has no /tmp, so the
// runner sets DELTA_GPU_DUMP_DIR to a writable path (e.g. the cwd under
// /data/local/tmp). Returned without a trailing slash.
const char *dumpDir() {
  const char *d = std::getenv("DELTA_GPU_DUMP_DIR");
  return (d && *d) ? d : "/tmp";
}

PFN_vkCmdBeginRenderingKHR p_vkCmdBeginRendering = nullptr;
PFN_vkCmdEndRenderingKHR p_vkCmdEndRendering = nullptr;

uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((typeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return 0;
}

// Pick a memory type matching `pref` if any exists, else fall back to `req`. Used
// for the readback buffer: the CPU READS it every frame (the scanout flip), so it
// must be HOST_CACHED -- reading from the default HOST_COHERENT (write-combined,
// uncached) staging memory byte-by-byte is ~30x slower and was dominating frame
// time. CACHED+COHERENT (present on desktop GPUs) needs no manual invalidate.
uint32_t findMemoryTypePref(uint32_t typeBits, VkMemoryPropertyFlags pref,
                            VkMemoryPropertyFlags req) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((typeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & pref) == pref)
      return i;
  return findMemoryType(typeBits, req);
}

void imageBarrier(VkCommandBuffer c, VkImage img, VkImageLayout from,
                   VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA,
                   uint32_t layers = 1, uint32_t mipLevels = 1) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, layers};
  b.srcAccessMask = srcA;
  b.dstAccessMask = dstA;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &b);
}

bool createDevice() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  app.pApplicationName = "prosperity-gpu";
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ic.pApplicationInfo = &app;
  VKOK(vkCreateInstance(&ic, nullptr, &g.instance));

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
  if (!n) { std::fprintf(stderr, "[gpuvk] no device\n"); return false; }
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(g.instance, &n, devs.data());

  // Prefer a real GPU over the llvmpipe software rasteriser (reported as type
  // CPU): discrete > integrated > virtual > CPU. The loader can enumerate both
  // a discrete GPU and llvmpipe on the same box, so picking devs[0] blindly may
  // land on software. DELTA_VK_GPU=<name-substring> forces a specific device.
  const char *want = std::getenv("DELTA_VK_GPU");
  int best = -1;
  g.phys = VK_NULL_HANDLE;
  for (VkPhysicalDevice d : devs) {
    uint32_t dqn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &dqn, nullptr);
    std::vector<VkQueueFamilyProperties> dq(dqn);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &dqn, dq.data());
    bool gfx = false;
    for (auto &q : dq)
      if (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) { gfx = true; break; }
    if (!gfx) continue;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    int score;
    switch (p.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 4; break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 3; break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score = 2; break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:            score = 0; break;  // llvmpipe
      default:                                     score = 1; break;
    }
    if (want && std::strstr(p.deviceName, want)) score = 100;
    if (score > best) { best = score; g.phys = d; }
  }
  if (g.phys == VK_NULL_HANDLE) { std::fprintf(stderr, "[gpuvk] no gfx device\n"); return false; }

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, qprops.data());
  bool found = false;
  for (uint32_t i = 0; i < qn; i++)
    if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { g.qfam = i; found = true; break; }
  if (!found) { std::fprintf(stderr, "[gpuvk] no gfx queue\n"); return false; }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qc.queueFamilyIndex = g.qfam;
  qc.queueCount = 1;
  qc.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan12Features avail12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 avail2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  avail2.pNext = &avail12;
  vkGetPhysicalDeviceFeatures2(g.phys, &avail2);
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.samplerMirrorClampToEdge = avail12.samplerMirrorClampToEdge;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.pNext = &f12;
  f13.dynamicRendering = VK_TRUE;
  VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dc.pNext = &f13;
  dc.queueCreateInfoCount = 1;
  dc.pQueueCreateInfos = &qc;
  // robustBufferAccess makes out-of-bounds storage-buffer loads/stores safe (return 0
  // / drop the write) so the compute path can't corrupt memory on a miscomputed index.
  VkPhysicalDeviceFeatures wantFeat{};
  if (avail2.features.robustBufferAccess) wantFeat.robustBufferAccess = VK_TRUE;
  if (avail2.features.samplerAnisotropy) wantFeat.samplerAnisotropy = VK_TRUE;
  g.samplerAnisotropy = wantFeat.samplerAnisotropy;
  g.samplerMirrorClamp = f12.samplerMirrorClampToEdge;
  dc.pEnabledFeatures = &wantFeat;
  VKOK(vkCreateDevice(g.phys, &dc, nullptr, &g.device));
  vkGetDeviceQueue(g.device, g.qfam, 0, &g.queue);

  p_vkCmdBeginRendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdBeginRendering");
  p_vkCmdEndRendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdEndRendering");
  if (!p_vkCmdBeginRendering) {
    p_vkCmdBeginRendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdBeginRenderingKHR");
    p_vkCmdEndRendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdEndRenderingKHR");
  }
  if (!p_vkCmdBeginRendering) { std::fprintf(stderr, "[gpuvk] no dynamic rendering\n"); return false; }

  VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pc.queueFamilyIndex = g.qfam;
  VKOK(vkCreateCommandPool(g.device, &pc, nullptr, &g.pool));
  VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VKOK(vkAllocateCommandBuffers(g.device, &ca, &g.cmd));
  VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VKOK(vkCreateFence(g.device, &fc, nullptr, &g.fence));

  // Vertex ring.
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = kVbRing;
  bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VKOK(vkCreateBuffer(g.device, &bi, nullptr, &g.vb));
  VkMemoryRequirements vr;
  vkGetBufferMemoryRequirements(g.device, g.vb, &vr);
  VkMemoryAllocateInfo va{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  va.allocationSize = vr.size;
  va.memoryTypeIndex = findMemoryType(vr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g.device, &va, nullptr, &g.vbMem));
  VKOK(vkBindBufferMemory(g.device, g.vb, g.vbMem, 0));
  VKOK(vkMapMemory(g.device, g.vbMem, 0, kVbRing, 0, (void **)&g.vbMap));

  // Index ring (host-visible, 32-bit indices).
  VkBufferCreateInfo ibi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ibi.size = kIbRing;
  ibi.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  VKOK(vkCreateBuffer(g.device, &ibi, nullptr, &g.ib));
  VkMemoryRequirements ir;
  vkGetBufferMemoryRequirements(g.device, g.ib, &ir);
  VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ia.allocationSize = ir.size;
  ia.memoryTypeIndex = findMemoryType(ir.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g.device, &ia, nullptr, &g.ibMem));
  VKOK(vkBindBufferMemory(g.device, g.ib, g.ibMem, 0));
  VKOK(vkMapMemory(g.device, g.ibMem, 0, kIbRing, 0, (void **)&g.ibMap));

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g.phys, &props);
  std::fprintf(stderr, "[gpuvk] device: %s\n", props.deviceName);

  // Recomp cbuffer ring + dynamic-UBO descriptors (set 1) + empty set-0 layout.
  g.uboAlign = (uint32_t)props.limits.minUniformBufferOffsetAlignment;
  if (g.uboAlign < 1) g.uboAlign = 1;
  {
    VkBufferCreateInfo ub{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ub.size = kUboRing;
    ub.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VKOK(vkCreateBuffer(g.device, &ub, nullptr, &g.uboBuf));
    VkMemoryRequirements ur;
    vkGetBufferMemoryRequirements(g.device, g.uboBuf, &ur);
    VkMemoryAllocateInfo um{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    um.allocationSize = ur.size;
    um.memoryTypeIndex = findMemoryType(ur.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKOK(vkAllocateMemory(g.device, &um, nullptr, &g.uboMem));
    VKOK(vkBindBufferMemory(g.device, g.uboBuf, g.uboMem, 0));
    VKOK(vkMapMemory(g.device, g.uboMem, 0, kUboRing, 0, (void **)&g.uboMap));

    VkDescriptorSetLayoutBinding ubs[8]{};
    for (uint32_t i = 0; i < 8; i++) {
      ubs[i].binding = i;
      ubs[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      ubs[i].descriptorCount = 1;
      ubs[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ul{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ul.bindingCount = 8; ul.pBindings = ubs;
    VKOK(vkCreateDescriptorSetLayout(g.device, &ul, nullptr, &g.uboLayout));
    VkDescriptorSetLayoutCreateInfo el{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    el.bindingCount = 0;
    VKOK(vkCreateDescriptorSetLayout(g.device, &el, nullptr, &g.emptyLayout));

    VkDescriptorPoolSize ups{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8};
    VkDescriptorPoolCreateInfo upi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    upi.maxSets = 1; upi.poolSizeCount = 1; upi.pPoolSizes = &ups;
    VKOK(vkCreateDescriptorPool(g.device, &upi, nullptr, &g.uboPool));
    VkDescriptorSetAllocateInfo uai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    uai.descriptorPool = g.uboPool; uai.descriptorSetCount = 1; uai.pSetLayouts = &g.uboLayout;
    VKOK(vkAllocateDescriptorSets(g.device, &uai, &g.uboSet));
    VkDescriptorBufferInfo ubinfo[8];
    VkWriteDescriptorSet uw[8];
    for (uint32_t i = 0; i < 8; i++) {
      ubinfo[i] = {g.uboBuf, 0, kCbufWindow};
      uw[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      uw[i].dstSet = g.uboSet; uw[i].dstBinding = i; uw[i].descriptorCount = 1;
      uw[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      uw[i].pBufferInfo = &ubinfo[i];
    }
    vkUpdateDescriptorSets(g.device, 8, uw, 0, nullptr);
  }
  return true;
}

VkShaderModule makeModule(const uint32_t *spv, size_t bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = spv;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(g.device, &ci, nullptr, &m);
  return m;
}

// GNM blend multiplier (CB_BLENDn_CONTROL factor field) -> Vulkan blend factor.
VkBlendFactor vkFactor(uint32_t f) {
  switch (f) {
    case 0:  return VK_BLEND_FACTOR_ZERO;
    case 1:  return VK_BLEND_FACTOR_ONE;
    case 2:  return VK_BLEND_FACTOR_SRC_COLOR;
    case 3:  return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 4:  return VK_BLEND_FACTOR_SRC_ALPHA;
    case 5:  return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 6:  return VK_BLEND_FACTOR_DST_ALPHA;
    case 7:  return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 8:  return VK_BLEND_FACTOR_DST_COLOR;
    case 9:  return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 11: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 12: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 13: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 14: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 15: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 16: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 17: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 18: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: return VK_BLEND_FACTOR_ONE;
  }
}
// GNM blend function (combine fcn) -> Vulkan blend op.
VkBlendOp vkBlendOp(uint32_t f) {
  switch (f) {
    case 0:  return VK_BLEND_OP_ADD;
    case 1:  return VK_BLEND_OP_SUBTRACT;
    case 2:  return VK_BLEND_OP_MIN;
    case 3:  return VK_BLEND_OP_MAX;
    case 4:  return VK_BLEND_OP_REVERSE_SUBTRACT;
    default: return VK_BLEND_OP_ADD;
  }
}
// Decode CB_BLEND0_CONTROL into a Vulkan colour-blend attachment. `en` is the
// per-target blend enable (bit 30). Falls back to a sensible src-alpha blend when
// the guest enables blend but the control word is zero (default state, not yet set).
VkPipelineColorBlendAttachmentState blendAttachment(uint32_t bc, bool en) {
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  // DELTA_GPU_NOBLEND: force opaque (diagnostic) to test whether a draw vanishes
  // because its src-alpha blend multiplies by a zero texel alpha (Doom64 3D walls).
  static const bool noBlend = std::getenv("DELTA_GPU_NOBLEND") != nullptr;
  if (noBlend) en = false;
  if (!en) { cba.blendEnable = VK_FALSE; return cba; }
  cba.blendEnable = VK_TRUE;
  uint32_t cs = bc & 0x1F, cf = (bc >> 5) & 0x7, cd = (bc >> 8) & 0x1F;
  bool sep = (bc >> 29) & 1;
  uint32_t as = sep ? (bc >> 16) & 0x1F : cs;
  uint32_t af = sep ? (bc >> 21) & 0x7 : cf;
  uint32_t ad = sep ? (bc >> 24) & 0x1F : cd;
  cba.srcColorBlendFactor = vkFactor(cs);
  cba.dstColorBlendFactor = vkFactor(cd);
  cba.colorBlendOp = vkBlendOp(cf);
  cba.srcAlphaBlendFactor = vkFactor(as);
  cba.dstAlphaBlendFactor = vkFactor(ad);
  cba.alphaBlendOp = vkBlendOp(af);
  return cba;
}

// Build a graphics pipeline for the colored (textured=false) or textured quad
// with the given colour-blend attachment. Shaders + layout selected by `textured`.
VkPipeline buildPipeline(bool textured, VkPipelineColorBlendAttachmentState cba) {
  VkShaderModule vs = makeModule(textured ? tex_vert_spv : quad_vert_spv,
                                 textured ? sizeof(tex_vert_spv) : sizeof(quad_vert_spv));
  VkShaderModule fs = makeModule(textured ? tex_frag_spv : quad_frag_spv,
                                 textured ? sizeof(tex_frag_spv) : sizeof(quad_frag_spv));
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs; stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs; stages[1].pName = "main";

  // Interleaved repacked vertex: pos.xy@0, color.rgba@8, uv.xy@24, stride 32.
  VkVertexInputBindingDescription bind{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[3] = {
      {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, 24},
  };
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = 3;
  vi.pVertexAttributeDescriptions = attrs;
  // GNM draws are indexed triangle LISTS (VGT_PRIMITIVE_TYPE 4); the previous
  // hardcoded strip connected separate sprites into long diagonal triangles.
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1; cb.pAttachments = &cba;
  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyns;
  VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &g.rtFormat;
  VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci; pi.stageCount = 2; pi.pStages = stages;
  pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
  pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &dss;
  pi.pColorBlendState = &cb; pi.pDynamicState = &dy;
  pi.layout = textured ? g.texLayout : g.layout;
  VkPipeline p = VK_NULL_HANDLE;
  vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &pi, nullptr, &p);
  vkDestroyShaderModule(g.device, vs, nullptr);
  vkDestroyShaderModule(g.device, fs, nullptr);
  return p;
}

// Pipeline for a draw's blend state, cached. Returns the default src-alpha pipeline
// when the per-state build fails so a draw never silently drops.
VkPipeline getPipeline(bool textured, uint32_t bc, bool en) {
  uint64_t key = (textured ? 1ull : 0) | (en ? 2ull : 0) |
                 ((uint64_t)(en ? (bc & 0x7FFFFFFFu) : 0u) << 2);
  auto it = g.pipeCache.find(key);
  if (it != g.pipeCache.end())
    return it->second;
  VkPipeline p = buildPipeline(textured, blendAttachment(bc, en));
  if (!p) p = textured ? g.texPipeline : g.pipeline;
  g.pipeCache[key] = p;
  return p;
}

bool createPipeline() {
  if (g.pipeline)
    return true;
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};  // mat4
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g.device, &li, nullptr, &g.layout));
  // Default colored pipeline: classic src-alpha (used as the fallback / for draws
  // that don't enable blend the cache builds an opaque one on demand).
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  g.pipeline = buildPipeline(false, cba);
  if (!g.pipeline) { std::fprintf(stderr, "[gpuvk] pipeline failed\n"); return false; }
  return true;
}

void uploadTexPixels(VkImage img, uint64_t base,
                     const gcn::TextureLayout32 &layout);  // defined below

bool createTexPipeline() {
  if (g.texPipeline)
    return true;
  // descriptor set layout: binding 0 = combined image sampler (fragment).
  VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount = 1;
  dl.pBindings = &b;
  VKOK(vkCreateDescriptorSetLayout(g.device, &dl, nullptr, &g.dsLayout));

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096};
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dp.maxSets = 4096;
  dp.poolSizeCount = 1;
  dp.pPoolSizes = &ps;
  VKOK(vkCreateDescriptorPool(g.device, &dp, nullptr, &g.dsPool));
  g.dsPools.push_back(g.dsPool);

  VkSamplerCreateInfo sc{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sc.magFilter = sc.minFilter = VK_FILTER_LINEAR;
  sc.addressModeU = sc.addressModeV = sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VKOK(vkCreateSampler(g.device, &sc, nullptr, &g.sampler));

  // Multi-texture path: an 8-binding set-0 layout + a pool, used only by recomp PS
  // that sample >1 texture (single-texture draws keep the 1-binding dsLayout/dsPool).
  {
    VkDescriptorSetLayoutBinding mb[State::kMaxTex];
    for (uint32_t i = 0; i < State::kMaxTex; i++)
      mb[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo ml{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ml.bindingCount = State::kMaxTex; ml.pBindings = mb;
    VKOK(vkCreateDescriptorSetLayout(g.device, &ml, nullptr, &g.texArrayLayout));
    VkDescriptorPoolSize mps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096 * State::kMaxTex};
    VkDescriptorPoolCreateInfo mp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    mp.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    mp.maxSets = 4096; mp.poolSizeCount = 1; mp.pPoolSizes = &mps;
    VKOK(vkCreateDescriptorPool(g.device, &mp, nullptr, &g.mtexPool));
    g.mtexPools.push_back(g.mtexPool);

    // 1x1 white default texture (for unresolved sampler bindings).
    VkImageCreateInfo wi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    wi.imageType = VK_IMAGE_TYPE_2D; wi.format = VK_FORMAT_R8G8B8A8_UNORM;
    wi.extent = {1, 1, 1}; wi.mipLevels = 1; wi.arrayLayers = 1;
    wi.samples = VK_SAMPLE_COUNT_1_BIT; wi.tiling = VK_IMAGE_TILING_OPTIMAL;
    wi.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VKOK(vkCreateImage(g.device, &wi, nullptr, &g.whiteImg));
    VkMemoryRequirements wmr; vkGetImageMemoryRequirements(g.device, g.whiteImg, &wmr);
    VkMemoryAllocateInfo wai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    wai.allocationSize = wmr.size;
    wai.memoryTypeIndex = findMemoryType(wmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKOK(vkAllocateMemory(g.device, &wai, nullptr, &g.whiteMem));
    vkBindImageMemory(g.device, g.whiteImg, g.whiteMem, 0);
    uint32_t white = 0xFFFFFFFFu;
    gcn::TextureLayout32 whiteLayout;
    gcn::buildTextureLayout32(whiteLayout, 1, 1, 1, 1, 1, 31, false);
    uploadTexPixels(g.whiteImg, reinterpret_cast<uint64_t>(&white), whiteLayout);
    VkImageViewCreateInfo wv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    wv.image = g.whiteImg; wv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    wv.format = VK_FORMAT_R8G8B8A8_UNORM;
    wv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VKOK(vkCreateImageView(g.device, &wv, nullptr, &g.whiteView));
    wv.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    VKOK(vkCreateImageView(g.device, &wv, nullptr, &g.whiteArrayView));

    VkDescriptorSetLayout layouts[2] = {g.dsLayout, g.dsLayout};
    VkDescriptorSet sets[2];
    VkDescriptorSetAllocateInfo wa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    wa.descriptorPool = g.dsPool; wa.descriptorSetCount = 2; wa.pSetLayouts = layouts;
    VKOK(vkAllocateDescriptorSets(g.device, &wa, sets));
    g.whiteSet = sets[0]; g.whiteArraySet = sets[1];
    VkDescriptorImageInfo infos[2] = {
        {g.sampler, g.whiteView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {g.sampler, g.whiteArrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkWriteDescriptorSet writes[2];
    for (uint32_t i = 0; i < 2; i++) {
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = sets[i]; writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g.device, 2, writes, 0, nullptr);
  }

  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 68};  // mat4 + clipUV flag
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1;
  li.pSetLayouts = &g.dsLayout;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g.device, &li, nullptr, &g.texLayout));

  // Default textured pipeline: src-alpha over (the common sprite blend). Per-draw
  // blend states build their own pipeline on demand via getPipeline().
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  g.texPipeline = buildPipeline(true, cba);
  if (!g.texPipeline) { std::fprintf(stderr, "[gpuvk] tex pipeline failed\n"); return false; }
  return true;
}

VkDescriptorSet allocateSamplerSet(VkDescriptorSetLayout layout, bool multi,
                                   VkDescriptorPool &owner) {
  auto &pools = multi ? g.mtexPools : g.dsPools;
  for (VkDescriptorPool pool : pools) {
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = pool; da.descriptorSetCount = 1; da.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(g.device, &da, &set) == VK_SUCCESS) {
      owner = pool;
      return set;
    }
  }

  VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            4096u * (multi ? State::kMaxTex : 1u)};
  VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  ci.maxSets = 4096; ci.poolSizeCount = 1; ci.pPoolSizes = &size;
  VkDescriptorPool pool;
  if (vkCreateDescriptorPool(g.device, &ci, nullptr, &pool) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  pools.push_back(pool);
  VkDescriptorSet set;
  VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = pool; da.descriptorSetCount = 1; da.pSetLayouts = &layout;
  if (vkAllocateDescriptorSets(g.device, &da, &set) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  owner = pool;
  return set;
}

VkSampler samplerFor(const SamplerKey &key) {
  if (!key.valid) return g.sampler;
  auto found = g_samplerCache.find(key);
  if (found != g_samplerCache.end()) return found->second;
  if (g_samplerCache.size() >= 4096) return g.sampler;

  auto addressMode = [](uint32_t mode) {
    switch (mode & 7) {
      case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
      case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
      case 3: case 5: case 7:
        return g.samplerMirrorClamp ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE
                                    : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
      case 4: case 6: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
      default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
  };
  VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  ci.addressModeU = addressMode(key.raw[0]);
  ci.addressModeV = addressMode(key.raw[0] >> 3);
  ci.addressModeW = addressMode(key.raw[0] >> 6);
  uint32_t mag = (key.raw[2] >> 20) & 3;
  uint32_t min = (key.raw[2] >> 22) & 3;
  ci.magFilter = (mag & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  ci.minFilter = (min & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  uint32_t mipFilter = (key.raw[2] >> 26) & 3;
  ci.mipmapMode = mipFilter == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                  : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  if (mipFilter) {
    uint32_t minLod = std::max(key.raw[1] & 0xFFF, key.imageMinLod);
    ci.minLod = static_cast<float>(minLod) / 256.0f;
    ci.maxLod = static_cast<float>((key.raw[1] >> 12) & 0xFFF) / 256.0f;
    ci.maxLod = std::max(ci.minLod, ci.maxLod);
  }
  int32_t bias = static_cast<int32_t>(key.raw[2] << 18) >> 18;
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g.phys, &props);
  ci.mipLodBias = std::clamp(static_cast<float>(bias) / 256.0f,
                             -props.limits.maxSamplerLodBias,
                             props.limits.maxSamplerLodBias);
  switch ((key.raw[3] >> 30) & 3) {
    case 1: ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; break;
    case 2: ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; break;
    default: ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK; break;
  }
  if (g.samplerAnisotropy && (mag >= 2 || min >= 2)) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g.phys, &props);
    ci.anisotropyEnable = VK_TRUE;
    ci.maxAnisotropy = std::min(static_cast<float>(1u << ((key.raw[0] >> 9) & 7)),
                                props.limits.maxSamplerAnisotropy);
  }
  VkSampler sampler = VK_NULL_HANDLE;
  if (vkCreateSampler(g.device, &ci, nullptr, &sampler) != VK_SUCCESS)
    return g.sampler;
  g_samplerCache.emplace(key, sampler);
  return sampler;
}

// Copy a complete guest mip chain into `img`. The layout module owns all physical
// offsets and swizzles; this function only packs the logical mip/layer images for
// Vulkan buffer-to-image copies.
void uploadTexPixels(VkImage img, uint64_t base,
                     const gcn::TextureLayout32 &layout) {
  static const bool noDetile = std::getenv("DELTA_GPU_NODETILE") != nullptr;
  std::vector<uint32_t> linear;
  uint64_t linearDwords = 0;
  for (uint32_t mip = 0; mip < layout.mipLevels; mip++)
    linearDwords += static_cast<uint64_t>(layout.mips[mip].width) *
                    layout.mips[mip].height * layout.layers;
  linear.resize(static_cast<size_t>(linearDwords));

  VkBufferImageCopy copies[16]{};
  uint64_t linearOffset = 0;
  const uint32_t *src = reinterpret_cast<const uint32_t *>(base);
  for (uint32_t mip = 0; mip < layout.mipLevels; mip++) {
    const auto &level = layout.mips[mip];
    copies[mip].bufferOffset = linearOffset * 4;
    copies[mip].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, layout.layers};
    copies[mip].imageExtent = {level.width, level.height, 1};
    for (uint32_t layer = 0; layer < layout.layers; layer++) {
      uint32_t *dst = linear.data() + linearOffset +
                      static_cast<uint64_t>(layer) * level.width * level.height;
      if (!noDetile) {
        gcn::detileTextureMip32(src, dst, layout, mip, layer);
      } else {
        const uint32_t *levelSrc = src + level.offset / 4 +
            static_cast<uint64_t>(layer) * level.pitch * level.storedHeight;
        for (uint32_t y = 0; y < level.height; y++)
          std::memcpy(dst + static_cast<size_t>(y) * level.width,
                      levelSrc + static_cast<size_t>(y) * level.pitch,
                      static_cast<size_t>(level.width) * 4);
      }
    }
    linearOffset += static_cast<uint64_t>(level.width) * level.height * layout.layers;
  }
  VkDeviceSize sz = linear.size() * sizeof(uint32_t);
  VkBuffer stg; VkDeviceMemory stgMem; void *map;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = sz; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  vkCreateBuffer(g.device, &bi, nullptr, &stg);
  VkMemoryRequirements br; vkGetBufferMemoryRequirements(g.device, stg, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(g.device, &ba, nullptr, &stgMem);
  vkBindBufferMemory(g.device, stg, stgMem, 0);
  vkMapMemory(g.device, stgMem, 0, sz, 0, &map);
  std::memcpy(map, linear.data(), sz);
  vkUnmapMemory(g.device, stgMem);

  VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g.pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
  VkCommandBuffer c; vkAllocateCommandBuffers(g.device, &ca, &c);
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(c, &cbi);
  imageBarrier(c, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               0, VK_ACCESS_TRANSFER_WRITE_BIT, layout.layers, layout.mipLevels);
  vkCmdCopyBufferToImage(c, stg, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         layout.mipLevels, copies);
  imageBarrier(c, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT, layout.layers, layout.mipLevels);
  vkEndCommandBuffer(c);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &c;
  uint64_t _t0 = nowNs();
  vkResetFences(g.device, 1, &g.fence);
  vkQueueSubmit(g.queue, 1, &si, g.fence);
  vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
  g_nsTexUp += nowNs() - _t0; g_texUps++;
  vkFreeCommandBuffers(g.device, g.pool, 1, &c);
  vkDestroyBuffer(g.device, stg, nullptr);
  vkFreeMemory(g.device, stgMem, nullptr);
}

void clearMultiTexCache();

void retireTextureImage(const TexImageKey &key) {
  bool retired = false;
  for (auto set = g_texCache.begin(); set != g_texCache.end();) {
    if (set->first.image == key) {
      g_retiredTexSets.push_back(set->second);
      set = g_texCache.erase(set);
      retired = true;
    } else {
      ++set;
    }
  }
  for (auto view = g_texViews.begin(); view != g_texViews.end();) {
    if (view->first.image == key) {
      g_retiredTexViews.push_back(view->second);
      view = g_texViews.erase(view);
      retired = true;
    } else {
      ++view;
    }
  }
  auto image = g_texImages.find(key);
  if (image != g_texImages.end()) {
    g_retiredTexImages.push_back(image->second);
    g_texImages.erase(image);
    retired = true;
  }
  if (retired) clearMultiTexCache();
}

// Upload a guest texture (linear 32bpp RGBA) and return a descriptor set bound
// to it. Cached by guest base; re-uploaded when the guest pixels change (the
// room art is composed/loaded into the same buffer after the first sample, so a
// once-only cache would serve a stale black frame).
VkDescriptorSet getTexture(uint64_t base, uint32_t w, uint32_t h,
                            uint32_t tiling = 8, uint32_t pitch = 0,
                            uint32_t layers = 1, uint32_t baseArray = 0,
                             uint32_t viewLayers = 1, uint32_t mipLevels = 1,
                             uint32_t baseMip = 0, uint32_t viewMips = 1,
                             uint32_t minLod = 0, bool pow2Pad = false,
                             const uint32_t *sampler = nullptr,
                             bool samplerValid = false, bool arrayed = false) {
  constexpr uint64_t kGuestBegin = 0x1000000000ull;
  constexpr uint64_t kGuestEnd = 0x20000000000ull;
  constexpr uint64_t kMaxTextureBytes = 256ull * 1024 * 1024;
  if (!w || !h || w > 8192 || h > 8192) return VK_NULL_HANDLE;
  if (!layers || baseArray >= layers) return VK_NULL_HANDLE;
  viewLayers = std::min(viewLayers, layers - baseArray);
  if (!arrayed) viewLayers = 1;
  if (!viewLayers) return VK_NULL_HANDLE;
  if (!mipLevels || baseMip >= mipLevels) return VK_NULL_HANDLE;
  viewMips = std::min(viewMips, mipLevels - baseMip);
  if (!viewMips) return VK_NULL_HANDLE;
  // Diagnostic override used to identify incorrectly described guest surfaces.
  tiling = textureTiling(tiling);
  gcn::TextureLayout32 layout;
  if (!gcn::buildTextureLayout32(layout, w, h, pitch ? pitch : w, layers,
                                  mipLevels, tiling, pow2Pad))
    return VK_NULL_HANDLE;
  uint64_t footprint = layout.size;
  if (base < kGuestBegin || footprint > kMaxTextureBytes || base > kGuestEnd - footprint)
    return VK_NULL_HANDLE;
  TexKey key = textureKey(base, w, h, tiling, pitch, layers, baseArray,
                            viewLayers, mipLevels, baseMip, viewMips, minLod,
                            pow2Pad, sampler, samplerValid, arrayed);
  // Diagnostic (DELTA_GPU_TEXDUMP): in deep gameplay, dump the first few large guest
  // textures sampled, so a non-tutorial room's floor texture can be inspected (is it
  // loaded/brown or black/zero?). Counts non-zero pixels too.
  static const bool texDump = std::getenv("DELTA_GPU_TEXDUMP") != nullptr;
  if (texDump && g.frameNum > 1200 && w >= 128 && h >= 128) {
    static int tdn = 0;
    if (tdn < 16) {
      const uint32_t *px = reinterpret_cast<const uint32_t *>(base);
      uint64_t cnt = (uint64_t)w * h, nz = 0, step = cnt > 8192 ? cnt / 8192 : 1;
      for (uint64_t i = 0; i < cnt; i += step) if (px[i] & 0x00FFFFFFu) nz++;
      char p[256];
      std::snprintf(p, sizeof(p), "%s/tex_%02d_%#lx_%ux%u.ppm", dumpDir(), tdn,
                    (unsigned long)base, w, h);
      FILE *f = std::fopen(p, "wb");
      if (f) {
        std::fprintf(f, "P6\n%u %u\n255\n", w, h);
        const uint8_t *b = reinterpret_cast<const uint8_t *>(base);
        for (uint64_t i = 0; i < cnt; i++) { std::fputc(b[i*4], f); std::fputc(b[i*4+1], f); std::fputc(b[i*4+2], f); }
        std::fclose(f);
      }
      std::fprintf(stderr, "[texdump] %d base=%#lx %ux%u nonzero=%lu/8192\n", tdn,
                   (unsigned long)base, w, h, (unsigned long)nz);
      tdn++;
    }
  }
  auto imageIt = g_texImages.find(key.image);
  uint64_t hsh = 0;
  if (imageIt == g_texImages.end() || imageIt->second.lastCheckedFrame != g.frameNum) {
    hsh = texHash(base, footprint);
    if (imageIt != g_texImages.end() && imageIt->second.hash != hsh) {
      retireTextureImage(key.image);
      imageIt = g_texImages.end();
    }
  }
  if (imageIt == g_texImages.end()) {
    if (g_texImages.size() >= 3000) return VK_NULL_HANDLE;
    TexImageEntry imageEntry;
    imageEntry.footprint = footprint;
    imageEntry.hash = hsh;
    imageEntry.lastCheckedFrame = g.frameNum;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {w, h, 1};
    ii.mipLevels = mipLevels; ii.arrayLayers = layers;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkCreateImage(g.device, &ii, nullptr, &imageEntry.image) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g.device, imageEntry.image, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(g.device, &ai, nullptr, &imageEntry.mem) != VK_SUCCESS) {
      vkDestroyImage(g.device, imageEntry.image, nullptr);
      return VK_NULL_HANDLE;
    }
    if (vkBindImageMemory(g.device, imageEntry.image, imageEntry.mem, 0) != VK_SUCCESS) {
      vkFreeMemory(g.device, imageEntry.mem, nullptr);
      vkDestroyImage(g.device, imageEntry.image, nullptr);
      return VK_NULL_HANDLE;
    }
    uploadTexPixels(imageEntry.image, base, layout);
    imageIt = g_texImages.emplace(key.image, imageEntry).first;
  } else {
    imageIt->second.lastCheckedFrame = g.frameNum;
  }

  auto it = g_texCache.find(key);
  if (it != g_texCache.end()) return it->second.set;
  if (g_texCache.size() >= 12000) return VK_NULL_HANDLE;
  TexViewKey viewKey = textureViewKey(key);
  auto viewIt = g_texViews.find(viewKey);
  if (viewIt == g_texViews.end()) {
    if (g_texViews.size() >= 12000) return VK_NULL_HANDLE;
    TexViewEntry viewEntry;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = imageIt->second.image;
    vci.viewType = arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, viewMips, baseArray,
                            arrayed ? viewLayers : 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &viewEntry.view) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    viewIt = g_texViews.emplace(viewKey, viewEntry).first;
  }

  TexEntry e;
  e.set = allocateSamplerSet(g.dsLayout, false, e.pool);
  if (!e.set) return VK_NULL_HANDLE;
  VkDescriptorImageInfo dii{samplerFor(key.sampler), viewIt->second.view,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr.dstSet = e.set; wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wr.pImageInfo = &dii;
  vkUpdateDescriptorSets(g.device, 1, &wr, 0, nullptr);

  g_texCache.emplace(key, e);
  return e.set;
}

// Image view for a guest texture (ensures it is cached/uploaded via getTexture).
VkImageView texViewFor(const DrawInfo::DrawTex &t) {
  if (!t.base || !t.w || !t.h) return VK_NULL_HANDLE;
  if (getTexture(t.base, t.w, t.h, t.tiling, t.pitch, t.layers, t.baseArray,
                   t.viewLayers, t.mipLevels, t.baseMip, t.viewMips,
                  t.minLod, t.pow2Pad, t.sampler, t.samplerValid,
                  t.arrayed) == VK_NULL_HANDLE)
    return VK_NULL_HANDLE;
  TexKey key = textureKey(t.base, t.w, t.h, textureTiling(t.tiling), t.pitch,
                          t.layers, t.baseArray, t.viewLayers, t.mipLevels,
                          t.baseMip, t.viewMips, t.minLod, t.pow2Pad, t.sampler,
                          t.samplerValid, t.arrayed);
  auto it = g_texViews.find(textureViewKey(key));
  return it != g_texViews.end() ? it->second.view : VK_NULL_HANDLE;
}

// An N-sampler descriptor set (set 0, bindings 0..kMaxTex-1) for a recomp PS that
// samples >1 texture. Cached by the combination of the textures' (base,w,h); any
// binding we cannot resolve gets the 1x1 white default so a diffuse*lightmap shader
// with a missing map shows the diffuse instead of going black.
struct MultiTexSet {
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
};
struct MultiTexKey {
  uint32_t nTexs = 0;
  TexKey tex[State::kMaxTex];
  bool operator==(const MultiTexKey &o) const {
    if (nTexs != o.nTexs) return false;
    for (uint32_t i = 0; i < nTexs; i++)
      if (!(tex[i] == o.tex[i])) return false;
    return true;
  }
};
struct MultiTexKeyHash {
  size_t operator()(const MultiTexKey &k) const {
    uint64_t h = hashWord(1469598103934665603ull, k.nTexs);
    for (uint32_t i = 0; i < k.nTexs; i++) h = hashWord(h, TexKeyHash{}(k.tex[i]));
    return static_cast<size_t>(h);
  }
};
std::unordered_map<MultiTexKey, MultiTexSet, MultiTexKeyHash> g_mtexCache;
std::vector<MultiTexSet> g_retiredMtex;
void clearMultiTexCache() {
  for (const auto &[key, entry] : g_mtexCache) {
    (void)key;
    if (entry.set) g_retiredMtex.push_back(entry);
  }
  g_mtexCache.clear();
}
void releaseRetiredTextures() {
  for (const MultiTexSet &entry : g_retiredMtex)
    vkFreeDescriptorSets(g.device, entry.pool, 1, &entry.set);
  g_retiredMtex.clear();
  for (const TexEntry &e : g_retiredTexSets)
    if (e.set) vkFreeDescriptorSets(g.device, e.pool, 1, &e.set);
  g_retiredTexSets.clear();
  for (const TexViewEntry &e : g_retiredTexViews)
    if (e.view) vkDestroyImageView(g.device, e.view, nullptr);
  g_retiredTexViews.clear();
  for (const TexImageEntry &e : g_retiredTexImages) {
    if (e.image) vkDestroyImage(g.device, e.image, nullptr);
    if (e.mem) vkFreeMemory(g.device, e.mem, nullptr);
  }
  g_retiredTexImages.clear();
}
VkDescriptorSet getMultiTexSet(const DrawInfo &d) {
  MultiTexKey key;
  key.nTexs = std::min(d.nTexs, State::kMaxTex);
  for (uint32_t i = 0; i < key.nTexs; i++) {
    const auto &t = d.texs[i];
    key.tex[i] = textureKey(t.base, t.w, t.h, textureTiling(t.tiling), t.pitch,
                            t.layers, t.baseArray, t.viewLayers, t.mipLevels,
                            t.baseMip, t.viewMips, t.minLod, t.pow2Pad, t.sampler,
                            t.samplerValid, t.arrayed);
  }
  for (uint32_t i = 0; i < key.nTexs; i++) texViewFor(d.texs[i]);
  auto ci = g_mtexCache.find(key);
  if (ci != g_mtexCache.end()) return ci->second.set;
  if (g_mtexCache.size() > 3500) return VK_NULL_HANDLE;
  // DELTA_GPU_FORCEWHITE: bind the 1x1 white default for every sampler (diagnostic).
  // Doom64's world textures are built by compute dispatches we don't execute, so the
  // atlases are all-zero and the alpha-blended world samples transparent-black (=
  // invisible). Forcing white makes the geometry render opaque, proving the 3D
  // transform/raster/depth path works and isolating the blackness to the texture data.
  static const bool forceWhite = std::getenv("DELTA_GPU_FORCEWHITE") != nullptr;
  VkImageView views[State::kMaxTex];
  for (uint32_t i = 0; i < State::kMaxTex; i++) {
    VkImageView v = (i < key.nTexs && !forceWhite) ? texViewFor(d.texs[i]) : VK_NULL_HANDLE;
    bool arrayed = i < key.nTexs && d.texs[i].arrayed;
    views[i] = v ? v : (arrayed ? g.whiteArrayView : g.whiteView);
  }
  MultiTexSet entry;
  entry.set = allocateSamplerSet(g.texArrayLayout, true, entry.pool);
  if (!entry.set) return VK_NULL_HANDLE;
  VkDescriptorImageInfo dii[State::kMaxTex];
  VkWriteDescriptorSet wr[State::kMaxTex];
  for (uint32_t i = 0; i < State::kMaxTex; i++) {
    SamplerKey sampler;
    if (i < key.nTexs) {
      sampler.valid = d.texs[i].samplerValid;
      std::memcpy(sampler.raw, d.texs[i].sampler, sizeof(sampler.raw));
      sampler.imageMinLod = d.texs[i].minLod;
    }
    dii[i] = {samplerFor(sampler), views[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[i].dstSet = entry.set; wr[i].dstBinding = i; wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr[i].pImageInfo = &dii[i];
  }
  vkUpdateDescriptorSets(g.device, State::kMaxTex, wr, 0, nullptr);
  g_mtexCache[key] = entry;
  return entry.set;
}

void ensureReadback(uint32_t w, uint32_t h) {
  VkDeviceSize need = (VkDeviceSize)w * h * 4;
  if (g.readback && need <= g.readbackSize)
    return;
  vkDeviceWaitIdle(g.device);
  if (g.readbackMap) vkUnmapMemory(g.device, g.readbackMem);
  if (g.readback) vkDestroyBuffer(g.device, g.readback, nullptr);
  if (g.readbackMem) vkFreeMemory(g.device, g.readbackMem, nullptr);
  g.readbackSize = need;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = need; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkCreateBuffer(g.device, &bi, nullptr, &g.readback);
  VkMemoryRequirements br; vkGetBufferMemoryRequirements(g.device, g.readback, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  // CPU reads this buffer every frame (the flip) -> prefer HOST_CACHED so reads hit
  // cache instead of streaming from write-combined memory (the dominant frame cost).
  ba.memoryTypeIndex = findMemoryTypePref(br.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(g.device, &ba, nullptr, &g.readbackMem);
  vkBindBufferMemory(g.device, g.readback, g.readbackMem, 0);
  vkMapMemory(g.device, g.readbackMem, 0, need, 0, &g.readbackMap);
}

// Find or create the render target at guest address `base` (dimensions w x h).
RTarget *getRT(uint64_t base, uint32_t w, uint32_t h) {
  auto it = g_rts.find(base);
  if (it != g_rts.end())
    return &it->second;
  if (g_rts.size() > 64 || !w || !h)
    return nullptr;
  RTarget t; t.w = w; t.h = h;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = g.rtFormat;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (vkCreateImage(g.device, &ii, nullptr, &t.image) != VK_SUCCESS) return nullptr;
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(g.device, t.image, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(g.device, &ai, nullptr, &t.mem);
  vkBindImageMemory(g.device, t.image, t.mem, 0);
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = g.rtFormat;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCreateImageView(g.device, &vci, nullptr, &t.view);
  // descriptor set so this RT can be sampled (render-to-texture).
  if (g.dsPool) {
    VkDescriptorPool owner;
    t.set = allocateSamplerSet(g.dsLayout, false, owner);
    if (t.set) {
      VkDescriptorImageInfo dii{g.sampler, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      wr.dstSet = t.set; wr.descriptorCount = 1;
      wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      wr.pImageInfo = &dii;
      vkUpdateDescriptorSets(g.device, 1, &wr, 0, nullptr);
    }
  }
  std::fprintf(stderr, "[gpuvk] new RT %#lx %ux%u\n", (unsigned long)base, w, h);
  g_rts[base] = t;
  registerRtPages(base, w, h);  // resource-model page table
  return &g_rts[base];
}

uint64_t rtByteSize(const RTarget &rt) { return rtByteSizeWH(rt.w, rt.h); }

// Transition a depth image (aspect = DEPTH) between layouts.
void depthBarrier(VkCommandBuffer c, VkImage img, VkImageLayout from,
                  VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from; b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  b.srcAccessMask = srcA; b.dstAccessMask = dstA;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Find or create the depth target at guest address `base` (dimensions w x h).
DepthTarget *getDepthRT(uint64_t base, uint32_t w, uint32_t h) {
  auto it = g_depths.find(base);
  if (it != g_depths.end()) return &it->second;
  if (g_depths.size() > 32 || !w || !h) return nullptr;
  DepthTarget t; t.w = w; t.h = h;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = kDepthFormat;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (vkCreateImage(g.device, &ii, nullptr, &t.image) != VK_SUCCESS) return nullptr;
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(g.device, t.image, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(g.device, &ai, nullptr, &t.mem);
  vkBindImageMemory(g.device, t.image, t.mem, 0);
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = kDepthFormat;
  vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  vkCreateImageView(g.device, &vci, nullptr, &t.view);
  std::fprintf(stderr, "[gpuvk] new depth %#lx %ux%u\n", (unsigned long)base, w, h);
  g_depths[base] = t;
  return &g_depths[base];
}

// Resolve a sampled texture address to the live RT image that backs it (the
// resource model's "the page table collects all overlappers" lookup). The game
// cycles/aliases RT addresses (double-buffered room layers, a pool of scene
// buffers), so a composite often samples an address that does not exactly match
// the RT base it was rendered into. We gather every RT whose footprint touches
// the sampled region's pages and resolve by IDENTITY, not recency: an exact base
// whose size matches wins outright (the guest bound that buffer, so a more recently
// rendered overlapping buffer must never override it); only when no exact match
// exists do we fall back to the best-fitting overlapper (dimension match, then the
// freshest). Returns the g_rts key, or 0.
uint64_t resolveSampledRT(uint64_t addr, uint32_t w, uint32_t h) {
  if (!addr) return 0;
  uint64_t reqSize = w && h ? (uint64_t)w * h * 4 : 4;
  uint64_t a0 = addr, a1 = addr + reqSize;
  // Exact-identity hit: the guest sampled this exact base and it is a live RT of the
  // requested size. That is unambiguously the right image -- return it before any
  // freshness comparison can pick an overlapping cycled buffer instead.
  auto ex = g_rts.find(addr);
  if (ex != g_rts.end() && ex->second.everRendered &&
      ((!w || !h) || (ex->second.w == w && ex->second.h == h)))
    return addr;
  uint64_t best = 0;
  long bestScore = -1;
  auto consider = [&](uint64_t b0) {
    auto it = g_rts.find(b0);
    if (it == g_rts.end()) return;
    const RTarget &rt = it->second;
    if (!rt.everRendered) return;  // never sample an RT with no content
    uint64_t b1 = b0 + rtByteSizeWH(rt.w, rt.h);
    if (!(a0 < b1 && b0 < a1)) return;  // no interval overlap
    bool dimMatch = (!w || !h) || (rt.w == w && rt.h == h);
    long score = (long)rt.lastFrame;
    if (dimMatch) score += 1L << 30;
    if (b0 == addr) score += 1L << 31;
    if (score > bestScore) { bestScore = score; best = b0; }
  };
  for (uint64_t p = a0 >> kRtPageShift; p <= (a1 - 1) >> kRtPageShift; p++) {
    auto it = g_rtPages.find(p);
    if (it != g_rtPages.end())
      for (uint64_t b0 : it->second) consider(b0);
  }
  return best;
}

// End the current dynamic-rendering region (if any), leaving its RTs readable.
void endRegion() {
  if (!g.curRt) return;
  p_vkCmdEndRendering(g.cmd);
  for (uint32_t i = 0; i < g.curMrtCount; i++) {
    auto it = g_rts.find(g.curMrt[i]);
    if (it == g_rts.end()) continue;
    auto &rt = it->second;
    imageBarrier(g.cmd, rt.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    rt.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }
  g.curRt = 0;
  g.curMrtCount = 0;
  g.curDepth = 0;
}

void setGuestViewport(const DrawInfo &d) {
  if (!std::isfinite(d.viewportXScale) || !std::isfinite(d.viewportXOffset) ||
      !std::isfinite(d.viewportYScale) || !std::isfinite(d.viewportYOffset) ||
      d.viewportXScale <= 0.0f || d.viewportYScale == 0.0f)
    return;
  VkViewport vp{
      d.viewportXOffset - d.viewportXScale,
      d.viewportYOffset - d.viewportYScale,
      d.viewportXScale * 2.0f,
      d.viewportYScale * 2.0f,
      0.0f,
      1.0f,
  };
  vkCmdSetViewport(g.cmd, 0, 1, &vp);
}

// Begin a dynamic-rendering region binding mrtCount color targets (mrtBase[0] is the
// primary). The common single-RT case (mrtCount == 1) binds exactly one attachment.
// depthBase != 0 additionally binds a depth attachment (cleared to depthClear on its
// first use each frame, loaded thereafter); depthBase == 0 leaves depth unbound (the
// 2D path).
void beginRegion(const uint64_t *mrtBase, uint32_t mrtCount, uint32_t w, uint32_t h,
                 uint64_t depthBase = 0, float depthClear = 1.0f) {
  static const bool lazyClear = [] { const char *e = std::getenv("DELTA_GPU_LAZYCLEAR");
    return !e || std::strcmp(e, "0") != 0; }();
  VkRenderingAttachmentInfo colors[8]{};
  g.curMrtCount = 0;
  for (uint32_t i = 0; i < mrtCount && i < 8; i++) {
    RTarget *rtp = getRT(mrtBase[i], w, h);
    RTarget &rt = *rtp;
    imageBarrier(g.cmd, rt.image, rt.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    rt.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    auto &color = colors[i];
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = rt.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // Lazy clear (DELTA_GPU_LAZYCLEAR, default on): persist RT content across frames
    // (LOAD), clearing only when the game explicitly requested a clear (clearPending) or
    // the RT was never rendered. The old per-frame auto-clear wiped baked-once content
    // (room floor) whose redraw lands on a different frame than its clear.
    if (lazyClear)
      color.loadOp = (rt.clearPending || !rt.everRendered) ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                           : VK_ATTACHMENT_LOAD_OP_LOAD;
    else
      color.loadOp = rt.usedThisFrame ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    rt.clearPending = false;
    rt.everRendered = true;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = rt.clearValue;
    rt.usedThisFrame = true;
    rt.lastFrame = g.frameNum;
    g.curMrt[g.curMrtCount++] = mrtBase[i];
  }
  RTarget &rt = *getRT(mrtBase[0], w, h);
  uint64_t base = mrtBase[0];
  // Depth attachment (3D). Cleared to the guest DB_DEPTH_CLEAR value on its first use
  // this frame, then loaded so multiple regions in a frame share one Z buffer.
  VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  DepthTarget *dt = depthBase ? getDepthRT(depthBase, rt.w, rt.h) : nullptr;
  if (dt) {
    depthBarrier(g.cmd, dt->image, dt->layout, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
    dt->layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.imageView = dt->view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    bool clearDepth = dt->clearPending || !dt->usedThisFrame;
    depthAtt.loadOp = clearDepth
                          ? VK_ATTACHMENT_LOAD_OP_CLEAR
                          : VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {dt->clearPending ? dt->clearValue : depthClear, 0};
    dt->clearPending = false;
    dt->usedThisFrame = true;
    dt->lastFrame = g.frameNum;
    g.curDepth = depthBase;
  }
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {rt.w, rt.h}};
  ri.layerCount = 1; ri.colorAttachmentCount = g.curMrtCount; ri.pColorAttachments = colors;
  if (dt) ri.pDepthAttachment = &depthAtt;
  p_vkCmdBeginRendering(g.cmd, &ri);
  // Negative-height (y-up) viewport: GCN/PS4 rasterises y-up, so we do too. This
  // stores render-target content upright, so render-to-texture composites (the
  // scene->scanout copy, effect overlays) sample it with aligned UVs when run through
  // the game's real recompiled shader, and the presented scanout is already upright
  // (no readback flip needed; DELTA_GPU_FLIP defaults to 0).
  VkViewport vpt{0, (float)rt.h, (float)rt.w, -(float)rt.h, 0, 1};
  vkCmdSetViewport(g.cmd, 0, 1, &vpt);
  VkRect2D sc{{0, 0}, {rt.w, rt.h}};
  vkCmdSetScissor(g.cmd, 0, 1, &sc);
  rt.usedThisFrame = true;
  rt.lastFrame = g.frameNum;
  if (rt.w >= 700 && rt.w <= 900) g.frameRoomBake = true;  // room background baked
  g.curRt = base;
  g.lastRt = base;
  static const bool regTrace = std::getenv("DELTA_GPU_REGTRACE") != nullptr;
  if (regTrace && rt.w < 1280)
    std::fprintf(stderr, "[reg] f%d begin RT %#lx %ux%u mrt=%u clear=%d\n", g.frameNum,
                 (unsigned long)base, rt.w, rt.h, g.curMrtCount,
                 colors[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
}

void writePpm(const char *path, const uint8_t *bgra, uint32_t w, uint32_t h) {
  FILE *f = std::fopen(path, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  for (uint32_t i = 0; i < w * h; i++) {
    std::fputc(bgra[i * 4 + 2], f);
    std::fputc(bgra[i * 4 + 1], f);
    std::fputc(bgra[i * 4 + 0], f);
  }
  std::fclose(f);
}

void dumpPpm(const uint8_t *bgra, uint32_t w, uint32_t h) {
  if (g_dumpedFrames >= 4) return;
  char path[256];
  std::snprintf(path, sizeof(path), "%s/gpu_frame_%d.ppm", dumpDir(), g_dumpedFrames++);
  writePpm(path, bgra, w, h);
  std::fprintf(stderr, "[gpuvk] dumped %s\n", path);
}

// Diagnostic: dump every render target used this frame to its own ppm so the
// per-buffer content (which 832 buffer holds the floor vs the walls, etc) can be
// inspected directly. One-shot per RT via a transient command buffer (slow; only
// for debugging). Call after the frame's main submit has completed.

}  // namespace

bool init() {
  if (g.ready) return true;
  if (!createDevice()) {
    std::fprintf(stderr, "[gpuvk] headless Vulkan unavailable; gpu disabled\n");
    return false;
  }
  g.ready = true;
  return true;
}

bool available() { return g.ready; }

// ---- compute dispatch -------------------------------------------------------
// A recompiled compute pipeline, cached by CS address: the SPIR-V + binding layout
// depend only on the code, so only the descriptor set + push constants + storage
// buffers are rebuilt per dispatch.
struct CsPipe {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  uint32_t nres = 0;
};
std::unordered_map<uint64_t, CsPipe> g_csPipes;

CsPipe *getCsPipe(const ComputeInfo &ci) {
  auto it = g_csPipes.find(ci.csAddr);
  if (it != g_csPipes.end())
    return it->second.nres == ci.nres ? &it->second : nullptr;
  CsPipe cp; cp.nres = ci.nres;
  VkDescriptorSetLayoutBinding binds[8];
  for (uint32_t i = 0; i < ci.nres; i++)
    binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sl.bindingCount = ci.nres; sl.pBindings = binds;
  if (vkCreateDescriptorSetLayout(g.device, &sl, nullptr, &cp.setLayout) != VK_SUCCESS)
    return nullptr;
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};  // 16 user-data dwords
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1; li.pSetLayouts = &cp.setLayout;
  li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(g.device, &li, nullptr, &cp.layout) != VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(g.device, cp.setLayout, nullptr); return nullptr; }
  VkShaderModule cs = makeModule(ci.recomp->spirv.data(), ci.recomp->spirv.size() * 4);
  VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pi.stage.module = cs; pi.stage.pName = "main";
  pi.layout = cp.layout;
  VkResult r = vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &pi, nullptr, &cp.pipe);
  vkDestroyShaderModule(g.device, cs, nullptr);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[gpuvk] compute pipeline failed: %d\n", (int)r);
    vkDestroyPipelineLayout(g.device, cp.layout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, cp.setLayout, nullptr);
    return nullptr;
  }
  g_csPipes[ci.csAddr] = cp;
  return &g_csPipes[ci.csAddr];
}

// Drop cached textures overlapping a written range so the graphics path re-uploads
// the fresh compute output (the content hash would trigger this too, but erasing is
// immediate and covers a size/format change).
void invalidateTexRange(uint64_t base, uint64_t size) {
  if (!size || base > UINT64_MAX - size) return;
  uint64_t end = base + size;
  std::vector<TexImageKey> overlap;
  for (const auto &[key, image] : g_texImages) {
    uint64_t texEnd = key.base + image.footprint;
    if (key.base < end && base < texEnd) overlap.push_back(key);
  }
  for (const TexImageKey &key : overlap) retireTextureImage(key);
}

// Persistent compute staging. Dispatches are serialized behind the fence, so one set
// of staging buffers (+ descriptor pool + command buffer) is reused across every
// dispatch: this avoids the per-dispatch vkCreateBuffer/vkAllocateMemory/pool/cmd-
// buffer churn (~3ms/frame). The buffers are HOST_CACHED so the copy-BACK read after
// the dispatch hits cache instead of stalling on write-combined memory (which was
// ~25ms/frame for Doom64's 8 MB atlas: the dominant compute cost).
struct CsStage {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void *map = nullptr;
  VkDeviceSize cap = 0;
};
CsStage g_csStage[8];
VkDescriptorPool g_csDescPool = VK_NULL_HANDLE;
VkCommandBuffer g_csCmd = VK_NULL_HANDLE;

// Ensure staging slot i can hold `size` bytes (grow-on-demand, kept mapped).
bool csEnsureStage(uint32_t i, VkDeviceSize size) {
  CsStage &s = g_csStage[i];
  if (s.buf && s.cap >= size) return true;
  if (s.map) { vkUnmapMemory(g.device, s.mem); s.map = nullptr; }
  if (s.buf) { vkDestroyBuffer(g.device, s.buf, nullptr); s.buf = VK_NULL_HANDLE; }
  if (s.mem) { vkFreeMemory(g.device, s.mem, nullptr); s.mem = VK_NULL_HANDLE; }
  VkDeviceSize cap = (size + 0xFFFFF) & ~VkDeviceSize(0xFFFFF);  // 1 MiB granularity
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = cap; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (vkCreateBuffer(g.device, &bi, nullptr, &s.buf) != VK_SUCCESS) { s.buf = VK_NULL_HANDLE; return false; }
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g.device, s.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryTypePref(mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(g.device, &ai, nullptr, &s.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g.device, s.buf, nullptr); s.buf = VK_NULL_HANDLE; s.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g.device, s.buf, s.mem, 0);
  vkMapMemory(g.device, s.mem, 0, cap, 0, &s.map);
  s.cap = cap;
  return true;
}

bool dispatch(const ComputeInfo &ci) {
  if (!g.ready || !ci.recomp || !ci.recomp->ok || !ci.nres || ci.nres > 8) return false;
  CsPipe *cp = getCsPipe(ci);
  if (!cp) return false;
  static const bool verbose = std::getenv("DELTA_GPU_CSGPU_VERBOSE") != nullptr;

  // Persistent command buffer + descriptor pool (created once, reused).
  if (g_csCmd == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g.pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g.device, &ca, &g_csCmd) != VK_SUCCESS) { g_csCmd = VK_NULL_HANDLE; return false; }
  }
  if (g_csDescPool == VK_NULL_HANDLE) {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g.device, &pci, nullptr, &g_csDescPool) != VK_SUCCESS) { g_csDescPool = VK_NULL_HANDLE; return false; }
  }

  // Stage each resource's guest range into its reused storage buffer.
  VkDeviceSize sz[8];
  for (uint32_t i = 0; i < ci.nres; i++) {
    sz[i] = ci.res[i].size ? ((ci.res[i].size + 3) & ~VkDeviceSize(3)) : 4;
    if (!csEnsureStage(i, sz[i])) return false;
    std::memcpy(g_csStage[i].map, reinterpret_cast<const void *>(ci.res[i].base), ci.res[i].size);
  }

  // Descriptor set binding the storage buffers (pool reset each dispatch).
  vkResetDescriptorPool(g.device, g_csDescPool, 0);
  VkDescriptorSet set;
  VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = g_csDescPool; da.descriptorSetCount = 1; da.pSetLayouts = &cp->setLayout;
  if (vkAllocateDescriptorSets(g.device, &da, &set) != VK_SUCCESS) return false;
  VkDescriptorBufferInfo dbi[8]; VkWriteDescriptorSet wr[8];
  for (uint32_t i = 0; i < ci.nres; i++) {
    dbi[i] = {g_csStage[i].buf, 0, sz[i]};
    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[i].dstSet = set; wr[i].dstBinding = ci.res[i].binding; wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(g.device, ci.nres, wr, 0, nullptr);

  // Record + submit the dispatch, wait for completion (synchronous: the result must
  // be back in guest memory before the following draws/texture uploads read it).
  vkResetCommandBuffer(g_csCmd, 0);
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g_csCmd, &cbi);
  vkCmdBindPipeline(g_csCmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
  vkCmdBindDescriptorSets(g_csCmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->layout, 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(g_csCmd, cp->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, ci.userData);
  vkCmdDispatch(g_csCmd, ci.groups[0], ci.groups[1], ci.groups[2]);
  vkEndCommandBuffer(g_csCmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &g_csCmd;
  vkResetFences(g.device, 1, &g.fence);
  if (vkQueueSubmit(g.queue, 1, &si, g.fence) != VK_SUCCESS) return false;
  vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);

  // Copy written ranges back to guest memory + invalidate any cached texture there.
  for (uint32_t i = 0; i < ci.nres; i++) {
    if (!ci.res[i].written) continue;
    std::memcpy(reinterpret_cast<void *>(ci.res[i].base), g_csStage[i].map, ci.res[i].size);
    invalidateTexRange(ci.res[i].base, ci.res[i].size);
    if (verbose) {
      const uint8_t *b = static_cast<const uint8_t *>(g_csStage[i].map);
      uint64_t nz = 0, step = ci.res[i].size > 65536 ? ci.res[i].size / 65536 : 1;
      for (uint64_t k = 0; k < ci.res[i].size; k += step) nz += b[k] != 0;
      std::fprintf(stderr, "[csgpu] wrote back base=%#lx size=%lu nonzero=%lu/%lu\n",
                   (unsigned long)ci.res[i].base, (unsigned long)ci.res[i].size,
                   (unsigned long)nz, (unsigned long)(ci.res[i].size / step));
    }
  }
  return true;
}

// ---- recompiled-shader path -------------------------------------------------
// GCN data format -> Vulkan vertex format.
VkFormat vfmt(uint32_t dfmt, uint32_t nfmt) {
  switch (dfmt) {
    case 1:  return VK_FORMAT_R8_UNORM;
    case 3:  return VK_FORMAT_R8G8_UNORM;
    case 4:  return nfmt == 4 ? VK_FORMAT_R32_UINT : VK_FORMAT_R32_SFLOAT;
    case 10: return nfmt == 4 ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_UNORM;
    case 11: return VK_FORMAT_R32G32_SFLOAT;
    case 13: return VK_FORMAT_R32G32B32_SFLOAT;
    case 14: return VK_FORMAT_R32G32B32A32_SFLOAT;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
  }
}

// VGT_PRIMITIVE_TYPE -> Vulkan topology. Unknown/2D types fall back to triangle list
// (the previous hardcoded topology), so the 2D path is unchanged.
VkPrimitiveTopology vkTopology(uint32_t prim) {
  switch (prim) {
    case 1:  return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case 2:  return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 3:  return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case 5:  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case 6:  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case 4:  // triangle list
    default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }
}

struct RecompPipe {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  bool textured = false;
  bool multiTex = false;  // PS samples >1 texture -> set 0 = the N-sampler layout
};
std::unordered_map<uint64_t, RecompPipe> g_recompPipes;

VkShaderModule makeModuleVec(const std::vector<uint32_t> &spv) {
  return makeModule(spv.data(), spv.size() * 4);
}

// Build (or fetch) the pipeline for a recompiled draw, keyed by the shader pair +
// blend state + vertex layout.
RecompPipe *getRecompPipe(const DrawInfo &d) {
  uint32_t mrtN = d.mrtCount ? (d.mrtCount > 8 ? 8 : d.mrtCount) : 1;
  // Depth + primitive-setup state folded into the pipeline key (mixed through an FNV
  // prime so it spreads across the whole 64-bit space, away from the blend/stride bits).
  uint32_t dstate = (d.depthBase ? 1u : 0u) | (d.depthTestEnable ? 2u : 0u) |
                    (d.depthWriteEnable ? 4u : 0u) | ((d.depthFunc & 7u) << 3) |
                    ((d.primType & 0x1Fu) << 6) | ((d.cullMode & 3u) << 11) |
                    (d.frontCCW ? 0u : (1u << 13));
  uint64_t key = d.vsAddr * 0x9e3779b97f4a7c15ull ^ d.psAddr ^
                 ((uint64_t)(d.blendEnable ? (d.blendControl & 0x7FFFFFFFu) : 0) << 1) ^
                 ((uint64_t)d.vertexStride << 33) ^ ((uint64_t)mrtN << 60) ^
                 ((uint64_t)dstate * 0x100000001b3ull);
  auto it = g_recompPipes.find(key);
  if (it != g_recompPipes.end()) return &it->second;
  RecompPipe rp;
  rp.textured = !d.recomp->psTexs.empty();
  rp.multiTex = d.recomp->psTexs.size() > 1;  // multi-sampler set-0 layout

  // set 0 = texture(s) (or an empty layout when untextured), set 1 = cbuffer UBO.
  // A multi-texture PS uses the 8-binding layout; single-texture keeps the 1-binding.
  VkDescriptorSetLayout set0 = !rp.textured ? g.emptyLayout
                             : rp.multiTex ? g.texArrayLayout : g.dsLayout;
  VkDescriptorSetLayout sls[2] = {set0, g.uboLayout};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 2;
  li.pSetLayouts = sls;
  if (vkCreatePipelineLayout(g.device, &li, nullptr, &rp.layout) != VK_SUCCESS) return nullptr;

  VkShaderModule vs = makeModuleVec(d.recomp->vsSpirv);
  VkShaderModule fs = makeModuleVec(d.recomp->fsSpirv);
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

  VkVertexInputBindingDescription bind{0, d.vertexStride, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[8];
  for (uint32_t i = 0; i < d.nvattrs; i++)
    attrs[i] = {d.vattrs[i].location, 0, vfmt(d.vattrs[i].dfmt, d.vattrs[i].nfmt), d.vattrs[i].offset};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = d.nvattrs; vi.pVertexAttributeDescriptions = attrs;

  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = vkTopology(d.primType);
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  // Face culling from PA_SU_SC_MODE_CNTL (CULL_FRONT[0]/CULL_BACK[1] map 1:1 onto the
  // Vulkan cull-mode bits). The render region uses a negative-height (y-up) viewport
  // to match GCN rasterisation, which flips triangle winding in framebuffer space, so
  // the guest's front-face sense is inverted here to compensate. Culling is opt-in
  // (DELTA_GPU_CULL=1) until the winding can be validated against visible 3D geometry:
  // Doom64's world textures are compute-built (unimplemented) so its geometry is not
  // yet visible, and depth already resolves occlusion, so the default stays cull-none
  // to avoid dropping correctly-drawn faces (some HUD draws set cull bits).
  static const bool doCull = std::getenv("DELTA_GPU_CULL") != nullptr;
  rs.cullMode = doCull ? (VkCullModeFlags)(d.cullMode & 0x3) : VK_CULL_MODE_NONE;
  rs.frontFace = d.frontCCW ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  // Depth test/write from DB_DEPTH_CONTROL (only when the draw bound a Z buffer; 2D
  // draws leave depthBase 0 so this stays fully disabled, unchanged from before).
  VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  if (d.depthBase) {
    dss.depthTestEnable = d.depthTestEnable ? VK_TRUE : VK_FALSE;
    dss.depthWriteEnable = d.depthWriteEnable ? VK_TRUE : VK_FALSE;
    dss.depthCompareOp = (VkCompareOp)(d.depthFunc & 0x7);  // ZFUNC maps 1:1
  }
  // One blend attachment per bound MRT target, each from its own CB_BLENDn_CONTROL
  // (mrtBlend[i] / mrtBlendMask bit i); target 0 mirrors blendControl/blendEnable so the
  // single-RT path is unchanged. Targets the PS does not export to are write-masked off
  // so they keep their loaded content.
  VkPipelineColorBlendAttachmentState cbAtt[8];
  for (uint32_t i = 0; i < mrtN; i++) {
    uint32_t bc = i == 0 ? d.blendControl : d.mrtBlend[i];
    bool en = i == 0 ? d.blendEnable : ((d.mrtBlendMask >> i) & 1u);
    cbAtt[i] = blendAttachment(bc, en);
    if (i && !(d.recomp->psMrtMask & (1u << i))) cbAtt[i].colorWriteMask = 0;
  }
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = mrtN; cb.pAttachments = cbAtt;
  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyns;
  VkFormat fmts[8];
  for (uint32_t i = 0; i < mrtN; i++) fmts[i] = g.rtFormat;
  VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = mrtN; rci.pColorAttachmentFormats = fmts;
  if (d.depthBase) rci.depthAttachmentFormat = kDepthFormat;
  VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci; pi.stageCount = 2; pi.pStages = stages;
  pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
  pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &dss;
  pi.pColorBlendState = &cb; pi.pDynamicState = &dy; pi.layout = rp.layout;
  VkResult r = vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &pi, nullptr, &rp.pipe);
  vkDestroyShaderModule(g.device, vs, nullptr);
  vkDestroyShaderModule(g.device, fs, nullptr);
  if (r != VK_SUCCESS) { std::fprintf(stderr, "[gpuvk] recomp pipeline failed: %d\n", (int)r);
    return nullptr; }
  g_recompPipes[key] = rp;
  return &g_recompPipes[key];
}

// Why a draw declined the recompiled path (falls back to the heuristic). Tallied
// per reason so the remaining heuristic draws can be driven to zero; dumped with the
// periodic frame log.
enum DeclineReason { DR_NORECOMP, DR_NOTEXPIPE, DR_SELF, DR_RING,
                     DR_GUESTTEX, DR_MIDREGION, DR_NOPIPE, DR_MAX };
static const char *kDeclineName[DR_MAX] = {
    "norecomp", "notexpipe", "self", "ring", "guesttex", "midregion", "nopipe"};
uint32_t g_decline[DR_MAX] = {0};
inline bool decline(DeclineReason r) { g_decline[r]++; return false; }

// Issue an indexed draw running the game's recompiled VS/PS. Returns false if the
// draw can't be handled (the caller falls back to the heuristic path).
bool drawRecomp(const DrawInfo &d) {
  static const bool drawTrace = std::getenv("DELTA_GPU_DRAWTRACE") != nullptr;
  if (drawTrace && d.indexCount >= 300) {
    static int n = 0;
    if (n++ < 40)
      std::fprintf(stderr, "[dt] enter recomp idx=%u rt=%#lx tex=%#lx nTexs=%u nvattrs=%u ok=%d\n",
                   d.indexCount, (unsigned long)d.rtBase, (unsigned long)d.texBase,
                   d.nTexs, d.nvattrs, d.recomp ? d.recomp->ok : 0);
  }
  if (!d.recomp || !d.recomp->ok || !d.nvattrs || !d.indexData || d.indexCount < 3)
    return decline(DR_NORECOMP);
  if (!g.texPipeline) return decline(DR_NOTEXPIPE);  // need the descriptor infra (dsPool/dsLayout)
  // Resolve the sampled texture address to an overlapping live RT (resource-model
  // page-table lookup), so an RT-as-texture sample binds the live image for
  // cycled/aliased RT addresses instead of stale guest memory. Additive: an exact
  // RT base resolves to itself.
  uint64_t texBase = d.texBase;
  if (texBase && !d.texArrayed && !g_rts.count(texBase)) {
    uint64_t r = resolveSampledRT(texBase, d.texW, d.texH);
    if (r) texBase = r;
  }
  bool rtAsTex = !d.texArrayed && texBase && texBase != d.rtBase && g_rts.count(texBase);
  if (rtAsTex && g_rts[texBase].w >= 700 && g_rts[texBase].w <= 900)
    g.frameHadRoom = true;
  if (texBase && texBase == d.rtBase) return decline(DR_SELF);
  // Index range -> vertex count.
  const uint16_t *i16 = nullptr; const uint32_t *i32 = nullptr;
  uint32_t maxIdx = 0;
  if (d.indexType == 1) { i32 = (const uint32_t *)d.indexData;
    for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = i32[i] > maxIdx ? i32[i] : maxIdx; }
  else { i16 = (const uint16_t *)d.indexData;
    for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = i16[i] > maxIdx ? i16[i] : maxIdx; }
  uint32_t nv = maxIdx + 1;
  if (nv > 200000u || !d.vertexStride) return decline(DR_NORECOMP);

  // A fullscreen, untextured, near-black REPLACE draw is the game CLEARING an RT.
  // Don't render it (that wipes the RT immediately); record a LAZY clear instead --
  // realised as loadOp=CLEAR only when content actually redraws this RT this frame
  // (see beginRegion). Baked-once content (the room floor) whose clear and redraw land
  // on different frames then survives. A COLOURED fullscreen REPLACE is real content
  // (e.g. the per-frame minimap redraw) and must NOT be treated as a clear.
  static const bool noWipe = [] {
    const char *e = std::getenv("DELTA_GPU_NOWIPE");
    return !e || std::strcmp(e, "0") != 0;
  }();
  static const bool lazyClear2 = [] { const char *e = std::getenv("DELTA_GPU_LAZYCLEAR");
    return !e || std::strcmp(e, "0") != 0; }();
  if (noWipe && d.recomp->psTexs.empty() && nv <= 8) {
    uint32_t cdst = (d.blendControl >> 8) & 0x1F, csrc = d.blendControl & 0x1F;
    bool replace = d.blendEnable && csrc == 1 && cdst == 0;
    if (replace) {
      const auto *vb = static_cast<const uint8_t *>(d.vertexData);
      bool nearBlack = true;
      float clearColor[4] = {0, 0, 0, 0};
      for (uint32_t a = 0; a < d.nvattrs; a++) {
        if (d.vattrs[a].numComps == 4 && d.vattrs[a].offset != 0) {
          const uint8_t *cb0 = vb + d.vattrs[a].offset;  // vertex 0's colour
          if (d.vattrs[a].dfmt == 10) {
            for (int i = 0; i < 4; i++) clearColor[i] = cb0[i] / 255.f;
          } else {
            const float *c = reinterpret_cast<const float *>(cb0);
            for (int i = 0; i < 4; i++) clearColor[i] = c[i];
          }
          if (clearColor[0] > 0.02f || clearColor[1] > 0.02f || clearColor[2] > 0.02f)
            nearBlack = false;
          break;
        }
      }
      const float *m = d.mvp;
      float nx0=1e9f,ny0=1e9f,nx1=-1e9f,ny1=-1e9f;
      for (uint32_t v = 0; v < nv; v++) {
        const float *p = reinterpret_cast<const float *>(vb + (size_t)v * d.vertexStride);
        float cw = m[3]*p[0]+m[7]*p[1]+m[15]; if (cw==0) cw=1;
        float nx=(m[0]*p[0]+m[4]*p[1]+m[12])/cw, ny=(m[1]*p[0]+m[5]*p[1]+m[13])/cw;
        nx0=nx<nx0?nx:nx0; nx1=nx>nx1?nx:nx1; ny0=ny<ny0?ny:ny0; ny1=ny>ny1?ny:ny1;
      }
      bool fullscreenBlack = nearBlack && (nx1-nx0) >= 1.8f && (ny1-ny0) >= 1.8f;
      if (fullscreenBlack && lazyClear2) {
        RTarget *rt = getRT(d.rtBase, d.rtW, d.rtH);  // create if needed
        if (rt) {
          rt->clearPending = true;
          std::memcpy(rt->clearValue.float32, clearColor, sizeof(clearColor));
        }
        // This draw also performs the guest's reverse-Z clear (depth write enabled,
        // ZFUNC=ALWAYS). Suppressing its color write must not discard that depth
        // effect, or stale depth rejects the following layer composites.
        if (d.depthBase && d.depthWriteEnable && d.depthFunc == 7) {
          DepthTarget *dt = getDepthRT(d.depthBase, d.rtW, d.rtH);
          if (dt) {
            dt->clearPending = true;
            dt->clearValue = d.depthClear;
          }
        }
        g.frameDraws++;
        return true;  // suppressed; the clear is applied lazily on the next redraw
      }
      // Legacy single-frame behaviour (DELTA_GPU_LAZYCLEAR=0): only suppress if the
      // RT already holds content this frame.
      auto rit = g_rts.find(d.rtBase);
      if (fullscreenBlack && rit != g_rts.end() && rit->second.draws > 0 &&
          rit->second.lastFrame == g.frameNum) {
        g.frameDraws++;
        return true;
      }
    }
  }

  VkDeviceSize vneed = (VkDeviceSize)nv * d.vertexStride;
  if (g.vbOffset + vneed > kVbRing) return decline(DR_RING);
  if (g.ibOffset + (VkDeviceSize)d.indexCount * 4 > kIbRing) return decline(DR_RING);

  RecompPipe *rp = getRecompPipe(d);
  if (!rp) return decline(DR_NOPIPE);
  // Guest-texture source resolved up front; an RT-as-texture source is resolved after
  // the region switch (transitioning it to readable must happen outside a region).
  VkDescriptorSet texSet = VK_NULL_HANDLE;
  if (rp->textured && !rtAsTex) {
    if (rp->multiTex) {
      texSet = getMultiTexSet(d);
    } else {
      texSet = getTexture(d.texBase, d.texW, d.texH, d.texTiling,
                           d.texPitch, d.texLayers, d.texBaseArray,
                           d.texViewLayers, d.texMipLevels, d.texBaseMip,
                           d.texViewMips, d.texMinLod, d.texPow2Pad, d.texSampler,
                          d.texSamplerValid, d.texArrayed);
      if (!texSet) texSet = d.texArrayed ? g.whiteArraySet : g.whiteSet;
    }
    if (!texSet) return decline(DR_GUESTTEX);
  }

  // Copy the raw interleaved vertex buffer and the indices into the rings.
  VkDeviceSize voff = g.vbOffset, ioff = g.ibOffset;
  std::memcpy(g.vbMap + voff, d.vertexData, (size_t)vneed);
  auto *idst = reinterpret_cast<uint32_t *>(g.ibMap + ioff);
  if (i32) std::memcpy(idst, i32, (size_t)d.indexCount * 4);
  else for (uint32_t i = 0; i < d.indexCount; i++) idst[i] = i16[i];

  // Switch render target. Re-begin when the primary target or the MRT count changes
  // (the open region's attachment count must match the pipeline's), or when a new
  // RT-as-texture source still needs a read transition. Barriers cannot be recorded
  // inside dynamic rendering; consecutive layer composites often keep the same target
  // while switching sources, so that source change must also close/reopen the region.
  uint32_t mrtN = d.mrtCount ? (d.mrtCount > 8 ? 8 : d.mrtCount) : 1;
  bool transitionSource = rtAsTex &&
      g_rts[texBase].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  bool pendingDepthClear = d.depthBase && g_depths.count(d.depthBase) &&
                           g_depths[d.depthBase].clearPending;
  if (g.curRt != d.rtBase || g.curMrtCount != mrtN || g.curDepth != d.depthBase ||
      transitionSource || pendingDepthClear) {
    endRegion();
    if (transitionSource) {
      auto &src = g_rts[texBase];
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        imageBarrier(g.cmd, src.image, src.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    RTarget *rt = getRT(d.rtBase, d.rtW, d.rtH);
    if (!rt) return true;  // RT cap hit: treat as handled (dropped)
    beginRegion(d.mrtBase, mrtN, d.rtW, d.rtH, d.depthBase, d.depthClear);
  }
  if (rtAsTex) {
    auto &src = g_rts[texBase];
    if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return decline(DR_MIDREGION);
    texSet = src.set;
    if (!texSet) return decline(DR_MIDREGION);
  }

  setGuestViewport(d);
  vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->pipe);
  // Copy each guest cbuffer window into the per-frame ring and bind set 1. Vulkan
  // requires one dynamic offset for every dynamic descriptor in the set layout.
  VkDeviceSize cbOff = (g.uboOffset + g.uboAlign - 1) & ~(VkDeviceSize)(g.uboAlign - 1);
  VkDeviceSize cbStride = (kCbufWindow + g.uboAlign - 1) & ~(VkDeviceSize)(g.uboAlign - 1);
  if (cbOff + cbStride * 8 > kUboRing) return decline(DR_RING);
  uint32_t dynOff[8];
  for (uint32_t i = 0; i < 8; i++) {
    VkDeviceSize bindingOff = cbOff + cbStride * i;
    dynOff[i] = static_cast<uint32_t>(bindingOff);
    uint8_t *cbDst = g.uboMap + bindingOff;
    std::memset(cbDst, 0, kCbufWindow);
    const auto &cb = d.cbufs[i];
    if (cb.base >= 0x1000000000ull && cb.base < 0x20000000000ull) {
      uint32_t n = cb.size < kCbufWindow ? cb.size : kCbufWindow;
      std::memcpy(cbDst, reinterpret_cast<const void *>(cb.base), n);
    } else if (i == 0) {
      std::memcpy(cbDst, d.mvp, sizeof(d.mvp));
    }
  }
  g.uboOffset = cbOff + cbStride * 8;
  vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->layout,
                          1, 1, &g.uboSet, 8, dynOff);
  if (texSet)
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->layout, 0, 1, &texSet, 0, nullptr);
  vkCmdBindVertexBuffers(g.cmd, 0, 1, &g.vb, &voff);
  vkCmdBindIndexBuffer(g.cmd, g.ib, ioff, VK_INDEX_TYPE_UINT32);
  if (drawTrace && d.indexCount >= 300) {
    static int n = 0;
    if (n++ < 40)
      std::fprintf(stderr, "[dt] RECOMP DREW idx=%u rt=%#lx nv=%u multiTex=%d\n",
                   d.indexCount, (unsigned long)d.rtBase, nv, rp->multiTex);
  }
  vkCmdDrawIndexed(g.cmd, d.indexCount, d.instanceCount ? d.instanceCount : 1, 0, 0, 0);
  g.vbOffset += vneed;
  g.ibOffset += (VkDeviceSize)d.indexCount * 4;
  g.frameDraws++;
  if (g.curRt) { auto &rt = g_rts[g.curRt];
    if (++rt.draws > g.busiestRtDraws) { g.busiestRtDraws = rt.draws; g.busiestRt = g.curRt; } }
  return true;
}

void beginFrame() {
  if (!g.ready) return;
  // endFrame waits for the prior frame's fence, so objects invalidated while that
  // command buffer was recording are safe to release now.
  releaseRetiredTextures();
  if (!createPipeline()) return;
  createTexPipeline();  // best-effort; colored path still works without it
  g.frameDraws = 0;
  g.frameHeuristic = 0;
  g.frameMaxIdx = 0;
  g.frameNum++;
  g.vbOffset = 0;
  g.ibOffset = 0;
  g.uboOffset = 0;
  g.curRt = 0;
  g.curDepth = 0;
  g.lastRt = 0;
  g.busiestRt = 0;
  g.busiestRtDraws = 0;
  g.frameHadRoom = false;
  g.frameRoomBake = false;
  for (auto &kv : g_rts) {
    kv.second.usedThisFrame = false;
    kv.second.draws = 0;
    // An orphaned lazy clear must not wipe persistent content when an unrelated
    // incremental draw touches this RT in a later frame.
    kv.second.clearPending = false;
  }
  for (auto &kv : g_depths)
    kv.second.usedThisFrame = false;

  vkResetCommandBuffer(g.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g.cmd, &bi);
  g.recording = true;
}

void draw(const DrawInfo &d) {
  if (!g.recording || !d.vertexData || !d.vertexStride)
    return;
  if (d.indexCount > g.frameMaxIdx) g.frameMaxIdx = d.indexCount;
  ScopeNs _t(&g_nsDraw);
  // Recompiled-shader path: run the game's actual VS/PS. Falls through to the
  // heuristic quad path when the draw can't be handled. On by default now that it
  // renders gameplay correctly; DELTA_GPU_RECOMP=0 forces the old heuristic path.
  static const bool recompPath = [] {
    const char *e = std::getenv("DELTA_GPU_RECOMP");
    return !e || std::strcmp(e, "0") != 0;
  }();
  if (recompPath && d.recomp && drawRecomp(d))
    return;
  // Indexed triangle list (the common GNM draw): the index buffer selects which
  // vertices form each triangle. Find how many vertices the indices reference so
  // we repack exactly that many (the V# num_records can be the whole shared batch).
  const uint16_t *idx16 = nullptr;
  const uint32_t *idx32 = nullptr;
  bool indexed = d.indexData && d.indexCount >= 3;
  uint32_t nv = d.vertexCount;
  if (indexed) {
    if (d.indexCount > 1500000u) return;
    uint32_t maxIdx = 0;
    if (d.indexType == 1) {
      idx32 = static_cast<const uint32_t *>(d.indexData);
      for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = idx32[i] > maxIdx ? idx32[i] : maxIdx;
    } else {
      idx16 = static_cast<const uint16_t *>(d.indexData);
      for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = idx16[i] > maxIdx ? idx16[i] : maxIdx;
    }
    nv = maxIdx + 1;
  }
  if (nv < 3 || nv > 200000u) return;  // sane cap
  VkDeviceSize need = (VkDeviceSize)nv * 32;  // pos.xy + color.rgba + uv.xy
  if (g.vbOffset + need > kVbRing)
    return;  // ring full this frame
  if (indexed && g.ibOffset + (VkDeviceSize)d.indexCount * 4 > kIbRing)
    return;
  // Repack pos / color / uv interleaved into the vertex ring (stride 32).
  auto *base = static_cast<const uint8_t *>(d.vertexData);
  auto *dst = reinterpret_cast<float *>(g.vbMap + g.vbOffset);
  for (uint32_t v = 0; v < nv; v++) {
    const uint8_t *vert = base + (size_t)v * d.vertexStride;
    auto *p = reinterpret_cast<const float *>(vert + d.posOffset);
    dst[v * 8 + 0] = p[0];
    dst[v * 8 + 1] = p[1];
    if (d.colorOffset != 0xFFFFFFFFu) {
      auto *c = reinterpret_cast<const float *>(vert + d.colorOffset);
      dst[v * 8 + 2] = c[0];
      dst[v * 8 + 3] = c[1];
      dst[v * 8 + 4] = c[2];
      dst[v * 8 + 5] = 1.0f;
    } else {
      dst[v * 8 + 2] = dst[v * 8 + 3] = dst[v * 8 + 4] = dst[v * 8 + 5] = 1.0f;
    }
    if (d.uvData && d.uvStride) {
      auto *u = reinterpret_cast<const float *>(vert + d.uvOffset);
      dst[v * 8 + 6] = u[0];
      dst[v * 8 + 7] = u[1];
    } else {
      dst[v * 8 + 6] = dst[v * 8 + 7] = 0.0f;
    }
  }

  // Resolve the sampled texture address to a render target via overlap (the
  // resource-model page-table lookup): an exact RT base, or an address whose
  // footprint overlaps a live RT, binds that RT's image instead of stale guest
  // memory. This replaces the old per-symptom FRESHRT/CYCLEREDIR/ROOMALPHA address
  // heuristics with one principled, game-agnostic lookup.
  uint64_t texBase = d.texBase;
  if (texBase && !d.texArrayed && !g_rts.count(texBase)) {
    uint64_t r = resolveSampledRT(texBase, d.texW, d.texH);
    if (r) texBase = r;
  }
  // Is this a render-to-texture sample (the draw samples another render target)?
  bool rtAsTex = !d.texArrayed && texBase && texBase != d.rtBase && g_rts.count(texBase);
  bool roomSrc = rtAsTex && g_rts[texBase].w >= 700 && g_rts[texBase].w <= 900;
  if (roomSrc) g.frameHadRoom = true;

  // Upload guest texture (independent of the render region) if not RT-as-texture.
  VkDescriptorSet texSet = VK_NULL_HANDLE;
  if (d.texBase && g.texPipeline && !rtAsTex && !d.texArrayed)
    texSet = getTexture(d.texBase, d.texW, d.texH, d.texTiling, d.texPitch,
                         d.texLayers, d.texBaseArray, d.texViewLayers,
                         d.texMipLevels, d.texBaseMip, d.texViewMips,
                         d.texMinLod, d.texPow2Pad, d.texSampler,
                         d.texSamplerValid, false);

  // Switch render target if this draw targets a different RT than the open region (or
  // the open region is multi-target/has a depth attachment: the heuristic path renders
  // to a single color attachment with no depth).
  if (g.curRt != d.rtBase || g.curMrtCount != 1 || g.curDepth != 0) {
    endRegion();
    if (rtAsTex) {  // make the sampled RT shader-readable before we render
      auto &src = g_rts[texBase];
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        imageBarrier(g.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    RTarget *rt = getRT(d.rtBase, d.rtW, d.rtH);
    if (!rt) { g.frameDraws++; return; }
    beginRegion(d.mrtBase, 1, d.rtW, d.rtH);  // heuristic path is single-RT
  }
  if (rtAsTex && g_rts[texBase].layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    texSet = g_rts[texBase].set;

  g.frameHeuristic++;
  setGuestViewport(d);
  VkDeviceSize off = g.vbOffset;
  if (texSet) {
    // Per-draw blend from the guest's CB_BLEND0_CONTROL, real vertex UVs.
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      getPipeline(true, d.blendControl, d.blendEnable));
    float pc[17];
    std::memcpy(pc, d.mvp, 64);
    reinterpret_cast<uint32_t *>(pc)[16] = 0u;  // clipUV: real per-vertex uv/colour
    vkCmdPushConstants(g.cmd, g.texLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 68, pc);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.texLayout, 0,
                            1, &texSet, 0, nullptr);
  } else {
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      getPipeline(false, d.blendControl, d.blendEnable));
    vkCmdPushConstants(g.cmd, g.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, d.mvp);
  }
  vkCmdBindVertexBuffers(g.cmd, 0, 1, &g.vb, &off);
  if (indexed) {
    // Widen the guest indices (16- or 32-bit) into the 32-bit index ring and draw.
    VkDeviceSize ioff = g.ibOffset;
    auto *idst = reinterpret_cast<uint32_t *>(g.ibMap + ioff);
    if (idx32)
      std::memcpy(idst, idx32, (size_t)d.indexCount * 4);
    else
      for (uint32_t i = 0; i < d.indexCount; i++) idst[i] = idx16[i];
    vkCmdBindIndexBuffer(g.cmd, g.ib, ioff, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(g.cmd, d.indexCount, d.instanceCount ? d.instanceCount : 1, 0, 0, 0);
    g.ibOffset += (VkDeviceSize)d.indexCount * 4;
  } else {
    vkCmdDraw(g.cmd, nv, d.instanceCount ? d.instanceCount : 1, 0, 0);
  }
  g.vbOffset += need;
  g.frameDraws++;
  if (g.curRt) {
    auto &rt = g_rts[g.curRt];
    if (++rt.draws > g.busiestRtDraws) { g.busiestRtDraws = rt.draws; g.busiestRt = g.curRt; }
  }
}

// Wall-clock FPS report. Always on (cheap): every ~2s of presented frames, log
// the average FPS over that window so perf changes can be measured empirically
// (DELTA_GPU_FPS=0 silences it).
void reportFps() {
  static const bool off = [] { const char *e = std::getenv("DELTA_GPU_FPS"); return e && e[0] == '0'; }();
  if (off) return;
  using clock = std::chrono::steady_clock;
  static auto last = clock::now();
  static int frames = 0;
  frames++;
  auto now = clock::now();
  double dt = std::chrono::duration<double>(now - last).count();
  if (dt >= 2.0) {
    double f = frames ? frames : 1;
    std::fprintf(stderr,
        "[fps] %.1f fps | per-frame gpu-code: draw=%.2fms end=%.2fms (readback=%.2fms) "
        "texup=%.2fms x%.1f\n",
        frames / dt, g_nsDraw / f / 1e6, g_nsEnd / f / 1e6, g_nsReadback / f / 1e6,
        g_nsTexUp / f / 1e6, g_texUps / f);
    last = now;
    frames = 0;
    g_nsDraw = g_nsEnd = g_nsReadback = g_nsTexUp = 0;
    g_texUps = 0;
  }
}

void endFrame(uint64_t scanoutBase) {
  if (!g.ready || !g.recording) return;
  g.recording = false;
  reportFps();
  ScopeNs _t(&g_nsEnd);
  endRegion();  // close any open region

  // Present the scanout RT (the flip buffer); fall back to the last RT rendered.
  uint64_t presentBase = g_rts.count(scanoutBase) ? scanoutBase : g.lastRt;
  // Debug: present the busiest RT (the scene) instead of the composited scanout.
  static const bool presentScene = std::getenv("DELTA_GPU_PRESENT_SCENE") != nullptr;
  if (presentScene && g.busiestRt) presentBase = g.busiestRt;
  // Debug: present the first RT matching DELTA_GPU_PRESENT_RTW x RTH (inspect a
  // specific render target, e.g. the 832x512 room buffer).
  static const int wantW = [] { const char *e = std::getenv("DELTA_GPU_PRESENT_RTW"); return e ? std::atoi(e) : 0; }();
  static const int wantH = [] { const char *e = std::getenv("DELTA_GPU_PRESENT_RTH"); return e ? std::atoi(e) : 0; }();
  if (wantW && wantH) {
    int best = -1000000;  // pick the FRESHEST match (room buffers cycle addresses)
    for (auto &kv : g_rts)
      if ((int)kv.second.w == wantW && (int)kv.second.h == wantH &&
          kv.second.lastFrame > best) { best = kv.second.lastFrame; presentBase = kv.first; }
  }
  // Debug: present a specific RT by guest address (addresses are stable per build).
  static const uint64_t wantAddr = [] {
    const char *e = std::getenv("DELTA_GPU_PRESENT_ADDR");
    return e ? strtoull(e, nullptr, 0) : 0ull;
  }();
  if (wantAddr && g_rts.count(wantAddr)) presentBase = wantAddr;
  auto it = g_rts.find(presentBase);
  if (it == g_rts.end()) {  // nothing rendered this frame
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &g.cmd;
    vkResetFences(g.device, 1, &g.fence);
    vkQueueSubmit(g.queue, 1, &si, g.fence);
    vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
    return;
  }
  RTarget &rt = it->second;
  ensureReadback(rt.w, rt.h);
  imageBarrier(g.cmd, rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {rt.w, rt.h, 1};
  vkCmdCopyImageToBuffer(g.cmd, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         g.readback, 1, &copy);
  vkEndCommandBuffer(g.cmd);

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g.cmd;
  uint64_t _tr0 = nowNs();
  vkResetFences(g.device, 1, &g.fence);
  vkQueueSubmit(g.queue, 1, &si, g.fence);
  vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
  g_nsReadback += nowNs() - _tr0;

  // Readback transform (DELTA_GPU_FLIP: 0=none 1=Y 2=X 3=XY). Default 0 (none): the
  // y-up (negative-height) viewport already stores render-target content upright, and
  // the scene->scanout copy runs the game's real recompiled shader (which samples that
  // upright content correctly), so the presented image needs no flip. (The old default
  // Y-flip existed only to undo the heuristic composite's upside-down output.)
  static const int flipMode = [] {
    const char *e = std::getenv("DELTA_GPU_FLIP");
    return e ? std::atoi(e) : 0;
  }();
  static std::vector<uint8_t> flipped;
  flipped.resize(static_cast<size_t>(rt.w) * rt.h * 4);
  const auto *rb = static_cast<const uint8_t *>(g.readbackMap);
  for (uint32_t y = 0; y < rt.h; y++) {
    uint32_t sy = (flipMode & 1) ? (rt.h - 1 - y) : y;
    const uint32_t *srow = reinterpret_cast<const uint32_t *>(rb + (size_t)sy * rt.w * 4);
    uint32_t *drow = reinterpret_cast<uint32_t *>(flipped.data() + (size_t)y * rt.w * 4);
    if (flipMode & 2)
      for (uint32_t x = 0; x < rt.w; x++) drow[x] = srow[rt.w - 1 - x];
    else
      std::memcpy(drow, srow, (size_t)rt.w * 4);
  }
  const uint8_t *pixels = flipped.data();
  // Minimal single-shot capture (DELTA_GPU_SNAP=N): write ONE ppm of the presented
  // scanout to <dumpdir>/gpu_snap.ppm at the first drawing frame >= N, then never
  // again. For verifying gfx without the rolling DELTA_GPU_DUMP firehose (hundreds
  // of MB per run). DELTA_GPU_SNAP_ROOM=1 waits for a gameplay-room frame.
  static const int snapAt = [] { const char *e = std::getenv("DELTA_GPU_SNAP"); return e ? std::atoi(e) : 0; }();
  static const bool snapRoom = std::getenv("DELTA_GPU_SNAP_ROOM") != nullptr;
  // Wait for a frame with at least this many draws before capturing, so a busy
  // scene frame is grabbed instead of a sparse HUD/transition frame (e.g. Doom64
  // gameplay where only some frames carry the full level geometry).
  static const int snapMinDraws = [] { const char *e = std::getenv("DELTA_GPU_SNAP_MINDRAWS"); return e ? std::atoi(e) : 0; }();
  // DELTA_GPU_SNAP_MININDICES: require a frame to contain a draw with at least this
  // many indices (3D level geometry, e.g. Doom64 with ~2400-index draws) instead of
  // counting draws -- a level frame can have few draws but huge index counts that a
  // draw-count gate (snapMinDraws/snapBest) misses.
  static const int snapMinIdx = [] { const char *e = std::getenv("DELTA_GPU_SNAP_MININDICES"); return e ? std::atoi(e) : 0; }();
  // DELTA_GPU_SNAP_BEST: instead of capturing the first qualifying frame, keep
  // re-capturing whenever this frame has more draws than any seen so far (after
  // snapAt). The final gpu_snap.ppm is then the busiest frame of the run -- a real
  // scene frame, not a sparse HUD/transition one, without guessing a frame number.
  static const bool snapBest = std::getenv("DELTA_GPU_SNAP_BEST") != nullptr;
  static int snapBestDraws = 0;
  static bool snapped = false;
  bool snapNow = snapAt && g.frameNum >= snapAt && g.frameDraws > 0 &&
                 (int)g.frameDraws >= snapMinDraws &&
                 (int)g.frameMaxIdx >= snapMinIdx &&
                 (!snapRoom || (g.frameHadRoom && g.frameDraws > 20));
  // With a min-indices gate, "best" tracks the largest index count seen (the busiest
  // 3D frame) rather than the draw count.
  if (snapBest && snapMinIdx)
    snapNow = snapNow && (int)g.frameMaxIdx > snapBestDraws;
  else if (snapBest)
    snapNow = snapNow && (int)g.frameDraws > snapBestDraws;
  else
    snapNow = snapNow && !snapped;
  if (snapNow) {
    snapBestDraws = snapMinIdx ? (int)g.frameMaxIdx : (int)g.frameDraws;
    char p[256]; std::snprintf(p, sizeof p, "%s/gpu_snap.ppm", dumpDir());
    writePpm(p, pixels, rt.w, rt.h);
    std::fprintf(stderr, "[snap] wrote %s (f%d %ux%u draws=%u)\n", p, g.frameNum, rt.w, rt.h, g.frameDraws);
    snapped = true;
  }
  // Sequence capture (DELTA_GPU_SNAPSEQ=K): write up to K numbered gameplay-room
  // frames, one every ~250 frames, to seq_NN.ppm. Bounded (K*6MB, cleaned up after);
  // lets a long explore run be inspected for non-start rooms without the firehose.
  static const int snapSeqN = [] { const char *e = std::getenv("DELTA_GPU_SNAPSEQ"); return e ? std::atoi(e) : 0; }();
  static int seqDone = 0, seqLastFrame = -10000;
  if (snapSeqN && seqDone < snapSeqN && g.frameHadRoom && g.frameDraws > 20 &&
      g.frameNum - seqLastFrame >= 250) {
    char p[256]; std::snprintf(p, sizeof p, "%s/seq_%02d.ppm", dumpDir(), seqDone);
    writePpm(p, pixels, rt.w, rt.h);
    std::fprintf(stderr, "[snapseq] %d -> f%d draws=%u\n", seqDone, g.frameNum, g.frameDraws);
    seqDone++; seqLastFrame = g.frameNum;
  }

  // Latch the gameplay signal once a run is clearly underway (sustained room
  // frames with real draw counts) so the headless autoskip stops opening menus
  // and stays in the run instead of bouncing back out via the pause menu.
  static int roomStreak = 0;
  if (g.frameHadRoom && g.frameDraws > 20 && ++roomStreak >= 4)
    gfx::setInGameplay(true);  // latch fast, before the autoskip re-pauses
  // A frame with a huge index count is 3D level geometry (Doom64: ~2400-index
  // draws; 2D titles stay well under this). Latch gameplay so the input autoskip/
  // sweep stops mashing menus and the loaded level stays stable for capture.
  if (g.frameMaxIdx >= 1500)
    gfx::setInGameplay(true);

  // Deterministic room capture: whenever this frame sampled a room RT, roll the
  // presented image to /tmp/gpu_room.ppm (atomic). The last write is guaranteed a
  // gameplay frame regardless of when the flaky autoskip enters/leaves a run. Skip
  // sparse transition frames (few draws) so the capture is representative gameplay.
  if (g_dump && g.frameHadRoom && g.frameDraws > 20) {
    char p[256], tmp[256];
    std::snprintf(p, sizeof(p), "%s/gpu_room.ppm", dumpDir());
    std::snprintf(tmp, sizeof(tmp), "%s/gpu_room.tmp", dumpDir());
    writePpm(tmp, pixels, rt.w, rt.h);
    std::rename(tmp, p);
  }
  if (g_dump && g.frameNum >= 1000 && g.frameNum % 2000 == 0 && g.frameDraws > 0)
    dumpPpm(pixels, rt.w, rt.h);
  // Rolling latest-frame capture (uncapped) so late transitions (menu/gameplay)
  // can be inspected from a long headless run without knowing the frame number.
  static const int latestEvery = [] { const char *e = std::getenv("DELTA_GPU_LATEST_EVERY");
    return e ? std::atoi(e) : 300; }();
  if (g_dump && g.frameNum % latestEvery == 0 && g.frameDraws > 0) {
    char latest[256];
    std::snprintf(latest, sizeof(latest), "%s/gpu_latest.ppm", dumpDir());
    writePpm(latest, pixels, rt.w, rt.h);
  }
  if (g_dump && g.frameNum % 200 == 0) {
    std::fprintf(stderr, "[gpuvk] frame %d draws=%u heuristic=%u rt=%#lx %ux%u  scanout=%#lx\n",
                 g.frameNum, g.frameDraws, g.frameHeuristic, (unsigned long)presentBase, rt.w, rt.h,
                 (unsigned long)scanoutBase);
    std::fprintf(stderr, "[gpuvk]   decline:");
    for (int i = 0; i < DR_MAX; i++)
      if (g_decline[i]) std::fprintf(stderr, " %s=%u", kDeclineName[i], g_decline[i]);
    std::fprintf(stderr, "\n");
    for (auto &kv : g_rts)
      if (kv.second.usedThisFrame)
        std::fprintf(stderr, "[gpuvk]    RT %#lx %ux%u draws=%u%s\n",
                     (unsigned long)kv.first, kv.second.w, kv.second.h,
                     kv.second.draws,
                     kv.first == scanoutBase ? " <-SCANOUT" : "");
  }
  // Present the rendered scanout into the window the VideoOut HLE opened. When
  // there is no display (headless) the window was never created, so we skip
  // present and rely on the readback/PPM path. DELTA_GPU_NOPRESENT forces that
  // headless path even on a display.
  static const bool noPresent = std::getenv("DELTA_GPU_NOPRESENT") != nullptr;
  // Bring the window up on the first presentable frame: the videoout HLE only
  // creates it from its own scanout-present path, which the GPU (Gnm) title
  // never takes, so the renderer owns window creation here. ensure() is
  // idempotent and runs on this (the presenting) thread.
  if (!noPresent && gfx::ensure("prosperity", rt.w, rt.h) && gfx::pumpEvents())
    gfx::present(pixels, rt.w, rt.h, rt.w * 4, gfx::PixelFormat::bgra8);
}

}  // namespace gpu::vk
