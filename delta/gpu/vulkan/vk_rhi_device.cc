/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cstdio>
#include <cstring>

#include <unistd.h>
#include "base/logging.h"

#include "base/algorithm.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "gpu/gpu_check.h"
#include "gpu/gpu_perf.h"
#include "gpu/vulkan/vk_rhi_internal.h"
#include "profile/profile.h"
#include "ui/shader_activity.h"

namespace gpu::vk {

using namespace rhi_impl;

namespace rhi_impl {

namespace {

constexpr VkDeviceSize kImageBlockSize = 256ull * 1024 * 1024;
constexpr u32 kSetsPerPool = 1024;

// Forwards a validation message with the command label stack that names the
// guest work which provoked it.
VKAPI_ATTR VkBool32 VKAPI_CALL
MessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                VkDebugUtilsMessageTypeFlagsEXT,
                const VkDebugUtilsMessengerCallbackDataEXT* data,
                void* user) {
  using Sink = void (*)(const char*, const char*, const char*, const char*);
  const char* level =
      severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT     ? "error"
      : severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "warning"
                                                                   : "info";
  base::String labels;
  for (u32 i = 0; i < data->cmdBufLabelCount; i++) {
    if (!labels.empty())
      labels += " > ";
    labels += data->pCmdBufLabels[i].pLabelName;
  }
  reinterpret_cast<Sink>(user)(
      level, data->pMessageIdName ? data->pMessageIdName : "?", labels.c_str(),
      data->pMessage ? data->pMessage : "");
  return VK_FALSE;
}

bool HasExtension(const base::Vector<VkExtensionProperties>& exts,
                  const char* name) {
  for (const auto& e : exts)
    if (!std::strcmp(e.extensionName, name))
      return true;
  return false;
}

base::Vector<u8> ReadFile(const base::String& path) {
  base::Vector<u8> out;
  if (path.empty())
    return out;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    return out;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    if (std::fread(out.data(), 1, out.size(), f) != out.size())
      out.clear();
  }
  std::fclose(f);
  return out;
}

VkFilter ToVkFilter(rhi::Filter f) {
  return f == rhi::Filter::kLinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

VkSamplerAddressMode ToVkAddress(rhi::AddressMode m) {
  switch (m) {
    case rhi::AddressMode::kRepeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case rhi::AddressMode::kMirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case rhi::AddressMode::kClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case rhi::AddressMode::kClampToBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case rhi::AddressMode::kMirrorClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
  }
  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
}

VkComponentSwizzle ToVkSwizzle(rhi::Swizzle s) {
  switch (s) {
    case rhi::Swizzle::kIdentity:
      return VK_COMPONENT_SWIZZLE_IDENTITY;
    case rhi::Swizzle::kZero:
      return VK_COMPONENT_SWIZZLE_ZERO;
    case rhi::Swizzle::kOne:
      return VK_COMPONENT_SWIZZLE_ONE;
    case rhi::Swizzle::kR:
      return VK_COMPONENT_SWIZZLE_R;
    case rhi::Swizzle::kG:
      return VK_COMPONENT_SWIZZLE_G;
    case rhi::Swizzle::kB:
      return VK_COMPONENT_SWIZZLE_B;
    case rhi::Swizzle::kA:
      return VK_COMPONENT_SWIZZLE_A;
  }
  return VK_COMPONENT_SWIZZLE_IDENTITY;
}

VkImageViewType ToVkViewType(rhi::ViewDim dim) {
  switch (dim) {
    case rhi::ViewDim::k1D:
      return VK_IMAGE_VIEW_TYPE_1D;
    case rhi::ViewDim::k1DArray:
      return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case rhi::ViewDim::k2D:
      return VK_IMAGE_VIEW_TYPE_2D;
    case rhi::ViewDim::k2DArray:
      return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case rhi::ViewDim::kCube:
      return VK_IMAGE_VIEW_TYPE_CUBE;
    case rhi::ViewDim::kCubeArray:
      return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case rhi::ViewDim::k3D:
      return VK_IMAGE_VIEW_TYPE_3D;
  }
  return VK_IMAGE_VIEW_TYPE_2D;
}

VkCompareOp ToVkCompare(rhi::CompareOp op) {
  return static_cast<VkCompareOp>(op);  // same order as VkCompareOp
}

VkBlendFactor ToVkBlendFactor(rhi::BlendFactor f) {
  switch (f) {
    case rhi::BlendFactor::kZero:
      return VK_BLEND_FACTOR_ZERO;
    case rhi::BlendFactor::kOne:
      return VK_BLEND_FACTOR_ONE;
    case rhi::BlendFactor::kSrcColor:
      return VK_BLEND_FACTOR_SRC_COLOR;
    case rhi::BlendFactor::kOneMinusSrcColor:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case rhi::BlendFactor::kDstColor:
      return VK_BLEND_FACTOR_DST_COLOR;
    case rhi::BlendFactor::kOneMinusDstColor:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case rhi::BlendFactor::kSrcAlpha:
      return VK_BLEND_FACTOR_SRC_ALPHA;
    case rhi::BlendFactor::kOneMinusSrcAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case rhi::BlendFactor::kDstAlpha:
      return VK_BLEND_FACTOR_DST_ALPHA;
    case rhi::BlendFactor::kOneMinusDstAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case rhi::BlendFactor::kConstantColor:
      return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case rhi::BlendFactor::kOneMinusConstantColor:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case rhi::BlendFactor::kConstantAlpha:
      return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case rhi::BlendFactor::kOneMinusConstantAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    case rhi::BlendFactor::kSrcAlphaSaturate:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case rhi::BlendFactor::kSrc1Color:
      return VK_BLEND_FACTOR_SRC1_COLOR;
    case rhi::BlendFactor::kOneMinusSrc1Color:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case rhi::BlendFactor::kSrc1Alpha:
      return VK_BLEND_FACTOR_SRC1_ALPHA;
    case rhi::BlendFactor::kOneMinusSrc1Alpha:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
  }
  return VK_BLEND_FACTOR_ONE;
}

VkBlendOp ToVkBlendOp(rhi::BlendOp op) {
  switch (op) {
    case rhi::BlendOp::kAdd:
      return VK_BLEND_OP_ADD;
    case rhi::BlendOp::kSubtract:
      return VK_BLEND_OP_SUBTRACT;
    case rhi::BlendOp::kReverseSubtract:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
    case rhi::BlendOp::kMin:
      return VK_BLEND_OP_MIN;
    case rhi::BlendOp::kMax:
      return VK_BLEND_OP_MAX;
  }
  return VK_BLEND_OP_ADD;
}

VkStencilOp ToVkStencilOp(rhi::StencilOp op) {
  return static_cast<VkStencilOp>(op);  // same order as VkStencilOp
}

VkStencilOpState ToVkStencil(const rhi::StencilFace& f) {
  VkStencilOpState s{};
  s.failOp = ToVkStencilOp(f.fail);
  s.passOp = ToVkStencilOp(f.pass);
  s.depthFailOp = ToVkStencilOp(f.depth_fail);
  s.compareOp = ToVkCompare(f.compare);
  s.compareMask = f.compare_mask;
  s.writeMask = f.write_mask;
  s.reference = f.reference;
  return s;
}

VkPrimitiveTopology ToVkTopology(rhi::Topology t) {
  switch (t) {
    case rhi::Topology::kPointList:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case rhi::Topology::kLineList:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case rhi::Topology::kLineStrip:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case rhi::Topology::kTriangleList:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case rhi::Topology::kTriangleStrip:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case rhi::Topology::kTriangleFan:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case rhi::Topology::kLineListAdjacency:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case rhi::Topology::kTriangleListAdjacency:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
  }
  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

VkShaderModule MakeModule(VkDevice device, const rhi::ShaderCode& code) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = code.count * 4;
  ci.pCode = code.words;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(device, &ci, nullptr, &m);
  return m;
}

}  // namespace

VkShaderStageFlags ToVkStages(u32 stages) {
  VkShaderStageFlags out = 0;
  if (stages & rhi::kStageVertex)
    out |= VK_SHADER_STAGE_VERTEX_BIT;
  if (stages & rhi::kStageGeometry)
    out |= VK_SHADER_STAGE_GEOMETRY_BIT;
  if (stages & rhi::kStageFragment)
    out |= VK_SHADER_STAGE_FRAGMENT_BIT;
  if (stages & rhi::kStageCompute)
    out |= VK_SHADER_STAGE_COMPUTE_BIT;
  if (stages & rhi::kStageMesh)
    out |= VK_SHADER_STAGE_MESH_BIT_EXT;
  return out;
}

VkDescriptorType ToVkDescriptorType(rhi::BindingType type) {
  switch (type) {
    case rhi::BindingType::kSampledTexture:
      return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case rhi::BindingType::kStorageTexture:
      return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case rhi::BindingType::kUniformBuffer:
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case rhi::BindingType::kStorageBuffer:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case rhi::BindingType::kUniformBufferDynamic:
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    case rhi::BindingType::kStorageBufferDynamic:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
  }
  return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

VkImageAspectFlags ToVkAspect(u8 aspect) {
  VkImageAspectFlags out = 0;
  if (aspect & rhi::kAspectColor)
    out |= VK_IMAGE_ASPECT_COLOR_BIT;
  if (aspect & rhi::kAspectDepth)
    out |= VK_IMAGE_ASPECT_DEPTH_BIT;
  if (aspect & rhi::kAspectStencil)
    out |= VK_IMAGE_ASPECT_STENCIL_BIT;
  return out;
}

bool ImageAllocator::Allocate(VulkanDevice& device,
                              VkImage image,
                              Allocation& out) {
  VkDevice dev = device.native.device;
  VkMemoryDedicatedRequirements dedicated{
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
  VkMemoryRequirements2 req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
  req.pNext = &dedicated;
  VkImageMemoryRequirementsInfo2 info{
      VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
  info.image = image;
  vkGetImageMemoryRequirements2(dev, &info, &req);
  const VkMemoryRequirements& mr = req.memoryRequirements;
  GPU_BUGCHECK(mr.memoryTypeBits != 0, "image reports no memory types");
  u32 type = device.FindMemoryType(mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (type == UINT32_MAX)
    type = device.FindMemoryType(mr.memoryTypeBits, 0);

  if (!dedicated.requiresDedicatedAllocation &&
      mr.size <= kImageBlockSize / 2) {
    base::LockGuard<base::Mutex> lock(mutex_);
    auto try_block = [&](Block& block) {
      u64 offset = 0;
      if (block.type != type ||
          !block.spans.Allocate(mr.size, mr.alignment, offset))
        return false;
      if (vkBindImageMemory(dev, image, block.memory, offset) != VK_SUCCESS) {
        block.spans.Free(offset, mr.size);
        return false;
      }
      out = {block.memory, offset, mr.size, type, false};
      return true;
    };
    for (auto& block : blocks_)
      if (try_block(block))
        return true;
    Block block;
    block.type = type;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = kImageBlockSize;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(dev, &ai, nullptr, &block.memory) == VK_SUCCESS) {
      block.spans.Reset(kImageBlockSize);
      blocks_.push_back(base::move(block));
      if (try_block(blocks_.back()))
        return true;
    }
  }

  VkMemoryDedicatedAllocateInfo di{
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  di.image = image;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &di;
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = type;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (vkAllocateMemory(dev, &ai, nullptr, &memory) != VK_SUCCESS)
    return false;
  if (vkBindImageMemory(dev, image, memory, 0) != VK_SUCCESS) {
    vkFreeMemory(dev, memory, nullptr);
    return false;
  }
  out = {memory, 0, mr.size, type, true};
  return true;
}

void ImageAllocator::Free(VulkanDevice& device, Allocation& allocation) {
  if (!allocation.memory)
    return;
  if (allocation.dedicated) {
    vkFreeMemory(device.native.device, allocation.memory, nullptr);
  } else {
    base::LockGuard<base::Mutex> lock(mutex_);
    bool found = false;
    for (auto& block : blocks_)
      if (block.memory == allocation.memory) {
        block.spans.Free(allocation.offset, allocation.size);
        found = true;
        break;
      }
    GPU_BUGCHECK(found, "image allocation freed against an unknown block");
  }
  allocation = {};
}

VulkanDevice::~VulkanDevice() {
  if (!native.device)
    return;
  vkDeviceWaitIdle(native.device);
  SavePipelineCache(true);
  for (VkDescriptorPool pool : set_pools_)
    vkDestroyDescriptorPool(native.device, pool, nullptr);
  if (timeline_)
    vkDestroySemaphore(native.device, timeline_, nullptr);
  if (command_pool)
    vkDestroyCommandPool(native.device, command_pool, nullptr);
  if (side_fence_)
    vkDestroyFence(native.device, side_fence_, nullptr);
  if (side_pool_)
    vkDestroyCommandPool(native.device, side_pool_, nullptr);
  if (native.pipeline_cache)
    vkDestroyPipelineCache(native.device, native.pipeline_cache, nullptr);
  // Device memory blocks and the device itself are left to process exit:
  // the renderer owns objects that may still reference them.
}

u32 VulkanDevice::FindMemoryType(u32 bits, VkMemoryPropertyFlags want) const {
  for (u32 i = 0; i < memory_properties.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (memory_properties.memoryTypes[i].propertyFlags & want) == want)
      return i;
  return UINT32_MAX;
}

bool VulkanDevice::Init(const VulkanOptions& options) {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  app.pApplicationName = "prosperity-gpu";
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ic.pApplicationInfo = &app;
  bool debug_utils = false;
  if (options.debug_utils || options.validation_layer) {
    u32 n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    base::Vector<VkExtensionProperties> exts(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
    debug_utils = HasExtension(exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  const char* instance_exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  if (debug_utils) {
    ic.enabledExtensionCount = 1;
    ic.ppEnabledExtensionNames = instance_exts;
  }
  const char* layer = options.validation_layer;
  VkValidationFeaturesEXT vf{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  static const VkValidationFeatureEnableEXT kSyncFeature =
      VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
  if (layer) {
    u32 n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    base::Vector<VkLayerProperties> layers(n);
    vkEnumerateInstanceLayerProperties(&n, layers.data());
    bool found = false;
    for (const auto& l : layers)
      found |= !std::strcmp(l.layerName, layer);
    if (found) {
      ic.enabledLayerCount = 1;
      ic.ppEnabledLayerNames = &layer;
      if (options.sync_validation) {
        vf.enabledValidationFeatureCount = 1;
        vf.pEnabledValidationFeatures = &kSyncFeature;
        ic.pNext = &vf;
      }
    } else {
      BASE_LOGI("vkval", "{} not available on this loader", layer);
    }
  }
  if (vkCreateInstance(&ic, nullptr, &native.instance) != VK_SUCCESS) {
    BASE_LOGI("gpuvk", "vkCreateInstance failed");
    return false;
  }
  if (debug_utils && options.debug_utils) {
    VkInstance inst = native.instance;
    set_object_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetInstanceProcAddr(inst, "vkSetDebugUtilsObjectNameEXT"));
    cmd_begin_label = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(inst, "vkCmdBeginDebugUtilsLabelEXT"));
    cmd_end_label = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(inst, "vkCmdEndDebugUtilsLabelEXT"));
    cmd_insert_label = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(inst, "vkCmdInsertDebugUtilsLabelEXT"));
    caps_.debug_labels = set_object_name != nullptr;
  }
  if (options.on_message && ic.enabledLayerCount && debug_utils) {
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(native.instance,
                              "vkCreateDebugUtilsMessengerEXT"));
    VkDebugUtilsMessengerCreateInfoEXT mi{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    mi.pfnUserCallback = MessageCallback;
    mi.pUserData = reinterpret_cast<void*>(options.on_message);
    if (create)
      create(native.instance, &mi, nullptr, &messenger_);
  }

  // Prefer a real GPU over a software rasteriser: discrete > integrated >
  // virtual > CPU, unless the filter names one.
  u32 n = 0;
  vkEnumeratePhysicalDevices(native.instance, &n, nullptr);
  base::Vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(native.instance, &n, devs.data());
  int best = -1;
  for (VkPhysicalDevice d : devs) {
    u32 qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
    base::Vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qs.data());
    int family = -1;
    for (u32 i = 0; i < qn && family < 0; i++)
      if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        family = static_cast<int>(i);
    if (family < 0)
      continue;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    int score = 1;
    switch (p.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score = 4;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score = 3;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score = 2;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        score = 0;
        break;
      default:
        break;
    }
    if (options.gpu_filter && std::strstr(p.deviceName, options.gpu_filter))
      score = 100;
    if (score > best) {
      best = score;
      native.phys = d;
      native.queue_family = static_cast<u32>(family);
      native.timestamp_valid_bits = qs[family].timestampValidBits;
    }
  }
  if (!native.phys) {
    BASE_LOGI("gpuvk", "no gfx device");
    return false;
  }
  VkPhysicalDevice phys = native.phys;
  vkGetPhysicalDeviceMemoryProperties(phys, &memory_properties);

  VkPhysicalDeviceVulkan12Features avail12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 avail{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  avail.pNext = &avail12;
  vkGetPhysicalDeviceFeatures2(phys, &avail);

  u32 en = 0;
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, nullptr);
  base::Vector<VkExtensionProperties> exts(en);
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, exts.data());

  VkPhysicalDeviceVulkan12Features f12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.samplerMirrorClampToEdge = avail12.samplerMirrorClampToEdge;
  f12.separateDepthStencilLayouts = avail12.separateDepthStencilLayouts;
  f12.timelineSemaphore = VK_TRUE;
  f12.bufferDeviceAddress =
      avail12.bufferDeviceAddress && avail.features.shaderInt64;
  VkPhysicalDeviceVulkan13Features f13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.pNext = &f12;
  f13.dynamicRendering = VK_TRUE;
  void** chain = &f13.pNext;
  (void)chain;

  base::Vector<const char*> dev_exts;
  auto want_ext = [&](const char* name) {
    if (!HasExtension(exts, name))
      return false;
    dev_exts.push_back(name);
    return true;
  };
  native.checkpoints =
      options.checkpoints &&
      want_ext(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
  native.push_descriptor = want_ext(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  caps_.host_import = want_ext(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);

  VkPhysicalDeviceFaultFeaturesEXT fault{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
  if (want_ext(VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
    native.device_fault = true;
    fault.deviceFault = VK_TRUE;
    fault.pNext = f13.pNext;
    f13.pNext = &fault;
  }
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  if (want_ext(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
    caps_.fragment_barycentric = true;
    bary.fragmentShaderBarycentric = VK_TRUE;
    bary.pNext = f13.pNext;
    f13.pNext = &bary;
  }
  VkPhysicalDeviceMeshShaderFeaturesEXT mesh{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
  if (HasExtension(exts, VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
    VkPhysicalDeviceFeatures2 query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    query.pNext = &mesh;
    vkGetPhysicalDeviceFeatures2(phys, &query);
    if (mesh.meshShader) {
      mesh = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
      mesh.meshShader = VK_TRUE;
      mesh.pNext = f13.pNext;
      f13.pNext = &mesh;
      dev_exts.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
      caps_.mesh_shader = true;
      VkPhysicalDeviceProperties2 props{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      props.pNext = &native.mesh_limits;
      vkGetPhysicalDeviceProperties2(phys, &props);
      const auto& ml = native.mesh_limits;
      caps_.mesh.max_threads = ml.maxMeshWorkGroupInvocations;
      caps_.mesh.max_threads_x = ml.maxMeshWorkGroupSize[0];
      caps_.mesh.max_shared_bytes = ml.maxMeshSharedMemorySize;
      caps_.mesh.max_output_vertices = ml.maxMeshOutputVertices;
      caps_.mesh.max_output_primitives = ml.maxMeshOutputPrimitives;
      caps_.mesh.max_groups[0] = ml.maxMeshWorkGroupCount[0];
      caps_.mesh.max_groups[1] = ml.maxMeshWorkGroupCount[1];
      caps_.mesh.max_total_groups = ml.maxMeshWorkGroupTotalCount;
    }
  }

  // robustBufferAccess: out-of-bounds storage accesses from a miscomputed
  // guest index read 0 / drop the write instead of faulting the device.
  VkPhysicalDeviceFeatures f{};
  const VkPhysicalDeviceFeatures& a = avail.features;
  f.shaderInt64 = f12.bufferDeviceAddress;
  f.textureCompressionBC = a.textureCompressionBC;
  f.robustBufferAccess = a.robustBufferAccess;
  f.depthClamp = a.depthClamp;
  f.samplerAnisotropy = a.samplerAnisotropy;
  f.geometryShader = a.geometryShader;
  f.independentBlend = a.independentBlend;
  f.dualSrcBlend = a.dualSrcBlend;
  f.shaderStorageImageWriteWithoutFormat =
      a.shaderStorageImageWriteWithoutFormat;
  f.vertexPipelineStoresAndAtomics = a.vertexPipelineStoresAndAtomics;
  {
    u32 qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
    base::Vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qs.data());
    caps_.sparse_buffer = a.sparseBinding && a.sparseResidencyBuffer &&
                          native.queue_family < qn &&
                          (qs[native.queue_family].queueFlags &
                           VK_QUEUE_SPARSE_BINDING_BIT);
    f.sparseBinding = caps_.sparse_buffer;
    f.sparseResidencyBuffer = caps_.sparse_buffer;
  }
  f.fragmentStoresAndAtomics = a.fragmentStoresAndAtomics;

  const float prios[2] = {1.0f, 1.0f};
  u32 family_queues = 1;
  {
    u32 qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
    base::Vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qs.data());
    if (native.queue_family < qn)
      family_queues = qs[native.queue_family].queueCount;
  }
  VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qc.queueFamilyIndex = native.queue_family;
  qc.queueCount = family_queues >= 2 ? 2 : 1;
  qc.pQueuePriorities = prios;
  VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dc.pNext = &f13;
  dc.queueCreateInfoCount = 1;
  dc.pQueueCreateInfos = &qc;
  dc.enabledExtensionCount = static_cast<u32>(dev_exts.size());
  dc.ppEnabledExtensionNames = dev_exts.data();
  dc.pEnabledFeatures = &f;
  if (vkCreateDevice(phys, &dc, nullptr, &native.device) != VK_SUCCESS) {
    BASE_LOGI("gpuvk", "vkCreateDevice failed");
    return false;
  }
  VkDevice dev = native.device;
  vkGetDeviceQueue(dev, native.queue_family, 0, &native.queue);
  if (qc.queueCount == 2)
    vkGetDeviceQueue(dev, native.queue_family, 1, &side_queue_);

  cmd_begin_rendering = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
      vkGetDeviceProcAddr(dev, "vkCmdBeginRendering"));
  cmd_end_rendering = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
      vkGetDeviceProcAddr(dev, "vkCmdEndRendering"));
  if (!cmd_begin_rendering) {
    cmd_begin_rendering = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
        vkGetDeviceProcAddr(dev, "vkCmdBeginRenderingKHR"));
    cmd_end_rendering = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
        vkGetDeviceProcAddr(dev, "vkCmdEndRenderingKHR"));
  }
  if (!cmd_begin_rendering) {
    BASE_LOGI("gpuvk", "no dynamic rendering");
    return false;
  }
  if (native.push_descriptor) {
    cmd_push_descriptor_set = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(dev, "vkCmdPushDescriptorSetKHR"));
    native.push_descriptor = cmd_push_descriptor_set != nullptr;
  }
  if (caps_.mesh_shader) {
    cmd_draw_mesh_tasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksEXT"));
    caps_.mesh_shader = cmd_draw_mesh_tasks != nullptr;
  }
  if (native.checkpoints)
    cmd_set_checkpoint = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(
        vkGetDeviceProcAddr(dev, "vkCmdSetCheckpointNV"));

  if (options.pipeline_cache_path)
    pipeline_cache_path_ = options.pipeline_cache_path;
  base::Vector<u8> blob = ReadFile(pipeline_cache_path_);
  VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  pci.initialDataSize = blob.size();
  pci.pInitialData = blob.empty() ? nullptr : blob.data();
  if (vkCreatePipelineCache(dev, &pci, nullptr, &native.pipeline_cache) !=
      VK_SUCCESS)
    native.pipeline_cache = VK_NULL_HANDLE;

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpi.queueFamilyIndex = native.queue_family;
  if (vkCreateCommandPool(dev, &cpi, nullptr, &command_pool) != VK_SUCCESS)
    return false;
  VkSemaphoreTypeCreateInfo sti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &sti;
  if (vkCreateSemaphore(dev, &sci, nullptr, &timeline_) != VK_SUCCESS)
    return false;
  if (side_queue_) {
    VkCommandBufferAllocateInfo cai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateCommandPool(dev, &cpi, nullptr, &side_pool_) != VK_SUCCESS ||
        (cai.commandPool = side_pool_, cai.commandBufferCount = 1,
         vkAllocateCommandBuffers(dev, &cai, &side_cmd_) != VK_SUCCESS) ||
        vkCreateFence(dev, &fci, nullptr, &side_fence_) != VK_SUCCESS)
      side_queue_ = VK_NULL_HANDLE;
    caps_.copy_after = side_queue_ != VK_NULL_HANDLE;
  }

  VkPhysicalDeviceSubgroupProperties subgroup{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
  subgroup.pNext = caps_.host_import ? &host : nullptr;
  VkPhysicalDeviceProperties2 props2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(phys, &props2);
  const VkPhysicalDeviceLimits& lim = props2.properties.limits;
  device_name_ = props2.properties.deviceName;

  caps_.backend = rhi::Backend::kVulkan;
  caps_.device_name = device_name_.c_str();
  caps_.geometry_shader = f.geometryShader;
  caps_.independent_blend = f.independentBlend;
  caps_.sampler_anisotropy = f.samplerAnisotropy;
  caps_.sampler_mirror_clamp = f12.samplerMirrorClampToEdge;
  caps_.storage_image_write_without_format =
      f.shaderStorageImageWriteWithoutFormat;
  caps_.buffer_address = f12.bufferDeviceAddress;
  caps_.host_import_alignment = host.minImportedHostPointerAlignment;
  caps_.texture_blit = true;
  caps_.dispatch_base = true;
  caps_.dispatch_indirect = true;
  caps_.timestamps =
      lim.timestampComputeAndGraphics && native.timestamp_valid_bits != 0;
  caps_.timestamp_period_ns = lim.timestampPeriod;
  caps_.timestamp_bits = native.timestamp_valid_bits;
  caps_.subgroup_size = subgroup.subgroupSize ? subgroup.subgroupSize : 32;
  caps_.max_dynamic_uniform_buffers = lim.maxDescriptorSetUniformBuffersDynamic;
  caps_.max_dynamic_storage_buffers = lim.maxDescriptorSetStorageBuffersDynamic;
  caps_.max_storage_buffers_per_stage = lim.maxPerStageDescriptorStorageBuffers;
  caps_.max_push_constant_bytes = lim.maxPushConstantsSize;
  caps_.max_storage_buffer_range = lim.maxStorageBufferRange;
  caps_.uniform_offset_alignment =
      static_cast<u32>(lim.minUniformBufferOffsetAlignment);
  caps_.storage_offset_alignment =
      static_cast<u32>(lim.minStorageBufferOffsetAlignment);
  if (!caps_.storage_offset_alignment)
    caps_.storage_offset_alignment = 1;
  caps_.max_texture_size = lim.maxImageDimension2D;
  caps_.max_texture_size_3d = lim.maxImageDimension3D;
  max_lod_bias_ = lim.maxSamplerLodBias;
  max_anisotropy_ = lim.maxSamplerAnisotropy;
  caps_.max_compute_resources =
      base::Min(lim.maxPerStageDescriptorStorageBuffers,
                lim.maxDescriptorSetStorageBuffers);
  constexpr VkMemoryPropertyFlags kUnified =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  caps_.unified_memory = FindMemoryType(~0u, kUnified) != UINT32_MAX;

  shader_stages_ = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  if (caps_.geometry_shader)
    shader_stages_ |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
  if (caps_.mesh_shader)
    shader_stages_ |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT;
  BASE_LOGI("gpuvk", "device: {}", device_name_.c_str());
  return true;
}

rhi::Buffer* VulkanDevice::CreateBuffer(const rhi::BufferDesc& desc) {
  auto buffer = base::MakeUnique<VulkanBuffer>(desc);
  VkDevice dev = native.device;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = desc.size;
  if (desc.sparse) {
    if (!caps_.sparse_buffer || desc.host_pointer)
      return nullptr;
    bi.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
               VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
  }
  if (desc.usage & rhi::kBufferVertex)
    bi.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (desc.usage & rhi::kBufferIndex)
    bi.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  if (desc.usage & rhi::kBufferUniform)
    bi.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (desc.usage & rhi::kBufferStorage)
    bi.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (desc.usage & rhi::kBufferIndirect)
    bi.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  if (desc.usage & rhi::kBufferCopySrc)
    bi.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (desc.usage & rhi::kBufferCopyDst)
    bi.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  const bool address =
      (desc.usage & rhi::kBufferAddress) && caps_.buffer_address;
  if (address)
    bi.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkExternalMemoryBufferCreateInfo ext{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  if (desc.host_pointer) {
    if (!caps_.host_import)
      return nullptr;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    bi.pNext = &ext;
  }
  if (vkCreateBuffer(dev, &bi, nullptr, &buffer->buffer) != VK_SUCCESS)
    return nullptr;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buffer->buffer, &mr);
  if (desc.sparse) {
    buffer->memory_type =
        FindMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    caps_.sparse_page = mr.alignment;
    if (buffer->memory_type == UINT32_MAX) {
      vkDestroyBuffer(dev, buffer->buffer, nullptr);
      return nullptr;
    }
    if (address) {
      VkBufferDeviceAddressInfo info{
          VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
      info.buffer = buffer->buffer;
      buffer->SetAddress(vkGetBufferDeviceAddress(dev, &info));
    }
    if (desc.name)
      SetName(&*buffer, desc.name);
    return gpu::rhi::Release(buffer);
  }

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  if (address) {
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    ai.pNext = &flags;
  }
  VkImportMemoryHostPointerInfoEXT import{
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
  u32 type = UINT32_MAX;
  constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (desc.host_pointer) {
    static PFN_vkGetMemoryHostPointerPropertiesEXT get_props =
        reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT"));
    VkMemoryHostPointerPropertiesEXT hp{
        VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    u32 bits = mr.memoryTypeBits;
    if (get_props &&
        get_props(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                  desc.host_pointer, &hp) == VK_SUCCESS)
      bits &= hp.memoryTypeBits;
    // Nothing maps or flushes these pages explicitly, so only a coherent
    // type is usable.
    type = FindMemoryType(bits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import.pHostPointer = desc.host_pointer;
    import.pNext = ai.pNext;
    ai.pNext = &import;
    ai.allocationSize = desc.size;
  } else {
    switch (desc.memory) {
      case rhi::MemoryKind::kDevice:
        type = FindMemoryType(mr.memoryTypeBits,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        break;
      case rhi::MemoryKind::kUpload:
        type = FindMemoryType(mr.memoryTypeBits, kHost);
        break;
      case rhi::MemoryKind::kUploadDevice:
        type = FindMemoryType(mr.memoryTypeBits,
                              kHost | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX)
          type = FindMemoryType(mr.memoryTypeBits, kHost);
        break;
      case rhi::MemoryKind::kReadback: {
        // Cached before device-local: the CPU reads these, and an uncached
        // read is ~30x slower.
        int best = -1;
        for (u32 i = 0; i < memory_properties.memoryTypeCount; i++) {
          const VkMemoryPropertyFlags fl =
              memory_properties.memoryTypes[i].propertyFlags;
          if (!(mr.memoryTypeBits & (1u << i)) || (fl & kHost) != kHost)
            continue;
          const int score =
              ((fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 4 : 0) +
              ((fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 2 : 0);
          if (score > best) {
            best = score;
            type = i;
          }
        }
        break;
      }
    }
    if (type == UINT32_MAX)
      type = FindMemoryType(mr.memoryTypeBits, 0);
  }
  ai.memoryTypeIndex = type;
  const VkResult allocated = type == UINT32_MAX
                                 ? VK_ERROR_FEATURE_NOT_PRESENT
                                 : vkAllocateMemory(dev, &ai, nullptr,
                                                    &buffer->memory);
  if (allocated == VK_SUCCESS && desc.host_pointer) {
    static u64 total = 0, count = 0;
    total += desc.size;
    if (++count <= 400 || count % 1000 == 0)
      BASE_LOGI("gpuvk", "host import #{} {:p}+{:#x} (total created {} MiB)",
                count, desc.host_pointer, desc.size, total >> 20);
  }
  if (allocated != VK_SUCCESS && desc.host_pointer) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGI("gpuvk", "host import {:p}+{:#x} failed: VkResult {} type {}",
                desc.host_pointer, desc.size, static_cast<int>(allocated),
                type);
  }
  if (allocated != VK_SUCCESS ||
      vkBindBufferMemory(dev, buffer->buffer, buffer->memory, 0) !=
          VK_SUCCESS) {
    if (buffer->memory)
      vkFreeMemory(dev, buffer->memory, nullptr);
    vkDestroyBuffer(dev, buffer->buffer, nullptr);
    return nullptr;
  }
  if (desc.host_pointer) {
    buffer->SetMapped(static_cast<u8*>(desc.host_pointer));
  } else if (desc.memory != rhi::MemoryKind::kDevice) {
    void* p = nullptr;
    if (vkMapMemory(dev, buffer->memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS)
      buffer->SetMapped(static_cast<u8*>(p));
  }
  if (address) {
    VkBufferDeviceAddressInfo info{
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer->buffer;
    buffer->SetAddress(vkGetBufferDeviceAddress(dev, &info));
  }
  if (desc.name)
    SetName(&*buffer, desc.name);
  return gpu::rhi::Release(buffer);
}

rhi::Texture* VulkanDevice::CreateTexture(const rhi::TextureDesc& desc) {
  auto texture = base::MakeUnique<VulkanTexture>(desc);
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(desc.format);
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = desc.dim == rhi::TextureDim::k1D   ? VK_IMAGE_TYPE_1D
                 : desc.dim == rhi::TextureDim::k3D ? VK_IMAGE_TYPE_3D
                                                    : VK_IMAGE_TYPE_2D;
  ii.format = ToVkFormat(desc.format);
  ii.extent = {desc.width, desc.height,
               desc.dim == rhi::TextureDim::k3D ? desc.depth : 1u};
  ii.mipLevels = desc.mips;
  ii.arrayLayers = desc.dim == rhi::TextureDim::k3D ? 1u : desc.layers;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (desc.usage & rhi::kTextureSampled)
    ii.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  if (desc.usage & rhi::kTextureStorage)
    ii.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  if (desc.usage & rhi::kTextureColorTarget)
    ii.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (desc.usage & rhi::kTextureDepthTarget)
    ii.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (desc.usage & rhi::kTextureCopySrc)
    ii.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (desc.usage & rhi::kTextureCopyDst)
    ii.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (desc.usage & rhi::kTextureMutableFormat)
    ii.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  if (desc.usage & rhi::kTextureCubeCompatible)
    ii.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  if (desc.usage & rhi::kTextureArrayCompatible)
    ii.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
  if (vkCreateImage(native.device, &ii, nullptr, &texture->image) != VK_SUCCESS)
    return nullptr;
  if (!images_.Allocate(*this, texture->image, texture->allocation)) {
    vkDestroyImage(native.device, texture->image, nullptr);
    return nullptr;
  }
  texture->aspects = fi.is_depth || fi.is_stencil
                         ? (fi.is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                               (fi.is_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
                         : VK_IMAGE_ASPECT_COLOR_BIT;
  if (desc.name)
    SetName(&*texture, desc.name);
  return gpu::rhi::Release(texture);
}

rhi::TextureView* VulkanDevice::CreateView(rhi::Texture* texture,
                                           const rhi::TextureViewDesc& desc) {
  auto* tex = static_cast<VulkanTexture*>(texture);
  auto view = base::MakeUnique<VulkanView>(texture, desc);
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = tex->image;
  vi.viewType = ToVkViewType(desc.dim);
  vi.format =
      ToVkFormat(desc.format == rhi::Format::kUndefined ? texture->desc().format
                                                        : desc.format);
  vi.components = {ToVkSwizzle(desc.swizzle[0]), ToVkSwizzle(desc.swizzle[1]),
                   ToVkSwizzle(desc.swizzle[2]), ToVkSwizzle(desc.swizzle[3])};
  vi.subresourceRange = {ToVkAspect(desc.aspect), desc.base_mip, desc.mips,
                         desc.base_layer, desc.layers};
  if (vkCreateImageView(native.device, &vi, nullptr, &view->view) != VK_SUCCESS)
    return nullptr;
  return gpu::rhi::Release(view);
}

rhi::Sampler* VulkanDevice::CreateSampler(const rhi::SamplerDesc& desc) {
  auto sampler = base::MakeUnique<VulkanSampler>();
  VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  si.magFilter = ToVkFilter(desc.mag);
  si.minFilter = ToVkFilter(desc.min);
  si.mipmapMode = desc.mip == rhi::Filter::kLinear
                      ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                      : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  auto address = [&](rhi::AddressMode m) {
    if (m == rhi::AddressMode::kMirrorClampToEdge &&
        !caps_.sampler_mirror_clamp)
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    return ToVkAddress(m);
  };
  si.addressModeU = address(desc.address_u);
  si.addressModeV = address(desc.address_v);
  si.addressModeW = address(desc.address_w);
  si.mipLodBias = base::Clamp(desc.lod_bias, -max_lod_bias_, max_lod_bias_);
  si.minLod = desc.min_lod;
  si.maxLod = desc.max_lod;
  si.anisotropyEnable = desc.max_anisotropy > 1.0f && caps_.sampler_anisotropy
                            ? VK_TRUE
                            : VK_FALSE;
  si.maxAnisotropy = base::Min(desc.max_anisotropy, max_anisotropy_);
  si.compareEnable = desc.compare_enable ? VK_TRUE : VK_FALSE;
  si.compareOp = ToVkCompare(desc.compare);
  switch (desc.border) {
    case rhi::BorderColor::kTransparentBlack:
      si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      break;
    case rhi::BorderColor::kOpaqueBlack:
      si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
      break;
    case rhi::BorderColor::kOpaqueWhite:
      si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      break;
  }
  if (vkCreateSampler(native.device, &si, nullptr, &sampler->sampler) !=
      VK_SUCCESS)
    return nullptr;
  return gpu::rhi::Release(sampler);
}

rhi::BindGroupLayout* VulkanDevice::CreateBindGroupLayout(
    const rhi::BindGroupLayoutDesc& desc) {
  auto layout = base::MakeUnique<VulkanBindGroupLayout>(desc);
  base::Vector<VkDescriptorSetLayoutBinding> bindings;
  for (const auto& b : desc.bindings) {
    VkDescriptorType type = ToVkDescriptorType(b.type);
    layout->types.push_back(type);
    if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
        type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
      layout->dynamic_count++;
    bindings.push_back({b.binding, type, 1, ToVkStages(b.stages), nullptr});
  }
  VkDescriptorSetLayoutCreateInfo li{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  li.bindingCount = static_cast<u32>(bindings.size());
  li.pBindings = bindings.data();
  if (desc.push && native.push_descriptor)
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  if (vkCreateDescriptorSetLayout(native.device, &li, nullptr,
                                  &layout->layout) != VK_SUCCESS)
    return nullptr;
  return gpu::rhi::Release(layout);
}

VkDescriptorPool VulkanDevice::GrowSetPool() {
  const VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSetsPerPool * 16},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSetsPerPool * 2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSetsPerPool},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSetsPerPool * 8},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 256},
  };
  VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pi.maxSets = kSetsPerPool;
  pi.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
  pi.pPoolSizes = sizes;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(native.device, &pi, nullptr, &pool) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  set_pools_.push_back(pool);
  return pool;
}

void VulkanDevice::FillWrite(VkDescriptorType type,
                             const rhi::BindingWrite& in,
                             VkWriteDescriptorSet& write,
                             VkDescriptorImageInfo& image,
                             VkDescriptorBufferInfo& buffer) const {
  write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstBinding = in.binding;
  write.descriptorCount = 1;
  write.descriptorType = type;
  switch (type) {
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
      auto* view = static_cast<VulkanView*>(in.view);
      image.imageView = view ? view->view : VK_NULL_HANDLE;
      image.sampler = in.sampler
                          ? static_cast<VulkanSampler*>(in.sampler)->sampler
                          : VK_NULL_HANDLE;
      image.imageLayout =
          type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
              ? VK_IMAGE_LAYOUT_GENERAL
              : ToVkLayout(in.view_state,
                           view ? view->desc().aspect : rhi::kAspectColor);
      write.pImageInfo = &image;
      break;
    }
    default:
      buffer.buffer = in.buffer ? static_cast<VulkanBuffer*>(in.buffer)->buffer
                                : VK_NULL_HANDLE;
      buffer.offset = in.offset;
      buffer.range = in.range ? in.range : VK_WHOLE_SIZE;
      write.pBufferInfo = &buffer;
      break;
  }
}

rhi::BindGroup* VulkanDevice::CreateBindGroup(const rhi::BindGroupDesc& desc) {
  auto* layout = static_cast<VulkanBindGroupLayout*>(desc.layout);
  auto group = base::MakeUnique<VulkanBindGroup>();
  group->layout = layout;
  {
    base::LockGuard<base::Mutex> lock(set_pool_mutex_);
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout->layout;
    for (VkDescriptorPool pool : set_pools_) {
      ai.descriptorPool = pool;
      if (vkAllocateDescriptorSets(native.device, &ai, &group->set) ==
          VK_SUCCESS) {
        group->pool = pool;
        break;
      }
    }
    if (!group->set) {
      ai.descriptorPool = GrowSetPool();
      if (!ai.descriptorPool ||
          vkAllocateDescriptorSets(native.device, &ai, &group->set) !=
              VK_SUCCESS)
        return nullptr;
      group->pool = ai.descriptorPool;
    }
  }
  UpdateBindGroup(&*group, desc.writes.data(),
                  static_cast<u32>(desc.writes.size()));
  return gpu::rhi::Release(group);
}

void VulkanDevice::UpdateBindGroup(rhi::BindGroup* group,
                                   const rhi::BindingWrite* writes,
                                   u32 count) {
  auto* g = static_cast<VulkanBindGroup*>(group);
  const auto& bindings = g->layout->desc().bindings;
  base::Vector<VkWriteDescriptorSet> vk_writes(count);
  base::Vector<VkDescriptorImageInfo> images(count);
  base::Vector<VkDescriptorBufferInfo> buffers(count);
  u32 n = 0;
  for (u32 i = 0; i < count; i++) {
    for (size_t b = 0; b < bindings.size(); b++) {
      if (bindings[b].binding != writes[i].binding)
        continue;
      FillWrite(g->layout->types[b], writes[i], vk_writes[n], images[n],
                buffers[n]);
      vk_writes[n].dstSet = g->set;
      n++;
      break;
    }
  }
  if (n)
    vkUpdateDescriptorSets(native.device, n, vk_writes.data(), 0, nullptr);
}

rhi::PipelineLayout* VulkanDevice::CreatePipelineLayout(
    const rhi::PipelineLayoutDesc& desc) {
  auto layout = base::MakeUnique<VulkanPipelineLayout>(desc);
  base::Vector<VkDescriptorSetLayout> sets;
  for (rhi::BindGroupLayout* g : desc.groups)
    sets.push_back(g ? static_cast<VulkanBindGroupLayout*>(g)->layout
                     : VK_NULL_HANDLE);
  layout->push_stages = ToVkStages(desc.push_constant_stages);
  VkPushConstantRange range{layout->push_stages, 0, desc.push_constant_bytes};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = static_cast<u32>(sets.size());
  li.pSetLayouts = sets.data();
  li.pushConstantRangeCount = desc.push_constant_bytes ? 1 : 0;
  li.pPushConstantRanges = &range;
  if (vkCreatePipelineLayout(native.device, &li, nullptr, &layout->layout) !=
      VK_SUCCESS)
    return nullptr;
  return gpu::rhi::Release(layout);
}

rhi::Pipeline* VulkanDevice::CreateGraphicsPipeline(
    const rhi::GraphicsPipelineDesc& desc) {
  DELTA_ZONE("vk.create_graphics_pipeline");
  const ui::ShaderCompilation compilation;
  auto* layout = static_cast<VulkanPipelineLayout*>(desc.layout);
  VkDevice dev = native.device;
  VkShaderModule modules[3] = {};
  VkPipelineShaderStageCreateInfo stages[3]{};
  u32 stage_count = 0;
  auto add_stage = [&](const rhi::ShaderCode& code,
                       VkShaderStageFlagBits stage) {
    if (code.empty())
      return true;
    VkShaderModule m = MakeModule(dev, code);
    if (!m)
      return false;
    modules[stage_count] = m;
    stages[stage_count] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[stage_count].stage = stage;
    stages[stage_count].module = m;
    stages[stage_count].pName = "main";
    stage_count++;
    return true;
  };
  bool ok = add_stage(desc.vertex, desc.mesh ? VK_SHADER_STAGE_MESH_BIT_EXT
                                             : VK_SHADER_STAGE_VERTEX_BIT) &&
            add_stage(desc.geometry, VK_SHADER_STAGE_GEOMETRY_BIT) &&
            add_stage(desc.fragment, VK_SHADER_STAGE_FRAGMENT_BIT);

  base::Vector<VkVertexInputBindingDescription> binds;
  for (size_t i = 0; i < desc.vertex_buffers.size(); i++)
    binds.push_back({static_cast<u32>(i), desc.vertex_buffers[i].stride,
                     desc.vertex_buffers[i].per_instance
                         ? VK_VERTEX_INPUT_RATE_INSTANCE
                         : VK_VERTEX_INPUT_RATE_VERTEX});
  base::Vector<VkVertexInputAttributeDescription> attrs;
  for (const auto& a : desc.vertex_attributes)
    attrs.push_back({a.location, a.buffer, ToVkFormat(a.format), a.offset});
  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = static_cast<u32>(binds.size());
  vi.pVertexBindingDescriptions = binds.data();
  vi.vertexAttributeDescriptionCount = static_cast<u32>(attrs.size());
  vi.pVertexAttributeDescriptions = attrs.data();
  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = ToVkTopology(desc.topology);
  ia.primitiveRestartEnable = desc.primitive_restart ? VK_TRUE : VK_FALSE;
  VkPipelineViewportStateCreateInfo vp{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = static_cast<VkCullModeFlags>(desc.cull);
  rs.frontFace = desc.front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                : VK_FRONT_FACE_CLOCKWISE;
  rs.lineWidth = 1.0f;
  rs.depthClampEnable = desc.depth_clamp ? VK_TRUE : VK_FALSE;
  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
  ds.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
  ds.depthCompareOp = ToVkCompare(desc.depth_compare);
  ds.stencilTestEnable = desc.stencil_test ? VK_TRUE : VK_FALSE;
  ds.front = ToVkStencil(desc.stencil_front);
  ds.back = ToVkStencil(desc.stencil_back);
  VkPipelineColorBlendAttachmentState att[8]{};
  for (u32 i = 0; i < desc.color_count && i < 8; i++) {
    const rhi::BlendAttachment& b = desc.blend[i];
    att[i].blendEnable = b.enable ? VK_TRUE : VK_FALSE;
    att[i].srcColorBlendFactor = ToVkBlendFactor(b.src_color);
    att[i].dstColorBlendFactor = ToVkBlendFactor(b.dst_color);
    att[i].colorBlendOp = ToVkBlendOp(b.color_op);
    att[i].srcAlphaBlendFactor = ToVkBlendFactor(b.src_alpha);
    att[i].dstAlphaBlendFactor = ToVkBlendFactor(b.dst_alpha);
    att[i].alphaBlendOp = ToVkBlendOp(b.alpha_op);
    att[i].colorWriteMask = b.write_mask;
  }
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = desc.color_count;
  cb.pAttachments = att;
  const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                VK_DYNAMIC_STATE_SCISSOR,
                                VK_DYNAMIC_STATE_BLEND_CONSTANTS};
  VkPipelineDynamicStateCreateInfo dy{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 3;
  dy.pDynamicStates = dyn;
  VkFormat formats[8];
  for (u32 i = 0; i < desc.color_count && i < 8; i++)
    formats[i] = ToVkFormat(desc.color_formats[i]);
  VkPipelineRenderingCreateInfo rci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = desc.color_count;
  rci.pColorAttachmentFormats = formats;
  rci.depthAttachmentFormat = ToVkFormat(desc.depth_format);
  rci.stencilAttachmentFormat = ToVkFormat(desc.stencil_format);
  VkGraphicsPipelineCreateInfo pi{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci;
  pi.stageCount = stage_count;
  pi.pStages = stages;
  pi.pVertexInputState = desc.mesh ? nullptr : &vi;
  pi.pInputAssemblyState = desc.mesh ? nullptr : &ia;
  pi.pViewportState = &vp;
  pi.pRasterizationState = &rs;
  pi.pMultisampleState = &ms;
  pi.pDepthStencilState = &ds;
  pi.pColorBlendState = &cb;
  pi.pDynamicState = &dy;
  pi.layout = layout->layout;
  auto pipeline = base::MakeUnique<VulkanPipeline>();
  pipeline->layout = layout;
  pipeline->bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  VkResult r = VK_ERROR_UNKNOWN;
  const u64 started = NowNs();
  if (ok) {
    ScopeNs t(&g_ns_pipe_build);
    r = vkCreateGraphicsPipelines(dev, native.pipeline_cache, 1, &pi, nullptr,
                                  &pipeline->pipeline);
    g_pipe_build_n++;
  }
  for (u32 i = 0; i < stage_count; i++)
    vkDestroyShaderModule(dev, modules[i], nullptr);
  if (r != VK_SUCCESS) {
    BASE_LOGI("gpuvk", "graphics pipeline failed: {}", (int)r);
    return nullptr;
  }
  last_pipeline_build_ns_ = NowNs();
  SaveAfterSlowBuild(started);
  if (desc.name)
    SetName(&*pipeline, desc.name);
  return gpu::rhi::Release(pipeline);
}

rhi::Pipeline* VulkanDevice::CreateComputePipeline(
    const rhi::ComputePipelineDesc& desc) {
  DELTA_ZONE("vk.create_compute_pipeline");
  const ui::ShaderCompilation compilation;
  auto* layout = static_cast<VulkanPipelineLayout*>(desc.layout);
  VkShaderModule module = MakeModule(native.device, desc.code);
  if (!module)
    return nullptr;
  VkComputePipelineCreateInfo ci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ci.stage.module = module;
  ci.stage.pName = "main";
  ci.layout = layout->layout;
  if (desc.dispatch_base)
    ci.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
  auto pipeline = base::MakeUnique<VulkanPipeline>();
  pipeline->layout = layout;
  pipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
  const u64 started = NowNs();
  const VkResult r =
      vkCreateComputePipelines(native.device, native.pipeline_cache, 1, &ci,
                               nullptr, &pipeline->pipeline);
  vkDestroyShaderModule(native.device, module, nullptr);
  if (r != VK_SUCCESS) {
    BASE_LOGI("gpuvk", "compute pipeline failed: {}", (int)r);
    return nullptr;
  }
  last_pipeline_build_ns_ = NowNs();
  SaveAfterSlowBuild(started);
  if (desc.name)
    SetName(&*pipeline, desc.name);
  return gpu::rhi::Release(pipeline);
}

rhi::TimestampPool* VulkanDevice::CreateTimestampPool(u32 count) {
  auto pool = base::MakeUnique<VulkanTimestampPool>();
  VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qi.queryCount = count;
  if (vkCreateQueryPool(native.device, &qi, nullptr, &pool->pool) != VK_SUCCESS)
    return nullptr;
  pool->count = count;
  return gpu::rhi::Release(pool);
}

rhi::CommandList* VulkanDevice::CreateCommandList() {
  auto list = base::MakeUnique<VulkanCommandList>(*this);
  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = command_pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(native.device, &ai, &list->cmd) != VK_SUCCESS)
    return nullptr;
  return gpu::rhi::Release(list);
}

void VulkanDevice::Destroy(rhi::Object* object) {
  if (!object)
    return;
  VkDevice dev = native.device;
  if (auto* b = dynamic_cast<VulkanBuffer*>(object)) {
    vkDestroyBuffer(dev, b->buffer, nullptr);
    vkFreeMemory(dev, b->memory, nullptr);
    for (VkDeviceMemory block : b->blocks)
      vkFreeMemory(dev, block, nullptr);
  } else if (auto* t = dynamic_cast<VulkanTexture*>(object)) {
    vkDestroyImage(dev, t->image, nullptr);
    images_.Free(*this, t->allocation);
  } else if (auto* v = dynamic_cast<VulkanView*>(object)) {
    vkDestroyImageView(dev, v->view, nullptr);
  } else if (auto* s = dynamic_cast<VulkanSampler*>(object)) {
    vkDestroySampler(dev, s->sampler, nullptr);
  } else if (auto* l = dynamic_cast<VulkanBindGroupLayout*>(object)) {
    vkDestroyDescriptorSetLayout(dev, l->layout, nullptr);
  } else if (auto* g = dynamic_cast<VulkanBindGroup*>(object)) {
    base::LockGuard<base::Mutex> lock(set_pool_mutex_);
    vkFreeDescriptorSets(dev, g->pool, 1, &g->set);
  } else if (auto* pl = dynamic_cast<VulkanPipelineLayout*>(object)) {
    vkDestroyPipelineLayout(dev, pl->layout, nullptr);
  } else if (auto* p = dynamic_cast<VulkanPipeline*>(object)) {
    vkDestroyPipeline(dev, p->pipeline, nullptr);
  } else if (auto* q = dynamic_cast<VulkanTimestampPool*>(object)) {
    vkDestroyQueryPool(dev, q->pool, nullptr);
  }
  delete object;
}

bool VulkanDevice::CopyAfter(rhi::Buffer* src,
                             u64 src_offset,
                             rhi::Buffer* dst,
                             u64 dst_offset,
                             u64 bytes,
                             u64 after) {
  if (!side_queue_ || !src || !dst || !bytes)
    return false;
  while (side_busy_.exchange(true, base::memory_order_acquire)) {
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(side_cmd_, 0);
  vkBeginCommandBuffer(side_cmd_, &bi);
  const VkBufferCopy region{src_offset, dst_offset, bytes};
  vkCmdCopyBuffer(side_cmd_, static_cast<VulkanBuffer*>(src)->buffer,
                  static_cast<VulkanBuffer*>(dst)->buffer, 1, &region);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(side_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                       nullptr);
  vkEndCommandBuffer(side_cmd_);
  const uint64_t wait_value = after;
  VkTimelineSemaphoreSubmitInfo ti{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  ti.waitSemaphoreValueCount = 1;
  ti.pWaitSemaphoreValues = &wait_value;
  const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &ti;
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &timeline_;
  si.pWaitDstStageMask = &stage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &side_cmd_;
  vkResetFences(native.device, 1, &side_fence_);
  VkResult r = vkQueueSubmit(side_queue_, 1, &si, side_fence_);
  if (r == VK_SUCCESS)
    r = vkWaitForFences(native.device, 1, &side_fence_, VK_TRUE, ~0ull);
  side_busy_.store(false, base::memory_order_release);
  return r == VK_SUCCESS;
}

bool VulkanDevice::CommitSparse(rhi::Buffer* buffer,
                                u64 offset,
                                u64 bytes) {
  auto* b = static_cast<VulkanBuffer*>(buffer);
  const u64 page = caps_.sparse_page;
  if (!b || !b->desc().sparse || !page || !bytes ||
      offset + bytes > b->desc().size)
    return false;
  // Pages are carved out of blocks of this size; a block is never returned
  // before the buffer is.
  constexpr u64 kBlock = 64ull << 20;
  base::Vector<VkSparseMemoryBind> binds;
  for (u64 at = offset & ~(page - 1); at < offset + bytes; at += page) {
    if (b->committed.count(at))
      continue;
    if (b->blocks.empty() || b->block_used + page > kBlock) {
      VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      VkMemoryAllocateFlagsInfo flags{
          VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
      if (b->address()) {
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        ai.pNext = &flags;
      }
      ai.allocationSize = kBlock;
      ai.memoryTypeIndex = b->memory_type;
      VkDeviceMemory block;
      if (vkAllocateMemory(native.device, &ai, nullptr, &block) != VK_SUCCESS)
        return false;
      b->blocks.push_back(block);
      b->block_used = 0;
    }
    VkSparseMemoryBind bind{};
    bind.resourceOffset = at;
    bind.size = page;
    bind.memory = b->blocks.back();
    bind.memoryOffset = b->block_used;
    b->block_used += page;
    b->committed.insert(at);
    binds.push_back(bind);
  }
  if (binds.empty())
    return true;
  VkSparseBufferMemoryBindInfo buffer_binds{b->buffer,
                                            static_cast<u32>(binds.size()),
                                            binds.data()};
  VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
  info.bufferBindCount = 1;
  info.pBufferBinds = &buffer_binds;
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  if (vkCreateFence(native.device, &fi, nullptr, &fence) != VK_SUCCESS)
    return false;
  VkResult r;
  {
    base::LockGuard<base::Mutex> lock(queue_mutex);
    r = vkQueueBindSparse(native.queue, 1, &info, fence);
  }
  if (r == VK_SUCCESS)
    r = vkWaitForFences(native.device, 1, &fence, VK_TRUE, ~0ull);
  vkDestroyFence(native.device, fence, nullptr);
  return r == VK_SUCCESS;
}

void VulkanDevice::SetName(rhi::Object* object, const char* name) {
  if (!set_object_name || !object || !name)
    return;
  VkObjectType type = VK_OBJECT_TYPE_UNKNOWN;
  u64 handle = 0;
  if (auto* b = dynamic_cast<VulkanBuffer*>(object)) {
    type = VK_OBJECT_TYPE_BUFFER;
    handle = reinterpret_cast<u64>(b->buffer);
  } else if (auto* t = dynamic_cast<VulkanTexture*>(object)) {
    type = VK_OBJECT_TYPE_IMAGE;
    handle = reinterpret_cast<u64>(t->image);
  } else if (auto* p = dynamic_cast<VulkanPipeline*>(object)) {
    type = VK_OBJECT_TYPE_PIPELINE;
    handle = reinterpret_cast<u64>(p->pipeline);
  } else {
    return;
  }
  VkDebugUtilsObjectNameInfoEXT info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  info.objectType = type;
  info.objectHandle = handle;
  info.pObjectName = name;
  set_object_name(native.device, &info);
}

bool VulkanDevice::SupportsFormat(rhi::Format format, u32 usage) const {
  VkFormatProperties props;
  vkGetPhysicalDeviceFormatProperties(native.phys, ToVkFormat(format), &props);
  const VkFormatFeatureFlags f = props.optimalTilingFeatures;
  if ((usage & rhi::kTextureSampled) &&
      !(f & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
    return false;
  if ((usage & rhi::kTextureStorage) &&
      !(f & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
    return false;
  if ((usage & rhi::kTextureColorTarget) &&
      !(f & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
    return false;
  if ((usage & rhi::kTextureDepthTarget) &&
      !(f & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return false;
  if ((usage & rhi::kTextureCopySrc) && !(f & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
      !(f & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))
    return false;
  return true;
}

bool VulkanDevice::SupportsBlit(rhi::Format src, rhi::Format dst) const {
  VkFormatProperties s, d;
  vkGetPhysicalDeviceFormatProperties(native.phys, ToVkFormat(src), &s);
  vkGetPhysicalDeviceFormatProperties(native.phys, ToVkFormat(dst), &d);
  return (s.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
         (d.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
}

void VulkanDevice::QueryMemoryBudget(u64* used, u64* budget) const {
  VkPhysicalDeviceMemoryBudgetPropertiesEXT b{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 props{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &b};
  vkGetPhysicalDeviceMemoryProperties2(native.phys, &props);
  *used = *budget = 0;
  for (u32 i = 0; i < props.memoryProperties.memoryHeapCount; i++)
    if (props.memoryProperties.memoryHeaps[i].flags &
        VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      *used += b.heapUsage[i];
      *budget += b.heapBudget[i];
    }
}

u64 VulkanDevice::Submit(rhi::CommandList* const* lists, u32 count) {
  DELTA_ZONE("vk.submit");
  base::Vector<VkCommandBuffer> cmds(count);
  for (u32 i = 0; i < count; i++)
    cmds[i] = static_cast<VulkanCommandList*>(lists[i])->cmd;
  base::LockGuard<base::Mutex> lock(queue_mutex);
  const uint64_t value = submitted_ + 1;
  VkTimelineSemaphoreSubmitInfo ti{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  ti.signalSemaphoreValueCount = 1;
  ti.pSignalSemaphoreValues = &value;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &ti;
  si.commandBufferCount = count;
  si.pCommandBuffers = cmds.data();
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &timeline_;
  const VkResult r = vkQueueSubmit(native.queue, 1, &si, VK_NULL_HANDLE);
  if (r != VK_SUCCESS) {
    BASE_LOGI("gpuvk", "vkQueueSubmit failed: {}", (int)r);
    ReportDeviceLoss();
    return submitted_;
  }
  submitted_ = value;
  return value;
}

bool VulkanDevice::IsComplete(u64 submission) {
  uint64_t value = 0;
  if (vkGetSemaphoreCounterValue(native.device, timeline_, &value) !=
      VK_SUCCESS)
    return false;
  return value >= submission;
}

bool VulkanDevice::Wait(u64 submission, u64 timeout_ns) {
  DELTA_ZONE("vk.wait");
  VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wi.semaphoreCount = 1;
  wi.pSemaphores = &timeline_;
  const uint64_t value = submission;
  wi.pValues = &value;
  const VkResult r = vkWaitSemaphores(native.device, &wi, timeout_ns);
  if (r == VK_ERROR_DEVICE_LOST)
    ReportDeviceLoss();
  return r == VK_SUCCESS;
}

void VulkanDevice::WaitIdle() {
  DELTA_ZONE("vk.wait_idle");
  base::LockGuard<base::Mutex> lock(queue_mutex);
  vkQueueWaitIdle(native.queue);
}

bool VulkanDevice::ReadTimestamps(rhi::TimestampPool* pool,
                                  u32 first,
                                  u32 count,
                                  u64* out) {
  auto* p = static_cast<VulkanTimestampPool*>(pool);
  return vkGetQueryPoolResults(native.device, p->pool, first, count,
                               count * sizeof(u64), out, sizeof(u64),
                               VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
}

void VulkanDevice::SavePipelineCache(bool force) {
  if (pipeline_cache_path_.empty() || !native.pipeline_cache)
    return;
  if (force) {
    while (cache_saving_.load())
      base::YieldCurrentThread();
  } else if (cache_saving_.load()) {
    return;
  }
  // A long-lived cache is hundreds of MB (GTA:SA's is 500): wait for a
  // compile burst to end (or a minute), and serialize and write it off the
  // thread that ends the frame. Inline, one save stalled a frame 0.7-7 s,
  // and the title's hang detector fires at 10.
  const u64 now = NowNs();
  const u64 last_build = last_pipeline_build_ns_.load();
  if (!force && last_build <= last_cache_write_ns_)
    return;
  if (!force && now - last_build < 3000000000ull &&
      now - last_cache_write_ns_ < 60000000000ull)
    return;
  last_cache_write_ns_ = now;
  if (force) {
    WritePipelineCache(true);
    return;
  }
  cache_saving_.store(true);
  base::SpawnDetachedThread("vk_rhi_device", [this] {
    WritePipelineCache(false);
    cache_saving_.store(false);
  });
}

void VulkanDevice::SaveAfterSlowBuild(u64 started) {
  const u64 now = NowNs();
  if (now - started < 1000000000ull || pipeline_cache_path_.empty() ||
      !native.pipeline_cache)
    return;
  // A build that lands while a save runs is picked up by one more pass.
  save_requested_.store(true);
  if (cache_saving_.exchange(true))
    return;
  last_cache_write_ns_ = now;
  base::SpawnDetachedThread("vk_rhi_device", [this] {
    for (;;) {
      while (save_requested_.exchange(false))
        WritePipelineCache(false);
      cache_saving_.store(false);
      if (!save_requested_.load() || cache_saving_.exchange(true))
        return;
    }
  });
}

void VulkanDevice::WritePipelineCache(bool force) {
  size_t size = 0;
  if (vkGetPipelineCacheData(native.device, native.pipeline_cache, &size,
                             nullptr) != VK_SUCCESS ||
      !size)
    return;
  if (!force && size == last_cache_size_)
    return;
  last_cache_size_ = size;
  base::Vector<u8> blob(size);
  if (vkGetPipelineCacheData(native.device, native.pipeline_cache, &size,
                             blob.data()) != VK_SUCCESS)
    return;
  // Write-then-rename: the runner SIGKILLs the emulator, and a torn blob
  // would cost the next run its whole cache.
  const base::String tmp =
      base::Format("{}.{}.tmp", pipeline_cache_path_, ::getpid());
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f)
    return;
  const bool ok = std::fwrite(blob.data(), 1, size, f) == size;
  std::fclose(f);
  if (ok)
    std::rename(tmp.c_str(), pipeline_cache_path_.c_str());
  else
    ::unlink(tmp.c_str());
}

void VulkanDevice::Maintain() {
  SavePipelineCache(false);
}

void VulkanDevice::ReportDeviceLoss() {
  if (fault_reported_)
    return;
  fault_reported_ = true;
  VkDevice dev = native.device;
  if (native.checkpoints) {
    auto get = reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(
        vkGetDeviceProcAddr(dev, "vkGetQueueCheckpointDataNV"));
    if (get) {
      u32 count = 0;
      get(native.queue, &count, nullptr);
      base::Vector<VkCheckpointDataNV> points(
          count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
      get(native.queue, &count, points.data());
      for (const auto& p : points)
        BASE_LOGI("gpuvk", "  checkpoint stage={:#x} marker={:#x}",
                  u32(p.stage),
                  (unsigned long long)reinterpret_cast<uintptr_t>(
                      p.pCheckpointMarker));
    }
  }
  if (!native.device_fault)
    return;
  auto get_fault = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
      vkGetDeviceProcAddr(dev, "vkGetDeviceFaultInfoEXT"));
  if (!get_fault)
    return;
  VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
  if (get_fault(dev, &counts, nullptr) < 0)
    return;
  base::Vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
  base::Vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
  VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
  info.pAddressInfos = addrs.data();
  info.pVendorInfos = vendors.data();
  counts.vendorBinarySize = 0;
  get_fault(dev, &counts, &info);
  BASE_LOGI("gpuvk", "device fault: '{}' addrs={} vendor={}", info.description,
            counts.addressInfoCount, counts.vendorInfoCount);
  for (const auto& a : addrs)
    BASE_LOGI("gpuvk", "  fault addr type={} va={:#x} prec={:#x}",
              (int)a.addressType, (unsigned long long)a.reportedAddress,
              (unsigned long long)a.addressPrecision);
  for (const auto& v : vendors)
    BASE_LOGI("gpuvk", "  vendor '{}' code={:#x} data={:#x}", v.description,
              (unsigned long long)v.vendorFaultCode,
              (unsigned long long)v.vendorFaultData);
}

}  // namespace rhi_impl

base::UniquePointer<rhi::Device> CreateVulkanDevice(
    const VulkanOptions& options) {
  auto device = base::MakeUnique<VulkanDevice>();
  if (!device->Init(options))
    return nullptr;
  return device;
}

const NativeDevice& Native(rhi::Device* device) {
  return Impl(device).native;
}
VkBuffer Native(rhi::Buffer* b) {
  return b ? static_cast<VulkanBuffer*>(b)->buffer : VK_NULL_HANDLE;
}
VkImage Native(rhi::Texture* t) {
  return t ? static_cast<VulkanTexture*>(t)->image : VK_NULL_HANDLE;
}
VkImageView Native(rhi::TextureView* v) {
  return v ? static_cast<VulkanView*>(v)->view : VK_NULL_HANDLE;
}
VkSampler Native(rhi::Sampler* s) {
  return s ? static_cast<VulkanSampler*>(s)->sampler : VK_NULL_HANDLE;
}
VkDescriptorSetLayout Native(rhi::BindGroupLayout* l) {
  return l ? static_cast<VulkanBindGroupLayout*>(l)->layout : VK_NULL_HANDLE;
}
VkDescriptorSet Native(rhi::BindGroup* g) {
  return g ? static_cast<VulkanBindGroup*>(g)->set : VK_NULL_HANDLE;
}
VkPipelineLayout Native(rhi::PipelineLayout* l) {
  return l ? static_cast<VulkanPipelineLayout*>(l)->layout : VK_NULL_HANDLE;
}
VkPipeline Native(rhi::Pipeline* p) {
  return p ? static_cast<VulkanPipeline*>(p)->pipeline : VK_NULL_HANDLE;
}
VkCommandBuffer Native(rhi::CommandList* c) {
  return c ? static_cast<VulkanCommandList*>(c)->cmd : VK_NULL_HANDLE;
}
void LockQueue(rhi::Device* device) {
  Impl(device).queue_mutex.lock();
}
void UnlockQueue(rhi::Device* device) {
  Impl(device).queue_mutex.unlock();
}

}  // namespace gpu::vk
