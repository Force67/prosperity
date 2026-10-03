/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cstring>

#include "base/logging.h"

// The IID constants are defined here and only here (dxguid on Windows).
#if !defined(_WIN32)
#define INITGUID
#endif
#include "base/algorithm.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "gpu/d3d12/d3d12_internal.h"

namespace gpu::d3d12 {

using namespace impl;

namespace impl {

namespace {

constexpr u64 AlignUp(u64 v, u64 a) {
  return (v + a - 1) / a * a;
}

D3D12_TEXTURE_ADDRESS_MODE ToAddress(rhi::AddressMode m) {
  switch (m) {
    case rhi::AddressMode::kRepeat:
      return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case rhi::AddressMode::kMirroredRepeat:
      return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case rhi::AddressMode::kClampToEdge:
      return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case rhi::AddressMode::kClampToBorder:
      return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    case rhi::AddressMode::kMirrorClampToEdge:
      return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
  }
  return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
}

u32 ToComponent(rhi::Swizzle s, u32 identity) {
  switch (s) {
    case rhi::Swizzle::kIdentity:
      return identity;
    case rhi::Swizzle::kZero:
      return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0;
    case rhi::Swizzle::kOne:
      return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1;
    case rhi::Swizzle::kR:
      return 0;
    case rhi::Swizzle::kG:
      return 1;
    case rhi::Swizzle::kB:
      return 2;
    case rhi::Swizzle::kA:
      return 3;
  }
  return identity;
}

// The view's swizzle, applied over `base` (which maps each output to a
// memory component; the stencil plane arrives in green).
UINT Mapping(const rhi::TextureViewDesc& desc, const u32 base[4]) {
  u32 c[4];
  for (u32 i = 0; i < 4; i++) {
    const u32 s = ToComponent(desc.swizzle[i], i);
    c[i] = s < 4 ? base[s] : s;
  }
  return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(c[0], c[1], c[2], c[3]);
}

base::String PackSampler(const rhi::SamplerDesc& d) {
  base::String key;
  auto put = [&](const void* p, size_t n) {
    key.append(static_cast<const char*>(p), n);
  };
  const u8 e[] = {u8(d.mag),
                  u8(d.min),
                  u8(d.mip),
                  u8(d.address_u),
                  u8(d.address_v),
                  u8(d.address_w),
                  u8(d.compare_enable),
                  u8(d.compare),
                  u8(d.border)};
  put(e, sizeof(e));
  put(&d.lod_bias, 4);
  put(&d.min_lod, 4);
  put(&d.max_lod, 4);
  put(&d.max_anisotropy, 4);
  return key;
}

}  // namespace

// CpuDescriptorPool

void CpuDescriptorPool::Init(ID3D12Device* device,
                             D3D12_DESCRIPTOR_HEAP_TYPE type,
                             u32 page_size) {
  device_ = device;
  type_ = type;
  page_size_ = page_size;
  increment_ = device->GetDescriptorHandleIncrementSize(type);
}

void CpuDescriptorPool::Shutdown() {
  for (Page& p : pages_)
    SafeRelease(p.heap);
  pages_.clear();
}

bool CpuDescriptorPool::Allocate(u32 count, CpuRange* out) {
  if (!count || count > page_size_)
    return false;
  base::LockGuard<base::Mutex> lock(mutex_);
  auto take = [&](u32 page_index) {
    Page& p = pages_[page_index];
    for (auto it = p.free.begin(); it != p.free.end(); ++it) {
      if (it->second < count)
        continue;
      const u32 offset = it->first;
      const u32 left = it->second - count;
      p.free.erase(it);
      if (left)
        p.free[offset + count] = left;
      out->cpu = {p.start.ptr + static_cast<size_t>(offset) * increment_};
      out->page = page_index;
      out->offset = offset;
      out->count = count;
      return true;
    }
    return false;
  };
  for (u32 i = 0; i < pages_.size(); i++)
    if (take(i))
      return true;
  D3D12_DESCRIPTOR_HEAP_DESC hd{};
  hd.Type = type_;
  hd.NumDescriptors = page_size_;
  Page page;
  if (FAILED(device_->CreateDescriptorHeap(
          &hd, IID_ID3D12DescriptorHeap, reinterpret_cast<void**>(&page.heap))))
    return false;
  page.start = page.heap->GetCPUDescriptorHandleForHeapStart();
  page.free[0] = page_size_;
  pages_.push_back(base::move(page));
  return take(static_cast<u32>(pages_.size() - 1));
}

void CpuDescriptorPool::Free(CpuRange& range) {
  if (!range.valid())
    return;
  base::LockGuard<base::Mutex> lock(mutex_);
  auto& free = pages_[range.page].free;
  u32 offset = range.offset, count = range.count;
  auto next = free.lower_bound(offset);
  if (next != free.end() && next->first == offset + count) {
    count += next->second;
    next = free.erase(next);
  }
  if (next != free.begin()) {
    auto prev = base::Prev(next);
    if (prev->first + prev->second == offset) {
      prev->second += count;
      range = {};
      return;
    }
  }
  free[offset] = count;
  range = {};
}

const GroupEntry* D3D12BindGroupLayout::Find(u32 binding) const {
  for (const GroupEntry& e : entries)
    if (e.binding == binding)
      return &e;
  return nullptr;
}

// D3D12Device

void D3D12Device::ReleaseSessionObjects() {
  WaitIdle();
  ReleaseObjects();
  shaders_ = {};
  for (auto& [key, pso] : blit_psos_)
    SafeRelease(pso);
  blit_psos_ = {};
  SafeRelease(blit_root_);
  blit_vs_ = {};
  blit_ps_ = {};
  sampler_ids_ = {};
  sampler_cpu_ = {};
  sampler_tables_ = {};
  sampler_heap_used_ = 0;
  ++sampler_generation_;
  views.Shutdown();
  rtvs.Shutdown();
  dsvs.Shutdown();
  samplers.Shutdown();
  Destroy(CreateSampler({}));
}

D3D12Device::~D3D12Device() {
  if (!device)
    return;
  WaitIdle();
  for (auto& [key, pso] : blit_psos_)
    SafeRelease(pso);
  SafeRelease(blit_root_);
  views.Shutdown();
  rtvs.Shutdown();
  dsvs.Shutdown();
  samplers.Shutdown();
  SafeRelease(ring_heap);
  SafeRelease(sampler_heap);
  SafeRelease(fence_);
  SafeRelease(queue);
  SafeRelease(device);
}

bool D3D12Device::Init(const D3D12Options& options) {
  if (!Dxc::Load()) {
    BASE_LOGI("gpud3d12", "DXC (dxcompiler) could not be loaded");
    return false;
  }
#if defined(_WIN32)
  if (options.debug_layer) {
    ID3D12Debug* debug = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_ID3D12Debug,
                                         reinterpret_cast<void**>(&debug)))) {
      debug->EnableDebugLayer();
      debug->Release();
    }
  }
  IDXGIFactory4* factory = nullptr;
  if (FAILED(CreateDXGIFactory2(0, IID_IDXGIFactory4,
                                reinterpret_cast<void**>(&factory))))
    return false;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT i = 0; !device; i++) {
    if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND)
      break;
    DXGI_ADAPTER_DESC1 ad{};
    adapter->GetDesc1(&ad);
    base::String name;
    for (const WCHAR* c = ad.Description; *c; c++)
      name += *c < 128 ? static_cast<char>(*c) : '?';
    const bool wanted = !(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                        (!options.gpu_filter ||
                         name.find(options.gpu_filter) != base::String::npos);
    if (wanted && SUCCEEDED(D3D12CreateDevice(
                      adapter, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device,
                      reinterpret_cast<void**>(&device))))
      device_name_ = name;
    adapter->Release();
  }
  factory->Release();
#else
  (void)options.gpu_filter;
  if (options.debug_layer) {
    ID3D12Debug* debug = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_ID3D12Debug,
                                         reinterpret_cast<void**>(&debug)))) {
      debug->EnableDebugLayer();
      debug->Release();
    }
  }
  if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                               IID_ID3D12Device,
                               reinterpret_cast<void**>(&device))))
    device = nullptr;
  if (device) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(vkd3d_get_vk_physical_device(device), &props);
    device_name_ = base::String("vkd3d: ") + props.deviceName;
  }
#endif
  if (!device) {
    BASE_LOGI("gpud3d12", "no D3D12 device");
    return false;
  }

  D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_5};
  if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm,
                                         sizeof(sm))) ||
      sm.HighestShaderModel < D3D_SHADER_MODEL_6_0) {
    BASE_LOGI("gpud3d12", "device has no shader model 6");
    return false;
  }
  shader_model_ =
      (sm.HighestShaderModel >> 4) * 10 + (sm.HighestShaderModel & 0xf);
  D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1{};
  device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1));
  D3D12_FEATURE_DATA_D3D12_OPTIONS3 o3{};
  device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &o3, sizeof(o3));
  D3D12_FEATURE_DATA_ARCHITECTURE arch{};
  device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch));

  D3D12_COMMAND_QUEUE_DESC qd{};
  qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (FAILED(device->CreateCommandQueue(&qd, IID_ID3D12CommandQueue,
                                        reinterpret_cast<void**>(&queue))) ||
      FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence,
                                 reinterpret_cast<void**>(&fence_))))
    return false;

  // The shader-visible ring: as large as the device allows, up to the
  // resource binding tier 1/2 limit of one million.
  for (u32 size : {1000000u, 500000u, 262144u, 65536u}) {
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = size;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (SUCCEEDED(device->CreateDescriptorHeap(
            &hd, IID_ID3D12DescriptorHeap,
            reinterpret_cast<void**>(&ring_heap)))) {
      ring_size_ = size;
      break;
    }
  }
  D3D12_DESCRIPTOR_HEAP_DESC sd{};
  sd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
  sd.NumDescriptors = kSamplerHeapSize;
  sd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  if (!ring_heap || FAILED(device->CreateDescriptorHeap(
                        &sd, IID_ID3D12DescriptorHeap,
                        reinterpret_cast<void**>(&sampler_heap))))
    return false;
  ring_cpu_ = ring_heap->GetCPUDescriptorHandleForHeapStart();
  ring_gpu_ = ring_heap->GetGPUDescriptorHandleForHeapStart();
  view_increment = device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  sampler_increment = device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
  for (u32 c = ring_size_ / kChunkSize; c-- > 0;)
    free_chunks_.push_back(c * kChunkSize);

  views.Init(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096);
  rtvs.Init(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1024);
  dsvs.Init(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1024);
  samplers.Init(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 256);
  // Sampler id 0: what an unwritten sampled-texture binding samples with.
  Destroy(CreateSampler({}));

  UINT64 frequency = 0;
  queue->GetTimestampFrequency(&frequency);

  caps_.backend = rhi::Backend::kD3D12;
  caps_.device_name = device_name_.c_str();
  caps_.geometry_shader = true;
  caps_.mesh_shader = false;
  caps_.fragment_barycentric = o3.BarycentricsSupported && shader_model_ >= 61;
  caps_.independent_blend = true;
  caps_.sampler_anisotropy = true;
  caps_.sampler_mirror_clamp = true;
  caps_.storage_image_write_without_format = true;
  caps_.buffer_address = false;
  caps_.host_import = false;
  caps_.unified_memory = arch.UMA && arch.CacheCoherentUMA;
  caps_.debug_labels = options.debug_labels;
  debug_labels = options.debug_labels;
  caps_.timestamps = frequency != 0;
  caps_.timestamp_period_ns = frequency ? 1e9 / double(frequency) : 1.0;
  caps_.subgroup_size =
      o1.WaveOps && o1.WaveLaneCountMin ? o1.WaveLaneCountMin : 32;
  caps_.max_dynamic_uniform_buffers = 8;
  caps_.max_dynamic_storage_buffers = 4;
  caps_.max_storage_buffers_per_stage = 64;
  caps_.max_push_constant_bytes = 128;
  caps_.max_storage_buffer_range = 128ull << 20;
  caps_.uniform_offset_alignment =
      D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
  caps_.storage_offset_alignment = D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT;
  caps_.max_texture_size = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
  caps_.max_texture_size_3d = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
  caps_.dispatch_base = true;  // an id offset in the lowered shader
  caps_.max_compute_resources = 64;
  caps_.texture_blit = InitBlit();
  BASE_LOGI("gpud3d12", "device: {} (shader model {}.{})", device_name_.c_str(),
            shader_model_ / 10, shader_model_ % 10);
  return true;
}

ID3D12Resource* D3D12Device::CreateBufferResource(u64 size,
                                                  D3D12_HEAP_TYPE heap,
                                                  D3D12_RESOURCE_FLAGS flags,
                                                  D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = heap;
  D3D12_RESOURCE_DESC rd{};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = size;
  rd.Height = 1;
  rd.DepthOrArraySize = 1;
  rd.MipLevels = 1;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  rd.Flags = flags;
  ID3D12Resource* resource = nullptr;
  if (FAILED(device->CreateCommittedResource(
          &hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_ID3D12Resource,
          reinterpret_cast<void**>(&resource))))
    return nullptr;
  return resource;
}

rhi::Buffer* D3D12Device::CreateBuffer(const rhi::BufferDesc& desc) {
  if (desc.host_pointer)
    return nullptr;
  auto buffer = base::MakeUnique<D3D12Buffer>(desc);
  // 256-byte multiples: a constant buffer view rounds its size up to that.
  buffer->alloc_size = AlignUp(base::Max<u64>(desc.size, 1), 256);
  D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
  switch (desc.memory) {
    case rhi::MemoryKind::kDevice:
      break;
    case rhi::MemoryKind::kUpload:
    case rhi::MemoryKind::kUploadDevice:
      heap = D3D12_HEAP_TYPE_UPLOAD;
      flags = D3D12_RESOURCE_FLAG_NONE;
      state = D3D12_RESOURCE_STATE_GENERIC_READ;
      buffer->fixed_state = true;
      break;
    case rhi::MemoryKind::kReadback:
      heap = D3D12_HEAP_TYPE_READBACK;
      flags = D3D12_RESOURCE_FLAG_NONE;
      state = D3D12_RESOURCE_STATE_COPY_DEST;
      buffer->fixed_state = true;
      break;
  }
  buffer->resource =
      CreateBufferResource(buffer->alloc_size, heap, flags, state);
  if (!buffer->resource)
    return nullptr;
  buffer->state = state;
  buffer->va = buffer->resource->GetGPUVirtualAddress();
  if (heap != D3D12_HEAP_TYPE_DEFAULT) {
    void* p = nullptr;
    const D3D12_RANGE none{0, 0};
    if (SUCCEEDED(buffer->resource->Map(
            0, heap == D3D12_HEAP_TYPE_UPLOAD ? &none : nullptr, &p)))
      buffer->SetMapped(static_cast<u8*>(p));
  }
  if (desc.name)
    SetName(&*buffer, desc.name);
  return gpu::rhi::Release(buffer, this);
}

rhi::Texture* D3D12Device::CreateTexture(const rhi::TextureDesc& desc) {
  const DxgiInfo& fi = Dxgi(desc.format);
  const rhi::FormatInfo& info = rhi::GetFormatInfo(desc.format);
  if (!fi.texture)
    return nullptr;
  auto texture = base::MakeUnique<D3D12Texture>(desc);
  const bool depth = info.is_depth || info.is_stencil;
  D3D12_RESOURCE_DESC rd{};
  rd.Dimension =
      desc.dim == rhi::TextureDim::k1D   ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
      : desc.dim == rhi::TextureDim::k3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                         : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  rd.Width = desc.width;
  rd.Height = desc.dim == rhi::TextureDim::k1D ? 1 : desc.height;
  rd.DepthOrArraySize = static_cast<UINT16>(
      desc.dim == rhi::TextureDim::k3D ? desc.depth : desc.layers);
  rd.MipLevels = static_cast<UINT16>(desc.mips);
  rd.Format = depth || (desc.usage & rhi::kTextureMutableFormat) ? fi.typeless
                                                                 : fi.format;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  if (depth) {
    rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
  } else {
    // Render-target capable whenever the format is, so ClearTexture and
    // BlitTexture can draw into a copy destination.
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{fi.format};
    const bool renderable =
        SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs,
                                              sizeof(fs))) &&
        (fs.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
    if ((desc.usage & rhi::kTextureColorTarget) ||
        (renderable && (desc.usage & rhi::kTextureCopyDst)))
      rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (desc.usage & rhi::kTextureStorage)
      rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  }
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  if (FAILED(device->CreateCommittedResource(
          &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
          IID_ID3D12Resource, reinterpret_cast<void**>(&texture->resource)))) {
    BASE_LOGI("gpud3d12", "texture {}x{} {} failed", desc.width, desc.height,
              rhi::FormatName(desc.format));
    return nullptr;
  }
  texture->flags = rd.Flags;
  texture->planes = info.is_depth && info.is_stencil ? 2 : 1;
  texture->states.assign(desc.mips * texture->array_size() * texture->planes,
                         D3D12_RESOURCE_STATE_COMMON);
  if (desc.name)
    SetName(&*texture, desc.name);
  return gpu::rhi::Release(texture, this);
}

rhi::TextureView* D3D12Device::CreateView(rhi::Texture* texture,
                                          const rhi::TextureViewDesc& desc) {
  D3D12Texture* tex = Tex(texture);
  const rhi::TextureDesc& td = texture->desc();
  const rhi::Format format =
      desc.format == rhi::Format::kUndefined ? td.format : desc.format;
  const DxgiInfo& fi = Dxgi(format);
  if (!fi.texture)
    return nullptr;
  if (format != td.format && fi.typeless != Dxgi(td.format).typeless) {
    // Casting across format families needs relaxed casting, which this
    // backend does not use.
    BASE_LOGI("gpud3d12", "view {} of a {} texture is not castable",
              rhi::FormatName(format), rhi::FormatName(td.format));
    return nullptr;
  }
  auto view = base::MakeUnique<D3D12View>(texture, desc);
  const bool is3d = td.dim == rhi::TextureDim::k3D;
  const bool arrayed = td.layers > 1 || desc.base_layer > 0;
  const bool stencil = desc.aspect == rhi::kAspectStencil;
  const bool depth = desc.aspect & (rhi::kAspectDepth | rhi::kAspectStencil);

  if (td.usage & rhi::kTextureSampled) {
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = stencil ? fi.stencil_srv : depth ? fi.srv : fi.format;
    const u32 identity[4] = {0, 1, 2, 3};
    const u32 green[4] = {1, 1, 1, 1};
    sv.Shader4ComponentMapping = Mapping(desc, stencil ? green : identity);
    const UINT plane = stencil ? 1 : 0;
    switch (desc.dim) {
      case rhi::ViewDim::k1D:
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
        sv.Texture1D = {desc.base_mip, desc.mips, 0.0f};
        if (!arrayed)
          break;
        [[fallthrough]];
      case rhi::ViewDim::k1DArray:
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
        sv.Texture1DArray = {desc.base_mip, desc.mips, desc.base_layer,
                             desc.layers, 0.0f};
        break;
      case rhi::ViewDim::k2D:
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D = {desc.base_mip, desc.mips, plane, 0.0f};
        if (!arrayed)
          break;
        [[fallthrough]];
      case rhi::ViewDim::k2DArray:
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        sv.Texture2DArray = {desc.base_mip, desc.mips, desc.base_layer,
                             desc.layers,   plane,     0.0f};
        break;
      case rhi::ViewDim::kCube:
        if (desc.base_layer == 0) {
          sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
          sv.TextureCube = {desc.base_mip, desc.mips, 0.0f};
          break;
        }
        [[fallthrough]];
      case rhi::ViewDim::kCubeArray:
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        sv.TextureCubeArray = {desc.base_mip, desc.mips, desc.base_layer,
                               base::Max(desc.layers / 6, 1u), 0.0f};
        break;
      case rhi::ViewDim::k3D:
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        sv.Texture3D = {desc.base_mip, desc.mips, 0.0f};
        break;
    }
    if (views.Allocate(1, &view->srv))
      device->CreateShaderResourceView(tex->resource, &sv, view->srv.cpu);
  }

  if ((tex->flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) && !depth) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
    uv.Format = fi.format;
    switch (desc.dim) {
      case rhi::ViewDim::k1D:
      case rhi::ViewDim::k1DArray:
        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
        uv.Texture1DArray = {desc.base_mip, desc.base_layer, desc.layers};
        if (desc.dim == rhi::ViewDim::k1D && !arrayed) {
          uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
          uv.Texture1D = {desc.base_mip};
        }
        break;
      case rhi::ViewDim::k3D:
        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uv.Texture3D = {desc.base_mip, 0, ~0u};
        break;
      default:
        if (desc.dim == rhi::ViewDim::k2D && !arrayed) {
          uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
          uv.Texture2D = {desc.base_mip, 0};
        } else if (is3d) {
          uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
          uv.Texture3D = {desc.base_mip, desc.base_layer, desc.layers};
        } else {
          uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
          uv.Texture2DArray = {desc.base_mip, desc.base_layer, desc.layers, 0};
        }
        break;
    }
    if (views.Allocate(1, &view->uav))
      device->CreateUnorderedAccessView(tex->resource, nullptr, &uv,
                                        view->uav.cpu);
  }

  if ((tex->flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) && !depth) {
    D3D12_RENDER_TARGET_VIEW_DESC rv{};
    rv.Format = fi.format;
    if (is3d) {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
      rv.Texture3D = {desc.base_mip, desc.base_layer, desc.layers};
    } else if (td.dim == rhi::TextureDim::k1D) {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
      rv.Texture1DArray = {desc.base_mip, desc.base_layer, desc.layers};
    } else if (arrayed) {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
      rv.Texture2DArray = {desc.base_mip, desc.base_layer, desc.layers, 0};
    } else {
      rv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
      rv.Texture2D = {desc.base_mip, 0};
    }
    if (rtvs.Allocate(1, &view->rtv))
      device->CreateRenderTargetView(tex->resource, &rv, view->rtv.cpu);
  }

  if (tex->flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    Dsv(&*view, 0);
  return gpu::rhi::Release(view, this);
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::Dsv(D3D12View* view, u32 variant) {
  D3D12Texture* tex = Tex(view->texture());
  const rhi::TextureDesc& td = tex->desc();
  if (!rhi::GetFormatInfo(td.format).is_stencil)
    variant &= 1;
  base::LockGuard<base::Mutex> lock(tex->view_mutex);
  CpuRange& range = view->dsv[variant];
  if (range.valid() || !dsvs.Allocate(1, &range))
    return range.cpu;
  const rhi::TextureViewDesc& desc = view->desc();
  D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
  dv.Format = Dxgi(td.format).format;
  dv.Flags = static_cast<D3D12_DSV_FLAGS>(
      ((variant & 1) ? D3D12_DSV_FLAG_READ_ONLY_DEPTH : 0) |
      ((variant & 2) ? D3D12_DSV_FLAG_READ_ONLY_STENCIL : 0));
  if (td.layers > 1 || desc.base_layer > 0) {
    dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    dv.Texture2DArray = {desc.base_mip, desc.base_layer, desc.layers};
  } else {
    dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    dv.Texture2D = {desc.base_mip};
  }
  device->CreateDepthStencilView(tex->resource, &dv, range.cpu);
  return range.cpu;
}

rhi::Sampler* D3D12Device::CreateSampler(const rhi::SamplerDesc& desc) {
  auto sampler = base::MakeUnique<D3D12Sampler>();
  const base::String key = PackSampler(desc);
  base::LockGuard<base::Mutex> lock(sampler_mutex_);
  auto it = sampler_ids_.find(key);
  if (it != sampler_ids_.end()) {
    sampler->id = it->second;
    return gpu::rhi::Release(sampler, this);
  }
  D3D12_SAMPLER_DESC sd{};
  const bool aniso = desc.max_anisotropy > 1.0f;
  const u32 filter = (u32(desc.min == rhi::Filter::kLinear) << 4) |
                     (u32(desc.mag == rhi::Filter::kLinear) << 2) |
                     u32(desc.mip == rhi::Filter::kLinear);
  sd.Filter = static_cast<D3D12_FILTER>(
      (aniso ? u32(D3D12_FILTER_ANISOTROPIC) : filter) |
      (desc.compare_enable ? 0x80u : 0u));
  sd.AddressU = ToAddress(desc.address_u);
  sd.AddressV = ToAddress(desc.address_v);
  sd.AddressW = ToAddress(desc.address_w);
  sd.MipLODBias = base::Clamp(desc.lod_bias, D3D12_MIP_LOD_BIAS_MIN,
                              D3D12_MIP_LOD_BIAS_MAX);
  sd.MaxAnisotropy = aniso
                         ? base::Min<u32>(D3D12_MAX_MAXANISOTROPY,
                                          static_cast<u32>(desc.max_anisotropy))
                         : 1u;
  sd.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(
      desc.compare_enable ? u32(desc.compare) + 1 : 1);
  const float border =
      desc.border == rhi::BorderColor::kOpaqueWhite ? 1.0f : 0.0f;
  sd.BorderColor[0] = sd.BorderColor[1] = sd.BorderColor[2] = border;
  sd.BorderColor[3] =
      desc.border == rhi::BorderColor::kTransparentBlack ? 0.0f : 1.0f;
  sd.MinLOD = desc.min_lod;
  sd.MaxLOD = desc.max_lod;
  CpuRange range;
  if (!samplers.Allocate(1, &range))
    return nullptr;
  device->CreateSampler(&sd, range.cpu);
  sampler->id = static_cast<u32>(sampler_cpu_.size());
  sampler_cpu_.push_back(range);
  sampler_ids_[key] = sampler->id;
  return gpu::rhi::Release(sampler, this);
}

D3D12_GPU_DESCRIPTOR_HANDLE D3D12Device::SamplerTable(
    const base::Vector<u32>& ids,
    u32* cached,
    u64* generation) {
  base::LockGuard<base::Mutex> lock(sampler_mutex_);
  const D3D12_GPU_DESCRIPTOR_HANDLE base =
      sampler_heap->GetGPUDescriptorHandleForHeapStart();
  if (cached && *cached != ~0u && *generation == sampler_generation_)
    return {base.ptr + u64(*cached) * sampler_increment};
  u32 offset;
  auto it = sampler_tables_.find(ids);
  if (it != sampler_tables_.end()) {
    offset = it->second;
  } else {
    if (sampler_heap_used_ + ids.size() > kSamplerHeapSize) {
      // Every table is rebuilt on its next bind. Work in flight still reads
      // the old ones, so it has to finish first.
      BASE_LOGI("gpud3d12", "sampler heap full: {} tables, starting over",
                sampler_tables_.size());
      WaitIdle();
      sampler_tables_.clear();
      sampler_heap_used_ = 0;
      sampler_generation_++;
    }
    offset = sampler_heap_used_;
    sampler_heap_used_ += static_cast<u32>(ids.size());
    const D3D12_CPU_DESCRIPTOR_HANDLE cpu =
        sampler_heap->GetCPUDescriptorHandleForHeapStart();
    for (size_t i = 0; i < ids.size(); i++)
      device->CopyDescriptorsSimple(
          1, {cpu.ptr + (offset + i) * sampler_increment},
          sampler_cpu_[ids[i] < sampler_cpu_.size() ? ids[i] : 0].cpu,
          D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    sampler_tables_[ids] = offset;
  }
  if (cached) {
    *cached = offset;
    *generation = sampler_generation_;
  }
  return {base.ptr + u64(offset) * sampler_increment};
}

bool D3D12Device::AcquireChunk(RingChunk* out) {
  base::LockGuard<base::Mutex> lock(chunk_mutex_);
  if (free_chunks_.empty())
    return false;
  out->first = free_chunks_.back();
  out->used = 0;
  free_chunks_.pop_back();
  return true;
}

void D3D12Device::ReleaseChunk(const RingChunk& chunk) {
  base::LockGuard<base::Mutex> lock(chunk_mutex_);
  free_chunks_.push_back(chunk.first);
}

rhi::BindGroupLayout* D3D12Device::CreateBindGroupLayout(
    const rhi::BindGroupLayoutDesc& desc) {
  auto layout = base::MakeUnique<D3D12BindGroupLayout>(desc);
  for (const rhi::BindingLayout& b : desc.bindings) {
    GroupEntry e;
    e.binding = b.binding;
    e.type = b.type;
    e.read_only = b.read_only;
    e.slot = layout->table_size++;
    if (b.type == rhi::BindingType::kSampledTexture)
      e.sampler_slot = layout->sampler_count++;
    layout->entries.push_back(e);
  }
  // Dynamic offsets arrive in binding order, as in Vulkan.
  for (u32 i = 0; i < layout->entries.size(); i++) {
    const rhi::BindingType t = layout->entries[i].type;
    if (t == rhi::BindingType::kUniformBufferDynamic ||
        t == rhi::BindingType::kStorageBufferDynamic)
      layout->dynamic_entries.push_back(i);
  }
  base::Sort(layout->dynamic_entries.begin(), layout->dynamic_entries.end(),
             [&](u32 a, u32 b) {
               return layout->entries[a].binding < layout->entries[b].binding;
             });
  for (u32 d = 0; d < layout->dynamic_entries.size(); d++)
    layout->entries[layout->dynamic_entries[d]].dynamic_index = d;
  return gpu::rhi::Release(layout, this);
}

void D3D12Device::WriteCbv(D3D12_CPU_DESCRIPTOR_HANDLE dst,
                           D3D12Buffer* buffer,
                           u64 offset,
                           u64 range) {
  if (!buffer) {
    device->CreateConstantBufferView(nullptr, dst);
    return;
  }
  if (!range || offset + range > buffer->alloc_size)
    range = buffer->alloc_size - base::Min(offset, buffer->alloc_size);
  D3D12_CONSTANT_BUFFER_VIEW_DESC cv{};
  cv.BufferLocation = buffer->va + offset;
  cv.SizeInBytes = static_cast<UINT>(base::Min<u64>(
      AlignUp(base::Min<u64>(range, 65536), 256), buffer->alloc_size - offset));
  device->CreateConstantBufferView(&cv, dst);
}

void D3D12Device::WriteRawView(D3D12_CPU_DESCRIPTOR_HANDLE dst,
                               D3D12Buffer* buffer,
                               u64 offset,
                               u64 range,
                               bool uav) {
  if (!buffer || (uav && buffer->fixed_state)) {
    WriteNull(dst, rhi::BindingType::kStorageBuffer, !uav);
    return;
  }
  const u64 size = buffer->desc().size;
  if (!range || offset + range > size)
    range = size - base::Min(offset, size);
  if (uav) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
    uv.Format = DXGI_FORMAT_R32_TYPELESS;
    uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uv.Buffer.FirstElement = offset / 4;
    uv.Buffer.NumElements = static_cast<UINT>(range / 4);
    uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device->CreateUnorderedAccessView(buffer->resource, nullptr, &uv, dst);
  } else {
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = DXGI_FORMAT_R32_TYPELESS;
    sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Buffer.FirstElement = offset / 4;
    sv.Buffer.NumElements = static_cast<UINT>(range / 4);
    sv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device->CreateShaderResourceView(buffer->resource, &sv, dst);
  }
}

void D3D12Device::WriteNull(D3D12_CPU_DESCRIPTOR_HANDLE dst,
                            rhi::BindingType type,
                            bool read_only) {
  switch (type) {
    case rhi::BindingType::kSampledTexture: {
      D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
      sv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      sv.Texture2D.MipLevels = 1;
      device->CreateShaderResourceView(nullptr, &sv, dst);
      break;
    }
    case rhi::BindingType::kStorageTexture: {
      D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
      uv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      device->CreateUnorderedAccessView(nullptr, nullptr, &uv, dst);
      break;
    }
    case rhi::BindingType::kUniformBuffer:
    case rhi::BindingType::kUniformBufferDynamic:
      device->CreateConstantBufferView(nullptr, dst);
      break;
    case rhi::BindingType::kStorageBuffer:
    case rhi::BindingType::kStorageBufferDynamic:
      if (read_only) {
        D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = DXGI_FORMAT_R32_TYPELESS;
        sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        device->CreateShaderResourceView(nullptr, &sv, dst);
      } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.Format = DXGI_FORMAT_R32_TYPELESS;
        uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(nullptr, nullptr, &uv, dst);
      }
      break;
  }
}

rhi::BindGroup* D3D12Device::CreateBindGroup(const rhi::BindGroupDesc& desc) {
  auto* layout = static_cast<D3D12BindGroupLayout*>(desc.layout);
  auto group = base::MakeUnique<D3D12BindGroup>();
  group->layout = layout;
  if (layout->table_size && !views.Allocate(layout->table_size, &group->table))
    return nullptr;
  for (const GroupEntry& e : layout->entries)
    WriteNull(views.At(group->table, e.slot), e.type, e.read_only);
  group->samplers.assign(layout->sampler_count, 0);
  group->dynamic.resize(layout->dynamic_entries.size());
  group->uses.resize(layout->entries.size());
  UpdateBindGroup(&*group, desc.writes.data(),
                  static_cast<u32>(desc.writes.size()));
  return gpu::rhi::Release(group, this);
}

void D3D12Device::UpdateBindGroup(rhi::BindGroup* bind_group,
                                  const rhi::BindingWrite* writes,
                                  u32 count) {
  auto* group = static_cast<D3D12BindGroup*>(bind_group);
  const D3D12BindGroupLayout* layout = group->layout;
  for (u32 i = 0; i < count; i++) {
    const rhi::BindingWrite& w = writes[i];
    const GroupEntry* e = layout->Find(w.binding);
    if (!e)
      continue;
    const size_t index = e - layout->entries.data();
    const D3D12_CPU_DESCRIPTOR_HANDLE dst = views.At(group->table, e->slot);
    D3D12Buffer* buffer = Buf(w.buffer);
    group->uses[index] = {};
    switch (e->type) {
      case rhi::BindingType::kSampledTexture:
        if (w.view && View(w.view)->srv.valid())
          device->CopyDescriptorsSimple(1, dst, View(w.view)->srv.cpu,
                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        else
          WriteNull(dst, e->type, true);
        group->samplers[e->sampler_slot] =
            w.sampler ? static_cast<D3D12Sampler*>(w.sampler)->id : 0;
        group->sampler_table = ~0u;
        break;
      case rhi::BindingType::kStorageTexture:
        if (w.view && View(w.view)->uav.valid())
          device->CopyDescriptorsSimple(1, dst, View(w.view)->uav.cpu,
                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        else
          WriteNull(dst, e->type, false);
        break;
      case rhi::BindingType::kUniformBuffer:
        WriteCbv(dst, buffer, w.offset, w.range);
        group->uses[index] = {buffer,
                              D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER};
        break;
      case rhi::BindingType::kStorageBuffer:
        WriteRawView(dst, buffer, w.offset, w.range, !e->read_only);
        group->uses[index] = {
            buffer, e->read_only
                        ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                              D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                        : D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
        break;
      case rhi::BindingType::kUniformBufferDynamic:
      case rhi::BindingType::kStorageBufferDynamic:
        group->dynamic[e->dynamic_index] = w;
        group->uses[index] = {
            buffer, e->type == rhi::BindingType::kUniformBufferDynamic
                        ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                    : e->read_only
                        ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                              D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                        : D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
        break;
    }
  }
}

rhi::TimestampPool* D3D12Device::CreateTimestampPool(u32 count) {
  auto pool = base::MakeUnique<D3D12TimestampPool>();
  D3D12_QUERY_HEAP_DESC qd{};
  qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  qd.Count = count;
  if (FAILED(device->CreateQueryHeap(&qd, IID_ID3D12QueryHeap,
                                     reinterpret_cast<void**>(&pool->heap))))
    return nullptr;
  pool->readback = CreateBufferResource(
      u64(count) * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_COPY_DEST);
  void* p = nullptr;
  if (!pool->readback || FAILED(pool->readback->Map(0, nullptr, &p))) {
    SafeRelease(pool->heap);
    SafeRelease(pool->readback);
    return nullptr;
  }
  pool->results = static_cast<const u64*>(p);
  pool->count = count;
  return gpu::rhi::Release(pool, this);
}

rhi::CommandList* D3D12Device::CreateCommandList() {
  auto list = base::MakeUnique<D3D12CommandList>(*this);
  if (!list->Init())
    return nullptr;
  return gpu::rhi::Release(list, this);
}

void D3D12Device::Destroy(rhi::Object* object) {
  Forget(object);
  if (!object)
    return;
  if (auto* b = dynamic_cast<D3D12Buffer*>(object)) {
    SafeRelease(b->resource);
  } else if (auto* t = dynamic_cast<D3D12Texture*>(object)) {
    for (auto& [key, r] : t->rtvs)
      rtvs.Free(r);
    for (auto& [key, r] : t->dsvs)
      dsvs.Free(r);
    SafeRelease(t->resource);
  } else if (auto* v = dynamic_cast<D3D12View*>(object)) {
    views.Free(v->srv);
    views.Free(v->uav);
    rtvs.Free(v->rtv);
    for (CpuRange& r : v->dsv)
      dsvs.Free(r);
  } else if (auto* g = dynamic_cast<D3D12BindGroup*>(object)) {
    views.Free(g->table);
  } else if (auto* pl = dynamic_cast<D3D12PipelineLayout*>(object)) {
    SafeRelease(pl->root);
  } else if (auto* p = dynamic_cast<D3D12Pipeline*>(object)) {
    SafeRelease(p->pso);
    SafeRelease(p->pso_restart32);
  } else if (auto* q = dynamic_cast<D3D12TimestampPool*>(object)) {
    SafeRelease(q->heap);
    SafeRelease(q->readback);
  }
  // Samplers stay in the deduplicated table: tables built from their id may
  // still be bound.
  delete object;
}

void D3D12Device::SetName(rhi::Object* object, const char* name) {
  if (!debug_labels || !object || !name)
    return;
  ID3D12Object* target = nullptr;
  if (auto* b = dynamic_cast<D3D12Buffer*>(object))
    target = b->resource;
  else if (auto* t = dynamic_cast<D3D12Texture*>(object))
    target = t->resource;
  else if (auto* p = dynamic_cast<D3D12Pipeline*>(object))
    target = p->pso;
  if (!target)
    return;
  base::Vector<WCHAR> wide;
  for (const char* c = name; *c; c++)
    wide.push_back(static_cast<WCHAR>(static_cast<u8>(*c)));
  wide.push_back(0);
  target->SetName(wide.data());
}

bool D3D12Device::SupportsFormat(rhi::Format format, u32 usage) const {
  const DxgiInfo& fi = Dxgi(format);
  if (!fi.texture)
    return false;
  auto support = [&](DXGI_FORMAT f) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{f};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs,
                                           sizeof(fs))))
      return u32(0);
    return u32(fs.Support1);
  };
  const u32 view = support(fi.format);
  if ((usage & rhi::kTextureSampled) &&
      !(support(fi.srv) & (D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE |
                           D3D12_FORMAT_SUPPORT1_SHADER_LOAD)))
    return false;
  if ((usage & rhi::kTextureStorage) &&
      !(view & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW))
    return false;
  if ((usage & rhi::kTextureColorTarget) &&
      !(view & D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
    return false;
  if ((usage & rhi::kTextureDepthTarget) &&
      !(view & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL))
    return false;
  return true;
}

D3D12_RESOURCE_STATES D3D12Device::TextureState(const D3D12Texture* texture,
                                                rhi::TextureState state) const {
  const D3D12_RESOURCE_STATES kRead =
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
      D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  switch (state) {
    case rhi::TextureState::kUndefined:
      return D3D12_RESOURCE_STATE_COMMON;
    case rhi::TextureState::kGeneral:
      return (texture->flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
                 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                 : D3D12_RESOURCE_STATE_COMMON;
    case rhi::TextureState::kColorTarget:
      return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case rhi::TextureState::kDepthTarget:
      return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    case rhi::TextureState::kDepthRead:
      return D3D12_RESOURCE_STATE_DEPTH_READ |
             ((texture->desc().usage & rhi::kTextureSampled)
                  ? kRead
                  : D3D12_RESOURCE_STATE_COMMON);
    case rhi::TextureState::kShaderRead:
      return kRead;
    case rhi::TextureState::kCopySrc:
      return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case rhi::TextureState::kCopyDst:
      return D3D12_RESOURCE_STATE_COPY_DEST;
  }
  return D3D12_RESOURCE_STATE_COMMON;
}

u64 D3D12Device::Submit(rhi::CommandList* const* lists, u32 count) {
  base::LockGuard<base::Mutex> lock(queue_mutex_);
  // One ExecuteCommandLists per list: buffers decay to COMMON between calls,
  // which is what each list's own state tracking starts from.
  for (u32 i = 0; i < count; i++) {
    ID3D12CommandList* cmd = static_cast<D3D12CommandList*>(lists[i])->cmd;
    queue->ExecuteCommandLists(1, &cmd);
  }
  const u64 value = submitted_ + 1;
  if (FAILED(queue->Signal(fence_, value))) {
    ReportDeviceLoss();
    return submitted_;
  }
  submitted_ = value;
  return value;
}

bool D3D12Device::IsComplete(u64 submission) {
  return fence_->GetCompletedValue() >= submission;
}

bool D3D12Device::Wait(u64 submission, u64 timeout_ns) {
  if (fence_->GetCompletedValue() >= submission)
    return true;
  const u32 ms = timeout_ns == ~0ull
                     ? ~0u
                     : static_cast<u32>(base::Min<u64>(
                           (timeout_ns + 999999) / 1000000, ~0u - 1));
  bool done = false;
#if defined(_WIN32)
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (SUCCEEDED(fence_->SetEventOnCompletion(submission, event)))
    done =
        WaitForSingleObject(event, ms == ~0u ? INFINITE : ms) == WAIT_OBJECT_0;
  CloseHandle(event);
#else
  HANDLE event = vkd3d_create_event();
  if (SUCCEEDED(fence_->SetEventOnCompletion(submission, event)))
    done = vkd3d_wait_event(event, ms == ~0u ? VKD3D_INFINITE : ms) ==
           VKD3D_WAIT_OBJECT_0;
  vkd3d_destroy_event(event);
#endif
  done = done || fence_->GetCompletedValue() >= submission;
  if (!done && FAILED(device->GetDeviceRemovedReason()))
    ReportDeviceLoss();
  return done;
}

void D3D12Device::WaitIdle() {
  Wait(submitted_, ~0ull);
}

bool D3D12Device::ReadTimestamps(rhi::TimestampPool* pool,
                                 u32 first,
                                 u32 count,
                                 u64* out) {
  auto* p = static_cast<D3D12TimestampPool*>(pool);
  if (first + count > p->count)
    return false;
  std::memcpy(out, p->results + first, count * sizeof(u64));
  return true;
}

void D3D12Device::ReportDeviceLoss() {
  if (fault_reported_)
    return;
  fault_reported_ = true;
  BASE_LOGI("gpud3d12", "device removed: {:#x}",
            static_cast<u32>(device->GetDeviceRemovedReason()));
}

}  // namespace impl

base::UniquePointer<rhi::Device> CreateD3D12Device(
    const D3D12Options& options) {
  auto device = base::MakeUnique<D3D12Device>();
  if (!device->Init(options))
    return nullptr;
  return device;
}

}  // namespace gpu::d3d12
