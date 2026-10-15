/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/image/bindless.h"

#include <cstring>

#include "base/algorithm.h"
#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/render/frame.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/rhi/device.h"
#include "guest/session.h"
#include "options/options.h"

namespace gpu::render::bindless {
namespace {
DELTA_OPTION(bool, kBindless, "DELTA_GPU_BINDLESS", false);

using gcn::kBindlessKeyWords;
using gcn::kBindlessKinds;
using gcn::kBindlessPending;
using gcn::kBindlessSlots;
using gcn::kBindlessStride;

constexpr u32 kTexturesPerKind = 4096;
constexpr u32 kSamplers = 2048;
constexpr u32 kSamplerPool = kBindlessKinds;
constexpr int kEvictFrames = 600;
constexpr int kReuseFrames = 3;

constexpr u64 TableBytes(u32 which) {
  return u64(kBindlessSlots[which]) * kBindlessStride[which] * 4 + 16;
}

struct Key {
  u32 kind = 0;
  u32 words[kBindlessKeyWords[0]] = {};
  bool operator==(const Key& o) const {
    return kind == o.kind && !std::memcmp(words, o.words, sizeof(words));
  }
};

struct KeyHash {
  size_t operator()(const Key& k) const {
    u32 h = (k.kind + 1) * 0x9E3779B1u;
    for (u32 w : k.words)
      h = (h ^ w) * 0x01000193u;
    return h ^ (h >> 15);
  }
};

struct Live {
  Key key;
  rhi::TextureView* view = nullptr;
  rhi::TextureState state = rhi::TextureState::kShaderRead;
  rhi::Sampler* sampler = nullptr;
  u32 index = 0;
  base::Vector<u32> slots;
};

struct Table {
  rhi::Buffer* buffer = nullptr;
  base::Vector<Live> live;
  base::HashMap<Key, u32, KeyHash> by_key;
  base::HashMap<u32, u32> owner;
  base::HashMap<u32, int> completing;
  base::Vector<Key> pending;
  base::Vector<u32> pending_slots;
};

struct Retired {
  u32 pool = 0;
  u32 index = 0;
  int frame = 0;
};

struct Update {
  u32 which = 0;
  u32 slot = 0;
  u32 words[kBindlessStride[0]] = {};
};

struct State {
  bool tried = false;
  bool ok = false;
  bool cleared = false;
  rhi::BindGroupLayout* layout = nullptr;
  rhi::BindGroup* group = nullptr;
  Table tables[2];
  rhi::Buffer* readback[2] = {};
  u32 next_index[kBindlessKinds + 1] = {};
  base::Vector<u32> free_index[kBindlessKinds + 1];
  base::Vector<Retired> retired;
  base::Vector<Update> updates;
  TextureResolver resolve_texture = nullptr;
  SamplerResolver resolve_sampler = nullptr;
};

State g;

rhi::TextureView* Placeholder(u32 kind) {
  const bool arrayed = kind & 1, is_3d = kind & 2, uint = kind & 4;
  if (is_3d)
    return uint ? g_tex.white_uint_3d_view : g_tex.zero_3d_view;
  if (arrayed)
    return uint ? g_tex.white_uint_array_view : g_tex.zero_array_view;
  return uint ? g_tex.white_uint_view : g_tex.zero_view;
}

void WriteTexture(u32 kind,
                  u32 index,
                  rhi::TextureView* view,
                  rhi::TextureState state = rhi::TextureState::kShaderRead) {
  rhi::BindingWrite write;
  write.binding = kind;
  write.array_index = index;
  write.view = view;
  write.view_state = state;
  Device().UpdateBindGroup(g.group, &write, 1);
}

void WriteSampler(u32 index, rhi::Sampler* sampler) {
  rhi::BindingWrite write;
  write.binding = gcn::kBindlessSamplerBinding;
  write.array_index = index;
  write.sampler = sampler;
  Device().UpdateBindGroup(g.group, &write, 1);
}

bool Init() {
  if (g.tried)
    return g.ok;
  if (!g_tex.descriptors_ready)
    return false;
  g.tried = true;
  if (!Supported())
    return false;
  rhi::Device& device = Device();
  constexpr u32 kStages =
      rhi::kStageVertex | rhi::kStageFragment | rhi::kStageCompute;
  rhi::BindGroupLayoutDesc desc;
  for (u32 kind = 0; kind < kBindlessKinds; kind++)
    desc.bindings.push_back({kind, rhi::BindingType::kTexture, kStages, true,
                             kTexturesPerKind, true});
  desc.bindings.push_back({gcn::kBindlessSamplerBinding,
                           rhi::BindingType::kSampler, kStages, true,
                           kSamplers, true});
  for (u32 which = 0; which < 2; which++)
    desc.bindings.push_back({gcn::kBindlessTableBinding + which,
                             rhi::BindingType::kStorageBuffer, kStages});
  g.layout = device.CreateBindGroupLayout(desc);
  if (!g.layout)
    return false;
  rhi::BindingWrite tables[2];
  for (u32 which = 0; which < 2; which++) {
    rhi::BufferDesc table;
    table.usage =
        rhi::kBufferStorage | rhi::kBufferCopySrc | rhi::kBufferCopyDst;
    table.size = TableBytes(which);
    table.name = which ? "bindless samplers" : "bindless textures";
    g.tables[which].buffer = device.CreateBuffer(table);
    if (!g.tables[which].buffer)
      return false;
    tables[which].binding = gcn::kBindlessTableBinding + which;
    tables[which].buffer = g.tables[which].buffer;
    tables[which].range = TableBytes(which);
  }
  rhi::BufferDesc readback;
  readback.usage = rhi::kBufferCopyDst;
  readback.memory = rhi::MemoryKind::kReadback;
  readback.size = TableBytes(0) + TableBytes(1);
  readback.name = "bindless readback";
  for (auto& buffer : g.readback) {
    buffer = device.CreateBuffer(readback);
    if (!buffer || !buffer->mapped())
      return false;
  }
  rhi::BindGroupDesc group;
  group.layout = g.layout;
  group.writes = {tables[0], tables[1]};
  g.group = device.CreateBindGroup(group);
  if (!g.group)
    return false;
  for (u32 kind = 0; kind < kBindlessKinds; kind++)
    WriteTexture(kind, 0, Placeholder(kind));
  WriteSampler(0, g_tex.sampler);
  for (u32& next : g.next_index)
    next = 1;
  g.ok = true;
  return true;
}

u32 TakeIndex(u32 pool) {
  if (!g.free_index[pool].empty()) {
    const u32 index = g.free_index[pool].back();
    g.free_index[pool].pop_back();
    return index;
  }
  const u32 limit = pool == kSamplerPool ? kSamplers : kTexturesPerKind;
  return g.next_index[pool] < limit ? g.next_index[pool]++ : 0;
}

void Retire(u32 pool, u32 index) {
  if (index)
    g.retired.push_back({pool, index, g_frame.num});
}

void QueueEntry(u32 which, u32 slot, const Key& key, u32 index) {
  Update& u = g.updates.emplace_back();
  u.which = which;
  u.slot = slot;
  u.words[0] = key.kind + 1;
  std::memcpy(&u.words[1], key.words, kBindlessKeyWords[which] * 4);
  u.words[kBindlessKeyWords[which] + 1] = index;
  u.words[kBindlessKeyWords[which] + 2] = g_frame.num;
}

void QueueClear(u32 which, u32 slot) {
  Update& u = g.updates.emplace_back();
  u.which = which;
  u.slot = slot;
}

void Resolve(u32 which, const Key& key, u32 slot) {
  Table& table = g.tables[which];
  auto it = table.by_key.find(key);
  if (it == table.by_key.end()) {
    Live live;
    live.key = key;
    if (which == 0) {
      live.view = g.resolve_texture
                      ? g.resolve_texture(key.kind, key.words, &live.state)
                      : nullptr;
      live.index = live.view ? TakeIndex(key.kind) : 0;
      if (live.index)
        WriteTexture(key.kind, live.index, live.view, live.state);
    } else {
      live.sampler = g.resolve_sampler
                         ? g.resolve_sampler(key.kind, key.words)
                         : nullptr;
      live.index = live.sampler ? TakeIndex(kSamplerPool) : 0;
      if (live.index)
        WriteSampler(live.index, live.sampler);
    }
    it = table.by_key.emplace(key, static_cast<u32>(table.live.size())).first;
    table.live.push_back(live);
  }
  Live& live = table.live[it->second];
  live.slots.push_back(slot);
  table.owner[slot] = it->second;
  QueueEntry(which, slot, key, live.index);
}

void Evict(u32 which, u32 slot) {
  Table& table = g.tables[which];
  const auto owner = table.owner.find(slot);
  if (owner == table.owner.end())
    return;
  const u32 at = owner->second;
  table.owner.erase(owner);
  QueueClear(which, slot);
  Live& live = table.live[at];
  base::EraseIf(live.slots, [slot](u32 s) { return s == slot; });
  if (!live.slots.empty())
    return;
  Retire(which ? kSamplerPool : live.key.kind, live.index);
  table.by_key.erase(live.key);
  if (at + 1 != table.live.size()) {
    live = base::move(table.live.back());
    table.by_key[live.key] = at;
    for (u32 s : live.slots)
      table.owner[s] = at;
  }
  table.live.pop_back();
}

void Refresh() {
  if (!g.resolve_texture)
    return;
  Table& table = g.tables[0];
  for (Live& live : table.live) {
    rhi::TextureState state = rhi::TextureState::kShaderRead;
    rhi::TextureView* view =
        g.resolve_texture(live.key.kind, live.key.words, &state);
    if (view == live.view && state == live.state)
      continue;
    const u32 index = view ? TakeIndex(live.key.kind) : 0;
    if (view && !index)
      continue;
    if (index)
      WriteTexture(live.key.kind, index, view, state);
    Retire(live.key.kind, live.index);
    live.view = view;
    live.state = state;
    live.index = index;
    for (u32 slot : live.slots)
      QueueEntry(0, slot, live.key, index);
  }
}

void ReleaseRetired() {
  base::EraseIf(g.retired, [](const Retired& r) {
    if (g_frame.num < r.frame + kReuseFrames)
      return false;
    g.free_index[r.pool].push_back(r.index);
    return true;
  });
}

void Collect(u32 which, const u32* words) {
  Table& table = g.tables[which];
  const u32 stride = kBindlessStride[which];
  const u32 key_words = kBindlessKeyWords[which];
  for (u32 slot = 0; slot < kBindlessSlots[which]; slot++) {
    const u32* entry = words + slot * stride;
    if (!entry[0])
      continue;
    if (!(entry[0] & kBindlessPending)) {
      const int stamp = static_cast<int>(entry[key_words + 2]);
      if (g_frame.num - stamp > kEvictFrames)
        Evict(which, slot);
      continue;
    }
    auto done = table.completing.find(slot);
    if (done != table.completing.end() &&
        g_frame.num < done->second + kReuseFrames)
      continue;
    table.completing[slot] = g_frame.num;
    Key key;
    key.kind = (entry[0] & ~kBindlessPending) - 1;
    std::memcpy(key.words, entry + 1, key_words * 4);
    table.pending.push_back(key);
    table.pending_slots.push_back(slot);
  }
}

const guest::SessionReset g_reset([] { g = {}; });
}  // namespace

bool Supported() {
  const rhi::Caps& caps = Device().caps();
  return kBindless && caps.bindless &&
         caps.max_bindless_textures >= kTexturesPerKind;
}

bool Enabled() {
  return kBindless && Init();
}

void SetResolvers(TextureResolver texture, SamplerResolver sampler) {
  g.resolve_texture = texture;
  g.resolve_sampler = sampler;
}

rhi::BindGroupLayout* Layout() {
  return Enabled() ? g.layout : nullptr;
}

rhi::BindGroup* Group() {
  return Enabled() ? g.group : nullptr;
}

void FrameBegin(rhi::CommandList* list) {
  if (!Enabled())
    return;
  if (!g.cleared) {
    g.cleared = true;
    for (const Table& table : g.tables)
      list->FillBuffer(table.buffer, 0, ~0ull, 0);
  }
  for (u32 which = 0; which < 2; which++) {
    Table& table = g.tables[which];
    for (size_t i = 0; i < table.pending.size(); i++)
      Resolve(which, table.pending[i], table.pending_slots[i]);
    table.pending.clear();
    table.pending_slots.clear();
  }
  Refresh();
  ReleaseRetired();
  list->Barrier(rhi::kAccessShaderRead | rhi::kAccessShaderWrite |
                    rhi::kAccessComputeRead | rhi::kAccessComputeWrite,
                rhi::kAccessCopyWrite);
  const u32 now = g_frame.num;
  for (u32 which = 0; which < 2; which++)
    list->UpdateBuffer(g.tables[which].buffer, TableBytes(which) - 16, 4,
                       &now);
  for (const Update& u : g.updates) {
    const u32 stride = kBindlessStride[u.which];
    list->UpdateBuffer(g.tables[u.which].buffer, u64(u.slot) * stride * 4,
                       stride * 4, u.words);
  }
  g.updates.clear();
  list->Barrier(rhi::kAccessCopyWrite,
                rhi::kAccessShaderRead | rhi::kAccessShaderWrite |
                    rhi::kAccessComputeRead | rhi::kAccessComputeWrite);
}

void FrameEnd(rhi::CommandList* list, u32 slot) {
  if (!Enabled())
    return;
  list->Barrier(rhi::kAccessShaderWrite | rhi::kAccessComputeWrite,
                rhi::kAccessCopyRead);
  list->CopyBuffer(g.readback[slot], 0, g.tables[0].buffer, 0, TableBytes(0));
  list->CopyBuffer(g.readback[slot], TableBytes(0), g.tables[1].buffer, 0,
                   TableBytes(1));
}

void FrameFinished(u32 slot) {
  if (!Enabled())
    return;
  const auto* words = reinterpret_cast<const u32*>(g.readback[slot]->mapped());
  Collect(0, words);
  Collect(1, words + TableBytes(0) / 4);
}

}  // namespace gpu::render::bindless
