#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/environment_variables.h"
#include "guest/session.h"
#include "kern/ipmi/savedata.h"
#include "base/logging.h"
#include "kern/ipmi/services.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/lv2/sys_info.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_procid.h"
#include "kern/lv2/sys_semaphore.h"
#include "kern/process.h"
#include "kern/save_storage.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kSavedataDir, "DELTA_SAVEDATA_DIR", nullptr);
DELTA_OPTION(bool, kSavedataTrace, "DELTA_SAVEDATA_TRACE", false);
}

namespace kern::ipmi {
base::String SaveDataRoot() {
  if (kSavedataDir)
    return kSavedataDir.get();
  base::StringU8 home;
  base::GetEnvironmentVariable(u8"HOME", home);
  return base::String(
             home.empty() ? "." : reinterpret_cast<const char*>(home.c_str())) +
         "/.prosperity/savedata";
}

namespace {
constexpr i32 kParameter = static_cast<i32>(0x809f0000u);
constexpr i32 kNotInitialized = static_cast<i32>(0x809f0001u);
constexpr i32 kBusy = static_cast<i32>(0x809f0003u);
constexpr i32 kNotMounted = static_cast<i32>(0x809f0004u);
constexpr i32 kExists = static_cast<i32>(0x809f0007u);
constexpr i32 kNotFound = static_cast<i32>(0x809f0008u);
constexpr i32 kInternal = static_cast<i32>(0x809f000bu);
constexpr i32 kNoIcon = static_cast<i32>(0x809f000eu);
constexpr i32 kNoEvent = static_cast<i32>(0x809f0018u);
constexpr i32 kUnsupported = -static_cast<i32>(SysError::eNOSYS);
struct Pending {
  std::array<u8, 128> header;
  std::vector<u8> data;
};
struct Mounted {
  base::String point, host;
  u32 mode, slot;
};
struct MemoryState {
  std::vector<base::String> memory_names;
  ObjectRef<Semaphore> memory_semaphore;
  u32 memory_semaphore_id = 0;
  u32 memory_app = 0;
  u64 memory_size = 0;
  u64 memory_total = 0;
  u64 memory_icon = 0;
  u32 memory_slot = UINT32_MAX;
};
struct MemoryWork {
  std::mutex mutex;
  std::condition_variable done;
  std::deque<std::array<u8, 60>> results;
  u32 pending = 0;
};
struct MemoryJob {
  std::shared_ptr<MemoryWork> work;
  MemoryState memory;
  base::String host, directory;
};
struct ClientState {
  bool ps5 = false;
  std::vector<Pending> pending;
  std::vector<Mounted> mounts;
  std::vector<u32> transaction_resources;
  MemoryState memory;
  std::unordered_map<std::string, MemoryState> memory_slots;
  base::String memory_directory = "sdmemory";
  std::unordered_map<std::string, u64> memory_listeners;
  ObjectRef<EventFlag> memory_event;
  u32 memory_event_id = 0;
  std::shared_ptr<MemoryWork> memory_work = std::make_shared<MemoryWork>();
  u32 memory_acknowledgements = 0;
  std::unordered_map<std::string, i32> restore_results;
};
std::mutex g_mutex;
u32 g_next_transaction_resource = 0;
std::unordered_map<u32, ClientState> g_clients;
std::array<bool, 16> g_mount_slots{};

std::mutex g_jobs_mutex;
std::condition_variable g_jobs_ready;
std::deque<MemoryJob> g_jobs;
std::unordered_map<std::string, u32> g_writing;
bool g_memory_worker = false;

bool MemoryWriting(const base::String& host) {
  std::lock_guard lock(g_jobs_mutex);
  return g_writing.count(host.c_str());
}
void WaitMemory(ClientState& state) {
  std::unique_lock lock(state.memory_work->mutex);
  state.memory_work->done.wait(lock,
                               [&] { return !state.memory_work->pending; });
}

const u8* Input(Invocation& inv, u32 slot, u64 expected) {
  u64 size = 0;
  const auto* data = static_cast<const u8*>(inv.Input(slot, size));
  return data && size == expected ? data : nullptr;
}
template <typename T>
T Read(const u8* data, size_t offset) {
  T value;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}
// Minecraft names its worlds BedrockWorld<id>@P1.
bool Component(const base::String& name) {
  if (name.empty())
    return false;
  for (char c : name)
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '@'))
      return false;
  return true;
}
bool String(const u8* data, size_t capacity, base::String& value) {
  const auto* end = static_cast<const u8*>(std::memchr(data, 0, capacity));
  if (!end)
    return false;
  value.assign(reinterpret_cast<const char*>(data), end - data);
  return true;
}
void ClearMemory(MemoryState& state) {
  for (const auto& name : state.memory_names)
    sys_shm_unlink(name.c_str());
  state.memory_names.clear();
  if (auto* process = Process::GetActive(); process && state.memory_semaphore) {
    auto object = process->GetObjTable().Get(state.memory_semaphore_id);
    if (object.get() == state.memory_semaphore.get())
      sys_osem_delete(state.memory_semaphore_id);
  }
  state.memory_semaphore = nullptr;
  state.memory_semaphore_id = 0;
  state.memory_app = 0;
  state.memory_size = 0;
  state.memory_total = 0;
  state.memory_icon = 0;
  state.memory_slot = UINT32_MAX;
}
void ClearMemoryEvent(ClientState& state) {
  state.memory_listeners.clear();
  if (state.memory_event) {
    state.memory_event->Cancel(0);
    if (auto* process = Process::GetActive();
        process && process->GetObjTable().Get(state.memory_event_id).get() ==
                       state.memory_event.get())
      sys_evf_delete(state.memory_event_id);
  }
  state.memory_event = nullptr;
  state.memory_event_id = 0;
}
void Clear(ClientState& state) {
  WaitMemory(state);
  ClearMemoryEvent(state);
  ClearMemory(state.memory);
  for (auto& [dir, memory] : state.memory_slots)
    ClearMemory(memory);
  state.memory_slots.clear();
  for (const auto& point : state.mounts) {
    vfs::Unmount(point.point.c_str());
    g_mount_slots[point.slot] = false;
  }
  state.mounts.clear();
  state.transaction_resources.clear();
  state.pending.clear();
  state.memory_work->results.clear();
  state.memory_acknowledgements = 0;
  state.restore_results.clear();
}

i32 Store(const base::String& path, const void* data, size_t size);
i32 MigrateMemory(const base::String& dir, const MemoryState& memory);

// The PS5 library names a slot's segments after an id it generates for each
// setup: /SceSaveData<app><user>_<slot>_<i>_<id>.
i32 SetupMemory(MemoryState& state,
                const Pending& pending,
                u32 slot = UINT32_MAX,
                const base::String& id = {}) {
  auto* process = Process::GetActive();
  if (!process)
    return kInternal;
  const auto* data = pending.data.data();
  const u64 size = Read<u64>(data, 8);
  const u64 icon = Read<u64>(data, 16);
  const u64 param = Read<u64>(data, 24);
  const u32 count = Read<u32>(data, 48);
  if (Read<u32>(data, 0) != 1 || !size || size > 0x2000000 || icon > 0x1c800 ||
      (param && param != 0x530) || !count || count > 2)
    return kParameter;
  if (Read<u32>(data, 32))
    return kUnsupported;
  if (!state.memory_names.empty())
    return kBusy;
  int mib[] = {1, 14, 35, sys_getpid()};
  std::array<u8, 72> info{};
  size_t info_size = info.size();
  if (sys_sysctl(mib, 4, info.data(), &info_size, nullptr, 0))
    return kInternal;
  const u32 app = Read<u32>(info.data(), 0);
  const u64 total = size + (icon ? icon + 64 : 0) + param;
  for (u32 i = 0; i <= count; ++i) {
    char name[64];
    if (i == count) {
      if (slot == UINT32_MAX)
        std::snprintf(name, sizeof(name), "/SceSaveDataI%08x%08x", app, 1);
      else if (id.empty())
        std::snprintf(name, sizeof(name), "/SceSaveDataI%08x%08x_%u", app, 1,
                      slot);
      else
        std::snprintf(name, sizeof(name), "/SceSaveDataI%08x%08x_%u_%s", app, 1,
                      slot, id.c_str());
    } else {
      if (slot == UINT32_MAX)
        std::snprintf(name, sizeof(name), "/SceSaveData%08x%08x_%u", app, 1, i);
      else if (id.empty())
        std::snprintf(name, sizeof(name), "/SceSaveData%08x%08x_%u_%u", app, 1,
                      slot, i);
      else
        std::snprintf(name, sizeof(name), "/SceSaveData%08x%08x_%u_%u_%s", app,
                      1, slot, i, id.c_str());
    }
    const auto result = ShmCreate(name, i == count ? 64 : total);
    if (result) {
      ClearMemory(state);
      return result == -static_cast<i32>(SysError::eEXIST) ? kBusy : kInternal;
    }
    state.memory_names.emplace_back(name);
    if (i != count &&
        ((icon && ShmWrite(name, size, &icon, sizeof(icon))) ||
         (param && ShmWrite(name, total - param, data + 0x34, 0x524)))) {
      ClearMemory(state);
      return kInternal;
    }
  }
  char name[64];
  if (slot == UINT32_MAX)
    std::snprintf(name, sizeof(name), "SceSaveData%08x%08x", app, 1);
  else
    std::snprintf(name, sizeof(name), "SceSaveData%08x%08x_%u", app, 1, slot);
  if (sys_osem_open(name) >= 0) {
    ClearMemory(state);
    return kBusy;
  }
  auto* semaphore = new Semaphore(process->GetObjTable(), name, 1, 1);
  semaphore->Retain();
  state.memory_semaphore = ObjectRef<Semaphore>(semaphore);
  state.memory_semaphore_id = semaphore->handle();
  state.memory_app = app;
  state.memory_size = size;
  state.memory_total = total;
  state.memory_icon = icon;
  state.memory_slot = slot;
  return 0;
}

i32 MemoryReady(MemoryState& state) {
  auto* process = Process::GetActive();
  if (!process || !state.memory_semaphore ||
      process->GetObjTable().Get(state.memory_semaphore_id).get() !=
          state.memory_semaphore.get())
    return kInternal;
  for (size_t i = 0; i < state.memory_names.size(); ++i)
    if (ShmRead(state.memory_names[i].c_str(),
                i + 1 == state.memory_names.size() ? 64 : state.memory_total,
                nullptr, 0))
      return kInternal;
  return 0;
}

i32 ConfirmMemory(MemoryState& state, const Pending& pending) {
  const auto* data = pending.data.data();
  if (Read<u32>(data, 4) != 1 || Read<u32>(data, 0) != state.memory_app ||
      !state.memory_size || Read<u64>(data, 8) != state.memory_size)
    return kParameter;
  return MemoryReady(state);
}

bool MemoryDirectory(const base::String& dir) {
  return dir == "sdmemory" || dir == "sdmemory1" || dir == "sdmemory2" ||
         dir == "sdmemory3";
}

i32 SetupMemorySlot(ClientState& state, const Pending& pending) {
  const auto* data = pending.data.data();
  base::String dir, id;
  const u32 slot = Read<u32>(data, 0x56c);
  if (!String(data + 0x558, 16, dir) || !MemoryDirectory(dir) || slot >= 4 ||
      data[0x568] != 1 ||
      std::any_of(data + 0x569, data + 0x56c,
                  [](u8 value) { return value != 0; }) ||
      data[0x570] > 1 || !String(data + 0x571, 23, id) ||
      (!id.empty() && !Component(id)) ||
      std::any_of(data + 0x588, data + 0x590,
                  [](u8 value) { return value != 0; }))
    return kParameter;
  if (state.memory_slots.count(dir.c_str()) ||
      std::any_of(state.memory_slots.begin(), state.memory_slots.end(),
                  [slot](const auto& entry) {
                    return entry.second.memory_slot == slot;
                  }))
    return kBusy;
  MemoryState memory;
  if (const auto result = SetupMemory(memory, pending, slot, id))
    return result;
  if (const auto result = MigrateMemory(dir, memory)) {
    ClearMemory(memory);
    return result;
  }
  state.memory_slots.emplace(dir.c_str(), std::move(memory));
  state.memory_directory = dir;
  return 0;
}

i32 ClearMemorySlot(ClientState& state, const u8* data) {
  base::String dir;
  if (Read<u32>(data, 0) != 1 || !String(data + 4, 16, dir) ||
      !MemoryDirectory(dir) ||
      std::any_of(data + 20, data + 60, [](u8 value) { return value != 0; }))
    return kParameter;
  const auto it = state.memory_slots.find(dir.c_str());
  if (it != state.memory_slots.end()) {
    WaitMemory(state);
    ClearMemory(it->second);
    state.memory_slots.erase(it);
    if (state.memory_directory == dir)
      state.memory_directory = "sdmemory";
    state.memory_listeners.erase(dir.c_str());
    if (state.memory_listeners.empty())
      ClearMemoryEvent(state);
  }
  return 0;
}

i32 ConfirmMemorySlot(ClientState& state, const Pending& pending) {
  const auto* data = pending.data.data();
  base::String dir;
  if (!String(data + 16, 16, dir) || !MemoryDirectory(dir) || data[32] != 1 ||
      std::any_of(data + 33, data + 36, [](u8 value) { return value != 0; }) ||
      std::any_of(data + 40, data + 72, [](u8 value) { return value != 0; }))
    return kParameter;
  const auto it = state.memory_slots.find(dir.c_str());
  if (it == state.memory_slots.end())
    return kParameter;
  auto& memory = it->second;
  if (Read<u32>(data, 0) != memory.memory_app || Read<u32>(data, 4) != 1 ||
      Read<u64>(data, 8) != memory.memory_size ||
      Read<u32>(data, 36) != memory.memory_slot)
    return kParameter;
  return MemoryReady(memory);
}

bool MemorySaveDirectory(const u8* data, base::String& dir) {
  base::String name;
  if (!String(data, 32, name))
    return false;
  for (const char* value : {"sdmemory", "sdmemory1", "sdmemory2", "sdmemory3"})
    if (name == base::String("sce_") + value) {
      dir = value;
      return true;
    }
  return false;
}

i32 MemorySubscription(ClientState& state, const Pending& pending, bool add) {
  const auto* data = pending.data.data();
  const size_t user_offset = add ? 8 : 4;
  const size_t title_offset = add ? 12 : 8;
  const size_t dir_offset = add ? 28 : 24;
  base::String dir;
  if (Read<u32>(data, 0) != static_cast<u32>(sys_getpid()) ||
      Read<u32>(data, user_offset) != 1 ||
      !MemorySaveDirectory(data + dir_offset, dir))
    return kParameter;
  if (std::any_of(data + title_offset, data + dir_offset,
                  [](u8 value) { return value != 0; }))
    return kUnsupported;
  if (!add) {
    state.memory_listeners.erase(dir.c_str());
    if (state.memory_listeners.empty())
      ClearMemoryEvent(state);
    return 0;
  }
  const u32 flags = Read<u32>(data, 72);
  const u64 size = Read<u64>(data, 80);
  const u64 limit = flags & 4 ? 0x2000000 : 0x800000;
  if (flags > 7 || !size || size > limit || ((flags & 2) && size > limit / 2) ||
      Read<u64>(data, 88) > 0x1c800 || Read<u32>(data, 60) ||
      Read<u32>(data, 76))
    return kParameter;
  const auto it = state.memory_slots.find(dir.c_str());
  auto* memory = it != state.memory_slots.end() ? &it->second
                 : dir == "sdmemory"            ? &state.memory
                                                : nullptr;
  if (!memory || !memory->memory_size ||
      Read<u32>(data, 4) != memory->memory_app || size != memory->memory_size ||
      Read<u64>(data, 88) != memory->memory_icon)
    return kParameter;
  if (const auto result = MemoryReady(*memory))
    return result;
  auto* process = Process::GetActive();
  if (state.memory_event &&
      process->GetObjTable().Get(state.memory_event_id).get() !=
          state.memory_event.get())
    return kInternal;
  if (!state.memory_event) {
    char name[64];
    std::snprintf(name, sizeof(name), "SceSaveDataMemoryRUI%08d", sys_getpid());
    if (sys_evf_open(name) >= 0)
      return kBusy;
    auto* event = new EventFlag(process->GetObjTable(), name, 0);
    event->Retain();
    state.memory_event = ObjectRef<EventFlag>(event);
    state.memory_event_id = event->handle();
  }
  state.memory_listeners[dir.c_str()] = Read<u64>(data, 64);
  return 0;
}

i32 SaveHost(const base::String& dir, base::String& host, bool& exists) {
  if (!Component(dir))
    return kParameter;
  base::String title = vfs::TitleId();
  if (title.empty())
    title = "SAVEDATA";
  if (!Component(title))
    return kParameter;
  const base::String root = SaveDataRoot();
  host = root + "/" + title + "/" + dir;
  std::error_code error;
  exists = std::filesystem::is_directory(host.c_str(), error);
  if (!exists) {
    error.clear();
    const auto legacy = root + "/" + dir;
    if (std::filesystem::is_directory(legacy.c_str(), error)) {
      host = legacy;
      exists = true;
    }
  }
  if (std::filesystem::is_symlink((root + "/" + title).c_str(), error) ||
      std::filesystem::is_symlink(host.c_str(), error))
    return kParameter;
  return 0;
}

i32 SaveHost(const u8* data, base::String& host, bool& exists) {
  base::String dir, title;
  if (Read<i32>(data, 0x400) != 1 || !String(data + 0x414, 32, dir) ||
      !Component(dir) || !String(data + 0x404, 16, title))
    return kParameter;
  if (!title.empty())
    return kUnsupported;
  return SaveHost(dir, host, exists);
}

i32 CreateSave(const base::String& host, u64 blocks) {
  std::error_code error;
  std::filesystem::create_directories(host.c_str(), error);
  if (error)
    return kInternal;
  if (const auto result = Store(host + "/.sce_quota.bin", &blocks, 8)) {
    std::filesystem::remove_all(host.c_str(), error);
    return result;
  }
  return 0;
}

i32 CreateMemorySave(ClientState& state, const Pending& pending) {
  const auto* data = pending.data.data();
  const u64 bytes = Read<u64>(data, 8);
  if (Read<u32>(data, 0) != 1 || Read<u32>(data, 4) || !bytes)
    return kParameter;
  std::array<u8, 0x434> save{};
  std::memcpy(save.data() + 0x400, data, 4);
  const auto dir = "sce_" + state.memory_directory;
  std::memcpy(save.data() + 0x414, dir.c_str(), dir.size() + 1);
  base::String host;
  bool exists = false;
  if (const auto result = SaveHost(save.data(), host, exists))
    return result;
  if (exists)
    return kExists;
  return CreateSave(host,
                    std::max<u64>(96, bytes / 0x8000 + (bytes % 0x8000 != 0)));
}

i32 MigrateMemory(const base::String& dir, const MemoryState& memory) {
  std::array<u8, 0x434> save{};
  const u32 user = 1;
  std::memcpy(save.data() + 0x400, &user, 4);
  const auto name = "sce_" + dir;
  std::memcpy(save.data() + 0x414, name.c_str(), name.size() + 1);
  base::String host;
  bool exists = false;
  if (const auto result = SaveHost(save.data(), host, exists))
    return result;
  if (MemoryWriting(host))
    return kBusy;
  std::error_code error;
  const auto target = host + "/memory.dat";
  const auto status = std::filesystem::symlink_status(target.c_str(), error);
  if (!error && status.type() != std::filesystem::file_type::not_found)
    return 0;
  if (error && error != std::errc::no_such_file_or_directory)
    return kInternal;
  const auto title =
      vfs::TitleId().empty() ? base::String("SAVEDATA") : vfs::TitleId();
  const auto source_root = SaveDataRoot() + "/" + title + "/sce_sdmemory";
  const auto slot = dir == "sdmemory" ? '0' : dir[dir.size() - 1];
  const auto source = source_root + "/memory" + base::String(1, slot) + ".bin";
  error.clear();
  const auto source_status =
      std::filesystem::symlink_status(source.c_str(), error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && source_status.type() == std::filesystem::file_type::not_found))
    return 0;
  if (error || !std::filesystem::is_regular_file(source_status) ||
      std::filesystem::is_symlink(source_root.c_str(), error))
    return kParameter;
  io::File file;
  if (!save_storage::OpenRead(file, source_root, source))
    return kInternal;
  const u64 bytes = file.GetSize();
  if (bytes > 0x2000000) {
    save_storage::Close(file);
    return kParameter;
  }
  std::vector<u8> contents(std::max(bytes, memory.memory_size));
  const bool read = !bytes || file.Read(contents.data(), bytes) == bytes;
  save_storage::Close(file);
  if (!read)
    return kInternal;
  if (!exists) {
    const u64 blocks = (contents.size() + 0x7fff) / 0x8000;
    if (const auto result = CreateSave(host, std::max<u64>(96, blocks)))
      return result;
  }
  const auto result = Store(target, contents.data(), contents.size());
  if (result && !exists)
    std::filesystem::remove_all(host.c_str(), error);
  return result;
}

i32 PersistMemory(const MemoryState& memory_state, const base::String& host) {
  if (memory_state.memory_semaphore->Wait(1, nullptr))
    return kInternal;
  struct Release {
    Semaphore* semaphore;
    ~Release() { semaphore->Post(1); }
  } release{memory_state.memory_semaphore.get()};
  u32 index = 0;
  if (ShmRead(memory_state.memory_names.back().c_str(), 0, &index,
              sizeof(index)) ||
      index >= memory_state.memory_names.size() - 1)
    return kInternal;
  std::vector<u8> memory(memory_state.memory_total);
  if (ShmRead(memory_state.memory_names[index].c_str(), 0, memory.data(),
              memory.size()))
    return kInternal;
  const u64 icon = memory_state.memory_icon;
  const size_t icon_offset = memory_state.memory_size;
  const size_t param_offset = icon_offset + (icon ? icon + 64 : 0);
  u64 icon_size = 0;
  if (icon) {
    icon_size = Read<u64>(memory.data(), icon_offset + 8);
    if (Read<u64>(memory.data(), icon_offset) != icon || icon_size > icon)
      return kInternal;
  }
  return [&] {
    if (const auto result = Store(host + "/memory.dat", memory.data(),
                                  memory_state.memory_size))
      return result;
    if (memory.size() != param_offset) {
      if (const auto result = Store(host + "/.sce_param.bin",
                                    memory.data() + param_offset, 0x530))
        return result;
    }
    if (icon_size)
      return Store(host + "/.sce_icon.bin", memory.data() + icon_offset + 64,
                   icon_size);
    return i32(0);
  }();
}

void MemoryWorker() {
  for (;;) {
    MemoryJob job;
    {
      std::unique_lock lock(g_jobs_mutex);
      while (g_jobs.empty() && !guest::Stopping())
        g_jobs_ready.wait_for(lock, std::chrono::milliseconds(20));
      if (g_jobs.empty())
        return;
      job = std::move(g_jobs.front());
      g_jobs.pop_front();
    }
    const i32 result =
        guest::Stopping() ? kInternal : PersistMemory(job.memory, job.host);
    std::array<u8, 60> event{};
    const u32 type = 3, user = 1;
    std::memcpy(event.data(), &type, 4);
    std::memcpy(event.data() + 4, &result, 4);
    std::memcpy(event.data() + 8, &user, 4);
    std::memcpy(event.data() + 28, job.directory.c_str(),
                job.directory.size() + 1);
    {
      std::lock_guard lock(g_jobs_mutex);
      auto it = g_writing.find(job.host.c_str());
      if (!--it->second)
        g_writing.erase(it);
    }
    {
      std::lock_guard lock(job.work->mutex);
      job.work->results.push_back(event);
      --job.work->pending;
      job.work->done.notify_all();
    }
  }
}

i32 SyncMemory(ClientState& state, const u8* data) {
  base::String dir;
  if (Read<u32>(data, 0) != 1 || !String(data + 4, 16, dir) ||
      !MemoryDirectory(dir) ||
      std::any_of(data + 24, data + 84, [](u8 value) { return value != 0; }))
    return kParameter;
  const u32 mode = Read<u32>(data, 20);
  if (mode > 1)
    return kParameter;
  if (mode) {
    std::lock_guard lock(state.memory_work->mutex);
    if (state.memory_work->results.size() + state.memory_work->pending +
                state.restore_results.size() >=
            32 ||
        state.memory_acknowledgements >= 32)
      return kBusy;
  }
  const auto it = state.memory_slots.find(dir.c_str());
  auto* configured = it != state.memory_slots.end() ? &it->second
                     : dir == "sdmemory"            ? &state.memory
                                                    : nullptr;
  if (!configured || !configured->memory_size)
    return kNotInitialized;
  if (const auto result = MemoryReady(*configured))
    return result;
  std::array<u8, 0x434> save{};
  std::memcpy(save.data() + 0x400, data, 4);
  const auto directory = "sce_" + dir;
  std::memcpy(save.data() + 0x414, directory.c_str(), directory.size() + 1);
  base::String host;
  bool exists = false;
  if (const auto result = SaveHost(save.data(), host, exists))
    return result;
  if (!exists)
    return kNotFound;
  for (const auto& [client, other] : g_clients)
    for (const auto& mount : other.mounts)
      if (mount.host == host)
        return kBusy;
  if (!mode)
    return MemoryWriting(host) ? kBusy : PersistMemory(*configured, host);
  if (guest::Stopping())
    return kInternal;
  {
    std::lock_guard lock(g_jobs_mutex);
    if (!g_memory_worker) {
      if (!guest::SpawnThread("savedata-memory", MemoryWorker))
        return kInternal;
      g_memory_worker = true;
    }
    {
      std::lock_guard work_lock(state.memory_work->mutex);
      ++state.memory_work->pending;
    }
    ++g_writing[host.c_str()];
    g_jobs.push_back({state.memory_work, *configured, host, dir});
    g_jobs_ready.notify_one();
  }
  ++state.memory_acknowledgements;
  return 0;
}

i32 MountHost(ClientState& state,
              const base::String& host,
              bool exists,
              u32 mode,
              u32 original_mode,
              u64 blocks,
              std::vector<u8>& reply) {
  if (MemoryWriting(host))
    return kBusy;
  if ((mode & ~0x2fu) || !(mode & 3) || (mode & 3) == 3 || !original_mode ||
      (original_mode & ~0x27u) || (original_mode & 0x24) == 0x24 ||
      ((original_mode & 1) && (original_mode & 0x24)) ||
      ((original_mode & 0x24) && blocks < 96))
    return kParameter;
  const auto available =
      std::find(g_mount_slots.begin(), g_mount_slots.end(), false);
  if (available == g_mount_slots.end())
    return kInternal;
  const u32 slot = available - g_mount_slots.begin();
  if (!exists && !(original_mode & 0x24))
    return kNotFound;
  if (exists && (original_mode & 4))
    return kExists;
  if (!exists) {
    if (const auto result = CreateSave(host, blocks))
      return result;
  }
  char point[16];
  std::snprintf(point, sizeof(point), "/savedata%u", slot);
  if (mode & 2)
    vfs::MountWritable(point, host.c_str(), true);
  else
    vfs::Mount(point, host.c_str(), true);
  state.mounts.push_back({point, host, mode, slot});
  g_mount_slots[slot] = true;
  std::memcpy(reply.data(), point, std::strlen(point) + 1);
  // The client reads this byte to report that the save was created.
  reply[24] = !exists;
  return 0;
}

// A save written before quotas were recorded reports the 60000-block
// default; titles that sum their saves (Minecraft, Shadow of the Colossus)
// then refuse to create another. Record the size the title asks for, or
// what is already used.
i32 RecordMissingQuota(const base::String& host, bool exists, u64 blocks) {
  const auto quota = host + "/.sce_quota.bin";
  std::error_code error;
  u64 total, free;
  if (!exists || std::filesystem::exists(quota.c_str(), error) || error ||
      save_storage::Info(host, total, free))
    return 0;
  const u64 size = std::max<u64>({blocks, total - free, 96});
  return Store(quota, &size, 8);
}

i32 Mount(ClientState& state, const Pending& pending, std::vector<u8>& reply) {
  const auto* data = pending.data.data();
  base::String host;
  bool exists = false;
  if (const auto result = SaveHost(data, host, exists))
    return result;
  if (data[0x434])
    return kUnsupported;
  const u64 blocks = Read<u64>(data, 0x488);
  if (const auto result = RecordMissingQuota(host, exists, blocks))
    return result;
  return MountHost(state, host, exists, Read<u32>(data, 0x490),
                   Read<u32>(data, 0x494), blocks, reply);
}

// The PS5 library's mount: user at 0, directory at 4, blocks at 0x28 and the
// mount mode at 0x38. The reply matches the PS4 one. PS5 titles pass create
// together with create-if-missing (0x20), and block counts below the PS4
// minimum; both mean create-if-missing here.
i32 MountPs5(ClientState& state,
             const Pending& pending,
             std::vector<u8>& reply) {
  const auto* data = pending.data.data();
  base::String dir, host;
  bool exists = false;
  if (Read<i32>(data, 0) != 1 || !String(data + 4, 32, dir))
    return kParameter;
  if (const auto result = SaveHost(dir, host, exists))
    return result;
  u32 mode = Read<u32>(data, 0x38);
  // Minecraft mounts every save, new ones included, as RDWR with 0x8 and
  // never sends a create, so a writable 0x8 means create-if-missing.
  if ((mode & 0xa) == 0xa)
    mode = (mode & ~0x8u) | 0x20;
  if (mode & 0x20)
    mode &= ~4u;
  const u64 blocks = std::max<u64>(Read<u64>(data, 0x28), 96);
  if (const auto result = RecordMissingQuota(host, exists, blocks))
    return result;
  return MountHost(state, host, exists, mode & 0x2f, mode & 0x27, blocks,
                   reply);
}

i32 UnmountPoint(ClientState& state,
                 const base::String& point,
                 u32 backup,
                 std::vector<u8>& reply) {
  if (backup > 1)
    return kParameter;
  const auto it =
      std::find_if(state.mounts.begin(), state.mounts.end(),
                   [&](const auto& mounted) { return mounted.point == point; });
  if (it == state.mounts.end())
    return kNotMounted;
  if (backup) {
    for (const auto& [client, other] : g_clients)
      for (const auto& mount : other.mounts)
        if (&mount != &*it && mount.host == it->host)
          return kBusy;
    const auto result = save_storage::Backup(it->host);
    if (result == -static_cast<i32>(SysError::eBUSY))
      return kBusy;
    if (result == -static_cast<i32>(SysError::eACCES))
      return kParameter;
    if (result)
      return result == -static_cast<i32>(SysError::eNOSPC)
                 ? static_cast<i32>(0x809f000au)
                 : kInternal;
  }
  std::memcpy(reply.data(), &it->mode, 4);
  vfs::Unmount(it->point.c_str());
  g_mount_slots[it->slot] = false;
  state.mounts.erase(it);
  return 0;
}

i32 Unmount(ClientState& state,
            const Pending& pending,
            std::vector<u8>& reply) {
  base::String point;
  if (!String(pending.data.data(), 16, point))
    return kParameter;
  return UnmountPoint(state, point, Read<u32>(pending.data.data(), 16), reply);
}

// The PS5 library's unmount: the PS4 request behind a word of flags (the
// upper half is set after a commit).
i32 UnmountPs5(ClientState& state,
               const Pending& pending,
               std::vector<u8>& reply) {
  base::String point;
  if (!String(pending.data.data() + 4, 16, point))
    return kParameter;
  return UnmountPoint(state, point, Read<u32>(pending.data.data(), 20), reply);
}

// The PS5 library's commit of a mount point (at 4). Writes already land in
// the host directory, so there is nothing to apply.
i32 CommitPs5(ClientState& state, const Pending& pending) {
  base::String point;
  if (!String(pending.data.data() + 4, 16, point))
    return kParameter;
  const bool mounted =
      std::any_of(state.mounts.begin(), state.mounts.end(),
                  [&](const auto& mount) { return mount.point == point; });
  return mounted ? 0 : kNotMounted;
}

i32 CommitTransactionPs5(ClientState& state, const Pending& pending) {
  const auto* data = pending.data.data();
  if (Read<u32>(data, 4) > 1 ||
      std::any_of(data + 8, data + 72, [](u8 value) { return value != 0; }))
    return kParameter;
  const auto resource = Read<u32>(data, 0);
  const auto& resources = state.transaction_resources;
  // Save writes already reach the host backing directory. The transaction
  // resource is acquired separately from the mount point.
  return std::find(resources.begin(), resources.end(), resource) !=
                 resources.end()
             ? 0
             : kParameter;
}

struct CommandSizes {
  u64 input, output;
};
CommandSizes Sizes(u32 command) {
  switch (command) {
    case 2:
      return {20, 132};
    case 3:
      return {16, 176};
    case 4:
      return {0x434, 128};
    case 6:
    case 118:
      return {192, 0x11308};
    case 7:
      return {0x1c814, 128};
    case 8:
      return {144, 0x1c884};
    case 9:
      return {0x1c818, 128};
    case 10:
      return {148, 0x1c804};
    case 36:
      return {136, 36};
    case 37:
      return {56, 0x1cd40};
    case 46:
      return {60, 32};
    case 33:
      return {0x558, 32};
    case 47:
      return {16, 32};
    case 78:
      return {0x590, 32};
    case 80:
      return {72, 32};
    case 51:
      return {96, 32};
    case 52:
      return {56, 32};
    case 35:
      return {16, 32};
    case 1:
    case 60:
      return {0x4a0, 32};
    case 90:
      return {200, 32};
    case 91:
      return {72, 64};
    case 92:
      return {148, 132};
    case 101:
      return {88, 64};
    default:
      return {};
  }
}

i32 Delete(const Pending& pending) {
  base::String host;
  bool exists = false;
  if (const auto result = SaveHost(pending.data.data(), host, exists))
    return result;
  if (MemoryWriting(host))
    return kBusy;
  for (const auto& [client, state] : g_clients)
    for (const auto& mount : state.mounts)
      if (mount.host == host)
        return kBusy;
  if (!exists)
    return kNotFound;
  const auto result = save_storage::RemoveSave(host);
  if (result == -static_cast<i32>(SysError::eBUSY))
    return kBusy;
  if (result == -static_cast<i32>(SysError::eNOENT))
    return kNotFound;
  return result ? kInternal : 0;
}

i32 Store(const base::String& path, const void* data, size_t size) {
  const auto root = std::filesystem::path(path.c_str()).parent_path().string();
  const auto result = save_storage::Replace(root.c_str(), path, data, size);
  if (result == -static_cast<i32>(SysError::eNOSPC))
    return static_cast<i32>(0x809f000au);
  return result ? kInternal : 0;
}

bool Matches(const std::string& name, const base::String& pattern) {
  size_t i = 0, j = 0, wildcard = SIZE_MAX, retry = 0;
  while (i < name.size()) {
    if (j < pattern.size() && (pattern[j] == '_' || pattern[j] == name[i])) {
      ++i;
      ++j;
    } else if (j < pattern.size() && pattern[j] == '%') {
      wildcard = j++;
      retry = i;
    } else if (wildcard != SIZE_MAX) {
      j = wildcard + 1;
      i = ++retry;
    } else {
      return false;
    }
  }
  while (j < pattern.size() && pattern[j] == '%')
    ++j;
  return j == pattern.size();
}

i32 LoadParam(const base::String& path, std::array<u8, 1328>& blob) {
  if (FILE* file = std::fopen(path.c_str(), "rb")) {
    std::fread(blob.data(), 1, blob.size(), file);
    const bool failed = std::ferror(file);
    std::fclose(file);
    return failed ? kInternal : 0;
  }
  return errno == ENOENT ? 0 : kInternal;
}

i32 StorageInfo(const base::String& host, u64& total, u64& free) {
  return save_storage::Info(host, total, free) ? kInternal : 0;
}

i32 MountInfo(ClientState& state,
              const Pending& pending,
              std::vector<u8>& reply) {
  base::String point;
  if (!String(pending.data.data(), 16, point))
    return kParameter;
  const auto mount =
      std::find_if(state.mounts.begin(), state.mounts.end(),
                   [&](const auto& m) { return m.point == point; });
  if (mount == state.mounts.end())
    return kNotMounted;
  u64 total, free;
  if (const auto result = StorageInfo(mount->host, total, free))
    return result;
  std::memcpy(reply.data(), &total, 8);
  std::memcpy(reply.data() + 8, &free, 8);
  return 0;
}

// `other_title` is the PS5 library's search of another title's saves (command
// 118), which names that title; command 6 searches the caller's own.
i32 Search(const Pending& pending,
           std::vector<u8>& reply,
           bool other_title,
           bool ps5) {
  const auto* data = pending.data.data();
  base::String title, filter;
  if (Read<i32>(data, 0) != 1 || !String(data + 4, 16, title) ||
      !String(data + 20, 32, filter))
    return kParameter;
  if (title.empty() == other_title)
    return other_title ? kParameter : kUnsupported;
  if (title.empty())
    title = vfs::TitleId();
  if (title.empty())
    title = "SAVEDATA";
  if (!Component(title))
    return kParameter;
  for (char c : filter)
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
          (c >= '0' && c <= '9') || c == '%' || c == '_' || c == '-' ||
          c == '.' || c == '@'))
      return kParameter;
  const u32 key = Read<u32>(data, 52), order = Read<u32>(data, 56);
  if (key > 5 || order > 1)
    return kParameter;
  if (key == 4)
    return kUnsupported;
  const auto root = SaveDataRoot() + "/" + title;
  std::error_code error;
  if (std::filesystem::is_symlink(root.c_str(), error))
    return kParameter;
  struct Entry {
    std::string name;
    std::array<u8, 1328> param{};
    u64 blocks = 0, free_blocks = 0;
  };
  std::vector<Entry> entries;
  std::filesystem::directory_iterator it(root.c_str(), error), end;
  if (error)
    return error == std::errc::no_such_file_or_directory ? 0 : kInternal;
  for (; it != end; it.increment(error)) {
    if (error)
      return kInternal;
    const auto status = it->symlink_status(error);
    if (error)
      return kInternal;
    if (!std::filesystem::is_directory(status))
      continue;
    const auto name = it->path().filename().string();
    if (name.size() >= 32 || !Component(name.c_str()) || name[0] == '_' ||
        name.starts_with("sce_") || (!filter.empty() && !Matches(name, filter)))
      continue;
    Entry entry{name};
    const base::String path(it->path().string().c_str());
    if (const auto result = LoadParam(path + "/.sce_param.bin", entry.param))
      return result;
    if (const auto result = StorageInfo(path, entry.blocks, entry.free_blocks))
      return result;
    // A save written before quotas were recorded would report the 60000-block
    // default, and Minecraft refuses to open any world once those add up;
    // Shadow of the Colossus sums its saves against 1 GiB and refuses to
    // create a new game. Report what it uses until a mount records a size.
    std::error_code quota_error;
    if (!std::filesystem::exists((path + "/.sce_quota.bin").c_str(),
                                 quota_error)) {
      const u64 used = entry.blocks - entry.free_blocks;
      entry.blocks = std::max<u64>(used, 96);
      entry.free_blocks = entry.blocks - used;
    }
    entries.push_back(std::move(entry));
  }
  if (error)
    return kInternal;
  std::sort(entries.begin(), entries.end(), [&](const auto& a, const auto& b) {
    int comparison = 0;
    if (key == 1) {
      const auto av = Read<u32>(a.param.data(), 1280);
      const auto bv = Read<u32>(b.param.data(), 1280);
      comparison = (av > bv) - (av < bv);
    } else if (key == 3) {
      const auto av = Read<i64>(a.param.data(), 1288);
      const auto bv = Read<i64>(b.param.data(), 1288);
      comparison = (av > bv) - (av < bv);
    } else if (key == 2 || key == 5) {
      const auto av = key == 2 ? a.blocks : a.free_blocks;
      const auto bv = key == 2 ? b.blocks : b.free_blocks;
      comparison = (av > bv) - (av < bv);
    }
    if (!comparison)
      comparison = a.name.compare(b.name);
    return order ? comparison > 0 : comparison < 0;
  });
  const u32 total = entries.size();
  const u32 offset = Read<u32>(data, 60);
  const u32 count = offset >= total ? 0 : std::min(50u, total - offset);
  std::memcpy(reply.data(), &total, 4);
  std::memcpy(reply.data() + 4, &count, 4);
  for (u32 i = 0; i < count; ++i) {
    const auto& entry = entries[offset + i];
    std::memcpy(reply.data() + 8 + i * 32, entry.name.c_str(),
                entry.name.size() + 1);
    std::memcpy(reply.data() + 0x648 + i * 1328, entry.param.data(), 1328);
    std::memcpy(reply.data() + 0x109a8 + i * 48, &entry.blocks, 8);
    std::memcpy(reply.data() + 0x109b0 + i * 48, &entry.free_blocks, 8);
  }
  return 0;
}

i32 Param(ClientState& state,
          const Pending& pending,
          std::vector<u8>& reply,
          bool write) {
  const auto* data = pending.data.data();
  base::String point;
  if (!String(data, 16, point))
    return kParameter;
  const auto mounted =
      std::find_if(state.mounts.begin(), state.mounts.end(),
                   [&](const auto& mount) { return mount.point == point; });
  if (mounted == state.mounts.end())
    return kNotMounted;
  if (write && !(mounted->mode & 2))
    return -static_cast<i32>(SysError::eROFS);
  const u32 type = Read<u32>(data, 16);
  constexpr std::array<u32, 6> offsets = {0, 0, 128, 256, 1280, 1288};
  constexpr std::array<u32, 6> lengths = {1328, 128, 128, 1024, 4, 8};
  if (type >= offsets.size())
    return kParameter;
  std::array<u8, 1328> blob{};
  const auto path = mounted->host + "/.sce_param.bin";
  if (const auto result = LoadParam(path, blob))
    return result;
  if (!write) {
    const u32 length = lengths[type];
    std::memcpy(reply.data(), blob.data() + offsets[type], length);
    std::memcpy(reply.data() + 0x1c800, &length, 4);
    return 0;
  }
  const u32 length = Read<u32>(data, 0x1c814);
  if (!length || length > lengths[type])
    return kParameter;
  std::memcpy(blob.data() + offsets[type], data + 20, length);
  return Store(path, blob.data(), blob.size());
}

i32 Icon(ClientState& state,
         const Pending& pending,
         std::vector<u8>& reply,
         bool write) {
  const auto* data = pending.data.data();
  base::String point;
  if (!String(data, 16, point))
    return kParameter;
  const auto mounted =
      std::find_if(state.mounts.begin(), state.mounts.end(),
                   [&](const auto& mount) { return mount.point == point; });
  if (mounted == state.mounts.end())
    return kNotMounted;
  if (write && !(mounted->mode & 2))
    return -static_cast<i32>(SysError::eROFS);
  const auto path = mounted->host + "/.sce_icon.bin";
  if (write) {
    const u32 size = Read<u32>(data, 0x1c810);
    if (!size || size > 0x1c800)
      return kParameter;
    return Store(path, data + 16, size);
  }
  std::error_code error;
  const auto size = std::filesystem::file_size(path.c_str(), error);
  if (error)
    return error == std::errc::no_such_file_or_directory ? kNoIcon : kInternal;
  if (size > 0x1c800)
    return kParameter;
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file)
    return kInternal;
  file.read(reinterpret_cast<char*>(reply.data()), size);
  if (!file)
    return kInternal;
  const u32 length = size;
  std::memcpy(reply.data() + 0x1c800, &length, 4);
  return 0;
}

i32 BackupHost(const u8* data, base::String& host) {
  std::array<u8, 0x434> save{};
  std::memcpy(save.data() + 0x400, data, 52);
  bool exists = false;
  if (const auto result = SaveHost(save.data(), host, exists))
    return result;
  std::error_code error;
  const auto status =
      std::filesystem::symlink_status((host + ".backup").c_str(), error);
  if ((!error && std::filesystem::exists(status)) ||
      (error && error != std::errc::no_such_file_or_directory))
    return 0;
  base::String dir;
  if (!String(data + 20, 32, dir))
    return kParameter;
  const auto legacy = SaveDataRoot() + "/" + dir;
  error.clear();
  const auto legacy_status =
      std::filesystem::symlink_status((legacy + ".backup").c_str(), error);
  if (!error && std::filesystem::exists(legacy_status))
    host = legacy;
  return 0;
}
i32 BackupResult(i32 result) {
  if (result == -static_cast<i32>(SysError::eNOENT))
    return kNotFound;
  if (result == -static_cast<i32>(SysError::eBUSY))
    return kBusy;
  if (result == -static_cast<i32>(SysError::eACCES))
    return kParameter;
  if (result == -static_cast<i32>(SysError::eNOSPC))
    return static_cast<i32>(0x809f000au);
  return result ? kInternal : 0;
}
i32 CheckBackup(const Pending& pending, std::vector<u8>& reply) {
  const auto* data = pending.data.data();
  if (data[52] > 1 || data[53] > 1 || data[54] || data[55])
    return kParameter;
  base::String host;
  if (const auto result = BackupHost(data, host))
    return result;
  if (const auto result = BackupResult(save_storage::CheckBackup(host)))
    return result;
  const auto backup = host + ".backup";
  if (data[52]) {
    std::array<u8, 1328> params{};
    if (const auto result = LoadParam(backup + "/.sce_param.bin", params))
      return result;
    std::memcpy(reply.data(), params.data(), params.size());
  }
  if (data[53]) {
    std::error_code error;
    const auto path = backup + "/.sce_icon.bin";
    const auto size = std::filesystem::file_size(path.c_str(), error);
    if (error && error != std::errc::no_such_file_or_directory)
      return kInternal;
    if (!error) {
      if (size > 0x1c800)
        return kInternal;
      std::ifstream file(path.c_str(), std::ios::binary);
      if (size &&
          !file.read(reinterpret_cast<char*>(reply.data() + 0x530), size))
        return kInternal;
      const u32 length = size;
      std::memcpy(reply.data() + 0x1cd30, &length, sizeof(length));
    }
  }
  return 0;
}
i32 RestoreBackup(ClientState& state, const Pending& pending) {
  const auto* data = pending.data.data();
  base::String host, dir;
  if (const auto result = BackupHost(data, host))
    return result;
  if (!String(data + 20, 32, dir))
    return kParameter;
  {
    std::lock_guard lock(state.memory_work->mutex);
    if (!state.restore_results.count(dir.c_str()) &&
        state.memory_work->results.size() + state.memory_work->pending +
                state.restore_results.size() >=
            32)
      return kBusy;
  }
  const i32 result = [&] {
    if (data[52])
      return kUnsupported;
    if (std::any_of(data + 53, data + 136, [](u8 value) { return value != 0; }))
      return kParameter;
    if (MemoryWriting(host))
      return kBusy;
    for (const auto& [client, other] : g_clients)
      for (const auto& mount : other.mounts)
        if (mount.host == host)
          return kBusy;
    return BackupResult(save_storage::RestoreBackup(host));
  }();
  state.restore_results[dir.c_str()] = result;
  return result;
}
i32 RestoreNotification(ClientState& state, const Pending& pending) {
  const auto* data = pending.data.data();
  base::String dir;
  if (Read<u32>(data, 0) != 1 || Read<u32>(data, 4) != 1 ||
      !String(data + 24, 32, dir) || !Component(dir))
    return kParameter;
  if (std::any_of(data + 8, data + 24, [](u8 value) { return value != 0; }))
    return kUnsupported;
  const auto it = state.restore_results.find(dir.c_str());
  if (it == state.restore_results.end() || it->second != Read<i32>(data, 56))
    return kParameter;
  std::lock_guard lock(state.memory_work->mutex);
  if (state.memory_work->results.size() + state.memory_work->pending >= 32)
    return kBusy;
  std::array<u8, 60> event{};
  std::memcpy(event.data(), data, 4);
  std::memcpy(event.data() + 4, data + 56, 4);
  std::memcpy(event.data() + 8, data + 4, 4);
  std::memcpy(event.data() + 28, data + 24, 32);
  state.memory_work->results.push_back(event);
  state.restore_results.erase(it);
  return 0;
}

struct SaveData : Service {
  const char* Name() const override { return "SceSaveData"; }
  // PS4 clients connect with their 4-byte SDK version and then initialize with
  // command 56. PS5 native ones connect with a 44-byte block and send no
  // initialize: the handshake is the whole of sceSaveDataInitialize.
  i32 Connect(u32 client, const void* data, u64 size) override {
    if (!data || (size != 4 && size != 44))
      return kParameter;
    if (size == 44) {
      std::lock_guard lock(g_mutex);
      g_clients.try_emplace(client).first->second.ps5 = true;
    }
    return 0;
  }
  void DestroyClient(u32 client) override {
    std::lock_guard lock(g_mutex);
    const auto it = g_clients.find(client);
    if (it != g_clients.end()) {
      Clear(it->second);
      g_clients.erase(it);
    }
  }
  bool ReceiveEvent(u32 client,
                    u32&,
                    void* data,
                    u64 capacity,
                    u64& size) override {
    std::lock_guard lock(g_mutex);
    const auto it = g_clients.find(client);
    if (it == g_clients.end() || !it->second.memory_acknowledgements ||
        capacity < 4)
      return false;
    const i32 result = 0;
    std::memcpy(data, &result, sizeof(result));
    size = sizeof(result);
    --it->second.memory_acknowledgements;
    return true;
  }
  void Invoke(Invocation& inv) override {
    std::lock_guard lock(g_mutex);
    if (inv.Method() == 0x12340002) {
      const auto* header = Input(inv, 0, 24);
      if (inv.NumIn() != 2 || inv.NumOut() != 2 || !header ||
          !inv.CanReply(1, 8)) {
        inv.SetResult(kParameter);
        return;
      }
      const u32 command = Read<u32>(header, 0);
      const u64 size = command == 34 || command == 42 || command == 81 ? 0
                       : command == 43                                 ? 60
                       : command == 56                                 ? 4
                       : command == 88                                 ? 68
                       : command == 89                                 ? 64
                                                                       : 140;
      if (command != 34 && command != 42 && command != 56 && command != 63 &&
          command != 81 && command != 43 && command != 88 && command != 89) {
        inv.SetResult(kUnsupported);
        return;
      }
      const u64 input_size = command == 42                    ? 84
                             : command == 81                  ? 60
                             : command == 88 || command == 89 ? 68
                                                              : 4;
      const auto* data = Input(inv, 1, input_size);
      if (!data || Read<u64>(header, 8) != input_size ||
          Read<u64>(header, 16) != size || !inv.CanReply(0, size)) {
        inv.SetResult(kParameter);
        return;
      }
      if (command == 34 || command == 42 || command == 43 || command == 81) {
        const auto it = g_clients.find(inv.Client());
        if (it == g_clients.end()) {
          inv.SetResult(kNotInitialized);
          return;
        }
        if (command == 43) {
          if (Read<u32>(data, 0)) {
            inv.SetResult(kParameter);
            return;
          }
          std::lock_guard work_lock(it->second.memory_work->mutex);
          if (it->second.memory_work->results.empty()) {
            inv.SetResult(kNoEvent);
            return;
          }
          const auto& event = it->second.memory_work->results.front();
          inv.Reply(0, event.data(), event.size());
          const u64 written = event.size();
          inv.Reply(1, &written, sizeof(written));
          it->second.memory_work->results.pop_front();
          return;
        } else if (command == 81) {
          if (const auto result = ClearMemorySlot(it->second, data)) {
            inv.SetResult(result);
            return;
          }
        } else if (command == 42) {
          if (const auto result = SyncMemory(it->second, data)) {
            inv.SetResult(result);
            return;
          }
        } else if (Read<u32>(data, 0) != 1) {
          inv.SetResult(kParameter);
          return;
        }
        if (command == 34) {
          WaitMemory(it->second);
          ClearMemory(it->second.memory);
          if (!it->second.memory_slots.count("sdmemory")) {
            it->second.memory_listeners.erase("sdmemory");
            if (it->second.memory_listeners.empty())
              ClearMemoryEvent(it->second);
          }
        }
      } else if (command == 56) {
        g_clients.try_emplace(inv.Client());
      } else if (command == 88) {
        if (std::any_of(data + 4, data + 68,
                        [](u8 value) { return value != 0; }) ||
            g_next_transaction_resource >= INT32_MAX) {
          inv.SetResult(kParameter);
          return;
        }
        auto& state = g_clients.try_emplace(inv.Client()).first->second;
        const u32 resource = g_next_transaction_resource++;
        state.transaction_resources.push_back(resource);
        std::array<u8, 68> reply{};
        std::memcpy(reply.data(), &resource, sizeof(resource));
        inv.Reply(0, reply.data(), reply.size());
        const u64 written = reply.size();
        inv.Reply(1, &written, sizeof(written));
        return;
      } else if (command == 89) {
        const auto client = g_clients.find(inv.Client());
        if (client == g_clients.end()) {
          inv.SetResult(kNotInitialized);
          return;
        }
        auto& resources = client->second.transaction_resources;
        const auto resource = Read<u32>(data, 0);
        const auto it = std::find(resources.begin(), resources.end(), resource);
        if (it == resources.end() ||
            std::any_of(data + 4, data + 68,
                        [](u8 value) { return value != 0; })) {
          inv.SetResult(kParameter);
          return;
        }
        resources.erase(it);
        const std::array<u8, 64> reply{};
        inv.Reply(0, reply.data(), reply.size());
        const u64 written = reply.size();
        inv.Reply(1, &written, sizeof(written));
        return;
      } else {
        const auto it = g_clients.find(inv.Client());
        if (it != g_clients.end()) {
          Clear(it->second);
          g_clients.erase(it);
        }
      }
      inv.ReplyEmpty();
      return;
    }
    const auto client = g_clients.find(inv.Client());
    if (client == g_clients.end()) {
      inv.SetResult(kNotInitialized);
      return;
    }
    auto& state = client->second;
    if (inv.Method() == 0x12340000) {
      const auto* size = Input(inv, 0, 8);
      const auto* header = Input(inv, 1, 128);

      if (inv.NumIn() != 3 || inv.NumOut() != 1 || !size || !header ||
          !inv.CanReply(0, 4)) {
        inv.SetResult(kParameter);
        return;
      }
      const u32 command = Read<u32>(header, 4);
      const auto sizes = Sizes(command);
      if (!sizes.input) {
        inv.SetResult(kUnsupported);
        return;
      }
      const u64 payload_size = sizes.input;
      const auto* data = Input(inv, 2, payload_size);
      if (!data || Read<u64>(size, 0) != payload_size ||
          state.pending.size() >= 32 ||
          std::any_of(state.pending.begin(), state.pending.end(),
                      [&](const auto& p) {
                        return std::memcmp(p.header.data(), header, 128) == 0;
                      })) {
        inv.SetResult(kParameter);
        return;
      }
      Pending pending{};
      std::memcpy(pending.header.data(), header, 128);
      pending.data.assign(data, data + payload_size);
      state.pending.push_back(pending);
      inv.ReplyU32(0, 0);
      return;
    }
    if (inv.Method() == 0x12340001) {
      const auto* size = Input(inv, 0, 8);
      const auto* header = Input(inv, 1, 132);
      if (inv.NumIn() != 2 || inv.NumOut() != 2 || !size || !header ||
          !inv.CanReply(1, 8)) {
        inv.SetResult(kParameter);
        return;
      }
      const auto it = std::find_if(
          state.pending.begin(), state.pending.end(), [&](const auto& p) {
            return std::memcmp(p.header.data(), header, 128) == 0;
          });
      if (it == state.pending.end()) {
        inv.SetResult(kParameter);
        return;
      }
      const u32 command = Read<u32>(it->header.data(), 4);
      u64 reply_size = Sizes(command).output;
      // The PS5 library's search reply carries one more u64 after the PS4
      // layout; it is added to each entry's block count when sorting by size.
      if (state.ps5 && (command == 6 || command == 118))
        reply_size += 8;
      if (Read<u64>(size, 0) != reply_size || !inv.CanReply(0, reply_size)) {
        inv.SetResult(kParameter);
        return;
      }
      std::vector<u8> reply(reply_size);
      i32 result = 0;
      switch (command) {
        case 36:
          result = RestoreBackup(state, *it);
          break;
        case 37:
          result = CheckBackup(*it, reply);
          break;
        case 46:
          result = RestoreNotification(state, *it);
          break;
        case 33:
          result = SetupMemory(state.memory, *it);
          if (!result) {
            result = MigrateMemory("sdmemory", state.memory);
            if (result)
              ClearMemory(state.memory);
            else
              state.memory_directory = "sdmemory";
          }
          break;
        case 78:
          result = SetupMemorySlot(state, *it);
          break;
        case 80:
          result = ConfirmMemorySlot(state, *it);
          break;
        case 51:
        case 52:
          result = MemorySubscription(state, *it, command == 51);
          break;
        case 47:
          result = ConfirmMemory(state.memory, *it);
          break;
        case 35:
          result = CreateMemorySave(state, *it);
          break;
        case 1:
        case 60:
          result = Mount(state, *it, reply);
          break;
        case 90:
          result = MountPs5(state, *it, reply);
          break;
        case 91:
          result = CommitTransactionPs5(state, *it);
          break;
        case 92:
          result = UnmountPs5(state, *it, reply);
          break;
        case 101:
          result = CommitPs5(state, *it);
          break;
        case 2:
          result = Unmount(state, *it, reply);
          break;
        case 3:
          result = MountInfo(state, *it, reply);
          break;
        case 4:
          result = Delete(*it);
          break;
        case 6:
        case 118:
          result = Search(*it, reply, command == 118, state.ps5);
          break;
        case 7:
          result = Icon(state, *it, reply, true);
          break;
        case 8:
          result = Icon(state, *it, reply, false);
          break;
        case 9:
          result = Param(state, *it, reply, true);
          break;
        case 10:
          result = Param(state, *it, reply, false);
          break;
      }
      state.pending.erase(it);
      if (kSavedataTrace)
        BASE_LOGI("savedata", "command {} -> {:#x}", command,
                  static_cast<u32>(result));
      if (result) {
        inv.SetResult(result);
        return;
      }
      inv.Reply(0, reply.data(), reply_size);
      const u64 written = reply_size;
      inv.Reply(1, &written, sizeof(written));
      return;
    }
    inv.SetResult(kUnsupported);
  }
};
SaveData g_service;
const guest::SessionReset g_reset([] {
  std::lock_guard lock(g_mutex);
  for (auto& [id, state] : g_clients)
    Clear(state);
  g_clients.clear();
  g_next_transaction_resource = 0;
  g_mount_slots.fill(false);
  std::lock_guard jobs_lock(g_jobs_mutex);
  g_jobs.clear();
  g_writing.clear();
  g_memory_worker = false;
});
}  // namespace
Service& SaveDataService() {
  return g_service;
}
}  // namespace kern::ipmi
