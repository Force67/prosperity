#include "kern/ipmi/common_dialog.h"

#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "base/logging.h"
#include "guest/session.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/lv2/sys_mem.h"

namespace kern::ipmi {
namespace {
constexpr size_t kPayloadSize = 0x250f0;
std::mutex g_mutex;
struct Dialog {
  std::string name;
  u64 identity;
  bool active = false;
  size_t reply_offset = 0;
  u32 mode = 0;
  std::vector<u8> reply;
};
std::map<u32, Dialog> g_dialogs;
bool g_worker = false;
constexpr u32 kDialogApp = 0x60000002;
constexpr char kProcessFlag[] = "SceLncUtilAppStatus60000002";
struct LaunchResult {
  u32 client, method;
  i32 result;
};
std::map<u32, LaunchResult> g_launch_results;
struct DialogProcess {
  bool running;
  u64 identity;
};
constexpr char kWorkName[] = "/60000001CdlgShWk";
std::map<u32, DialogProcess> g_processes;
u32 g_next_request = 1;

u64 ProcessStatus() {
  for (const auto& [client, process] : g_processes)
    if (process.running)
      return 1;
  return 4;
}
void PublishProcessStatus() {
  const u64 status = ProcessStatus();
  EvfClearByNameSubstr(kProcessFlag, ~(u64(1) | 4));
  EvfSetByNameSubstr(kProcessFlag, status);
}

u32 Read32(const void* data, size_t offset) {
  u32 value;
  std::memcpy(&value, static_cast<const u8*>(data) + offset, sizeof(value));
  return value;
}
bool Name(Invocation& inv, std::string& name) {
  u64 size;
  const auto* input = static_cast<const char*>(inv.Input(0, size));
  if (inv.NumIn() != 1 || inv.NumOut() || !input || size != 24 ||
      input[0] != '/' || std::memcmp(input + 9, "SaveDataDialog", 15))
    return false;
  for (size_t i = 1; i < 9; ++i)
    if (!((input[i] >= '0' && input[i] <= '9') ||
          (input[i] >= 'a' && input[i] <= 'f') ||
          (input[i] >= 'A' && input[i] <= 'F')))
      return false;
  name.assign(input, size - 1);
  return true;
}
void Worker() {
  while (!guest::Stopping()) {
    {
      std::lock_guard lock(g_mutex);
      for (auto& [client, process] : g_processes) {
        if (!process.running)
          continue;
        u64 heartbeat;
        if (ShmRead(kWorkName, 0x18, &heartbeat, sizeof(heartbeat),
                    process.identity)) {
          process.running = false;
          PublishProcessStatus();
          continue;
        }
        ++heartbeat;
        ShmWrite(kWorkName, 0x18, &heartbeat, sizeof(heartbeat),
                 process.identity);
      }
      for (auto& [client, dialog] : g_dialogs) {
        if (!dialog.active)
          continue;
        std::array<u8, 48> header;
        if (ShmRead(dialog.name.c_str(), 0, header.data(), header.size(),
                    dialog.identity)) {
          dialog.active = false;
          continue;
        }
        if ((dialog.mode == 5 && !header[0x1c]) ||
            Read32(header.data(), 0x18) == 1)
          continue;
        if (header[0x20] == 3) {
          const u32 result = Read32(header.data(), 0x24);
          if (result > 1)
            continue;
          std::memcpy(dialog.reply.data() + 0x24b94, &result, sizeof(result));
        }
        const u32 result = 0;
        const u32 finished = 1;
        if (ShmWrite(dialog.name.c_str(), dialog.reply_offset,
                     dialog.reply.data(), dialog.reply.size(),
                     dialog.identity) ||
            ShmWrite(dialog.name.c_str(), 0x30, &result, sizeof(result),
                     dialog.identity) ||
            ShmWrite(dialog.name.c_str(), 0x18, &finished, sizeof(finished),
                     dialog.identity))
          dialog.active = false;
      }
    }
    guest::SleepForMilliseconds(1);
  }
}
i32 Launch(Dialog& dialog) {
  std::array<u8, 32> header;
  if (ShmRead(dialog.name.c_str(), 0, header.data(), header.size(),
              dialog.identity))
    return -static_cast<i32>(SysError::eNOENT);
  if (Read32(header.data(), 0) != 1 || Read32(header.data(), 4) != 1 ||
      Read32(header.data(), 8) != 0x400 ||
      Read32(header.data(), 0x10) != kPayloadSize ||
      Read32(header.data(), 0x14) != kPayloadSize)
    return -static_cast<i32>(SysError::eINVAL);
  u8 last;
  if (ShmRead(dialog.name.c_str(), 0x400 + kPayloadSize - 1, &last, 1,
              dialog.identity))
    return -static_cast<i32>(SysError::eINVAL);
  std::vector<u8> reply(kPayloadSize);
  if (ShmRead(dialog.name.c_str(), 0x400, reply.data(), reply.size(),
              dialog.identity))
    return -static_cast<i32>(SysError::eNOENT);
  const u32 mode = Read32(reply.data(), 0x38);
  if (!mode || mode > 7)
    return -static_cast<i32>(SysError::eNOSYS);
  if (Read32(reply.data(), 0x805c) > 1024)
    return -static_cast<i32>(SysError::eINVAL);
  for (const auto& [client, other] : g_dialogs)
    if (&other != &dialog && other.active)
      return -static_cast<i32>(SysError::eBUSY);
  if (guest::Stopping())
    return -static_cast<i32>(SysError::eIO);
  if (!g_worker) {
    if (!guest::SpawnThread("common-dialog", Worker))
      return -static_cast<i32>(SysError::eIO);
    g_worker = true;
  }
  const u64 ready = 0x102;
  if (ShmWrite(dialog.name.c_str(), 0x28, &ready, sizeof(ready),
               dialog.identity))
    return -static_cast<i32>(SysError::eNOENT);
  std::memset(reply.data() + 0x24b90, 0, 0x560);
  const u32 button = 1;
  std::memcpy(reply.data() + 0x24b98, &button, sizeof(button));
  if (Read32(reply.data(), 0x805c))
    std::memcpy(reply.data() + 0x24b9c, reply.data() + 0x5c, 32);
  dialog.mode = mode;
  dialog.reply = std::move(reply);
  dialog.reply_offset = 0x400;
  dialog.active = true;
  return 0;
}
const guest::SessionReset g_reset([] {
  std::lock_guard lock(g_mutex);
  g_dialogs.clear();
  g_worker = false;
  g_launch_results.clear();
  g_processes.clear();
});
}  // namespace

bool InvokeCommonDialog(Invocation& inv) {
  if (inv.Method() != 0x30044 && inv.Method() != 0x30045 &&
      inv.Method() != 0x30033 && inv.Method() != 0x30034)
    return false;
  std::lock_guard lock(g_mutex);
  const auto fail = [&](SysError error) {
    inv.SetResult(-static_cast<i32>(error));
  };
  if (inv.Method() == 0x30044 || inv.Method() == 0x30045) {
    std::string name;
    if (!Name(inv, name)) {
      fail(SysError::eINVAL);
      return true;
    }
    if (inv.Method() == 0x30045) {
      const auto it = g_dialogs.find(inv.Client());
      if (it == g_dialogs.end() || it->second.name != name)
        fail(SysError::eNOENT);
      else
        g_dialogs.erase(it);
      return true;
    }
    if (g_dialogs.contains(inv.Client())) {
      fail(SysError::eBUSY);
      return true;
    }
    for (const auto& [client, dialog] : g_dialogs)
      if (dialog.name == name) {
        fail(SysError::eBUSY);
        return true;
      }
    const u64 identity = ShmIdentity(name.c_str());
    std::array<u8, 32> header;
    if (!identity ||
        ShmRead(name.c_str(), 0, header.data(), header.size(), identity)) {
      fail(SysError::eNOENT);
      return true;
    }
    g_dialogs.emplace(inv.Client(), Dialog{std::move(name), identity});
    return true;
  }
  if (inv.NumIn() || inv.NumOut()) {
    fail(SysError::eINVAL);
    return true;
  }
  const auto it = g_dialogs.find(inv.Client());
  if (it == g_dialogs.end()) {
    if (inv.Method() == 0x30033)
      fail(SysError::eNOENT);
    return true;
  }
  if (inv.Method() == 0x30033) {
    const i32 result = Launch(it->second);
    if (result)
      BASE_LOGW("cdlg", "save data dialog {} refused: {}", it->second.name.c_str(),
                result);
    inv.SetResult(result);
  } else
    it->second.active = false;
  return true;
}

i32 LaunchCommonDialogProcess(Invocation& inv, u32& request) {
  if (inv.Method() != 0x30040 && inv.Method() != 0x30042)
    return -static_cast<i32>(SysError::eNOSYS);
  std::lock_guard lock(g_mutex);
  if (guest::Stopping())
    return -static_cast<i32>(SysError::eIO);
  if (g_launch_results.size() >= 64)
    return -static_cast<i32>(SysError::eBUSY);
  u64 size;
  if (inv.Method() == 0x30040) {
    const auto* user = inv.Input(0, size);
    if (inv.NumIn() != 4 || !user || size != 4)
      return -static_cast<i32>(SysError::eINVAL);
    const auto* title = inv.Input(1, size);
    if (!title || size != 10 || std::memcmp(title, "NPXS22010", 10))
      return -static_cast<i32>(SysError::eNOSYS);
    inv.Input(2, size);
    if (size)
      return -static_cast<i32>(SysError::eNOSYS);
    const auto* param = inv.Input(3, size);
    if (!param || size != 24 || Read32(param, 0) != 24)
      return -static_cast<i32>(SysError::eINVAL);
    const u64 identity = ShmIdentity(kWorkName);
    u64 heartbeat;
    if (!identity ||
        ShmRead(kWorkName, 0x18, &heartbeat, sizeof(heartbeat), identity))
      return -static_cast<i32>(SysError::eNOENT);
    if (!g_worker) {
      if (!guest::SpawnThread("common-dialog", Worker))
        return -static_cast<i32>(SysError::eIO);
      g_worker = true;
    }
    g_processes[inv.Client()] = DialogProcess{true, identity};
    PublishProcessStatus();
  } else {
    const auto* app = inv.Input(0, size);
    if (inv.NumIn() != 1 || !app || size != 4 || Read32(app, 0) != kDialogApp ||
        (!g_processes.contains(inv.Client()) ||
         !g_processes.at(inv.Client()).running))
      return -static_cast<i32>(SysError::eNOENT);
    g_processes.at(inv.Client()).running = false;
    if (auto dialog = g_dialogs.find(inv.Client()); dialog != g_dialogs.end())
      dialog->second.active = false;
    PublishProcessStatus();
  }
  request = g_next_request++;
  g_launch_results.emplace(
      request,
      LaunchResult{inv.Client(), inv.Method() + 1,
                   inv.Method() == 0x30040 ? static_cast<i32>(kDialogApp) : 0});
  return 0;
}

i32 GetCommonDialogLaunchResult(u32 client,
                                u32 method,
                                u32 request,
                                i32& result) {
  std::lock_guard lock(g_mutex);
  const auto it = g_launch_results.find(request);
  if (it == g_launch_results.end() || it->second.client != client ||
      it->second.method != method)
    return -static_cast<i32>(SysError::eNOENT);
  result = it->second.result;
  g_launch_results.erase(it);
  return 0;
}

u64 CommonDialogProcessStatus(const char* name) {
  if (std::strcmp(name, kProcessFlag))
    return 0;
  std::lock_guard lock(g_mutex);
  return ProcessStatus();
}

void DestroyCommonDialogClient(u32 client) {
  std::lock_guard lock(g_mutex);
  g_dialogs.erase(client);
  g_processes.erase(client);
  PublishProcessStatus();
  std::erase_if(g_launch_results, [client](const auto& item) {
    return item.second.client == client;
  });
}
}  // namespace kern::ipmi
