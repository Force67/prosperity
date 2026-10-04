#include <array>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "base/environment_variables.h"
#include "formats/title_metadata.h"
#include "guest/session.h"
#include "kern/ipmi/services.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kAppContentDir, "DELTA_APPCONTENT_DIR", nullptr);
}

namespace kern::ipmi {
namespace {
constexpr i32 kParameter = static_cast<i32>(0x80d90002u);
constexpr i32 kBusy = static_cast<i32>(0x80d90003u);
constexpr i32 kNotFound = static_cast<i32>(0x80d90005u);
constexpr i32 kInternal = static_cast<i32>(0x80d9000au);
constexpr char kTempPoint[] = "/temp0";
std::mutex g_mutex;
bool g_mounted = false;
std::unordered_map<u32, bool> g_modern_clients;

bool Read(Invocation& inv, u32 slot, void* destination, u64 bytes) {
  u64 size = 0;
  const void* source = inv.Input(slot, size);
  if (!source || size < bytes) {
    inv.SetResult(kParameter);
    return false;
  }
  std::memcpy(destination, source, bytes);
  return true;
}

bool Output(Invocation& inv, u64 bytes) {
  if (inv.CanReply(0, bytes))
    return true;
  inv.SetResult(kParameter);
  return false;
}

base::String TempHostDir() {
  base::String root;
  if (kAppContentDir && *kAppContentDir) {
    root = kAppContentDir;
  } else {
    base::StringU8 home;
    base::GetEnvironmentVariable(u8"HOME", home);
    root = base::String(home.empty()
                            ? "."
                            : reinterpret_cast<const char*>(home.c_str())) +
           "/.prosperity/appcontent";
  }
  const auto& title = vfs::TitleId();
  for (char c : title)
    if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')))
      return {};
  return root + "/" + (title.empty() ? base::String("APPCONTENT") : title) +
         "/temp0";
}

i32 UserDefinedParam(u32 id) {
  auto file = vfs::OpenRead("/app0/sce_sys/param.json");
  if (!file.IsOpen()) {
    auto sfo = vfs::OpenRead("/app0/sce_sys/param.sfo");
    if (!sfo.IsOpen() || sfo.GetSize() > 0x100000)
      return 0;
    std::vector<u8> data(sfo.GetSize());
    if (sfo.Read(data.data(), data.size()) != data.size())
      return 0;
    const auto key = "USER_DEFINED_PARAM_" + std::to_string(id);
    return static_cast<i32>(
        formats::SfoGetU32(data.data(), data.size(), key.c_str()));
  }
  const u64 size = file.GetSize();
  if (size > 0x100000)
    return 0;
  base::String json(static_cast<size_t>(size), '\0');
  if (file.Read(json.data(), json.size()) != json.size())
    return 0;
  const base::String key = "userDefinedParam" + std::to_string(id);
  int depth = 0;
  for (size_t pos = 0; pos < json.size(); ++pos) {
    const char c = json[pos];
    if (c == '{' || c == '[') {
      ++depth;
    } else if (c == '}' || c == ']') {
      --depth;
    } else if (c == '"') {
      const size_t start = ++pos;
      while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\')
          ++pos;
        ++pos;
      }
      if (depth != 1 || json.substr(start, pos - start) != key)
        continue;
      auto whitespace = [&] {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                     json[pos] == '\r' || json[pos] == '\n'))
          ++pos;
      };
      ++pos;
      whitespace();
      if (pos >= json.size() || json[pos++] != ':')
        return 0;
      whitespace();
      i64 value = 0;
      const auto result =
          std::from_chars(json.data() + pos, json.data() + json.size(), value);
      if (result.ec != std::errc() || value < std::numeric_limits<i32>::min() ||
          value > std::numeric_limits<u32>::max())
        return 0;
      pos = result.ptr - json.data();
      whitespace();
      if (pos == json.size() || (json[pos] != ',' && json[pos] != '}'))
        return 0;
      return static_cast<i32>(value);
    }
  }
  return 0;
}

bool MountPoint(Invocation& inv, u32 slot, base::String& point) {
  std::array<char, 16> data;
  if (!Read(inv, slot, data.data(), data.size()))
    return false;
  const auto* end =
      static_cast<const char*>(std::memchr(data.data(), 0, data.size()));
  if (!end) {
    inv.SetResult(kParameter);
    return false;
  }
  point.assign(data.data(), end - data.data());
  return true;
}

bool Format(Invocation& inv, const base::String& host) {
  std::error_code error;
  std::filesystem::remove_all(host.c_str(), error);
  if (!error)
    std::filesystem::create_directories(host.c_str(), error);
  if (!error)
    return true;
  inv.SetResult(kInternal);
  return false;
}

// PS4 11.00 and PS5 08.40 have an extra method before temporary-data mount.
// Mount and free-space signatures identify which method table the client uses.
struct AppContent : Service {
  const char* Name() const override { return "SceAppContent"; }
  void DestroyClient(u32 client) override {
    std::lock_guard lock(g_mutex);
    g_modern_clients.erase(client);
  }
  void Invoke(Invocation& inv) override {
    std::lock_guard lock(g_mutex);
    u32 method = inv.Method();
    const auto client = g_modern_clients.find(inv.Client());
    bool modern = client != g_modern_clients.end() && client->second;
    if (((method == 0x2000a || method == 0x2000b) && inv.NumIn() == 1 &&
         inv.NumOut() == 1) ||
        ((method == 0x2000e || method == 0x2000f) && inv.NumIn() == 2 &&
         inv.NumOut() == 1)) {
      const bool request_modern = method == 0x2000b || method == 0x2000f;
      if (client != g_modern_clients.end() && modern != request_modern) {
        inv.SetResult(kParameter);
        return;
      }
      modern = request_modern;
    } else if (method >= 0x2000b && method <= 0x2000d &&
               client == g_modern_clients.end()) {
      inv.SetResult(kNotFound);
      return;
    }
    if (modern && method >= 0x2000b)
      --method;
    switch (method) {
      case 0x20000: {
        u32 sdk;
        std::array<u8, 32> init;
        if (!Read(inv, 0, &sdk, sizeof(sdk)) ||
            !Read(inv, 1, init.data(), init.size()) || !Output(inv, 40))
          return;
        for (u8 byte : init)
          if (byte) {
            inv.SetResult(kParameter);
            return;
          }
        const std::array<u8, 40> boot{};
        inv.Reply(0, boot.data(), boot.size());
        return;
      }
      case 0x20001: {
        u32 id;
        if (!Read(inv, 0, &id, sizeof(id)) || !Output(inv, sizeof(i32)))
          return;
        if (id > 4) {
          inv.SetResult(kParameter);
          return;
        }
        const i32 value = id ? UserDefinedParam(id) : 3;
        inv.Reply(0, &value, sizeof(value));
        return;
      }
      case 0x2000a: {
        u32 option;
        if (!Read(inv, 0, &option, sizeof(option)) || !Output(inv, 16))
          return;
        if (option > 1) {
          inv.SetResult(kParameter);
          return;
        }
        if (g_mounted) {
          inv.SetResult(kBusy);
          return;
        }
        const auto host = TempHostDir();
        if (host.empty()) {
          inv.SetResult(kParameter);
          return;
        }
        if (option && !Format(inv, host))
          return;
        std::error_code error;
        std::filesystem::create_directories(host.c_str(), error);
        if (error) {
          inv.SetResult(kInternal);
          return;
        }
        vfs::MountWritable(kTempPoint, host.c_str());
        g_mounted = true;
        g_modern_clients[inv.Client()] = modern;
        inv.Reply(0, kTempPoint, sizeof(kTempPoint));
        return;
      }
      case 0x2000b:
      case 0x2000c: {
        base::String point;
        if (!MountPoint(inv, 0, point))
          return;
        if (point != kTempPoint || !g_mounted) {
          inv.SetResult(kNotFound);
          return;
        }
        if (method == 0x2000c) {
          Format(inv, vfs::ResolveWritable(kTempPoint));
        } else {
          vfs::Unmount(kTempPoint);
          g_mounted = false;
        }
        return;
      }
      case 0x2000e: {
        u32 kind;
        base::String point;
        if (!Read(inv, 0, &kind, sizeof(kind)) || !MountPoint(inv, 1, point) ||
            !Output(inv, sizeof(u64)))
          return;
        if (kind > 1 || (kind == 1 && point != kTempPoint) ||
            (kind == 0 && point != "/download0" && point != "/download1")) {
          inv.SetResult(kParameter);
          return;
        }
        const auto host = vfs::ResolveWritable(point.c_str());
        if (host.empty()) {
          inv.SetResult(kNotFound);
          return;
        }
        std::error_code error;
        const auto space = std::filesystem::space(host.c_str(), error);
        if (error) {
          inv.SetResult(kInternal);
          return;
        }
        const u64 available = space.available / 1024;
        g_modern_clients[inv.Client()] = modern;
        inv.Reply(0, &available, sizeof(available));
        return;
      }
      default:
        inv.SetResult(kInternal);
        return;
    }
  }
};

AppContent g_app_content;
const guest::SessionReset g_reset([] {
  std::lock_guard lock(g_mutex);
  if (g_mounted)
    vfs::Unmount(kTempPoint);
  g_mounted = false;
  g_modern_clients.clear();
});
}  // namespace

Service& AppContentService() {
  return g_app_content;
}
}  // namespace kern::ipmi
