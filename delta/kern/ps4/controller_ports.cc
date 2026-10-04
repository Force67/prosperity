#include "kern/ps4/controller_ports.h"

#include <mutex>
#include <unordered_map>

#include "guest/session.h"
#include "kern/lv2/error_table.h"

namespace kern {
namespace {
struct Handle {
  bool bound = false;
  ControllerPort port{};
};
std::mutex g_mutex;
std::unordered_map<u32, Handle> g_handles;
u32 g_next_handle = 1;
bool (*g_poll)(guest::ControllerState&) = nullptr;
void (*g_open)(int, int, int) = nullptr;
void (*g_rumble)(u8, u8) = nullptr;
u32 g_rumble_handle = 0;
void (*g_light)(u8, u8, u8) = nullptr;
u32 g_light_handle = 0;

void RestoreLight(u32 handle) {
  if (!handle || g_light_handle != handle)
    return;
  // The single local user's default controller color is blue.
  if (g_light)
    g_light(0, 0, 255);
  g_light_handle = 0;
}

void StopRumble(u32 handle) {
  if (!handle || g_rumble_handle != handle)
    return;
  if (g_rumble)
    g_rumble(0, 0);
  g_rumble_handle = 0;
}

void Reset() {
  std::lock_guard lock(g_mutex);
  if (g_rumble_handle)
    StopRumble(g_rumble_handle);
  if (g_light_handle)
    RestoreLight(g_light_handle);
  g_handles.clear();
}
guest::SessionReset g_reset(Reset);
}  // namespace

i32 AllocateControllerHandle() {
  std::lock_guard lock(g_mutex);
  if (g_next_handle > 0x7fffffff)
    return -SysError::eNOMEM;
  const u32 handle = g_next_handle++;
  g_handles.emplace(handle, Handle{});
  return handle;
}

bool ReleaseControllerHandle(u32 handle) {
  std::lock_guard lock(g_mutex);
  StopRumble(handle);
  RestoreLight(handle);
  return g_handles.erase(handle) != 0;
}

bool IsControllerHandle(u32 handle) {
  std::lock_guard lock(g_mutex);
  return g_handles.count(handle) != 0;
}

i32 BindControllerPort(u32 handle, const ControllerPort& port) {
  std::lock_guard lock(g_mutex);
  auto it = g_handles.find(handle);
  if (it == g_handles.end())
    return -SysError::eBADF;
  StopRumble(handle);
  RestoreLight(handle);
  it->second = {true, port};
  if (g_open)
    g_open(port.user, port.type, port.index);
  return 0;
}

i32 UnbindControllerPort(u32 handle, u32 pid) {
  std::lock_guard lock(g_mutex);
  auto it = g_handles.find(handle);
  if (it == g_handles.end())
    return -SysError::eBADF;
  if (it->second.bound && it->second.port.pid != pid)
    return -SysError::eINVAL;
  StopRumble(handle);
  RestoreLight(handle);
  it->second = {};
  return 0;
}

bool GetControllerPort(u32 handle, ControllerPort& port) {
  std::lock_guard lock(g_mutex);
  auto it = g_handles.find(handle);
  if (it == g_handles.end() || !it->second.bound)
    return false;
  port = it->second.port;
  return true;
}

void SetControllerSource(bool (*poll)(guest::ControllerState&)) {
  std::lock_guard lock(g_mutex);
  g_poll = poll;
}

void SetControllerOpen(void (*open)(int, int, int)) {
  std::lock_guard lock(g_mutex);
  g_open = open;
}

bool ReadControllerPort(u32 handle, guest::ControllerState& state) {
  std::lock_guard lock(g_mutex);
  auto it = g_handles.find(handle);
  if (it == g_handles.end() || !it->second.bound || !g_poll)
    return false;
  const auto& port = it->second.port;
  return port.user == 1 && port.type == 0 && port.index == 0 && g_poll(state);
}

bool ControllerPortConnected(u32 handle) {
  std::lock_guard lock(g_mutex);
  auto it = g_handles.find(handle);
  if (it == g_handles.end() || !it->second.bound || !g_poll)
    return false;
  const auto& port = it->second.port;
  return port.user == 1 && port.type == 0 && port.index == 0;
}

bool HasControllerSource() {
  std::lock_guard lock(g_mutex);
  return g_poll != nullptr;
}

void SetControllerRumble(void (*rumble)(u8, u8)) {
  std::lock_guard lock(g_mutex);
  if (g_rumble_handle)
    StopRumble(g_rumble_handle);
  g_rumble = rumble;
}

i32 VibrateControllerPort(u32 handle, u8 large, u8 small) {
  std::lock_guard lock(g_mutex);
  auto it = g_handles.find(handle);
  if (it == g_handles.end())
    return -static_cast<i32>(SysError::eBADF);
  if (!it->second.bound || !g_poll)
    return -static_cast<i32>(SysError::eNODEV);
  const auto& port = it->second.port;
  if (port.user != 1 || port.type != 0 || port.index != 0)
    return -static_cast<i32>(SysError::eNODEV);
  if (g_rumble)
    g_rumble(large, small);
  g_rumble_handle = large || small ? handle : 0;
  return 0;
}

void SetControllerLight(void (*light)(u8, u8, u8)) {
  std::lock_guard lock(g_mutex);
  if (g_light_handle)
    RestoreLight(g_light_handle);
  g_light = light;
}

i32 LightControllerPort(u32 handle, u8 red, u8 green, u8 blue) {
  std::lock_guard lock(g_mutex);
  auto it = g_handles.find(handle);
  if (it == g_handles.end())
    return -static_cast<i32>(SysError::eBADF);
  const auto& port = it->second.port;
  if (!it->second.bound || !g_poll || port.user != 1 || port.type != 0 ||
      port.index != 0)
    return -static_cast<i32>(SysError::eNODEV);
  if (g_light)
    g_light(red, green, blue);
  g_light_handle = handle;
  return 0;
}

i32 ResetControllerLight(u32 handle) {
  return LightControllerPort(handle, 0, 0, 255);
}
}  // namespace kern
