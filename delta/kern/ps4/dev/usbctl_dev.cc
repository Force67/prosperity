#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <vector>
#include "base/arch.h"

#include "base/time/time.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"

#include "kern/ps4/controller_ports.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/usb_ds4.h"
#include "kern/ps4/dev/usbctl_dev.h"

namespace kern {
namespace {
struct ReadDirectory {
  void* data;
  u32 start_entry;
  u32 max_length;
};
static_assert(sizeof(ReadDirectory) == 16);

constexpr u8 kDirectoryEntry[] = {19,  'u', 'g', 'e', 'n', '1', '.',
                                  '1', 0,   'u', 's', 'b', '/', '1',
                                  '.', '1', '.', '0', 0,   0};
constexpr u8 kDeviceDescriptor[] = {18,   1,    0,    2, 0, 0, 0, 64, 0x4c,
                                    0x05, 0xc4, 0x05, 0, 1, 1, 2, 0,  1};
constexpr u8 kConfigurationDescriptor[] = {9,
                                           2,
                                           41,
                                           0,
                                           1,
                                           1,
                                           0,
                                           0xc0,
                                           250,
                                           9,
                                           4,
                                           0,
                                           0,
                                           2,
                                           3,
                                           0,
                                           0,
                                           0,
                                           9,
                                           0x21,
                                           0x11,
                                           1,
                                           0,
                                           1,
                                           0x22,
                                           sizeof(kDs4ReportDescriptor) & 0xff,
                                           sizeof(kDs4ReportDescriptor) >> 8,
                                           7,
                                           5,
                                           0x84,
                                           3,
                                           64,
                                           0,
                                           5,
                                           7,
                                           5,
                                           0x03,
                                           3,
                                           64,
                                           0,
                                           5};
std::atomic<u8> g_configuration{0};
std::atomic<u64> g_generation{0};
std::mutex g_usb_mutex;
const UgenState* g_interface_owner = nullptr;
std::unordered_set<u32> g_usb_controllers;
void ResetUsb(u8 configuration = 0) {
  std::lock_guard lock(g_usb_mutex);
  g_configuration = configuration;
  ++g_generation;
  g_interface_owner = nullptr;
  for (u32 handle : g_usb_controllers)
    ReleaseControllerHandle(handle);
  g_usb_controllers.clear();
}
const guest::SessionReset g_usb_reset([] { ResetUsb(); });

struct GenericDescriptor {
  void* data;
  u16 language, maximum, actual, offset;
  u8 configuration, string, interface, alternate, endpoint, report_type;
  u8 reserved[8];
};
static_assert(sizeof(GenericDescriptor) == 32);

struct Setup {
  u8 type, request;
  u16 value, index, length;
};
static_assert(sizeof(Setup) == 8);

struct ControlRequest {
  void* data;
  u16 flags, actual;
  u8 address;
  u8 setup[8];
};
static_assert(sizeof(ControlRequest) == 24);
static_assert(offsetof(ControlRequest, setup) == 13);

struct FsEndpoint {
  void** buffers;
  u32* lengths;
  u32 frames, actual_frames;
  u16 flags, timeout, completion_time;
  i32 status;
};
static_assert(sizeof(FsEndpoint) == 40);
struct FsInit {
  FsEndpoint* endpoints;
  u8 count, reserved[7];
};
static_assert(sizeof(FsInit) == 16);
struct FsOpen {
  u32 maximum_buffer, maximum_frames;
  u16 packet_length;
  u8 device, index, endpoint, reserved[3];
};
static_assert(sizeof(FsOpen) == 16);

std::vector<u8> StringDescriptor(u8 index, u16 language) {
  if (!index)
    return {4, 3, 9, 4};
  if (language != 0x409)
    return {};
  const char* text = index == 1   ? "Sony Computer Entertainment"
                     : index == 2 ? "Wireless Controller"
                                  : nullptr;
  if (!text)
    return {};
  std::vector<u8> result{static_cast<u8>(2 + 2 * std::strlen(text)), 3};
  for (; *text; ++text) {
    result.push_back(*text);
    result.push_back(0);
  }
  return result;
}

i32 CopyDescriptor(GenericDescriptor& descriptor,
                   const u8* bytes,
                   size_t size) {
  const size_t offset = std::min<size_t>(descriptor.offset, size);
  const size_t length = std::min<size_t>(descriptor.maximum, size - offset);
  if (length && !host_memory::IsMemoryRangeMapped(descriptor.data, length))
    return -SysError::eFAULT;
  if (length)
    std::memcpy(descriptor.data, bytes + offset, length);
  descriptor.actual = length;
  return 0;
}

struct DeviceInfo {
  u16 product, vendor, release, power;
  u8 bus, address, index, device_class, subclass, protocol;
  u8 configuration, configuration_index;
  u8 speed, mode, ports, hub_address, hub_index, hub_port;
  u8 power_mode, suspended, reserved[16];
  char product_name[128], vendor_name[128], serial[64], release_name[8];
};
static_assert(sizeof(DeviceInfo) == 0x170);
}  // namespace

struct UgenState {
  struct Transfer {
    bool open = false, pending = false, cancelled = false;
    u8 endpoint = 0;
    u32 maximum_buffer = 0, maximum_frames = 0;
    u16 flags = 0, timeout = 0;
    u64 generation = 0;
    std::vector<void*> buffers;
    std::vector<u32> lengths;
    base::TimeTicks ready, deadline;
  };
  std::mutex mutex;
  FsEndpoint* endpoints = nullptr;
  std::vector<Transfer> transfers;
  std::deque<u8> completions;
  u32 controller = 0;
  u8 counter = 0;
};

UsbctlDevice::UsbctlDevice(ObjectTable& objects) : Device(objects) {}

i32 UsbctlDevice::Ioctl(u32 cmd, void* data) {
  if (cmd != 0x8010557f)
    return -SysError::eNOTTY;
  if (!host_memory::IsMemoryRangeMapped(data, sizeof(ReadDirectory)))
    return -SysError::eFAULT;
  const auto& request = *static_cast<ReadDirectory*>(data);
  if (!request.max_length)
    return 0;
  const bool include_entry = !request.start_entry && request.max_length >= 19;
  const size_t length = include_entry ? (request.max_length > 19 ? 20 : 19) : 1;
  if (!host_memory::IsMemoryRangeMapped(request.data, length))
    return -SysError::eFAULT;
  if (!include_entry)
    *static_cast<u8*>(request.data) = 0;
  else
    std::memcpy(request.data, kDirectoryEntry, length);
  return 0;
}

UgenDevice::UgenDevice(ObjectTable& objects)
    : Device(objects), state_(std::make_unique<UgenState>()) {}

UgenDevice::~UgenDevice() {
  std::lock_guard lock(g_usb_mutex);
  if (g_interface_owner == state_.get())
    g_interface_owner = nullptr;
  g_usb_controllers.erase(state_->controller);
  if (state_->controller)
    ReleaseControllerHandle(state_->controller);
}

bool UgenDevice::EnsureController() {
  std::lock_guard lock(g_usb_mutex);
  if (state_->controller && IsControllerHandle(state_->controller))
    return true;
  g_usb_controllers.erase(state_->controller);
  const i32 handle = AllocateControllerHandle();
  if (handle <= 0)
    return false;
  state_->controller = handle;
  g_usb_controllers.insert(handle);
  return BindControllerPort(handle, {1, 0, 0, 0, 0}) == 0;
}

i32 UgenDevice::Ioctl(u32 cmd, void* data) {
  std::lock_guard lock(state_->mutex);
  if ((cmd & 0xff) >= 192 && (cmd & 0xff) <= 199)
    return TransferIoctl(cmd, data);
  switch (cmd) {
    case 0x4004557e:  // USB_GET_PLUGTIME
    case 0x40125569:  // USB_GET_DEVICE_DESC
    case 0x41705570:  // USB_GET_DEVICEINFO
    case 0x41785570:  // PS5 adds two device-information words
    case 0x40045564:  // USB_GET_CONFIG
    case 0x80045506:  // USB_DEVICEENUMERATE
    case 0x80045565:  // USB_SET_CONFIG
    case 0x4009556a:  // USB_GET_CONFIG_DESC
    case 0xc0025566:  // USB_GET_ALTINTERFACE
    case 0xc0025567:  // USB_SET_ALTINTERFACE
    case 0xc020556d:  // USB_GET_FULL_DESC
    case 0xc020556e:  // USB_GET_STRING_DESC
    case 0xc018556f:  // USB_DO_REQUEST
    case 0xc0205579:  // USB_GET_IFACE_DRIVER
    case 0x8004557c:  // USB_IFACE_DRIVER_ACTIVE
    case 0x8004557d:  // USB_IFACE_DRIVER_DETACH
    case 0x80045598:  // Sony claim interface (interface | alternate << 8)
    case 0x80045599:  // Sony release interface
      break;
    default:
      return -SysError::eNOTTY;
  }
  if (!host_memory::IsMemoryRangeMapped(data, (cmd >> 16) & 0x1fff))
    return -SysError::eFAULT;
  switch (cmd) {
    case 0x80045506:
      if (*static_cast<i32*>(data))
        return -SysError::eINVAL;
      ResetUsb();
      break;
    case 0x40045564:
      *static_cast<i32*>(data) = g_configuration;
      break;
    case 0x80045565: {
      const i32 index = *static_cast<i32*>(data);
      if (index != 0 && index != 255)
        return -SysError::eINVAL;
      if (g_configuration != index)
        ResetUsb(index);
      break;
    }
    case 0x4009556a:
      if (g_configuration != 0)
        return -SysError::eNXIO;
      std::memcpy(data, kConfigurationDescriptor, 9);
      break;
    case 0xc0025566:
    case 0xc0025567: {
      auto* interface = static_cast<u8*>(data);
      if (interface[0] || (cmd == 0xc0025567 && interface[1]))
        return -SysError::eINVAL;
      interface[1] = 0;
      break;
    }
    case 0xc020556d: {
      auto& descriptor = *static_cast<GenericDescriptor*>(data);
      if (descriptor.configuration)
        return -SysError::eINVAL;
      return CopyDescriptor(descriptor, kConfigurationDescriptor,
                            sizeof(kConfigurationDescriptor));
    }
    case 0xc020556e: {
      auto& descriptor = *static_cast<GenericDescriptor*>(data);
      auto bytes = StringDescriptor(descriptor.string, descriptor.language);
      if (bytes.empty())
        return -SysError::eINVAL;
      return CopyDescriptor(descriptor, bytes.data(), bytes.size());
    }
    case 0xc018556f: {
      auto& request = *static_cast<ControlRequest*>(data);
      Setup setup;
      std::memcpy(&setup, request.setup, sizeof(setup));
      return Request(setup.type, setup.request, setup.value, setup.index,
                     request.data, setup.length, request.actual);
    }
    case 0xc0205579:
      return -SysError::eNXIO;
    case 0x8004557c:
      return -SysError::eNXIO;
    case 0x8004557d:
      return *static_cast<i32*>(data) == 0 ? 0 : -SysError::eINVAL;
    case 0x80045598: {
      std::lock_guard lock(g_usb_mutex);
      if (*static_cast<u32*>(data) || g_configuration != 0)
        return -SysError::eINVAL;
      if (g_interface_owner)
        return -SysError::eBUSY;
      g_interface_owner = state_.get();
      break;
    }
    case 0x80045599: {
      std::lock_guard lock(g_usb_mutex);
      if (*static_cast<u32*>(data) || g_interface_owner != state_.get())
        return -SysError::eINVAL;
      g_interface_owner = nullptr;
      break;
    }
    case 0x4004557e:
      *static_cast<u32*>(data) = 1;
      break;
    case 0x40125569:
      std::memcpy(data, kDeviceDescriptor, sizeof(kDeviceDescriptor));
      break;
    case 0x41705570:
    case 0x41785570: {
      DeviceInfo info{};
      info.product = 0x05c4;
      info.vendor = 0x054c;
      info.release = 0x0100;
      info.power = 500;
      info.bus = info.address = info.index = 1;
      info.configuration = g_configuration == 0 ? 1 : 0;
      info.configuration_index = g_configuration;
      info.speed = 2;  // USB full speed
      info.hub_port = 1;
      std::strcpy(info.product_name, "Wireless Controller");
      std::strcpy(info.vendor_name, "Sony Computer Entertainment");
      std::strcpy(info.release_name, "1.00");
      if (cmd == 0x41785570)
        std::memset(data, 0, 0x178);
      std::memcpy(data, &info, sizeof(info));
      break;
    }
  }
  return 0;
}

i32 UgenDevice::Request(u8 type,
                        u8 request,
                        u16 value,
                        u16 index,
                        void* data,
                        u16 length,
                        u16& actual) {
  actual = 0;
  if (length && !host_memory::IsMemoryRangeMapped(data, length))
    return -SysError::eFAULT;
  const u8* bytes = nullptr;
  size_t size = 0;
  std::vector<u8> temporary;
  if (request == 6 && (type == 0x80 || type == 0x81)) {
    switch (value >> 8) {
      case 1:
        if (type != 0x80 || (value & 0xff))
          return -SysError::ePIPE;
        bytes = kDeviceDescriptor;
        size = sizeof(kDeviceDescriptor);
        break;
      case 2:
        if (type != 0x80 || (value & 0xff))
          return -SysError::ePIPE;
        bytes = kConfigurationDescriptor;
        size = sizeof(kConfigurationDescriptor);
        break;
      case 3:
        if (type != 0x80)
          return -SysError::ePIPE;
        temporary = StringDescriptor(value & 0xff, index);
        bytes = temporary.data();
        size = temporary.size();
        break;
      case 0x21:
        if (type != 0x81 || index || (value & 0xff))
          return -SysError::ePIPE;
        bytes = kConfigurationDescriptor + 18;
        size = 9;
        break;
      case 0x22:
        if (type != 0x81 || index || (value & 0xff))
          return -SysError::ePIPE;
        bytes = kDs4ReportDescriptor;
        size = sizeof(kDs4ReportDescriptor);
        break;
    }
  } else if (request == 8 && type == 0x80 && !value && !index) {
    temporary = {static_cast<u8>(g_configuration == 0 ? 1 : 0)};
    bytes = temporary.data();
    size = temporary.size();
  } else if (request == 9 && type == 0 && !index && !length && value <= 1) {
    const u8 configuration = value ? 0 : 255;
    if (g_configuration != configuration)
      ResetUsb(configuration);
    return 0;
  } else if (request == 0 && type == 0x80 && !value && !index) {
    temporary = {1, 0};  // Self-powered device
    bytes = temporary.data();
    size = temporary.size();
  } else if (request == 10 && type == 0x81 && !value && !index) {
    temporary = {0};
    bytes = temporary.data();
    size = temporary.size();
  } else if (request == 11 && type == 1 && !value && !index && !length) {
    return 0;
  } else if (request == 1 && type == 2 && !value && !length &&
             (index == 0x84 || index == 3)) {
    return 0;  // CLEAR_FEATURE(ENDPOINT_HALT)
  } else if (request == 9 && type == 0x21 && value == 0x205 && !index) {
    if (length != 32 || static_cast<const u8*>(data)[0] != 5 ||
        !EnsureController())
      return -SysError::ePIPE;
    const auto* report = static_cast<const u8*>(data);
    if (report[1] & 1)
      VibrateControllerPort(state_->controller, report[5], report[4]);
    if (report[1] & 2)
      LightControllerPort(state_->controller, report[6], report[7], report[8]);
    actual = length;
    return 0;
  }
  if (!bytes || !size)
    return -SysError::ePIPE;
  actual = std::min<size_t>(length, size);
  if (actual)
    std::memcpy(data, bytes, actual);
  return 0;
}

namespace {
bool TransferReady(const UgenState::Transfer& transfer) {
  const auto now = base::TimeTicks::Now();
  return transfer.cancelled || transfer.generation != g_generation ||
         now >= transfer.ready ||
         (transfer.timeout && now >= transfer.deadline);
}

void BuildReport(u8* report, const guest::ControllerState& input, u8 counter) {
  std::memset(report, 0, 64);
  report[0] = 1;
  report[1] = input.lx;
  report[2] = input.ly;
  report[3] = input.rx;
  report[4] = input.ry;
  const u32 buttons = input.buttons;
  const bool up = buttons & 0x10, right = buttons & 0x20, down = buttons & 0x40,
             left = buttons & 0x80;
  const u8 hat = up      ? (right  ? 1
                            : left ? 7
                                   : 0)
                 : down  ? (right  ? 3
                            : left ? 5
                                   : 4)
                 : right ? 2
                 : left  ? 6
                         : 8;
  report[5] = hat | ((buttons & 0x8000) ? 0x10 : 0) |
              ((buttons & 0x4000) ? 0x20 : 0) |
              ((buttons & 0x2000) ? 0x40 : 0) | ((buttons & 0x1000) ? 0x80 : 0);
  report[6] = ((buttons & 0x400) ? 1 : 0) | ((buttons & 0x800) ? 2 : 0) |
              ((buttons & 0x100) ? 4 : 0) | ((buttons & 0x200) ? 8 : 0) |
              ((buttons & 1) ? 0x10 : 0) | ((buttons & 8) ? 0x20 : 0) |
              ((buttons & 2) ? 0x40 : 0) | ((buttons & 4) ? 0x80 : 0);
  report[7] = ((buttons & 0x10000) ? 1 : 0) | ((buttons & 0x100000) ? 2 : 0) |
              ((counter & 63) << 2);
  report[8] = input.l2;
  report[9] = input.r2;
  report[24] = 0x20;  // Stationary acceleration: +1g on Z (8192 units/g)
  report[30] = 0x1b;  // USB connected, full battery
  report[33] = 1;
  report[34] = counter;
  report[35] = report[39] = 0x80;  // No active touch contacts
}
}  // namespace

u16 UgenDevice::Poll(u16 events) {
  std::lock_guard lock(state_->mutex);
  return (events & 1) &&
                 std::any_of(state_->completions.begin(),
                             state_->completions.end(),
                             [&](u8 index) {
                               return TransferReady(state_->transfers[index]);
                             })
             ? 1
             : 0;
}

i32 UgenDevice::TransferIoctl(u32 cmd, void* data) {
  switch (cmd) {
    case 0x801055c3:
    case 0x800155c4:
    case 0xc01055c5:
    case 0x400155c2:
    case 0x800155c6:
    case 0x800155c1:
    case 0x800155c7:
    case 0x800155c0:
      break;
    default:
      return -SysError::eNOTTY;
  }
  const u32 size = (cmd >> 16) & 0x1fff;
  if (!host_memory::IsMemoryRangeMapped(data, size))
    return -SysError::eFAULT;
  if (cmd == 0x801055c3) {
    if (state_->endpoints)
      return -SysError::eBUSY;
    const auto& init = *static_cast<FsInit*>(data);
    if (!init.count)
      return -SysError::eINVAL;
    if (!host_memory::IsMemoryRangeMapped(init.endpoints,
                                          init.count * sizeof(FsEndpoint)))
      return -SysError::eFAULT;
    state_->endpoints = init.endpoints;
    state_->transfers.resize(init.count);
    return 0;
  }
  if (cmd == 0x800155c4) {
    state_->endpoints = nullptr;
    state_->transfers.clear();
    state_->completions.clear();
    return 0;
  }
  if (!state_->endpoints)
    return -SysError::eINVAL;
  if (cmd == 0xc01055c5) {
    auto& open = *static_cast<FsOpen*>(data);
    if (open.index >= state_->transfers.size() || !open.maximum_frames ||
        open.maximum_frames > 4096 || !open.maximum_buffer ||
        open.maximum_buffer > (1 << 18) ||
        (open.endpoint != 0 && open.endpoint != 0x80 && open.endpoint != 0x84 &&
         open.endpoint != 3))
      return -SysError::eINVAL;
    if (g_configuration != 0 && (open.endpoint & 0x7f))
      return -SysError::eNXIO;
    auto& transfer = state_->transfers[open.index];
    if (transfer.open)
      return -SysError::eBUSY;
    transfer.open = true;
    transfer.endpoint = open.endpoint;
    transfer.maximum_buffer = open.maximum_buffer;
    transfer.maximum_frames = open.maximum_frames;
    open.packet_length = 64;
    return 0;
  }
  if (cmd == 0x400155c2) {
    auto ready = std::find_if(
        state_->completions.begin(), state_->completions.end(),
        [&](u8 index) { return TransferReady(state_->transfers[index]); });
    if (ready == state_->completions.end())
      return -SysError::eBUSY;
    const u8 index = *ready;
    auto& transfer = state_->transfers[index];
    auto* endpoint = state_->endpoints + index;
    if (!host_memory::IsMemoryRangeMapped(endpoint, sizeof(*endpoint)) ||
        !host_memory::IsMemoryRangeMapped(
            endpoint->lengths, transfer.lengths.size() * sizeof(u32)))
      return -SysError::eFAULT;
    i32 status = (transfer.cancelled || transfer.generation != g_generation) ? 5
                 : transfer.timeout &&
                         base::TimeTicks::Now() >= transfer.deadline &&
                         transfer.deadline < transfer.ready
                     ? 20
                     : 0;
    std::vector<u32> lengths(transfer.lengths.size(), 0);
    u32 frames = 0;
    if (!status && (transfer.endpoint & 0x7f) && g_configuration != 0)
      status = 19;
    if (!status) {
      for (size_t frame = 0; frame < transfer.buffers.size(); ++frame) {
        if (transfer.lengths[frame] &&
            !host_memory::IsMemoryRangeMapped(transfer.buffers[frame],
                                              transfer.lengths[frame])) {
          status = 6;
          break;
        }
      }
    }
    if (!status && !(transfer.endpoint & 0x7f)) {
      if (transfer.lengths.empty() || transfer.lengths[0] != 8 ||
          transfer.lengths.size() > 2) {
        status = 3;
      } else {
        Setup setup;
        std::memcpy(&setup, transfer.buffers[0], sizeof(setup));
        if (setup.length && (transfer.lengths.size() != 2 ||
                             transfer.lengths[1] < setup.length)) {
          status = 7;
        } else {
          u16 actual = 0;
          const i32 result =
              Request(setup.type, setup.request, setup.value, setup.index,
                      setup.length ? transfer.buffers[1] : nullptr,
                      setup.length, actual);
          status = result == 0                                     ? 0
                   : result == -static_cast<i32>(SysError::eFAULT) ? 6
                                                                   : 22;
          lengths[0] = 8;
          if (lengths.size() > 1)
            lengths[1] = actual;
          frames = status ? 0 : lengths.size();
        }
      }
    } else if (!status && transfer.endpoint == 0x84) {
      guest::ControllerState input;
      if (!EnsureController() ||
          !ReadControllerPort(state_->controller, input)) {
        status = 18;
      } else {
        u8 report[64];
        BuildReport(report, input, state_->counter++);
        for (size_t i = 0; i < lengths.size(); ++i) {
          lengths[i] = std::min<u32>(transfer.lengths[i], sizeof(report));
          if (lengths[i])
            std::memcpy(transfer.buffers[i], report, lengths[i]);
          ++frames;
          if (lengths[i] < transfer.lengths[i] && !(transfer.flags & 3)) {
            status = 21;
            break;
          }
          if (lengths[i] < transfer.lengths[i] && !(transfer.flags & 2))
            break;
        }
      }
    } else if (!status && transfer.endpoint == 3) {
      for (size_t i = 0; i < lengths.size(); ++i) {
        if (transfer.lengths[i] != 32) {
          status = 22;
          break;
        }
        u16 actual = 0;
        if (Request(0x21, 9, 0x205, 0, transfer.buffers[i], transfer.lengths[i],
                    actual)) {
          status = 22;
          break;
        }
        lengths[i] = actual;
        ++frames;
      }
    }
    std::copy(lengths.begin(), lengths.end(), endpoint->lengths);
    endpoint->actual_frames = frames;
    endpoint->status = status;
    endpoint->completion_time = 0;
    transfer.pending = false;
    transfer.buffers.clear();
    transfer.lengths.clear();
    state_->completions.erase(ready);
    *static_cast<u8*>(data) = index;
    return 0;
  }
  const u8 index = *static_cast<u8*>(data);
  if (index >= state_->transfers.size() || !state_->transfers[index].open)
    return -SysError::eINVAL;
  auto& transfer = state_->transfers[index];
  if (cmd == 0x800155c6) {
    state_->completions.erase(std::remove(state_->completions.begin(),
                                          state_->completions.end(), index),
                              state_->completions.end());
    transfer = {};
    return 0;
  }
  if (cmd == 0x800155c1) {
    if (transfer.pending)
      transfer.cancelled = true;
    return 0;
  }
  if (cmd == 0x800155c7)
    return transfer.pending ? -SysError::eBUSY : 0;
  if (cmd != 0x800155c0)
    return -SysError::eNOTTY;
  if (transfer.pending)
    return -SysError::eBUSY;
  const auto* endpoint = state_->endpoints + index;
  if (!host_memory::IsMemoryRangeMapped(endpoint, sizeof(*endpoint)))
    return -SysError::eFAULT;
  const u32 frames = endpoint->frames;
  if (!frames || frames > transfer.maximum_frames || (endpoint->flags & ~15u))
    return -SysError::eINVAL;
  if (!host_memory::IsMemoryRangeMapped(endpoint->buffers,
                                        frames * sizeof(void*)) ||
      !host_memory::IsMemoryRangeMapped(endpoint->lengths,
                                        frames * sizeof(u32)))
    return -SysError::eFAULT;
  size_t total = 0;
  for (u32 i = 0; i < frames; ++i) {
    total += endpoint->lengths[i];
    if (total > transfer.maximum_buffer)
      return -SysError::eINVAL;
    if (endpoint->lengths[i] && !host_memory::IsMemoryRangeMapped(
                                    endpoint->buffers[i], endpoint->lengths[i]))
      return -SysError::eFAULT;
  }
  transfer.buffers.assign(endpoint->buffers, endpoint->buffers + frames);
  transfer.lengths.assign(endpoint->lengths, endpoint->lengths + frames);
  transfer.flags = endpoint->flags;
  transfer.generation = g_generation;
  transfer.timeout = endpoint->timeout;
  const auto now = base::TimeTicks::Now();
  transfer.ready = now + base::Milliseconds(transfer.endpoint == 0x84 ? 5 : 0);
  transfer.deadline = now + base::Milliseconds(transfer.timeout);
  transfer.pending = true;
  transfer.cancelled = false;
  state_->completions.push_back(index);
  return 0;
}

int UgenDevice::Fstat(void* stat) {
  if (!host_memory::IsMemoryRangeMapped(stat, sizeof(SceKernelStat)))
    return -SysError::eFAULT;
  FillStat(*static_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}

i64 UsbctlDevice::Lseek(i64, int) {
  return 0;
}

int UsbctlDevice::Fstat(void* stat) {
  if (!host_memory::IsMemoryRangeMapped(stat, sizeof(SceKernelStat)))
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace kern
