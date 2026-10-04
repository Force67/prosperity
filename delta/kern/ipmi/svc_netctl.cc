#include <array>
#include <cstring>

#include "kern/ipmi/services.h"

namespace kern::ipmi {
namespace {
constexpr i32 kInvalidCode = static_cast<i32>(0x80412102u);
constexpr i32 kInvalidArgument = static_cast<i32>(0x80412107u);
constexpr u32 kIpObtained = 3;

// The mounted guest has one wired interface. These are its virtual LAN
// settings, independent of the host's interface names and address layout.
bool InterfaceInfo(u32 selector, std::array<u8, 256>& info) {
  auto word = [&](u32 value) {
    std::memcpy(info.data(), &value, sizeof(value));
  };
  auto text = [&](const char* value) {
    std::memcpy(info.data(), value, std::strlen(value) + 1);
  };
  switch (selector) {
    case 1:  // Wired device.
      word(0);
      break;
    case 2: {
      const u8 address[] = {0x02, 0x50, 0x54, 0x00, 0x00, 0x01};
      std::memcpy(info.data(), address, sizeof(address));
      break;
    }
    case 3:
      word(1500);
      break;
    case 4:  // Link up.
      word(1);
      break;
    case 5:   // BSSID.
    case 6:   // SSID.
    case 7:   // Wireless security.
    case 8:   // RSSI dBm.
    case 9:   // RSSI percentage.
    case 10:  // Wireless channel.
    case 11:  // DHCP.
    case 13:  // PPPoE name.
    case 19:  // Proxy disabled.
    case 20:  // Proxy server.
    case 21:  // Proxy port.
      break;
    case 12:
      text("ps4");
      break;
    case 14:
      text("192.168.1.100");
      break;
    case 15:
      text("255.255.255.0");
      break;
    case 16:
      text("192.168.1.1");
      break;
    case 17:
      text("8.8.8.8");
      break;
    case 18:
      text("8.8.4.4");
      break;
    default:
      return false;
  }
  return true;
}

struct NetCtl : Service {
  const char* Name() const override { return "SceNetCtl"; }

  void Invoke(Invocation& inv) override {
    switch (inv.Method()) {
      case 0x20004:
        if (!inv.CanReply(0, sizeof(u32)))
          inv.SetResult(kInvalidArgument);
        else
          inv.ReplyU32(0, kIpObtained);
        break;
      case 0x20005: {
        u64 size = 0;
        const void* input = inv.Input(0, size);
        if (!input || size != sizeof(u32)) {
          inv.SetResult(kInvalidArgument);
          break;
        }
        u32 selector;
        std::memcpy(&selector, input, sizeof(selector));
        std::array<u8, 256> info{};
        if (!InterfaceInfo(selector, info))
          inv.SetResult(kInvalidCode);
        else if (!inv.CanReply(0, info.size()))
          inv.SetResult(kInvalidArgument);
        else
          inv.Reply(0, info.data(), info.size());
        break;
      }
      default:
        inv.ReplyEmpty();
        break;
    }
  }
};

NetCtl g_net_ctl;
}  // namespace

Service& NetCtlService() {
  return g_net_ctl;
}
}  // namespace kern::ipmi
