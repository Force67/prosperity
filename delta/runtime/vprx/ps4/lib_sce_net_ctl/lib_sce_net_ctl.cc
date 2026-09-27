#include "runtime/vprx/ps4/lib_sce_net_ctl/lib_sce_net_ctl.h"
#include "base/arch.h"
#include "guest_abi.h"

#include <cstring>

// A single wired ethernet interface with a static LAN config. Values are only
// read back by titles for display/logging; nothing routes through them.
namespace {
constexpr i32 kStateIpObtained = 3;  // SCE_NET_CTL_STATE_IPOBTAINED

// SceNetCtlInfo selector codes (the union member each one fills)
enum : i32 {
  kInfoDevice = 1,            // u32: 0 = wired
  kInfoEtherAddr = 2,         // u8[6]
  kInfoMtu = 3,               // u32
  kInfoLink = 4,              // u32: 1 = up
  kInfoBssid = 5,             // u8[6]
  kInfoSsid = 6,              // char[33]
  kInfoWifiSecurity = 7,      // u32
  kInfoRssiDbm = 8,           // u8
  kInfoRssiPercentage = 9,    // u8
  kInfoChannel = 10,          // u8
  kInfoIpConfig = 11,         // u32: 0 = dhcp
  kInfoDhcpHostname = 12,     // char[256]
  kInfoPppoeAuthName = 13,    // char[128]
  kInfoIpAddress = 14,        // char[16]
  kInfoNetmask = 15,          // char[16]
  kInfoDefaultRoute = 16,     // char[16]
  kInfoPrimaryDns = 17,       // char[16]
  kInfoSecondaryDns = 18,     // char[16]
  kInfoHttpProxyConfig = 19,  // u32: 0 = off
  kInfoHttpProxyServer = 20,  // char[256]
  kInfoHttpProxyPort = 21,    // u16
};
constexpr int kErrorInvalidCode = 0x80412102;  // SCE_NET_CTL_ERROR_INVALID_CODE
}  // namespace

int PS4ABI sceNetCtlInit() {
  return 0;
}
int PS4ABI sceNetCtlTerm() {
  return 0;
}

int PS4ABI sceNetCtlGetState(i32* state) {
  if (!state)
    return -1;
  *state = kStateIpObtained;
  return 0;
}

int PS4ABI sceNetCtlGetInfo(i32 code, void* info) {
  if (!info)
    return -1;
  // the union's largest members are 256 bytes
  std::memset(info, 0, 256);
  auto str = [&](const char* s) { std::strcpy(static_cast<char*>(info), s); };
  auto word = [&](u32 v) { std::memcpy(info, &v, 4); };
  switch (code) {
    case kInfoDevice:
      word(0);
      break;
    case kInfoEtherAddr: {
      static const u8 kMac[6] = {0x02, 0x50, 0x54, 0x00, 0x00, 0x01};
      std::memcpy(info, kMac, 6);
      break;
    }
    case kInfoMtu:
      word(1500);
      break;
    case kInfoLink:
      word(1);
      break;
    case kInfoBssid:
      break;  // wired: zeroed
    case kInfoSsid:
      break;
    case kInfoWifiSecurity:
      break;
    case kInfoRssiDbm:
      break;
    case kInfoRssiPercentage:
      break;
    case kInfoChannel:
      break;
    case kInfoIpConfig:
      word(0);
      break;  // dhcp
    case kInfoDhcpHostname:
      str("ps4");
      break;
    case kInfoPppoeAuthName:
      break;
    case kInfoIpAddress:
      str("192.168.1.100");
      break;
    case kInfoNetmask:
      str("255.255.255.0");
      break;
    case kInfoDefaultRoute:
      str("192.168.1.1");
      break;
    case kInfoPrimaryDns:
      str("8.8.8.8");
      break;
    case kInfoSecondaryDns:
      str("8.8.4.4");
      break;
    case kInfoHttpProxyConfig:
      word(0);
      break;
    case kInfoHttpProxyServer:
      break;
    case kInfoHttpProxyPort:
      break;
    default:
      return kErrorInvalidCode;
  }
  return 0;
}

int PS4ABI sceNetCtlRegisterCallback(void* func, void* arg, i32* cid) {
  // The state never changes, so the callback never needs to fire; hand out a
  // fixed id for the matching Unregister.
  if (cid)
    *cid = 0;
  return 0;
}

int PS4ABI sceNetCtlUnregisterCallback(i32 cid) {
  return 0;
}
int PS4ABI sceNetCtlCheckCallback() {
  return 0;
}

// The ForNpToolkit entry points differ only in taking the callback id back
// through the return value rather than an out pointer; the link never changes
// state either way, so nothing is ever queued for them to deliver.
int PS4ABI sceNetCtlRegisterCallbackForNpToolkit(void* func,
                                                 void* arg,
                                                 i32* cid) {
  if (cid)
    *cid = 0;
  return 0;
}

int PS4ABI sceNetCtlUnregisterCallbackForNpToolkit(i32 cid) {
  return 0;
}
int PS4ABI sceNetCtlCheckCallbackForNpToolkit() {
  return 0;
}
