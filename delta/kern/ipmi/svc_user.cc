/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <array>
#include <cstring>

#include "kern/ipmi/services.h"

namespace kern::ipmi {
namespace {
constexpr i32 kUserId = 1;
constexpr i32 kInvalidArgument = static_cast<i32>(0x80960005u);
constexpr i32 kNotLoggedIn = static_cast<i32>(0x80960009u);
constexpr i32 kBufferTooShort = static_cast<i32>(0x8096000au);

struct User : Service {
  const char* Name() const override { return "SceUserService"; }

  void Invoke(Invocation& inv) override {
    switch (inv.Method()) {
      case 0x3000e:
      case 0x3000f: {
        const u64 size = inv.Method() == 0x3000e ? 16 : 80;
        std::array<i32, 20> users;
        users.fill(-1);
        users[0] = kUserId;
        if (inv.NumIn() || inv.NumOut() != 1 || !inv.CanReply(0, size) ||
            !inv.Reply(0, users.data(), size))
          inv.SetResult(kInvalidArgument);
        break;
      }
      case 0x3000c:  // Foreground user.
      case 0x30011:  // Initial user.
        if (!inv.ReplyU32(0, kUserId))
          inv.SetResult(kInvalidArgument);
        break;
      case 0x30004: {  // String setting: {user id, key, buffer size}.
        u64 size = 0;
        const auto* input = static_cast<const u32*>(inv.Input(0, size));
        if (!input || size < 16) {
          inv.SetResult(kInvalidArgument);
          break;
        }
        if (input[0] != kUserId) {
          inv.SetResult(kNotLoggedIn);
          break;
        }
        if (input[1] == 1) {
          u64 capacity;
          std::memcpy(&capacity, input + 2, sizeof(capacity));
          if (capacity < sizeof("Player"))
            inv.SetResult(kBufferTooShort);
          else if (!inv.Reply(0, "Player", sizeof("Player")))
            inv.SetResult(kInvalidArgument);
        } else {
          inv.ReplyEmpty();
        }
        break;
      }
      default:
        inv.ReplyEmpty();
        break;
    }
  }

  bool ReceiveEvent(u32,
                    u32& cursor,
                    void* data,
                    u64 capacity,
                    u64& size) override {
    if (cursor || capacity < 8)
      return false;
    const i32 event[] = {0, kUserId};
    std::memcpy(data, event, sizeof(event));
    size = sizeof(event);
    ++cursor;
    return true;
  }
};

User g_user_service;
}  // namespace

Service& UserService() {
  return g_user_service;
}
}  // namespace kern::ipmi
