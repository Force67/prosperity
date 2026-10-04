#include <cstring>

#include "kern/ipmi/services.h"

namespace kern::ipmi {
namespace {
constexpr i32 kNotOpened = static_cast<i32>(0x80bc0001u);
constexpr i32 kNoKeyboard = static_cast<i32>(0x80bc0004u);
constexpr i32 kInvalidUser = static_cast<i32>(0x80bc0010u);
constexpr i32 kInvalidParameter = static_cast<i32>(0x80bc0032u);
constexpr i32 kInternal = static_cast<i32>(0x80bc00ffu);

struct Ime : Service {
  const char* Name() const override { return "SceImeService"; }

  i32 Connect(const void* data, u64 size) override {
    if (!data || size != 24)
      return kInvalidParameter;
    u32 type, user;
    std::memcpy(&type, data, sizeof(type));
    std::memcpy(&user, static_cast<const u8*>(data) + 20, sizeof(user));
    if (type != 4)
      return kInternal;
    for (u32 i = 12; i < 20; ++i)
      if (static_cast<const u8*>(data)[i])
        return kInvalidParameter;
    if (user != 1 && user != 0xfe)
      return kInvalidUser;
    // The virtual input inventory has a controller, but no IME keyboard.
    return kNoKeyboard;
  }

  void Invoke(Invocation& inv) override { inv.SetResult(kNotOpened); }
};

Ime g_ime;
}  // namespace

Service& ImeService() {
  return g_ime;
}
}  // namespace kern::ipmi
