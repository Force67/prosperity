#include <cstring>

#include "kern/ipmi/services.h"
#include "kern/ps4/audio_in.h"

namespace kern::ipmi {
namespace {
struct AudioSystem : Service {
  const char* Name() const override { return "SceSysAudioSystemIpc"; }
  void Invoke(Invocation& inv) override {
    switch (inv.Method()) {
      case 0x12340018:
      case 0x12340019: {
        u64 size = 0;
        const void* input = inv.Input(0, size);
        if (!input || size != 8) {
          inv.SetResult(static_cast<i32>(0x80260106u));
          return;
        }
        u32 request[2];
        std::memcpy(request, input, sizeof(request));
        inv.SetResult(ps4::SetAudioInPort(request[0], request[1],
                                          inv.Method() == 0x12340018));
        break;
      }
      default:
        inv.ReplyEmpty();
        break;
    }
  }
};
AudioSystem g_audio;
}  // namespace
Service& AudioSystemService() {
  return g_audio;
}
}  // namespace kern::ipmi
