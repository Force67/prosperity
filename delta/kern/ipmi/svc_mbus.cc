#include <cstring>

#include "kern/ipmi/services.h"
#include "kern/lv2/error_table.h"
#include "kern/ps4/controller_ports.h"

namespace kern::ipmi {
namespace {
struct Mbus : Service {
  const char* Name() const override { return "SceMbusIpc"; }

  void Invoke(Invocation& inv) override {
    const u32 method = inv.Method();
    if (method != 0xce110007 && method != 0xce11000a) {
      inv.SetResult(-SysError::eNOTSUP);
      return;
    }
    u64 size = 0;
    const void* input = inv.Input(0, size);
    const u64 expected = method == 0xce110007 ? 28 : 12;
    if (!input || size != expected || inv.NumIn() != 1 || inv.NumOut() != 0) {
      inv.SetResult(-SysError::eINVAL);
      return;
    }
    // PS4 message bus client sends seven words to register, three to unregister.
    u32 args[7]{};
    std::memcpy(args, input, size);
    if (args[0] == 1)
      return;  // Audio ports continue to use the audio daemon's own handles.
    if (args[0] != 0) {
      inv.SetResult(-SysError::eNOTSUP);
      return;
    }
    if (method == 0xce110007) {
      ControllerPort port{args[2], args[3], args[4], args[5], args[6]};
      inv.SetResult(BindControllerPort(args[1], port));
    } else {
      inv.SetResult(UnbindControllerPort(args[1], args[2]));
    }
  }
};
Mbus g_mbus;
}  // namespace

Service& MbusService() {
  return g_mbus;
}
}  // namespace kern::ipmi
