#pragma once

#include "kern/ipmi/ipmi.h"

namespace kern::ipmi {
bool InvokeCommonDialog(Invocation& inv);
i32 LaunchCommonDialogProcess(Invocation& inv, u32& request);
i32 GetCommonDialogLaunchResult(u32 client,
                                u32 method,
                                u32 request,
                                i32& result);
u64 CommonDialogProcessStatus(const char* name);
void DestroyCommonDialogClient(u32 client);
}  // namespace kern::ipmi
