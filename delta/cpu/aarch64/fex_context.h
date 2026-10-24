#pragma once

#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/SignalDelegator.h>
#include <FEXCore/Debug/InternalThreadState.h>

namespace cpu::fex {

inline FEXCore::Context::Context* g_context = nullptr;
// FEX fills in the dispatcher entry points when the core initialises.
inline const FEXCore::SignalDelegator* g_signal_delegator = nullptr;

inline thread_local FEXCore::Core::InternalThreadState* t_thread = nullptr;

}  // namespace cpu::fex
