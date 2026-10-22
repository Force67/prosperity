#pragma once

#include <FEXCore/Core/Context.h>
#include <FEXCore/Debug/InternalThreadState.h>

namespace cpu::fex {

inline FEXCore::Context::Context* g_context = nullptr;

inline thread_local FEXCore::Core::InternalThreadState* t_thread = nullptr;

}  // namespace cpu::fex
