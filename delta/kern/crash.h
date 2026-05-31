#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

namespace krnl {
// Install signal handlers that symbolize a guest fault (rip + frame-pointer
// backtrace) to <module>+offset using the loaded module table, then exit.
void installCrashHandler();
}  // namespace krnl
