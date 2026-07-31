/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#pragma once

#include <base/option.h>

// Every runtime knob is a base::Option whose name doubles as the environment
// variable it reads, so DELTA_GPU_TRACE=1 in the environment, +DELTA_GPU_TRACE=1
// in an options file and the same on the command line all set one option.
// Declare it at namespace scope in the file that reads it (never as a function
// local: initOptions can only fill options that already registered):
//
//   namespace {
//   DELTA_OPTION(bool, kGpuTrace, "DELTA_GPU_TRACE", false, "trace PM4 packets");
//   }
//
// The description is optional. Options an unrelated module also reads belong in
// that module's header as an extern instead of being declared twice.
#define DELTA_OPTION(type, var, name, default_value, ...)                      \
  base::Option<type> var { name, default_value, name, "" __VA_ARGS__ }

// The same for an option a header declares: one instance across the whole
// program, so it registers once however many files include it.
#define DELTA_OPTION_INLINE(type, var, name, default_value, ...)               \
  inline base::Option<type> var { name, default_value, name, "" __VA_ARGS__ }

namespace utl {

// Applies the environment, then the options files named by --options=<path> or
// DELTA_OPTIONS, then '+Name=Value' arguments, in that order: the later source
// wins. Consumes the arguments it handles (--options=, --dump-options, +Name=)
// so the caller's own parsing never sees them. Call once at startup, before
// anything reads an option.
void initOptions(int &argc, char **argv);

// The same, for hosts that have no command line of their own (the Android
// activity). Options come from the environment and DELTA_OPTIONS.
void initOptions();

// Applies a single options file. `optional` is for the paths we probe rather
// than the ones someone asked for: a missing file then passes without a word.
bool loadOptionFile(const char *path, bool optional = false);

} // namespace utl
