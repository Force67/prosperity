# Code conventions

All of `delta/`, `shared/` and `tools/` follow
[Chromium C++ style](https://chromium.googlesource.com/chromium/src/+/main/styleguide/c++/c++.md),
enforced by the root `.clang-format` and `.clang-tidy` (naming):

- Files are `snake_case.cc` / `.h`, named after the unit inside. Platform
  variants are separate files the build selects (`window_sdl.cc`,
  `window_android.cc`), not `#ifdef`s around a whole file.
- Types `CamelCase`; functions `CamelCase()`; trivial accessors may be
  `snake_case()`; variables, parameters and struct members `snake_case`;
  private class members `trailing_`; constants and `DELTA_OPTION` knobs
  `kCamelCase`; mutable globals `g_snake_case`; thread-locals `t_snake_case`;
  macros `UPPER_CASE`. A namespace matches its directory (`kern`, `host`).
- In-repo includes are quoted and spelled from their include root:
  `"kern/process.h"` (delta), `"io/file.h"` (shared), `"base/arch.h"`
  (equilibrium). Angle brackets are for system and third-party headers.
- Unit tests live in each module's `tests/`, registered through
  `add_delta_test`.
- Every module has a README saying what each unit hides. Module dependencies
  are one-way and checked by `delta/tests/check_module_layering.py` (and
  `delta/gpu/tests/check_layering.py` inside gpu).

Names that mirror a specification keep its spelling, behind `NOLINT` markers,
so they can be grepped against the source they come from: guest entry points
(`sce*`, `sys_*`, `unk_<nid>`), BSD/SCE ABI structs and constants
(`kevent_t`, `ePERM`, `O_CREAT`), AMD register and PM4 mnemonics, x86 register
names, and JNI exports (`Java_*`). A `k`-prefixed mnemonic (`kEVFILT_READ`) is
used where the bare spelling collides with a libc macro. The `prosperity_*`
functions are C-linkage bridges between modules and keep their names.

To check a file: `clang-tidy -p <build> <file>` and `clang-format --dry-run <file>`.
