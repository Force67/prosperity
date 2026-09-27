# delta/cpu

Runs guest x86-64 code. `backend.h` is the whole interface: a `cpu::Backend`
that maps images and creates/runs guest threads, plus the per-thread guest
state (fs base, registers) the kernel and crash handler read.

The build picks one implementation by host architecture:

- `native_backend.cc` (x86-64): guest code runs directly; the lifter has
  already rewritten syscalls and fs accesses, so entering the guest is a call.
- `fex_backend.cc` (aarch64): guest code runs through the embedded FEXCore JIT;
  HLE calls cross over through magic syscalls.
