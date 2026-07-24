# Stackless native FS lifting
Status: implemented

## Context
PS4 code uses the FS segment for guest thread-local storage. On the native
backend, Prosperity rewrites each supported FS-relative `mov` to call a rip-zone
stub, which calls a host helper and saves temporary registers on the guest
stack. The original instruction does not touch the stack.

This corrupts live red-zone data and memory below a fiber stack pointer. In
Uncharted 2, a worker switches `rsp` to the end of its current fiber context and
an injected helper call overwrites the adjacent job-scheduler bucket.

## Decision
Rewrite supported FS-relative moves to jump to a stackless rip-zone stub. The
stub reads the guest FS base directly from host thread-local storage, performs
the guest load or store, and jumps to the instruction following the original
move.

Loads use their destination register as the address temporary. Stores preserve
one temporary register in a dedicated host thread-local scratch slot. The
translation uses only moves and jumps, preserving guest flags and leaving guest
stack memory unchanged.

## Alternatives
- Keep helper calls and reserve stack space: rejected because any guest stack
  write changes the semantics of the original instruction.
- Set the host FS base to the guest value: rejected because host C++ and libc
  require the host thread pointer.
- Run the title through FEX: rejected because native translation must preserve
  guest semantics independently of another backend.

## Consequences
Native FS translation no longer depends on helper calling conventions or guest
stack alignment. It adds one host thread-local scratch word for translated
stores and requires the rip zone to hold a final relative jump.

## Acceptance
- Runtime tests verify that translated FS loads and stores branch to stackless
  stubs rather than call stubs.
- Existing tests pass.
- In combination with correct VFS root metadata, Uncharted 2 no longer
  overwrites bucket 7's pending-buffer pointer from the fiber stack and
  advances beyond `big2-ps4_Shipping+0xb3cf4a`.
