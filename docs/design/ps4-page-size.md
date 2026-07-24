# PS4 page size reporting
Status: implemented

## Context
PS4 virtual memory uses 16 KiB pages, and Prosperity already rounds mappings,
residency queries, and guest stack backing to `0x4000`. The `kern.pagesize`
sysctl instead reports 4096. Guest allocators can therefore size adjacent
objects using a smaller granularity than the kernel uses for their mappings.

The mismatch was found while investigating an Uncharted 2 startup fault. The
reported page size affects guest allocator behavior, but correcting it does not
resolve that fault.

## Decision
Return `0x4000` from the PS4 `kern.pagesize` sysctl. Keep the existing 16 KiB
rounding in the VM implementation unchanged.

## Alternatives
- Patch the title's queue code: rejected because the invalid layout originates
  in a kernel ABI mismatch and affects other titles.
- Serialize the title's worker threads: rejected because it hides the overlap
  and materially changes guest scheduling.

## Consequences
Guest allocators and the emulated VM now agree on page granularity. Allocation
layouts that were produced using the incorrect 4 KiB value can change.

## Acceptance
- A unit test verifies that `kern.pagesize` returns `0x4000`.
- Existing tests pass.
