# Memory debugger

`Scope(Bucket)` tags C++ allocations on the current thread. Replacement
ordinary and aligned new/delete keep the tag in a fixed header beside each
allocation, so freeing on another thread uses the original bucket. Standard
library nothrow allocation retains its normal behavior. Tracking uses no
dynamic containers or allocation registry on this path: a TLS tag, a header
(32 bytes on 64-bit hosts, plus alignment padding), and relaxed atomic updates
to 32 static thread shards. Snapshots are approximate during concurrent
allocation and exact after updates settle.

`Record` belongs to an explicitly managed resource and removes its counters
when destroyed or reset. Vulkan buffers, sparse backing blocks, textures, and
image allocator blocks use these records. Texture and render-target tiles
share the image allocator's physical backing. The image backing tile shows
their combined occupancy, also shown as `Pool %` and `Pool free` directly in
the texture tiles; that live total is not counted twice. Imported
guest-memory buffers are aliases, reported separately from allocated GPU
memory. Empty retained image blocks remain in backing until released.

FEX host mappings use a fixed 8192-slot table with at most 64 probes on normal
insertion and exact removal. Partial unmap and fixed replacement scan the
bounded table. Failed insertions increment the visible dropped-record count.
Inaccessible address reservations are excluded from these host buckets.
Guest mapped/reserved totals use the existing host mapping registry.

The UI takes snapshots at 4 Hz while visible and stores 256 fixed history
samples. It does not sample RSS or mappings while hidden. Debugger UI heap
allocations are excluded from the tagged host total. Tracking stays active
while the overlay is hidden, so opening it shows current consumption.

Coverage: C++ heap allocations, tagged FEX mappings, guest mapping totals,
and Vulkan allocations managed by the emulator. Native OS thread stacks,
C-library allocations (including FFmpeg internals), graphics driver internals,
and OpenGL/D3D12 device allocations are not individually attributed. Linux
process RSS includes resident guest pages and these untracked areas, so it
does not equal the sum of host and GPU tiles. Heap backing is usable payload
capacity, excluding tracking headers and allocator metadata. FEX mapping
backing is virtual mapped capacity, not residency.

Build with `-DDELTA_MEMORY_DEBUG=OFF` to remove replacement operators,
allocation counters, resource record storage, and the memory overlay.
