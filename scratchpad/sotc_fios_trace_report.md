# SOTC libSceFios2 guest-function trace — tool, root cause, fix

Session 2026-07-22 (branch ps5-runtime-bringup, HEAD df3a477). Goal: build an
aarch64/FEX-compatible guest-function hook, use it to trace SotC's libSceFios2
whole-file API for the world-op container that hangs `CGame::LoadInitialWorld`'s
main-loop gate (`[op+0x98]==0xb`), root-cause the `FHGetSize=0`/`actualCount 0`,
and fix it emulator-side. int3 hooks (crash.cpp, DELTA_FNWATCH) are
`#if __x86_64__` = INERT under FEX on this ARM host, so a new facility was needed.

## 1. The hook facility (env-gated, off by default)

**`cpu::makeGuestReturnHook(realTarget, hookId, loggerFn, name)`** — new, in
`delta/cpu/fex_backend.cpp` (declared `delta/cpu/cpu_backend.h`; native-backend
no-op stub in `delta/cpu/native_backend.cpp`). It emits a ~51-byte guest x86-64
trampoline into the existing FEX-registered RWX thunk pool (the one
`makeHostThunk` uses, so FEX JITs it) that WRAPS an already-resolved guest
function and captures **both args and its return value**:

```
push rdi/rsi/rdx/rcx        ; save a0..a3
sub rsp,8                    ; 16-align
movabs r11, realTarget ; call r11   ; run the REAL libSceFios2 export -> rax=ret
add rsp,8
mov r9, rax                  ; logger arg6 = ret
pop r8/rcx/rdx/rsi           ; a3..a0 back into logger arg regs
mov edi, hookId
push rax                     ; preserve ret across the syscall
mov r10,rcx ; mov eax, (0x42000000|loggerIdx) ; syscall   ; -> native logger
pop rax ; ret                ; return realTarget's value to the caller
```

The magic syscall (`kHostThunkSyscallBase`) is the project's guest->native bridge
already dispatched in `FexSyscallHandler::HandleSyscall`; the native logger
receives `(hookId, a0,a1,a2,a3, ret)` (SysV). All transient state is on the guest
stack, so the wrapper is reentrant/thread-safe. A **return-capturing** wrapper
(not just an entry log) was mandatory because the smoking gun is a *return value*.

**Installation point = import resolution, not a GOT patch at load.** `delta/kern/
proc.cpp::maybeWrapFiosImport(nidName, realAddr)` is called from
`smodule::resolveImports` (`delta/kern/module.cpp`) for every PLT import; when
`DELTA_FIOS_TRACE` is set and the NID matches one of the libSceFios2 whole-file
APIs it returns the wrapper in place of `realAddr` before the GOT slot is written.
**Why not a GOT patch in proc::create (the first thing I tried):** the eboot's
FIOS2 PLT jump-slots are *lazy* and still hold the stub `thunk+6` at proc::create
(observed: GOT `0x1600130` = `0xf8f686`); `resolveImports` binds them later when
the guest runs `sys_dynlib_process_needed_and_relocate`. Wrapping at resolution
time installs exactly when the slot is bound and it stays for the whole run.

Wrapped NIDs (verified live — `fiostrace: wrapped ... real=0x2072000xxxx`):
`sceFiosFHOpen` er6TkQFUvp0, `sceFiosFHGetSize` FdjoqFQOlt0, `sceFiosFHRead`
cg-VoPqZYss, `sceFiosFHPread` rR8wq7YFRZs, `sceFiosOpGetActualCount` +FRvKknUj1I.

The native logger (`fiosTraceLogger`, proc.cpp) is **path-agnostic**: it records
every FHOpen `(pOutFH,path)`, and flags the anomaly — **any FHGetSize<=0** or
FHRead/FHPread with **len==0** or OpGetActualCount<=0 — then reverse-maps the
`SceFiosFH` back to the path that opened it (`*pOutFH`, populated by the async
open by the time the size/read is issued). Zero events are deduped per fh/op so
the post-drain retry churn (millions of calls) can't flood the log.

Companion diagnostic: **`DELTA_FIOS_PROBE=path,path,...`** (proc.cpp
`probeFiosPaths`) resolves guest paths through the VFS at runtime (fired once at
FHOpen #2000, after PlayGo chunks mount) and logs Exists/size.

**Isaac gate (facility off): 820-877 fps, 0 crashes — unperturbed.** With
`DELTA_FIOS_TRACE` unset, `maybeWrapFiosImport` is identity and nothing is emitted.

## 2. What the trace established (decisive structural findings)

- **FIOS2 is LLE (guest /app0/sce_module/libSceFios2.prx) and does ALL member
  resolution.** The emulator's `/app0` VirtualProvider (PFS from the .pkg) serves
  only raw files. `DELTA_OPEN_TRACE` shows the ONLY physical file types opened are
  `.prx`, **`.psarc` (6 of them)**, `.txt`, `.plt`, `.bin` — **NOT** a single
  `.calt/.csdr/.ctxr/.cgpr/.cmat`. So the ~18k resources (including the world-op
  container) are **members inside psarc archives**, resolved by guest FIOS2 via
  in-memory TOC + `pread`; `FHGetSize` is an in-memory TOC lookup (no syscall).
  The 6 archives (all open with correct sizes, incl. the 12.8 GB one):
  `cachedfiles.psarc 37MB, pak0 364MB, pak1 464MB, pak2 16MB, pak3 720MB,
  pakn.psarc 12838532222 (12.8GB)`.
- **`$` resolves to `/app0`; the eboot's literal `****` segment resolves to
  `_cmn` or `_ps4`** (platform/common variant) and FIOS2 **lowercases** paths
  (eboot `cp3Characters.calt` -> FHOpen arg `/app0/characters/shadow/_cmn/
  cp3characters.calt`). The FHOpen *path arg* is a FIOS2-virtual path; it is NOT
  a loose file (direct `openRead` of it returns MISSING — only the psarc file it
  lives in is real). So the fix cannot be "drop the file in /app0".
- **`mainMenuPrecacheList` is NOT the hanging file.** The 18247-resource counter
  is already populated at 837 ms — the precache LISTS (cp3characters.calt,
  cp3shadowprecache.calt, mainMenuPrecacheList.calt) are read and parsed
  *successfully* early (they enumerate the 18k children). The report's prior
  identification of the container was an inference; the true world-op is a
  different file loaded by `LoadInitialWorld` from `game+0x38`, revealed only by
  the runtime `FHGetSize==0`/`actualCount 0` signal. mainmenu.cgpr (menu script)
  also loads fine (size 64857).

## 3. The world-op read-completion sink (static, confirms the hang shape)

`LoadFileInternal` 0x14b980 read tail (disasm): `OpGetActualCount` (`[vt+0xc0]`)
-> `cdqe` -> `[ldr+0x78]`; no-footer path `0x14bcb0: mov r13,[ldr+0x78]; sub
r13,r12(=footerLen=0); mov [ldr+0x80],r13`. So **payload size `[ldr+0x80]` =
actualCount** and a 0-byte read leaves `[ldr+0x80]=0, err[ldr+0x9c]=0` -> the
silent infinite retry (report §8/§10). The read errors are never checked
(`IsOk` consulted after OPEN at 0x14bb29 but ignored after READ) — so any I/O
failure that reports `actualCount 0` reproduces the hang, while an OPEN failure
(err=2 at 0x14bcca) would *advance* the game.

## 4. Root cause — the given premise is FALSIFIED by direct measurement

**There is no `FHGetSize==0`, no zero-length `FHRead`, and no retry churn.** A full
instrumented run to REMAINING=0 (world load completes, matching the round-9
marathon) and held there for minutes shows, from the wrapped libSceFios2 exports:

- `FHGetSize` returned 0 for **zero** files across the entire load (the
  path-agnostic detector fired 0 times).
- `FHRead`/`FHPread` were called with len==0 **zero** times.
- **No post-drain retry churn**: the `FHOpen` count and the OpGetActualCount-zero
  count are frozen once the counter hits 0 — nothing is being re-attempted. (The
  66k OpGetActualCount==0 events are all `open`-op counts, which are legitimately
  0; they stop growing at drain, so no read op is looping.)
- ps4delta stays at ~2.3 cores / ~4 fps with the render loop alive (black frame,
  draws declined as `norecomp`), NOT the "4 workers hot-spinning" churn.

So the round-10 model — "world-container `sceFiosFHRead` yields actualCount 0 →
payload 0 → CommitResult retry-churn forever" — **does not occur**. Combined with
the earlier force-payload finding that `[ldr+0x90]` (the read buffer) is NULL
(LoadFileInternal never allocated it), the evidence points to the world-op's
`LoadFileInternal` **job never running to completion at all** (no FHOpen for the
world file, no size query, no read) — i.e. a JobSystem dispatch / event-flag wake
deadlock (report §6 H2/H3), not an I/O size/short-read bug. The post-drain state
is a FREEZE (round-10 update-2's "all threads parked in libkernel syscall-wait"),
not a livelock.

### The actual mechanism (post-drain DELTA_WATCHDOG thread dump, decisive)

A full run to REMAINING=0, held frozen, with `DELTA_WATCHDOG` gives the exact
deadlock topology (module bases: eboot 0x201400000000, libkernel 0x200000000000):

| thread | state | meaning |
|---|---|---|
| tid=0 (game main) | `sys_evf_wait(0x110)`, scN advancing | normal ~4fps frame loop; polls the world-op gate `[op+0x98]==0xb` each frame — never passes |
| **tid=9 (Resource Loading, core 6)** | **`sys_evf_wait(0xe8)`, scN=80 FROZEN** | the loader/coordinator parked forever on the "job done" flag that never fires |
| tid=4 | holds scheduler umutex **0x200004140** (word=5), `sys_evf_set(0x98)`, scN=**17.3M** | request-pump spinning "work available" while holding the scheduler mutex |
| tid=5,6,7 | `sys_umtx_op(0x200004140, MUTEX_WAIT)`, scN 7–15M | the other 3 workers hot-spinning for the scheduler mutex |

**No thread is at an eboot RIP inside `LoadFileInternal` (0x14b98x) or any FIOS2
read** — the job *body* never executes. That is exactly why the FIOS hook records
zero I/O anomalies and zero churn: the hang is entirely in the **BPE JobSystem
scheduler**, upstream of any file I/O.

SotC creates a **fixed 4 job-worker threads** (`BPE JobWorkerThread CPU1..CPU4`,
= tid 4-7) plus the core-6 "Resource Loading" thread — independent of the affinity
mask. The 18247 resources (incl. the initial world "home" = the Shrine of
Worship, whose files DO appear in the 33268-path inventory) all load through these
4 workers. **One final job — the world-op finalize/commit — is submitted but the 4
workers perpetually fail to claim it** (report §5/§9: claim tests
`job_affinity & (1<<worker_ordinal)`; the workers exit `DoJobScheduling` seeing a
pending-work bit, lock 0x200004140, `0x38d40` fails to claim, unlock, respin — a
work-visible-but-unclaimable **livelock**). The loader (tid=9) waits forever on
that job's done-flag (evf 0xe8); the game (tid=0) loops the loading screen.

## 5. Fix

**The `FHGetSize=0` fix the task anticipated does not apply — that condition never
occurs.** The real fix must unblock the JobSystem so the final world-op job
dispatches. Attempt made this session, env-gated:

- **`DELTA_SOTC_7CORE`** (`delta/kern/ps4/lv2/sys_info.cpp`): grant core 6 in
  `sys_cpuset_getaffinity` (0x3F→0x7F), on the hypothesis (report §9.2) that the
  finalize job is pinned to core 6 (SotC hardcodes its Resource-Loading thread to
  mask 0x40) and no worker holds the matching ordinal. **Result: SotC still creates
  exactly 4 workers (CPU1-CPU4) — the worker count is fixed, not derived from the
  affinity popcount — so this lever does not add the needed ordinal.** (Kept env-
  gated/off; Isaac/Doom64 unaffected. See §6 for the drain outcome.)

The genuine fix lies in the BPE JobSystem claim/scheduler interaction (report
§6-H3): either (a) our `sys_umtx_op` MUTEX handoff on 0x200004140 lets the
request-pump thread (tid=4) monopolize the lock so no worker ever claims+runs the
pending job (fairness — wake a queued waiter, don't let the releaser re-acquire;
df3a477 began this), or (b) the final job carries an affinity/ordinal none of the
4 fixed workers satisfies and must be routed to a worker that exists. Pinning this
down needs runtime inspection of the pending job's `[job+0x998]` affinity vs the
worker ordinals at claim time (`0x38d40`), which the guest-fn hook facility built
here can now target directly (wrap `0x38d40`/`0x35480` the same way).

## 6. End-to-end result

- Hook facility: **complete and working**; Isaac gate 820-877fps, 0 crashes with
  it off. It delivered the decisive result: the assumed `FHGetSize=0` / retry-churn
  root cause is **falsified by direct measurement**, and the true hang is localized
  to the JobSystem dispatch livelock (§4).
- World load: reaches REMAINING=0 (world data incl. "home"/Shrine loads), then the
  game does **not** advance past the loading-screen gate — `DELTA_GPU_NOCS` frames
  keep ticking at ~4fps drawing nothing (`norecomp` declines), scanout black.
- `DELTA_SOTC_7CORE`: **falsified.** Full drain to REMAINING=0, held frozen 9
  loadwatch samples, **no advance** (0 new MSG lines after "Starting main loop"),
  still exactly 4 workers (CPU1-CPU4). The affinity mask does not change SotC's
  worker count, so it cannot supply the missing ordinal — the finalize job stays
  unclaimable. Left in-tree, env-gated/off.
- Isaac regression gate on the final binary (all env vars off): **825-847 fps, 0
  crashes** — the hook facility, the resolveImports wrapping, the probe, and the
  7-core option are all inert when their env vars are unset.

## 7. Bottom line / handoff

The task's premise (world-container `sceFiosFHGetSize` returns 0 → fix the VFS/FIOS
backing) is **falsified by the very tool it asked for**: across a complete load,
`FHGetSize` never returns 0, no read returns 0 bytes, and there is no retry churn.
The real hang is a **BPE JobSystem scheduler livelock**: after all 18247 resources
(incl. the initial world "home"/Shrine) load, one final world-op finalize job is
never dispatched — 4 worker threads hot-spin on scheduler umutex 0x200004140
(scN 7-17M) failing to claim it, the core-6 Resource-Loading coordinator parks
forever on its evf "job-done" flag (0xe8), and the game loops the loading screen
polling `[op+0x98]==0xb`. No file I/O is involved (no thread ever enters
`LoadFileInternal`).

Next step (uses the facility built here): detour-wrap the JobSystem claim
`0x38d40` and kick `0x35480` to log the pending finalize job's affinity
`[job+0x998]` vs the 4 workers' ordinals at claim time — that decides between
(a) an affinity/ordinal the 4 workers can't satisfy (route it / add the ordinal)
vs (b) a direct-assign to the parked core-6 thread vs (c) a umtx-fairness convoy
where tid=4 monopolizes 0x200004140. `makeGuestReturnHook` already does the
wrapping; wrapping an internal (non-import) function additionally needs a small
entry detour (relocate the prologue) since those aren't reached through a GOT.

## Files changed (all env-gated, no commits)
- `delta/cpu/fex_backend.cpp` — `makeGuestReturnHook` (return-capturing guest hook).
- `delta/cpu/cpu_backend.h`, `delta/cpu/native_backend.cpp` — decl + native stub.
- `delta/kern/proc.cpp` — `maybeWrapFiosImport` (DELTA_FIOS_TRACE), `fiosTraceLogger`,
  `probeFiosPaths` (DELTA_FIOS_PROBE), DELTA_FIOS_ALLOPEN.
- `delta/kern/proc.h` — `maybeWrapFiosImport` decl.
- `delta/kern/module.cpp` — call `maybeWrapFiosImport` in `resolveImports`.
- `delta/kern/ps4/lv2/sys_info.cpp` — `DELTA_SOTC_7CORE` (0x3F→0x7F affinity).

## 5. The fix

_(pending root cause)_

## 6. End-to-end result

_(pending)_
