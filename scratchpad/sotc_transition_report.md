# SOTC post-LoadInitialWorld gate — static RE report

Binary: `/tmp/sotc_eboot.elf` (Shadow_Shipping, guest base 0x201400000000).
All addresses below are **module offsets** (add 0x201400000000 for runtime; file offset = offset + 0x4000 — verified: getter at ELF vaddr 0x147f00 disassembles exactly as the known 10-queue sum).
Tools: capstone linear disasm + a vectorized rel32/RIP-rel xref scanner (`scratchpad/sdis.py`, `scratchpad/xref.py` in the session scratchpad).

## Executive summary

**There is no separate "completion detector" polling the resource counter. The counter is
logging-only.** `CGame::LoadInitialWorld` (real entry **0x3c5170**) is *synchronous*: it creates an
async world-load **op** object and then **busy-polls, on the game thread, the op's state dword at
`op+0x98` until it equals 0xb (11 = DONE)**. Nothing after the loop is reached until that happens;
the caller (boot-phase driver at 0x25c080, call site **0x25c1fa**) continues the boot sequence
inline after the call returns.

The value 0xb is written by exactly one runtime-relevant function: the op's **commit** at
**0x14d670**, reached from the resource **finalize** fn **0x147160**, which runs inside a
JobSystem job named `"ResourceFactory::ProcessLoadingThreadLoop"` submitted by a dedicated
loader thread (**0x1464d0**). Every hand-off in that chain is a **sceKernelEventFlag**
(always bit 1, always **infinite timeout** — the timeout wrapper 0x3b4e0 has *zero* xrefs), so a
single lost wake or a starved worker deadlocks the transition permanently, with the counter
sitting at 0 — exactly the observed state.

---

## 1. The wait side: CGame::LoadInitialWorld (0x3c5170)

Known log fn "0x3c515f" is actually the stack-cookie-fail tail of the previous fn; the real
prologue is **0x3c5170**. Anatomy:

- `0x3c51a2 call 0x17070` (new 0x18) + `0x3c51ad call 0x1d2d90` — allocate a 3-word *tracker*
  {op*, flag, state}; stored at `game+0x28` (`0x3c51e2 mov [r15+0x28], r13`).
- `0x3c51e6 lea rax,[rip+0x2b1db0b]` → **0x2ee2cf8** = ResourceFactory-owner singleton slot
  (8 bytes below the known counter object slot 0x2ee2d00).
  `0x3c522b call r14` where `r14=[[obj]+0x18]` — **virtual call vtbl+0x18 = "create async
  world-load op"** returning a refcounted op into `[rsp+0xe0]`.
- `0x3c52cf call 0x14cf20(op, 0xd, name)` — pins the op (lock xadd `[op+0x40]`, +1).
- `0x3c52e7 call 0x1d2dd0(tracker, &op)` — arm tracker, state=1.
- First log at `0x3c5487` (counter getter called at **0x3c53d9**, obj from slot **0x2ee2d00**).

**The poll loop — the game thread parks here forever (0x3c54e0..0x3c55fd):**

```
3c54e0: call 0x32430            ; elapsed-seconds of a periodic log timer
3c54e8: vucomiss xmm0,[rip+..]  ; below threshold -> skip logging
3c54f0: jbe 0x3c55f2
  ...  (log-channel-enabled check 0x3c54fe..0x3c551b can also skip the log)
3c5521: lea rax,[rip+0x2b1d7d8] ; = 0x2ee2d00  (counter obj slot)
3c5545: call 0x147f00           ; %d for the log line ONLY
3c55e1: call 0x3e7e0            ; the second known log site
3c55f2: mov rdi,[rbx+0x28]      ; tracker
3c55f6: call 0x1d2f90           ; "is the op done?"
3c55fb: test al,al
3c55fd: je 0x3c54e0             ; NOT DONE -> loop. No sleep, no kernel wait.
3c5603: ...epilogue, ret        ; continuation = simply RETURN to caller 0x25c1ff
```

Why only two log lines ever: the loop keeps spinning silently once the periodic-log timer /
log-channel gate stops passing; **absence of further log lines does NOT mean the loop exited.**

**The gate predicate** `0x1d2f90(tracker)`:

```
1d2fae: mov eax,[rbx+0x14]      ; tracker state: 0 -> return 0 (never done!)
1d2fbe: mov rdi,[rbx]           ; the op
1d2fc1: call 0x14d2c0           ; busy?
1d2fd4: call 0x14cf00           ; on completion: fetch result, state=2, return 1
```

`0x14d2c0` — **THE gate, 3 instructions**:

```
14d2c0: cmp dword ptr [rdi+0x98], 0xb
14d2c7: setne al
14d2ca: ret
```

`0x14cf00`: done means `[op+0x98]==0xb && [op+0x20]!=0` (result pointer set).

**Mission 2 answer:** wait = memory poll of `[op+0x98] != 0xb` at 0x3c55f6/0x14d2c0 (no kernel
primitive on this side); continuation = fall through to 0x3c5603 → return to 0x25c1ff, where the
boot-phase driver (fn 0x25c080) proceeds with the rest of init (registers the loaded world at
`[rax+0x2054e8]`, `0x25c229` refcount dance, then continues phase setup from 0x25c289).
Runtime: `op = **(void***)(game+0x28)`, i.e. `tracker=[game+0x28]`, `op=[tracker+0]`.

## 2. Mission 1: the "completion detector" — there isn't one

Exhaustive rel32 xref scan over all 22 MB of text found **exactly two** calls to getter 0x147f00:
`0x3c53d9` and `0x3c5545` — both are the log sites inside LoadInitialWorld. RIP-relative refs to
the counter-object slot 0x2ee2d00 all live in the ResourceFactory's own code (0x144xxx–0x14dxxx)
plus the two log sites. **Nobody compares the sum to 0 and branches.** The counter reaching 0 is
epiphenomenal; the transition trigger is the op-state write below.

## 3. Who writes state 0xb: the commit fn 0x14d670

Scan for `mov dword [reg+0x98], imm` found the 0xb writers: 0x14c8f4, 0x14ca5f (op
cancel/abandon paths in fns 0x14c8a0/0x14c9a0) and the live path **0x14d670** (writes at
**0x14d701 / 0x14d73e / 0x14d76a**). 0x14d670 = `AsyncOp::CommitResult(op=rdi, result=rsi)`:

```
14d690: spinlock acquire on op+0x50   (pause loop; after 0x2711 spins -> sceKernelUsleep(1) via 0x3b570)
14d6f1: mov [r15+0x20], r14           ; store result
14d6f5: cmp dword [r15+0x40], 0       ; pin refcount
14d6fc:   refs>0 && result!=0 -> 14d701: [op+0x98]=0xb ; notify owner [op+0x68] vtbl+0x40 / vtbl+0x68
14d73e:   refs==0            ->        [op+0x98]=0xb
14d763:   result==0 && [op+0x60]!=0 -> 14d76a: [op+0x98]=0xb   (error done)
14d777:   result==0 && [op+0x60]==0 -> NO 0xb: [op+0x68] vtbl+0x68 -> queue vtbl+0x18(op) = RE-ENQUEUE (retry)
```

Only caller: **0x1473d0**, inside the finalize fn **0x147160** (`ResourceFactory::FinalizeResource`),
guarded by `0x1473c3 cmp qword [op+0x20],0; jne skip`. 0x147160 gets the payload via
`0x1471ae call 0x14c000`; **if the payload is NULL it commits result=0** (`0x1471b9 test r14,r14;
je 0x1473b1` → r12=0 → `0x14d670(op, 0)`) which lands in the **retry branch 0x14d777** →
infinite re-enqueue churn (see §5 hypothesis H3).

On success 0x147160's tail also does the waiter hand-off:
`0x147439 lock factory+0x70` → if `byte [factory+0x90]` set: clear it,
**SetEventFlag(factory+0xa8)** at `0x14745f` → unlock (this is the "ready-callback →
0x3b450 → 0x3b3c0" sequence from the known facts; the containing fn is 0x147160, and the
per-resource queue-move state machine is the *next* fn, real prologue **0x147490** — the known
"0x147488" is its predecessor's cookie-fail tail. Its `dec/inc [r15+idx*0x28+0x110]` pair is at
0x147545/0x147586; its own transition signals are Sets at 0x14760d/0x147b2b/0x147c39).

## 4. The producer chain and its event flags

One object — call it **ResourceFactory** (global instance ptr at **0x2ee2d00**; queues at
+0x110..+0x278) — owns every primitive. All event flags are created by wrapper **0x3b3d0** with
**attr 0x21** (`EVF_ATTR_MULTI|TH_FIFO`); Set is always **bit 1**; Wait mode is
**0x11 (AND|CLEAR_ALL)** when the ef's auto-clear byte (+8) is 0, else **0x01 (AND)**;
**timeout is always NULL/infinite** (timed wrapper 0x3b4e0: zero xrefs).

| offset in factory | role | Set by | Waited by |
|---|---|---|---|
| +0x98 | "work available" | **0x14857a** (in request-pump fn 0x1480f0), external kickers 0x146aaa | loader thread **0x1465a7** |
| +0xa8 | "a resource finalized" (gated by byte +0x90) | **0x14745f** (finalize 0x147160) | in-job wait paths (0x146de4 area) |
| +0xb8 | "loader idle" notify (gated by byte +0x91) | loader **0x146709** | external WaitForAll loop **0x146aba** |
| +0xc8 | "job completed" | job body **0x1491f6** | loader thread **0x1466c2** |
| +0x70 | pthread mutex guarding the flag bytes 0x90/0x91/0x92 | lock 0x3b390 / unlock 0x3b3c0 | — |

**Loader thread loop** (`0x1464d0`, string `"ResourceFactory::ProcessLoadingThreadLoop"` at vaddr
0xfa9698 names the job it spawns):

```
1465a7: WaitEventFlag(factory+0x98)  [infinite]      ; wait for work
1465af: ClearEventFlag(factory+0xc8, ~1)
1465cc: call 0x35590 (JobSystem enqueue; singleton at [0x2ea4718])
1466ba: call 0x35480 (JobSystem kick: alloc slot, sets job affinity mask
                      [job+0x998] (default = [jobsys+0x3c]) and prio [job+0x9a0],
                      publish 0x34e60, wake workers 0x35250 -> per-worker SetEventFlag)
1466c2: WaitEventFlag(factory+0xc8)  [infinite]      ; wait for the job to finish
1466eb..14670e: lock +0x70; if byte+0x91 {clear; SetEventFlag(+0xb8)}; unlock
146721: PollEventFlag(exit-ef); loop
```

The job body (giant fn region 0x1480f0..0x1499xx; queue processing core 0x1451a0/0x146ea0)
drains the 10 queues, calls **0x147160 finalize** per resource (in-worker help loop
`0x146c20..0x146c84` loops `0x146ea0` + `0x147160` until the *target op itself* is finalized),
then **SetEventFlag(factory+0xc8) at 0x1491f6** before returning to the worker.

## 5. Mission 3: the JobSystem gate ("DoJobScheduling")

- Worker thread entry: **0x38700** (spawned from 0x33582). Per-worker event flag object at
  `jobsys + idx*0x90 + 0xc0`.
- **0x38b40 is "DoJobScheduling"** — the hot spin observed at runtime:

```
38be0: al = [jobsys+0]                      ; shutdown flag
38bfd..38c34: OR together 8 priority buckets [jobsys+0x1478 + i*0x2078 (+worker offs)]
              -> if any pending-work bit set, RETURN (caller then locks & picks)
38c3d: pause                                ; nothing pending:
38c3f..38c6a: clock 0x323c0 vs last-wake stamp [jobsys+idx*8+0xd88]; within grace -> spin again
38c76: call 0x39cc0  ("may I park?")  -> true:
38bb7:   WaitEventFlag(jobsys+idx*0x90+0xc0) [infinite]; Clear; restamp clock; respin
```

- Caller loop in 0x38700: `0x387a3 call 0x38b40` → `0x387c3 lock jobsys+0xb80`
  (**pthread mutex — this is the umutex observed at guest 0x200004140**) →
  `0x3885b call 0x38d40` (claim a job: first the direct-assign slot `[jobsys+wk*0x120+0xeb0]`,
  else shared buckets honoring the **job affinity bitmask `[job+0x998]`** — `bt edx,esi` loops at
  0x386e0/0x388a0) → run job (`0x38e43 call [job+0xe90]`) or **no-claim path 0x38af0: unlock and
  jump straight back to 0x387a0 — no wait**.

**Lost-wakeup vs work-starvation:** a parked worker (lost ef wake) would be *invisible* (asleep).
Four workers **hot-spinning on the scheduler mutex** means they repeatedly *exit* 0x38b40
(pending-work bits set) → lock 0xb80 → **0x38d40 fails to claim** → unlock → see the bits again.
That is **work-visible-but-unclaimable**, not a classic lost wake on the worker side.

## 6. Failure hypotheses, ranked (with runtime discriminators)

All three produce the exact observed state (counter 0, no logs, main thread mute, compositor
unaffected). Discriminate by sampling guest RIPs / breakpoints — module base 0x201400000000:

**H1 — Retry churn from a NULL payload (best fit for the 4 spinners).**
`0x14c000` returns NULL for the world op's payload → `0x147160` commits result=0 →
`0x14d670` branch **0x14d777** re-enqueues the op instead of writing 0xb → job re-submitted
forever. Workers perpetually see pending work, contend umutex 0x200004140, claim, run a no-op
iteration, repeat. *Check:* breakpoint 0x201400147777 (retry branch) and 0x20140014d701/73e/76a
(the three 0xb writes). If 0x14d777 fires repeatedly and no 0xb write ever fires → H1. Then the
real bug is whatever HLE/GPU call inside payload creation returns null (cf. PT texture-commit).

**H2 — Lost event-flag wake in the loader chain.**
The loader thread is parked forever in `WaitEventFlag` at **0x1465a7** (factory+0x98) or
**0x1466c2** (factory+0xc8); the final Set (0x14857a resp. 0x1491f6) was dropped or its latched
bit was consumed. All waits are infinite → permanent deadlock. *Check:* is any thread's RIP
inside the import called from 0xf8f990 with a waiting pattern arg of 1? If the loader thread sits
at 0x2014001465ac/0x2014001466c7 return addresses → H2. Kernel-side invariant to verify:
`sceKernelSetEventFlag(ef, 1)` with **no current waiter must latch bit 1** so a later
`WaitEventFlag(1, AND|CLEAR_ALL)` returns immediately; with `EVF_ATTR_MULTI` (attr 0x21) and one
Set + one waiter, exactly that waiter must wake and CLEAR_ALL must not eat wakes for others.

**H3 — Scheduler-mutex starvation (umutex 0x200004140).**
The job IS claimable, but the 4 spinning workers' lock/unlock storm on `jobsys+0xb80`
(pthread mutex → `_umtx_op` on 0x200004140) starves either the loader thread or the worker
that claimed the ResourceFactory job (e.g. our umtx wake handing the lock to a spinner instead
of the queued waiter). Commit df3a477 ("honor umtx wake counts") touches exactly this — re-test
on it. *Check:* does the ProcessLoadingThreadLoop job's worker make progress between mutex
acquisitions (RIP inside 0x148xxx/0x145xxx vs. stuck at the 0x3b390 import)?

Also worth one glance: the park-grace clock `0x323c0` — if it returns a frozen/regressing time,
workers can never take the `0x39cc0`→park path and will spin by design (feeds H3).

## 7. Mission 4: patch levers (emulator-level, game-agnostic)

1. **Event-flag latch/wake correctness** (serves H2, and H1's hand-offs): in our
   sceKernelEventFlag impl verify: (a) Set with no waiter persists the pattern; (b) waiter with
   `WAITMODE_AND|CLEAR_ALL (0x11)` atomically consumes-and-clears on wake, and a Set racing the
   Clear→Wait window (loader clears +0xc8 at 0x1465af *before* submitting) is never lost;
   (c) `EVF_ATTR_MULTI (0x21)` allows multiple simultaneous waiters. The game's whole phase
   transition is 4 event flags with bit 1 and infinite waits — zero tolerance for a lost wake.
2. **umtx fairness/wake-count on contended pthread mutexes** (H3): ensure `_umtx_op`
   WAKE on 0x200004140 wakes the queued waiter (not just releases to whoever spins next), and
   that a woken waiter isn't required to win a race against 4 hot re-lockers. df3a477 is the
   right area; validate with the loader thread's priority vs workers.
3. If runtime confirms **H1**, the lever is not sync at all: instrument `0x20140014c000`'s
   return and walk into whichever resource-creation HLE returns NULL for the world op — same
   class of bug as PT's missing texture commit.

## Appendix: address table (module offsets)

| addr | what |
|---|---|
| 0x3c5170 | CGame::LoadInitialWorld real entry (wrapper fn 0x3c4fa0; called from 0x25c1fa in boot driver 0x25c080) |
| 0x3c54e0–0x3c55fd | main-thread busy-poll loop; gate call 0x3c55f6 |
| 0x1d2f90 / 0x14d2c0 / 0x14cf00 | IsDone tri-state / `cmp [op+0x98],0xb` / fetch-result |
| 0x147f00 | counter getter — xrefs ONLY 0x3c53d9, 0x3c5545 (log sites) |
| 0x2ee2d00 / 0x2ee2cf8 | ResourceFactory instance slot / owner-singleton slot |
| 0x14d670 | CommitResult; 0xb writes 0x14d701/0x14d73e/0x14d76a; **retry branch 0x14d777** |
| 0x147160 | FinalizeResource (calls commit at 0x1473d0; payload fetch 0x1471ae→0x14c000; +0xa8 Set 0x14745f) |
| 0x147490 | per-resource queue state machine (dec/inc at 0x147545/0x147586) |
| 0x1464d0 | loader thread loop; waits 0x1465a7 (+0x98), 0x1466c2 (+0xc8) |
| 0x1480f0 | request pump; work-ef Set 0x14857a; job body signals +0xc8 at 0x1491f6 |
| 0x35590 / 0x35480 / 0x35250 | JobSystem enqueue / kick (affinity [job+0x998]) / worker wake |
| 0x38700 / 0x38b40 / 0x38d40 | worker main / **DoJobScheduling spin (0x38be0–0x38c6a)** / claim job |
| jobsys+0xb80 | scheduler pthread mutex = runtime umutex 0x200004140; jobsys singleton ptr at 0x2ea4718 |
| 0x3b3d0/0x3b450/0x3b460/0x3b470/0x3b4a0/0x3b4e0 | ef create(attr 0x21)/Set(bit1)/Clear(~1)/Poll/Wait(inf)/Wait(timed, UNUSED) |
| 0x3b390/0x3b3c0/0x3b570 | pthread_mutex_lock/unlock, sceKernelUsleep (imports 0xf8f8e0/0xf8f900/0xf8fa20; ef Set/Wait imports 0xf8f980/0xf8f990) |

Unresolved statically: which concrete import NIDs back 0xf8f9xx (no section headers/symbol names
in the dump — identified by calling convention + known-fact anchors 0x3b450/0x3b3c0); and which
of H1/H2/H3 is live — that needs the runtime discriminators above (three breakpoints decide it).
