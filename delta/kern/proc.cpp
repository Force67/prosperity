/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <thread>
#include <chrono>
#include <base.h>
#include <utl/file.h>
#include <utl/mem.h>
#include <utl/path.h>

#include "crash.h"
#include "module.h"
#include "proc.h"

#include "ps5/lv2/ctor_probe.h"
#include "ps5/lv2/initial_tcb.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
#include "ps4/lv2/sys_dynlib.h"
#include "ps4/lv2/sys_mem.h"
#include "runtime/vprx/vprx.h"

#include <cstdlib>
#include <cstring>

namespace krnl {
static proc *g_activeProc{nullptr};

// The guest fs base (TLS) and how the guest entry is run are backend-specific
// (see delta/cpu): native uses a host thread_local + direct call, FEX uses the
// FEXCore CPUState + JIT. setThreadFsBase() is defined by the active backend.

proc::proc() : vmem(env) { g_activeProc = this; }

proc *proc::getActive() { return g_activeProc; }

static void bringUpRebirthEbootRegistry(smodule &m);
static void investigateDcbGate(smodule &m);

bool proc::create(const base::String &path, bool fromVfs) {
  /*register HLE prx overrides*/
  runtime::vprx_init();

  /*init memory manager*/
  LOG_ASSERT(vmem.init());

  /*reserve slot for main module*/
  auto first = utl::make_ref<smodule>(this);
  first->getInfo().handle = 0;

  modules.emplace_back(first);

  const bool ps5 = plat == platform::ps5;
  if (ps5 && !std::getenv("DELTA_PS5_MODULES"))
    LOG_WARNING("PS5 title but DELTA_PS5_MODULES is unset; system modules "
                "(libkernel etc.) won't be found");

  /*pre-load required modules
   (the kernel does it, so do we)*/
  if (!loadModule(base::StringRef("libkernel")) ||
      !loadModule(base::StringRef("libSceLibcInternal"))) {
    LOG_ERROR("unable to preload sys modules");
    return false;
  }

  // PS4 libkernel is a thin forwarder: a chunk of its exports (memory-pool
  // helpers) live in libkernel_sys. PS5 has no such split, so only preload it
  // for PS4. Tolerate absence on minimal module sets.
  if (!ps5)
    loadModule(base::StringRef("libkernel_sys"));

  bool loaded = fromVfs ? first->fromVfs(path) : first->fromFile(path);
  if (!loaded) {
    LOG_ERROR("unable to load main process module");
    return false;
  }
  // PS5 modules carry no DT_SCE_MODULEINFO, so name the main module ourselves.
  if (ps5 && first->getInfo().name.empty())
    first->getInfo().name = base::String("eboot");

  // Engine bring-up: give Isaac's surface-name registry valid empty storage so
  // main-init doesn't deref a null bucket array (self-gated by ctor signature).
  if (ps5) {
    bringUpRebirthEbootRegistry(*first);
    investigateDcbGate(*first);
    // DELTA_PS5_GLYPHGUARD: recover the first-frame unbound-font null derefs in
    // the UI/text renderer so the render reaches real draws (diagnostic; the real
    // fix binds the font before rendererFrame).
    if (std::getenv("DELTA_PS5_GLYPHGUARD")) {
      auto *base8 = first->getInfo().base;
      auto eb = reinterpret_cast<uintptr_t>(base8);
      // movzx esi,[rdi+rcx*2+0x2e] (glyph cmap count), rdi==0
      krnl::setNullGuard(eb + 0x5cab56, krnl::GuardReg::rsi, 5);
      // mov rax,[rax+0x28]; mov rax,[rax+0x18] (chained font-object load), rax==0
      krnl::setNullGuard(eb + 0x5c7c53, krnl::GuardReg::rax, 8);
      // ROOT FIX: the renderer-init chain 0x5535d0 bails at its gate checks
      // (`test al,al; je 0x55365d`) when VOInit (gate C, 0x58fb10) returns false
      // -- a GPU render-context vtable step that fails in our env -- SKIPPING the
      // Shape-Renderer install at 0x55361b (0x58ec90). That leaves the global
      // active renderer *(0x9854f0) null, which is the source of the whole
      // first-frame null-object cascade. Force the chain past its three bail
      // branches so the game installs the renderer + builds its RTs/fonts itself.
      // DELTA_PS5_NOFORCE: skip the RenderInit gate force-through. Now that the PS5
      // videoout NIDs are HLE'd (RegisterBuffers returns 0), VOInit (gate C) should
      // return TRUE on its own -- forcing past it leaves an INVALID render context
      // (null pipelines / zero shader PGM). Test whether it succeeds naturally.
      struct { uint32_t off; uint8_t b1; } gates[] = {
          {0x553602, 0x59}, {0x553612, 0x49}, {0x553622, 0x39}};
      bool noForce = std::getenv("DELTA_PS5_NOFORCE") != nullptr;
      for (auto &g : gates) {
        if (noForce) break;
        uint8_t *c = base8 + g.off;
        if (c[0] == 0x74 && c[1] == g.b1) {  // je 0x55365d
          utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                          0x1000, utl::pageProtection::rwx);
          c[0] = 0x90;  // NOP the bail so the chain runs the Shape-Renderer install
          c[1] = 0x90;
        }
      }
      LOG_INFO("ps5 glyphguard: forced renderer-init chain through the Shape-Renderer install");
    }
  }

  return true;
}

// DELTA_PS5_DCBWATCH: diagnose the null frame-0 DrawCommandBuffer gate.
// (1) poll the DCB pointer slot manager[0] @ eboot+0x985a00 (renderer eboot+
//     0x985508 + idx0*0x600 + 0x138 + 0x3c0) + adjacent manager fields, logging
//     every change -> answers "is the DCB ever created before the render uses it?"
// (2) int3 call-order trace over the renderer/DCB-creation entry points so the
//     actual execution order (and which are reached) is visible before the crash.
static void investigateDcbGate(smodule &m) {
  if (!std::getenv("DELTA_PS5_DCBWATCH"))
    return;
  uint8_t *base = m.getInfo().base;
  struct { uint32_t off; const char *label; } pts[] = {
      {0x425ef0, "app_main(0x425ef0)"},   {0x4cc830, "app_render(0x4cc830)"},
      {0x5535d0, "RenderInit(0x5535d0)"}, {0x58fb10, "VOInit(0x58fb10)"},
      {0x58fd50, "DCBframeInit(0x58fd50)"}, {0x5901a0, "rendererFrame(0x5901a0)"},
  };
  for (auto &pt : pts) {
    auto *c = base + pt.off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
    if (c[0] == 0x55) {
      c[0] = 0xCC;
      setOrderTrace(reinterpret_cast<uintptr_t>(c), pt.label);
    } else {
      LOG_WARNING("dcbwatch: {} first byte {:#x} != push rbp", pt.label, c[0]);
    }
  }
  // 0x69e720 is the graphics-subsystem run-once init that VOInit calls before
  // sceVideoOutOpen; when it returns non-zero VOInit bails and the DCB is never
  // created. It accumulates its error in ebx from three sub-calls; trace each
  // `mov ebx,eax` return so we see which import fails.
  struct { uint32_t off; const char *label; } rets[] = {
      {0x69e761, "0x69e720:vSMAm3cxYTY#1"},
      {0x69e78f, "0x69e720:vSMAm3cxYTY#2"},
      {0x69e7aa, "0x69e720:23LRUSvYu1M"},
  };
  for (auto &r : rets) {
    auto *c = base + r.off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
    if (c[0] == 0x89 && c[1] == 0xc3) {
      c[0] = 0xCC;
      setRetTrace(reinterpret_cast<uintptr_t>(c), r.label);
    } else {
      LOG_WARNING("dcbwatch: {} bytes {:#x} {:#x} != mov ebx,eax", r.label, c[0], c[1]);
    }
  }
  // Identify the imports 0x69e720 calls by symbolizing their resolved GOT slots
  // (imports are already bound at this point).
  auto *p = proc::getActive();
  struct { uint32_t got; const char *nid; } gots[] = {
      {0x8e8e38, "vSMAm3cxYTY"}, {0x8e8e40, "23LRUSvYu1M(FAILING)"},
      {0x8e8e48, "2JtWUUiYBXs"}, {0x8e8d80, "1jfXLRVzisc"},
  };
  for (auto &g : gots) {
    uint64_t tgt = *reinterpret_cast<uint64_t *>(base + g.got);
    const char *mod = "??";
    uint64_t off = tgt;
    if (p)
      for (auto &mm : p->getModuleList()) {
        auto &mi = mm->getInfo();
        auto tb = reinterpret_cast<uint64_t>(mi.textSeg.addr);
        if (tb && tgt >= tb && tgt < tb + mi.textSeg.size) {
          mod = mi.name.c_str();
          off = tgt - tb;
          break;
        }
      }
    // Does any loaded module export this NID's hash? (is it resolvable?)
    uint64_t hid = 0;
    const char *expMod = "NONE";
    uint64_t expAddr = 0;
    if (runtime::decode_nid(g.nid, 11, hid) && p)
      for (auto &mm : p->getModuleList())
        if (uintptr_t a = mm->getExport(hid)) {
          expMod = mm->getInfo().name.c_str();
          expAddr = a;
          break;
        }
    std::printf("[dcbimp] %s got=%s+%#llx exportedBy=%s(%#llx)\n", g.nid, mod,
                (unsigned long long)off, expMod, (unsigned long long)expAddr);
  }
  auto *slot = reinterpret_cast<volatile uint64_t *>(base + 0x985a00);
  std::thread([slot] {
    uint64_t last = ~1ull;
    for (int i = 0; i < 400000; i++) {
      uint64_t v = *slot;
      if (v != last) {
        std::printf("[dcbwatch t=%dms] manager[0] (eboot+0x985a00) = %#llx\n",
                    i / 2, (unsigned long long)v);
        last = v;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }).detach();
}

modulePtr proc::getModule(base::StringRef name) {
  for (auto &mod : modules) {
    // module name is base::String, compare via c_str.
    if (name == base::StringRef(mod->getInfo().name))
      return mod;
  }
  return {nullptr};
}

modulePtr proc::getModule(uint32_t handle) {
  for (auto &mod : modules) {
    if (mod->getInfo().handle == handle)
      return mod;
  }
  return {nullptr};
}

/*does not expect an extension*/
// Isaac-specific surface setup. The game's global surface-name registry, a
// fixed-bucket hash map<string,surface> at rebirth+0x687a90, is constructed
// with a NULL bucket-array pointer (its ctor at +0x1e9bcd just zeroes it) and is
// meant to get its storage lazily when the renderer registers the base surfaces
// ("Floor Surface"/"Wall Surface"). That renderer path is gated on the Gnm->
// Vulkan graphics device, which isn't brought up yet, so the bucket array is
// never allocated and every registry find/insert dereferences null + idx*0x20
// (rebirth+0x1e8c8b on a worker insert, +0x1e7e09 on a main-thread find). Until
// the real gfx/renderer init exists, hand the registry valid empty storage up
// front: allocate the N=0x20 * 0x20-byte zeroed bucket array, point the registry
// at it, and NOP the ctor's null-write so it can't clobber the pointer when it
// later runs. The map then works as a valid empty registry independent of order
// or the missing gfx init. (Verified offsets via the decrypted rebirth.elf.)
static void bringUpRebirthSurfaceRegistry(smodule &m) {
  uint8_t *base = m.getInfo().base;
  constexpr uint32_t kRegistryOff = 0x687a90; // bucket-array base pointer
  constexpr uint32_t kCtorZeroOff = 0x1e9bcd; // `mov qword [registry], 0` (11 bytes)
  constexpr size_t kBucketBytes = 0x20 * 0x20; // N buckets * stride

  uint8_t *buckets = allocLowGuest(kBucketBytes);
  if (!buckets) {
    LOG_ERROR("rebirth surface-registry: bucket alloc failed");
    return;
  }
  *reinterpret_cast<uint64_t *>(base + kRegistryOff) =
      reinterpret_cast<uint64_t>(buckets);

  uint8_t *ctor = base + kCtorZeroOff;
  utl::protectMem(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ctor) & ~0xFFFull),
      0x1000, utl::pageProtection::rwx);
  std::memset(ctor, 0x90, 11); // NOP the ctor's null-write

  LOG_INFO("rebirth surface-registry: installed empty buckets@{} -> [+{:#x}]",
           (void *)buckets, kRegistryOff);

  // DIAGNOSTIC (DELTA_GFXCTX_WATCH): poll the rebirth GfxContext singleton's
  // command-buffer pointer (rebirth+0x687b30, field +0x38). The both-LLE render
  // thread faults at rebirth+0x23f027 dereferencing this when it is null; this
  // shows whether/when it gets allocated. Logs every transition.
  if (std::getenv("DELTA_GFXCTX_WATCH")) {
    auto *slot = reinterpret_cast<volatile uint64_t *>(base + 0x687b30 + 0x38);
    std::thread([slot] {
      uint64_t last = ~0ull;
      for (int i = 0; i < 200000; i++) {
        uint64_t v = *slot;
        if (v != last) {
          std::printf("[gfxctx] +0x38 = %#llx  (t=%dms)\n",
                      (unsigned long long)v, i / 2);
          last = v;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
    }).detach();
  }
}

// PS5 bring-up (mirror of bringUpRebirthSurfaceRegistry above). The PS5 eboot
// statically links the same "rebirth" engine, so it has the same global surface-
// name registry, here at eboot+0x985458 (bucket-array base pointer, a fixed 0x20
// buckets * 0x20-byte stride). Its ctor at eboot+0x56a5d0 zeroes the pointer
// (the null-write at eboot+0x56a5e7) and never allocates the array -- that
// storage is meant to come from the renderer registering the base surfaces,
// which needs the AGC/GPU device that isn't up yet. Main-init's first registry
// find (eboot+0x568840) then iterates null+idx*0x20 and faults (eboot+0x56889e).
// Hand it a valid empty bucket array up front and NOP the ctor's null-write, as
// on PS4. Guarded by the exact ctor-instruction bytes so a different PS5 title's
// eboot is left untouched. (Offsets verified against the decrypted eboot.)
static void bringUpRebirthEbootRegistry(smodule &m) {
  uint8_t *base = m.getInfo().base;
  constexpr uint32_t kRegistryOff = 0x985458; // bucket-array base pointer
  constexpr uint32_t kCtorZeroOff = 0x56a5e7; // `mov qword [registry], 0` (11 bytes)
  constexpr size_t kBucketBytes = 0x20 * 0x20; // N buckets * stride

  // `mov qword [rip+0x41ae66], 0` -> eboot+0x985458. Only this build has it.
  static const uint8_t kCtorBytes[] = {0x48, 0xc7, 0x05, 0x66, 0xae, 0x41,
                                       0x00, 0x00, 0x00, 0x00, 0x00};
  if (std::memcmp(base + kCtorZeroOff, kCtorBytes, sizeof(kCtorBytes)) != 0)
    return; // not this title's eboot; nothing to bring up

  uint8_t *buckets = allocLowGuest(kBucketBytes);
  if (!buckets) {
    LOG_ERROR("rebirth eboot-registry: bucket alloc failed");
    return;
  }
  *reinterpret_cast<uint64_t *>(base + kRegistryOff) =
      reinterpret_cast<uint64_t>(buckets);

  uint8_t *ctor = base + kCtorZeroOff;
  utl::protectMem(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ctor) & ~0xFFFull),
      0x1000, utl::pageProtection::rwx);
  std::memset(ctor, 0x90, 11); // NOP the ctor's null-write

  LOG_INFO("rebirth eboot-registry: installed empty buckets@{} -> [+{:#x}]",
           (void *)buckets, kRegistryOff);
}

// DIAGNOSTIC (DELTA_VO_PATCH): patch real libSceVideoOut export entries to
// `mov eax,<v>; ret`, to isolate which real setup function's error return makes
// rebirth skip command-buffer creation. List: open,regbuf,fliprate,addflip,getlabel.
// open returns 1 (a valid handle); the rest return 0 (SCE_OK).
// DIAGNOSTIC (DELTA_VO_WATCH): poll libSceVideoOut's internal display-config state
// during the NATURAL flow (its init runs via pthread_once on first Open). Shows
// how count[0x1cb30]/idx[0x1cb40]/cfg[*].f0[0x1cb50 stride 0x140] evolve, to pin
// exactly what the driver fails to set (the display-connected state f0==4 Open needs).
static void watchVideoOutState(smodule &m) {
  if (!std::getenv("DELTA_VO_WATCH"))
    return;
  uint8_t *base = m.getInfo().base;
  std::thread([base] {
    int32_t lc = 0x7fffffff, li = 0x7fffffff;
    uint32_t lf[3] = {0xdead, 0xdead, 0xdead};
    for (int i = 0; i < 120000; i++) {
      int32_t c = *reinterpret_cast<volatile int32_t *>(base + 0x1cb30);
      int32_t idx = *reinterpret_cast<volatile int32_t *>(base + 0x1cb40);
      uint32_t f[3];
      for (int s = 0; s < 3; s++)
        f[s] = *reinterpret_cast<volatile uint32_t *>(base + 0x1cb50 + s * 0x140);
      // port[0] @ vaddr 0x1d550 (stride 0xb0): field@0x14=open flag; field@0x48 set
      // to 0xfffffff3 once Open reaches the deep success path (just before op@0x580).
      uint32_t portOpen = *reinterpret_cast<volatile uint32_t *>(base + 0x1d564);
      uint32_t port48 = *reinterpret_cast<volatile uint32_t *>(base + 0x1d550 + 0x48);
      static uint32_t lpo = 0xdead, lp48 = 0xdead;
      if (c != lc || idx != li || f[0] != lf[0] || f[1] != lf[1] || f[2] != lf[2] ||
          portOpen != lpo || port48 != lp48) {
        std::printf("[vowatch t=%dms] count=%d idx=%d cfg.f0=[%#x %#x %#x] "
                    "port0.open=%u port0.f48=%#x\n",
                    i / 2, c, idx, f[0], f[1], f[2], portOpen, port48);
        lc = c; li = idx; lf[0] = f[0]; lf[1] = f[1]; lf[2] = f[2]; lpo = portOpen;
        lp48 = port48;
      }
      // DELTA_VO_FORCE_CONNECT: once the driver registered the placeholder into
      // cfg[0] (f0 went -1 -> 0) but Open reads cfg[idx] (still free -1), make a
      // connected display: copy the populated cfg[0] slot into cfg[idx] and mark
      // it connected (f0=4). Proves the display-config mechanism end to end.
      static bool patched = false;
      if (std::getenv("DELTA_VO_FORCE_CONNECT") && !patched && idx >= 1 &&
          idx < 8 && f[0] == 0 && f[1] == 0xffffffff) {
        uint8_t *cfg0 = base + 0x1cb50;
        uint8_t *cfgi = base + 0x1cb50 + (size_t)idx * 0x140;
        std::memcpy(cfgi, cfg0, 0x140);                 // copy ops/vtable
        *reinterpret_cast<uint32_t *>(cfgi) = 4;        // f0 = connected
        patched = true;
        std::printf("[vowatch] FORCE_CONNECT: cfg[%d] <- cfg[0], f0=4\n", idx);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }).detach();
}

// Host hook for the videoout busType/index map-op (vaddr 0x1020). Open calls it
// with esi=userId, edx=busType, ecx=index, r8=param. Via makeHostThunk those land
// in args (rsi,rdx,r10,r8) -> a2,a3,a4,a5. Logs the title's real Open() args and
// returns 0 (the op's value for the main display). If this never logs, Open failed
// before the op (count/f0/param gate).
static uint64_t PS4ABI voOpMapLog(uint64_t a1, uint64_t userId, uint64_t busType,
                                  uint64_t index, uint64_t param, uint64_t a6) {
  uint32_t pv = 0;
  if (param > 0x10000 && param < 0x800000000000ull)
    pv = *reinterpret_cast<uint32_t *>(param);
  std::printf("[voop] sceVideoOutOpen(userId=%#lx busType=%ld index=%ld param=%#lx "
              "[param]=%#x [param]&0xf=%#x) -> map-op returns 0\n",
              (unsigned long)userId, (long)busType, (long)index,
              (unsigned long)param, pv, pv & 0xf);
  return 0;
}

static void patchVideoOutDiag(smodule &m) {
  watchVideoOutState(m);
  if (std::getenv("DELTA_VO_OPLOG")) {
    uintptr_t thunk = cpu::makeHostThunk(reinterpret_cast<void *>(&voOpMapLog));
    uint8_t *o = m.getInfo().base + 0x1020;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(o) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    o[0] = 0x48; o[1] = 0xb8;                       // mov rax, imm64
    *reinterpret_cast<uint64_t *>(o + 2) = thunk;
    o[10] = 0xff; o[11] = 0xe0;                     // jmp rax
    std::printf("[voop] hooked map-op @ +0x1020 -> thunk %#lx\n",
                (unsigned long)thunk);
  }
  // TEST (DELTA_VO_SKIP_580): nop the `js error` after Open's `call op@0x580`
  // (config-validate op). If Open then progresses, op@0x580's return was a gate.
  if (std::getenv("DELTA_VO_SKIP_580")) {
    uint8_t *c = m.getInfo().base + 0xaeb8;  // js 0xef09 after the op call
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    if (c[0] == 0x78) {  // js rel8
      c[0] = 0x90; c[1] = 0x90;
      std::printf("[votest] nop'd op@0x580 error-js @ +0xaeb8\n");
    } else {
      std::printf("[votest] op@0x580 js bytes mismatch: %#x %#x\n", c[0], c[1]);
    }
  }
  const char *list = std::getenv("DELTA_VO_PATCH");
  if (!list)
    return;
  uint8_t *base = m.getInfo().base;
  struct { const char *name; uint32_t off; uint8_t ret; } fns[] = {
      {"open", 0xaad0, 1}, {"regbuf", 0xb620, 0}, {"fliprate", 0xbde0, 0},
      {"addflip", 0xc6c0, 0}, {"getlabel", 0xbb80, 0}};
  for (auto &fn : fns) {
    if (!std::strstr(list, fn.name))
      continue;
    uint8_t *c = base + fn.off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    c[0] = 0xb8; c[1] = fn.ret; c[2] = 0; c[3] = 0; c[4] = 0;  // mov eax, imm32
    c[5] = 0xc3;                                               // ret
    std::printf("[vopatch] libSceVideoOut!%s -> return %d\n", fn.name, fn.ret);
  }
}

modulePtr proc::loadModule(base::StringRef name) {
  auto mod = getModule(name);
  if (mod)
    return mod;

  auto lib = utl::make_ref<smodule>(this);
  lib->getInfo().handle = handleCounter;
  handleCounter++;

  modules.emplace_back(lib);

  base::String sname;
  sname.append(name.data(), name.length());

  // PS5 titles use a coherent Prospero module set: system modules from a
  // firmware dump (DELTA_PS5_MODULES, a ':'-separated list of dirs holding
  // <name>.sprx), the title's own SDK modules from its decrypted/ tree. A PS5
  // process never touches the PS4 modules/ dir - the ABIs are incompatible
  // (FreeBSD 11 vs 9 syscall numbers, different struct layouts).
  if (plat == platform::ps5) {
    lib->getInfo().name = sname; // PS5 modules have no DT_SCE_MODULEINFO
    bool ok = false;
    if (const char *env = std::getenv("DELTA_PS5_MODULES")) {
      for (const char *p = env; *p && !ok;) {
        const char *sep = std::strchr(p, ':');
        size_t len = sep ? static_cast<size_t>(sep - p) : std::strlen(p);
        if (len) {
          base::String hp;
          hp.append(p, len);
          if (hp.back() != '/')
            hp += "/";
          hp += sname.c_str();
          hp += ".sprx";
          if (utl::File(hp, utl::fileMode::read).IsOpen() && lib->fromFile(hp))
            ok = true;
        }
        p = sep ? sep + 1 : p + len;
      }
    }
    if (!ok) {
      const char *roots[] = {"/app0/decrypted/sce_module/", "/app0/decrypted/",
                             "/app0/sce_module/"};
      for (const char *root : roots) {
        base::String vp(root);
        vp += sname.c_str();
        vp += ".prx";
        if (vfs::openRead(vp.c_str()).Exists() && lib->fromVfs(vp)) {
          ok = true;
          break;
        }
      }
    }
    if (!ok) {
      // Drop the placeholder we emplaced above: a failed module left in the list
      // gets enumerated and libkernel calls its null module_start (crash). Its
      // imports fall through to the badcall stub instead, which is non-fatal.
      LOG_ERROR("unable to load ps5 module {}", sname.c_str());
      modules.pop_back();
      return nullptr;
    }
    return lib;
  }

  const bool isPackagedSdkModule =
      name == base::StringRef("libc") || name == base::StringRef("libSceFios2");
  auto loadPackagedModule = [&] {
    // Retail applications provide these SDK compatibility modules in
    // /app0/sce_module rather than using the firmware copies. The firmware-
    // derived classification used here is documented by shadPS4:
    // https://github.com/shadps4-emu/shadPS4/blob/2c9caf6bfbe7e1dc7a1b4565af8d84c56469dd56/src/core/libraries/sysmodule/sysmodule_table.h#L363-L372
    const char *roots[] = {"/app0/sce_module/", "/app0/"};
    for (const char *root : roots) {
      base::String vfsPath(root);
      vfsPath.append(name.data(), name.length());
      vfsPath += ".prx";
      if (!vfs::openRead(vfsPath.c_str()).Exists())
        continue;
      if (!lib->fromVfs(vfsPath))
        return false;
      if (name == base::StringRef("rebirth"))
        bringUpRebirthSurfaceRegistry(*lib);
      return true;
    }
    return false;
  };

  if (isPackagedSdkModule && loadPackagedModule())
    return lib;

  // HLE/system modules ship with the emulator; prefer those for modules that
  // are not supplied by the application.
  base::String hostRel("modules/");
  hostRel.append(name.data(), name.length());
  hostRel += ".sprx";
  base::String hostPath = utl::make_abs_path(hostRel);
  const bool isRebirth = name == base::StringRef("rebirth");
  const bool isVideoOut = name == base::StringRef("libSceVideoOut");
  if (utl::File(hostPath, utl::fileMode::read).IsOpen()) {
    if (lib->fromFile(hostPath)) {
      if (isRebirth)
        bringUpRebirthSurfaceRegistry(*lib);
      if (isVideoOut)
        patchVideoOutDiag(*lib);
      return lib;
    }
  } else {
    // The game's own modules live inside the pkg: SDK prx under
    // /app0/sce_module, the title's own prx at the app root.
    if (loadPackagedModule())
      return lib;
  }

  LOG_ERROR("unable to load module {}", sname.c_str());
  return nullptr;
}

// Patch a guest function to `xor eax,eax; ret`. Steps over libkernel-internal
// validation that rejects our externally-loaded module set (11.00 offsets).
static void forceReturn0(proc &p, const char *mod, uint32_t off) {
  auto m = p.getModule(base::StringRef(mod));
  if (!m)
    return;
  uint8_t *c = m->getInfo().base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  c[0] = 0x31;  // xor eax, eax
  c[1] = 0xC0;
  c[2] = 0xC3;  // ret
}

// Patch a (rdi=paramId, rsi=int* out) getter to `*out = val; return 0`.
static void forceGetterOk(proc &p, const char *mod, uint32_t off, uint32_t val) {
  auto m = p.getModule(base::StringRef(mod));
  if (!m)
    return;
  uint8_t *c = m->getInfo().base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  c[0] = 0xC7; c[1] = 0x06;  // mov dword [rsi], imm32
  std::memcpy(c + 2, &val, 4);
  c[6] = 0x31; c[7] = 0xC0;  // xor eax, eax
  c[8] = 0xC3;               // ret
}

#if defined(DELTA_BACKEND_NATIVE)
// PS5 libc-heap / pthread-mutex bootstrap fix. Once we hand libc a sceLibcParam
// (so its C++ operator-new arena can grow past the tiny 16 MiB default), libc's
// malloc turns thread-safe and locks a static-initialised mutex whose kernel
// state libkernel lazily allocates through the libc-malloc callback held at a
// libkernel data slot (fw 01.14.00: +0x5cfd8, loaded into rdx before every
// call to the lazy-init helper 0x34a10; was +0x68ec0 on fw 12.60). That malloc
// re-locks the still-uninitialised mutex -> unbounded recursion (stack overflow)
// or deadlock. Interpose the allocator with
// a per-thread re-entrancy guard: the outer call delegates to the real allocator
// (so ordinary pthread objects are heap-backed and freed normally), while a
// re-entrant call -- the bootstrap -- is served from a small malloc-free bump
// pool so the mutex can finish initialising. Native x86 only (the thunk is a host
// function the guest calls directly); PS5 only.
using PthreadAllocFn = uint64_t(PS4ABI *)(uint64_t op, uint64_t arg);
static PthreadAllocFn g_origPthreadAlloc = nullptr;
static thread_local int g_pthreadAllocDepth = 0;

static uint64_t PS4ABI ps5PthreadAlloc(uint64_t op, uint64_t arg) {
  if (op == 1 && g_pthreadAllocDepth > 0) {
    static std::atomic<size_t> off{0};
    static uint8_t pool[64 * 1024];
    size_t sz = (arg + 0xF) & ~size_t(0xF);
    size_t o = off.fetch_add(sz, std::memory_order_relaxed);
    return o + sz <= sizeof(pool) ? reinterpret_cast<uint64_t>(pool + o) : 0;
  }
  if (!g_origPthreadAlloc)
    return 0;
  ++g_pthreadAllocDepth;
  uint64_t r = g_origPthreadAlloc(op, arg);
  --g_pthreadAllocDepth;
  return r;
}
#endif

// libkernel populates its pthread-object allocator pointer at runtime, after
// boot patches run, so install our interpose lazily the first time it is
// non-null (called from thread creation, which happens after libc init but before
// the multithreaded malloc-mutex bootstrap). Idempotent; PS5 + native only.
constexpr uintptr_t kPthreadAllocSlot = 0x5cfd8;  // fw 01.14.00 libkernel data
void ps5MaybeInterposePthreadAlloc() {
#if defined(DELTA_BACKEND_NATIVE)
  static std::atomic<bool> done{false};
  auto *p = proc::getActive();
  if (!p || p->getPlatform() != proc::platform::ps5 || done.load())
    return;
  auto k = p->getModule(base::StringRef("libkernel"));
  if (!k)
    return;
  auto *slot = reinterpret_cast<uint64_t *>(k->getInfo().base + kPthreadAllocSlot);
  uint64_t cur = *slot;
  if (!cur || cur == reinterpret_cast<uint64_t>(&ps5PthreadAlloc))
    return;  // not populated yet, or already ours
  if (done.exchange(true))
    return;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(slot) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  g_origPthreadAlloc = reinterpret_cast<PthreadAllocFn>(cur);
  *slot = reinterpret_cast<uint64_t>(&ps5PthreadAlloc);
  LOG_INFO("interposed libkernel pthread-state alloc (+{:#x}) orig={:#x}",
           kPthreadAllocSlot, cur);
#endif
}

// Boot patches applied before entering the guest, needed by every boot path
// (modexec and the real pkg boot), not just the modexec harness.
static void applyBootPatches(proc &p) {
  // Redirect libkernel's __tls_get_addr (NID vNe1w4diLCs) to our per-thread HLE;
  // libkernel's own dynamic-TLS allocator leaves DTV entries null. NID lookup is
  // firmware-independent.
  //
  // NATIVE ONLY: this patches the guest to `jmp` a *host* function pointer,
  // which only works when the host runs x86 directly. Under the FEXCore JIT
  // (aarch64) that address is ARM code and jumping to it as x86 faults wildly.
  // TODO(boot/fex): redirect __tls_get_addr via a FEXCore thunk trampoline, or
  // satisfy guest dynamic TLS another way.
#if defined(DELTA_BACKEND_NATIVE)
  if (auto k = p.getModule(base::StringRef("libkernel"))) {
    if (uintptr_t a = k->getSymbolByNid("vNe1w4diLCs")) {
      auto *c = reinterpret_cast<uint8_t *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0x48;  // movabs rax, imm64
      c[1] = 0xB8;
      *reinterpret_cast<uint64_t *>(c + 2) =
          reinterpret_cast<uint64_t>(&guest_tls_get_addr);
      c[10] = 0xFF;  // jmp rax
      c[11] = 0xE0;
      LOG_INFO("patched libkernel __tls_get_addr -> host HLE");
    }
  }
#else  // DELTA_BACKEND_FEX
  // FEX path: a host jump is invalid inside the x86 JIT, so patch the export to
  // a tiny `mov eax, <magic>; syscall; ret` stub that the FEX syscall handler
  // bridges to krnl::guest_tls_get_addr (tls_index ptr arrives in rdi).
  if (auto k = p.getModule(base::StringRef("libkernel"))) {
    if (uintptr_t a = k->getSymbolByNid("vNe1w4diLCs")) {
      auto *c = reinterpret_cast<uint8_t *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0xB8; // mov eax, imm32
      *reinterpret_cast<uint32_t *>(c + 1) = cpu::kTlsGetAddrSyscall;
      c[5] = 0x0F; // syscall
      c[6] = 0x05;
      c[7] = 0xC3; // ret
      LOG_INFO("patched libkernel __tls_get_addr -> magic syscall (FEX)");
    }
  }
#endif
  // DEBUG (DELTA_TRAP_VADDR=0xADDR[,0xADDR...]): plant an int3 at guest code
  // address(es) so reaching them traps into the crash handler, which dumps the
  // guest RIP/registers/backtrace + stack scan. Lets us capture the context of a
  // deterministic-but-hard-to-breakpoint site (e.g. a fatal-error spin) without
  // gdb, which is far too slow under the boot's threading.
  if (const char *t = std::getenv("DELTA_TRAP_VADDR")) {
    base::String spec(t);
    char *cur = const_cast<char *>(spec.c_str());
    while (cur && *cur) {
      uint64_t addr = std::strtoull(cur, &cur, 0);
      if (addr) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        c[0] = 0xCC; // int3 -> SIGTRAP -> crash handler dump
        LOG_INFO("DELTA_TRAP_VADDR: planted int3 @ {:#x}", addr);
      }
      while (*cur == ',' || *cur == ' ') cur++;
    }
  }
  // DEBUG (DELTA_ALLOC_TRACE=0xADDR[,minMB]): trace large allocations through a
  // guest allocator whose entry (ADDR) begins with `push rbp`. Replace it with
  // int3; the fatal handler logs the size (rsi) and emulates the push. Lets us
  // see what fills a fixed heap (e.g. SOTTR's 1 GiB pool) without gdb.
  if (const char *at = std::getenv("DELTA_ALLOC_TRACE")) {
    char *end = nullptr;
    uint64_t addr = std::strtoull(at, &end, 0);
    uint64_t minB = 0x1000000;
    if (end && *end == ',') minB = std::strtoull(end + 1, nullptr, 0) * 1024 * 1024;
    // DELTA_ALLOC_TRACE doubles as a boolean toggle for the [lowalloc] tracer in
    // sys_mem.cpp, so a bare "=1" is legitimate and must NOT be treated as a code
    // address here: addr=1 would protect/deref page 0 (host null-deref SIGSEGV).
    // A real allocator entry is a guest .text vaddr (>= 64 KiB); ignore anything
    // smaller so tracing can be enabled without planting an int3 at a bogus addr.
    if (addr >= 0x10000) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) {  // push rbp
        c[0] = 0xCC;       // int3
        setAllocTrace(addr, minB);
        LOG_INFO("DELTA_ALLOC_TRACE: hooked alloc entry {:#x} (>= {} MB)", addr,
                 minB / (1024 * 1024));
      } else {
        LOG_WARNING("DELTA_ALLOC_TRACE: {:#x} first byte {:#x} != push rbp", addr,
                    c[0]);
      }
    }
  }
  // DELTA_HEAP_PROF=0xADDR: plant int3 at an operator-new/malloc entry (push rbp,
  // size in rdi) and aggregate bytes+count by guest caller; SIGUSR1 dumps top sites.
  if (const char *hp = std::getenv("DELTA_HEAP_PROF")) {
    char *cur = const_cast<char *>(hp);
    while (cur && *cur) {
      uint64_t addr = std::strtoull(cur, &cur, 0);
      if (addr >= 0x10000) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setHeapProf(addr);
          LOG_INFO("DELTA_HEAP_PROF: hooked alloc entry {:#x}", addr);
        } else {
          LOG_WARNING("DELTA_HEAP_PROF: {:#x} first byte {:#x} != push rbp", addr, c[0]);
        }
      }
      while (*cur == ',' || *cur == ' ') cur++;
    }
  }
  if (const char *ct = std::getenv("DELTA_CNT_TRACE")) {
    uint64_t addr = std::strtoull(ct, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setCntTrace(addr); }
    }
  }
  if (const char *ft = std::getenv("DELTA_FATAL_TRACE")) {
    uint64_t addr = std::strtoull(ft, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setFatalTrace(addr); }
    }
  }
  if (const char *ht = std::getenv("DELTA_HDR_TRACE")) {
    // Comma-separated list of consumer entry vaddrs to hook (e.g. 0x606150,0x6063a0).
    const char *s = ht;
    while (*s) {
      char *end = nullptr;
      uint64_t addr = std::strtoull(s, &end, 0);
      if (addr) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setHdrTrace(addr); }
      }
      s = (end && *end == ',') ? end + 1 : (end ? end : s + 1);
      if (!*s || (end && *end != ',')) break;
    }
  }
  if (const char *ro = std::getenv("DELTA_RDOFF_FIX")) {
    uint64_t addr = std::strtoull(ro, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setRdoffFix(addr); }
    }
  }
  if (const char *sf = std::getenv("DELTA_SKIP_FN")) {
    const char *s2 = sf;
    while (*s2) {
      char *end = nullptr;
      uint64_t addr = std::strtoull(s2, &end, 0);
      if (addr) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setSkipFn(addr); }
      }
      s2 = (end && *end == ',') ? end + 1 : (end ? end : s2 + 1);
      if (!*s2 || (end && *end != ',')) break;
    }
  }
  ps5::maybePrependCtor(p);
  forceReturn0(p, "libkernel", 0x287e0);            // module-gen lib-id validator
  forceReturn0(p, "libSceAppContentUtil", 0x1a00);  // AppContent IPMI init
  // AppContent's IPMI client is stubbed, so the real Initialize/AppParamGetInt
  // error out and the engine's GameSystemInit (big.cpp:1298/1302) asserts. Force
  // both to SCE_OK: Initialize (bootParam out is pre-zeroed = normal boot) and
  // AppParamGetInt returning SKU_FLAG = FULL (3).
  forceReturn0(p, "libSceAppContentUtil", 0x1610);     // sceAppContentInitialize
  forceGetterOk(p, "libSceAppContentUtil", 0x1630, 3); // sceAppContentAppParamGetInt
}

void proc::start() {
  LOG_ASSERT(modules[1]->getInfo().name == "libkernel");

  installCrashHandler();
  applyBootPatches(*this);

  auto &info = modules[0]->getInfo();
  auto &kinfo = modules[1]->getInfo();

  if (!info.entry) {
    LOG_WARNING("entry missing for {}", info.name.c_str());
    return;
  }

  union stack_entry {
    const void *ptr;
    uint64_t val;
  } stack[128];

  stack[0].val = 1 + 0; // argc
  auto s = reinterpret_cast<stack_entry *>(&stack[1]);
  (*s++).ptr = info.name.c_str();
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;
  (*s++).val = 9ull;
  (*s++).ptr = (const void *)(info.entry);
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;

  // Enter libkernel's entry with the PS4 convention (arg block in rdi). The
  // backend runs it natively (x86 host) or via the FEXCore JIT (aarch64 host).
  // PS5 starts with the TCB its kernel would have installed; libkernel reads
  // fs:0x10 before it gets around to setting up its own (see makeInitialTcb).
  const uint64_t fsbase = plat == platform::ps5 ? ps5::makeInitialTcb() : 0;
  cpu::backend().enterGuest(reinterpret_cast<uintptr_t>(kinfo.entry), stack,
                            fsbase);
}
}  // namespace krnl
