/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <utl/file.h>
#include <utl/mem.h>
#include <utl/path.h>
#include <xbyak.h>

#include "module.h"
#include "proc.h"
#include "runtime/vprx/vprx.h"

namespace krnl {
static proc *g_activeProc{nullptr};

proc::proc() : vmem(env) { g_activeProc = this; }

proc *proc::getActive() { return g_activeProc; }

bool proc::create(const base::String &path) {
  /*register HLE prx overrides*/
  runtime::vprx_init();

  /*init memory manager*/
  LOG_ASSERT(vmem.init());

  /*reserve slot for main module*/
  auto first = utl::make_ref<smodule>(this);
  first->getInfo().handle = 0;

  modules.emplace_back(first);

  /*pre-load required modules
   (the kernel does it, so do we)*/
  if (!loadModule(base::StringRef("libkernel")) ||
      !loadModule(base::StringRef("libSceLibcInternal"))) {
    LOG_ERROR("unable to preload sys modules");
    return false;
  }

  if (!first->fromFile(path)) {
    LOG_ERROR("unable to load main process module");
    return false;
  }

  return true;
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
modulePtr proc::loadModule(base::StringRef name) {
  auto mod = getModule(name);
  if (!mod) {
    auto lib = utl::make_ref<smodule>(this);
    lib->getInfo().handle = handleCounter;
    handleCounter++;

    modules.emplace_back(lib);

    base::String nameFull("modules/");
    nameFull.append(name.data(), name.length());
    nameFull += ".sprx";
    if (!lib->fromFile(utl::make_abs_path(nameFull))) {
      LOG_ERROR("unable to load module {}", nameFull.c_str());
      return nullptr;
    }

    return lib;
  }
  return mod;
}

void proc::start() {
  LOG_ASSERT(modules[1]->getInfo().name == "libkernel");

  auto &info = modules[0]->getInfo();
  auto &kinfo = modules[1]->getInfo();

  if (!info.entry) {
    LOG_WARNING("entry missing for {}", info.name.c_str());
    return;
  }

  auto func = (void *(PS4ABI *)(void *))kinfo.entry;

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
  func(stack);
}
}  // namespace krnl
