#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <kern/proc.h>

#include <base/containers/vector.h>
#include <base/strings/xstring.h>
#include <base/memory/unique_pointer.h>

class deltaCore {
public:
  using argvList = base::Vector<base::String>;

  deltaCore();
  ~deltaCore();

  bool init();
  void boot(const base::String& fromdir);

  argvList argv;

private:
  base::UniquePointer<krnl::proc> proc;
};

extern "C" int dcoreMain(int argc, char** argv);
