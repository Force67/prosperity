#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "kern/process.h"

#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"

class Launcher {
 public:
  using ArgvList = base::Vector<base::String>;

  Launcher();
  ~Launcher();

  bool Init();
  void Boot(const base::String& game_path);
  void Stop();

  ArgvList argv;

 private:
  base::UniquePointer<kern::Process> proc_;
};
