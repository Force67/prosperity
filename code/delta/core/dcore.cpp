/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <thread>

#include "dcore.h"
#include <logger/logger.h>
#include <utl/file.h>

#include "formats/pup_object.h"

deltaCore::deltaCore() = default;
deltaCore::~deltaCore() = default;

bool deltaCore::init() {
  LOG_INFO("Initializing deltaCore " rsc_copyright);
  return true;
}

void deltaCore::boot(const base::String& xdir) {
  base::String dir = xdir;

#ifdef _WIN32
  for (auto& c : dir) if (c == '/') c = '\\';
#endif

  std::thread ctx([dir = std::move(dir)]() {
    auto p = base::MakeUnique<krnl::proc>();
    if (!p->create(dir))
      return;

    p->start();
  });

  ctx.detach();
}
