#pragma once

// based off:
// https://github.com/xenia-project/xenia/blob/master/src/xenia/kernel/util/object_table.h

#include <logger/logger.h>
#include "base/arch.h"
#include <memory>
#include <mutex>
#include <utl/object_ref.h>

#include "kern/object.h"

namespace krnl {
class kObject;

class objectTable {
public:
  objectTable();
  ~objectTable();

  void reset();
  void purge();
  bool add(kObject *, u32 &);
  bool remove(u32);
  bool release(u32);
  bool keep(u32);
  kObject *get(u32);

private:
  bool resize(u32 newCap);
  bool findSlot(u32 &out);

  // recursive: release() holds the lock and calls remove(), which re-locks.
  std::recursive_mutex omutex;

  struct entry {
    int refCount = 0;
    kObject *obj = nullptr;
  };

  entry *findEntry(u32);

  u32 tableCap = 0;
  u32 lastFreeEntry = 0;
  entry *table = nullptr;
};
}