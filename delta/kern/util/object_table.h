#pragma once

// based off:
// https://github.com/xenia-project/xenia/blob/master/src/xenia/kernel/util/object_table.h

#include <logger/logger.h>
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
  bool add(kObject *, uint32_t &);
  bool remove(uint32_t);
  bool release(uint32_t);
  bool keep(uint32_t);
  kObject *get(uint32_t);

private:
  bool resize(uint32_t newCap);
  bool findSlot(uint32_t &out);

  std::mutex omutex;

  struct entry {
    int refCount = 0;
    kObject *obj = nullptr;
  };

  entry *findEntry(uint32_t);

  uint32_t tableCap = 0;
  uint32_t lastFreeEntry = 0;
  entry *table = nullptr;
};
}