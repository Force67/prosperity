#pragma once

// based off:
// https://github.com/xenia-project/xenia/blob/master/src/xenia/kernel/util/object_table.h

#include "base/arch.h"
#include "kern/object_ref.h"
#include "logger/logger.h"

#include "base/threading/recursive_mutex.h"
#include "kern/object.h"

namespace kern {
class Object;

class ObjectTable {
 public:
  ObjectTable();
  ~ObjectTable();

  void Reset();
  void Purge();
  bool Add(Object*, u32&);
  bool Remove(u32);
  bool Release(u32);
  bool Keep(u32);
  Object* Get(u32);

 private:
  bool Resize(u32 new_cap);
  bool FindSlot(u32& out);

  // recursive: release() holds the lock and calls remove(), which re-locks.
  base::RecursiveMutex omutex_;

  struct Entry {
    int ref_count = 0;
    Object* obj = nullptr;
  };

  Entry* FindEntry(u32);

  u32 table_cap_ = 0;
  u32 last_free_entry_ = 0;
  Entry* table_ = nullptr;
};
}  // namespace kern