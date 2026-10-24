#pragma once

// based off:
// https://github.com/xenia-project/xenia/blob/master/src/xenia/kernel/util/object_table.h

#include "base/arch.h"
#include "kern/object_ref.h"
#include "logger/logger.h"

#include "base/containers/hash_map.h"
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
  ObjectRef<Object> Get(u32);

  // Another handle for `handle`'s object: the lowest free one >= `min` (dup,
  // F_DUPFD). Both share the object, so they share its file offset.
  bool Dup(u32 handle, u32 min, u32& out);
  // Makes `target` name `handle`'s object, closing whatever it named (dup2).
  // A target that is not a slot handle (0..3, or not a multiple of 4: the
  // stdio numbers) becomes an alias.
  bool Dup2(u32 handle, u32 target);
  // True when `handle` is an alias dup2 made.
  bool IsAlias(u32 handle);

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
  static bool IsSlotHandle(u32 h) { return h && !(h & 3); }
  bool PlaceAt(Object* obj, u32 slot);

  base::HashMap<u32, Entry> aliases_;

  u32 table_cap_ = 0;
  u32 last_free_entry_ = 0;
  Entry* table_ = nullptr;
};
}  // namespace kern