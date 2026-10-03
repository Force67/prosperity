
#include <cstring>

#include "base/algorithm.h"
#include "base/arch.h"
#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "kern/object_table.h"
#include "logger/logger.h"

namespace kern {
ObjectTable::ObjectTable() {}

ObjectTable::~ObjectTable() {
  Reset();
}

void ObjectTable::Reset() {
  base::LockGuard lock(omutex_);

  // Release all objects.
  for (u32 n = 0; n < table_cap_; n++) {
    auto& entry = table_[n];
    if (entry.obj)
      entry.obj->Release();
  }

  table_cap_ = 0;
  last_free_entry_ = 0;

  if (table_) {
    free(table_);
    table_ = nullptr;
  }
}

void ObjectTable::Purge() {
  base::LockGuard lock(omutex_);

  for (u32 slot = 0; slot < table_cap_; slot++) {
    auto& entry = table_[slot];
    if (entry.obj) {
      entry.ref_count = 0;
      entry.obj->Release();
      entry.obj = nullptr;
    }
  }
}

bool ObjectTable::Resize(u32 new_cap) {
  u32 new_size = new_cap * sizeof(Entry);
  u32 old_size = table_cap_ * sizeof(Entry);

  auto new_table = reinterpret_cast<Entry*>(realloc(table_, new_size));
  if (!new_table)
    return false;

  // Zero out new entries.
  if (new_size > old_size)
    std::memset(reinterpret_cast<u8*>(new_table) + old_size, 0,
                new_size - old_size);

  last_free_entry_ = table_cap_;
  table_cap_ = new_cap;
  table_ = new_table;

  return true;
}

bool ObjectTable::FindSlot(u32& out) {
  u32 slot = last_free_entry_;
  u32 scan_count = 0;
  while (scan_count < table_cap_) {
    auto& entry = table_[slot];
    if (!entry.obj) {
      out = slot;
      return true;
    }
    scan_count++;
    slot = (slot + 1) % table_cap_;
    if (slot == 0) {
      // Never allow 0 handles.
      scan_count++;
      slot++;
    }
  }

  // Table out of slots, expand.
  u32 new_table_capacity = base::Max(16 * 1024u, table_cap_ * 2);
  if (!Resize(new_table_capacity)) {
    LOG_ERROR("unable to resize handle table");
    return false;
  }

  // Never allow 0 handles.
  slot = ++last_free_entry_;
  out = slot;

  return true;
}

ObjectTable::Entry* ObjectTable::FindEntry(u32 handle) {
  u32 slot = handle >> 2;

  if (slot < table_cap_)
    return &table_[slot];

  return nullptr;
}

bool ObjectTable::Keep(u32 handle) {
  base::LockGuard lock(omutex_);

  auto* e = FindEntry(handle);
  if (e) {
    e->ref_count++;
    return true;
  }

  return false;
}

bool ObjectTable::Add(Object* obj, u32& handle_out) {
  base::LockGuard lock(omutex_);

  u32 slot = 0, handle = 0;

  bool result = FindSlot(slot);
  if (result) {
    // stash
    auto& entry = table_[slot];
    entry.obj = obj;
    entry.ref_count = 1;

    handle = slot << 2;
    obj->handles().push_back(handle);

    // Retain so long as the object is in the table.
    obj->Retain();

    handle_out = handle;
    return true;
  }

  handle_out = -1;
  return false;
}

bool ObjectTable::Remove(u32 handle) {
  base::LockGuard lock(omutex_);

  auto* e = FindEntry(handle);
  if (e && e->obj) {
    auto* object = e->obj;
    e->obj = nullptr;
    e->ref_count = 0;

    auto& handles = object->handles();

    auto it = base::Find(handles.begin(), handles.end(), handle);
    if (it != handles.end())
      handles.erase(it);

    object->Release();
    return true;
  }

  return false;
}

bool ObjectTable::Release(u32 handle) {
  base::LockGuard lock(omutex_);

  auto* e = FindEntry(handle);
  if (!e) {
    return false;
  }

  if (--e->ref_count == 0)
    return Remove(handle);

  return true;
}

ObjectRef<Object> ObjectTable::Get(u32 handle) {
  base::LockGuard lock(omutex_);

  // Lower 2 bits are ignored.
  u32 slot = handle >> 2;
  Object* obj = nullptr;

  // Verify slot.
  if (slot < table_cap_) {
    auto& entry = table_[slot];
    if (entry.obj)
      obj = entry.obj;
  }

  // Retain the object pointer.
  if (obj) {
    obj->Retain();
    return ObjectRef<Object>(obj);
  }

  return nullptr;
}
}  // namespace kern
