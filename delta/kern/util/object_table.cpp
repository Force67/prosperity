
#include <base.h>
#include "object_table.h"
#include <algorithm>
#include <logger/logger.h>

namespace krnl {
objectTable::objectTable() {}

objectTable::~objectTable() { reset(); }

void objectTable::reset() {
  std::lock_guard lock(omutex);

  // Release all objects.
  for (uint32_t n = 0; n < tableCap; n++) {
    auto &entry = table[n];
    if (entry.obj)
      entry.obj->release();
  }

  tableCap = 0;
  lastFreeEntry = 0;

  if (table) {
    free(table);
    table = nullptr;
  }
}

void objectTable::purge() {
  std::lock_guard lock(omutex);

  for (uint32_t slot = 0; slot < tableCap; slot++) {
    auto &entry = table[slot];
    if (entry.obj) {
      entry.refCount = 0;
      entry.obj->release();
      entry.obj = nullptr;
    }
  }
}

bool objectTable::resize(uint32_t newCap) {
  uint32_t new_size = newCap * sizeof(entry);
  uint32_t old_size = tableCap * sizeof(entry);

  auto new_table = reinterpret_cast<entry *>(realloc(table, new_size));
  if (!new_table)
    return false;

  // Zero out new entries.
  if (new_size > old_size)
    std::memset(reinterpret_cast<uint8_t *>(new_table) + old_size, 0,
                new_size - old_size);

  lastFreeEntry = tableCap;
  tableCap = newCap;
  table = new_table;

  return true;
}

bool objectTable::findSlot(uint32_t &out) {
  uint32_t slot = lastFreeEntry;
  uint32_t scan_count = 0;
  while (scan_count < tableCap) {
    auto &entry = table[slot];
    if (!entry.obj) {
      out = slot;
      return true;
    }
    scan_count++;
    slot = (slot + 1) % tableCap;
    if (slot == 0) {
      // Never allow 0 handles.
      scan_count++;
      slot++;
    }
  }

  // Table out of slots, expand.
  uint32_t new_table_capacity = std::max(16 * 1024u, tableCap * 2);
  if (!resize(new_table_capacity)) {
    LOG_ERROR("unable to resize handle table");
    return false;
  }

  // Never allow 0 handles.
  slot = ++lastFreeEntry;
  out = slot;

  return true;
}

objectTable::entry *objectTable::findEntry(uint32_t handle) {
  uint32_t slot = handle >> 2;

  if (slot < tableCap)
    return &table[slot];

  return nullptr;
}

bool objectTable::keep(uint32_t handle) {
  std::lock_guard lock(omutex);

  auto *e = findEntry(handle);
  if (e) {
    e->refCount++;
    return true;
  }

  return false;
}

bool objectTable::add(kObject *obj, uint32_t &handleOut) {
  std::lock_guard lock(omutex);

  uint32_t slot = 0, handle = 0;

  bool result = findSlot(slot);
  if (result) {

    // stash
    auto &entry = table[slot];
    entry.obj = obj;
    entry.refCount = 1;

    handle = slot << 2;
    obj->handles().push_back(handle);

    // Retain so long as the object is in the table.
    obj->retain();

    handleOut = handle;
    return true;
  }

  handleOut = -1;
  return false;
}

bool objectTable::remove(uint32_t handle) {
  
    //FOR NOW DANGEROUS
  //std::lock_guard lock(omutex);

  auto *e = findEntry(handle);
  if (e && e->obj) {
    auto *object = e->obj;
    e->obj = nullptr;
    e->refCount = 0;

    auto &handles = object->handles();

    auto it = std::find(handles.begin(), handles.end(), handle);
    if (it != handles.end())
      handles.erase(it);

    object->release();
    return true;
  }

  return false;
}

bool objectTable::release(uint32_t handle) {
  std::lock_guard lock(omutex);

  auto *e = findEntry(handle);
  if (!e) {
    return false;
  }

  if (--e->refCount == 0)
    return remove(handle);

  return true;
}

kObject *objectTable::get(uint32_t handle) {
  std::lock_guard lock(omutex);

  // Lower 2 bits are ignored.
  uint32_t slot = handle >> 2;
  kObject *obj = nullptr;

  // Verify slot.
  if (slot < tableCap) {
    auto &entry = table[slot];
    if (entry.obj)
      obj = entry.obj;
  }

  // Retain the object pointer.
  if (obj) {
    obj->retain();
    return obj;
  }

  return nullptr;
}
} // namespace krnl
