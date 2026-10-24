#include <gtest/gtest.h>

#include "kern/object_table.h"

using kern::ObjectTable;

// A freshly constructed table owns nothing, so every lookup or mutation against
// an unknown handle must fail instead of crashing or returning garbage.
TEST(ObjectTable, EmptyTableLookupsFail) {
  ObjectTable table;
  EXPECT_EQ(table.Get(0x4), nullptr);
  EXPECT_EQ(table.Get(0x1234), nullptr);
  EXPECT_FALSE(table.Remove(0x4));
  EXPECT_FALSE(table.Release(0x4));
  EXPECT_FALSE(table.Keep(0x4));
}

// reset()/purge() on an empty table are no-ops and leave it usable.
TEST(ObjectTable, ResetAndPurgeAreSafeWhenEmpty) {
  ObjectTable table;
  table.Reset();
  table.Purge();
  EXPECT_EQ(table.Get(0x4), nullptr);
}

namespace {
class CountedObject : public kern::Object {
 public:
  CountedObject(ObjectTable& table, int& destroyed)
      : Object(table, Type::kDevice), destroyed_(destroyed) {}
  ~CountedObject() override { ++destroyed_; }

 private:
  int& destroyed_;
};
}  // namespace

TEST(ObjectTable, LookupOwnsReferenceUntilItLeavesScope) {
  ObjectTable table;
  int destroyed = 0;
  auto* object = new CountedObject(table, destroyed);
  const auto handle = object->handle();
  {
    auto lookup = table.Get(handle);
    ASSERT_EQ(lookup.get(), object);
    EXPECT_TRUE(table.Release(handle));
    EXPECT_EQ(destroyed, 0);
  }
  EXPECT_EQ(destroyed, 1);
}

TEST(ObjectTable, ResetReleasesObjectsAfterRepeatedLookups) {
  ObjectTable table;
  int destroyed = 0;
  auto* object = new CountedObject(table, destroyed);
  const auto handle = object->handle();
  for (int i = 0; i < 100; ++i) {
    auto lookup = table.Get(handle);
    ASSERT_EQ(lookup.get(), object);
  }
  table.Reset();
  EXPECT_EQ(destroyed, 1);
}

// dup hands out a second handle for the same object; the object lives until
// both are closed.
TEST(ObjectTable, DupSharesTheObject) {
  ObjectTable table;
  int destroyed = 0;
  auto* obj = new CountedObject(table, destroyed);
  const u32 h = obj->handle();
  u32 copy = 0;
  ASSERT_TRUE(table.Dup(h, 0, copy));
  EXPECT_NE(copy, h);
  EXPECT_EQ(table.Get(copy).get(), obj);
  table.Release(h);
  EXPECT_EQ(destroyed, 0);
  EXPECT_EQ(table.Get(copy).get(), obj);
  table.Release(copy);
  EXPECT_EQ(destroyed, 1);
}

// dup2 onto an occupied handle closes its object; onto a stdio number it makes
// an alias that resolves and closes like any handle.
TEST(ObjectTable, Dup2ReplacesAndAliases) {
  ObjectTable table;
  int a_gone = 0, b_gone = 0;
  auto* a = new CountedObject(table, a_gone);
  auto* b = new CountedObject(table, b_gone);
  const u32 ha = a->handle(), hb = b->handle();
  ASSERT_TRUE(table.Dup2(ha, hb));
  EXPECT_EQ(b_gone, 1);
  EXPECT_EQ(table.Get(hb).get(), a);

  ASSERT_TRUE(table.Dup2(ha, 1));
  EXPECT_TRUE(table.IsAlias(1));
  EXPECT_EQ(table.Get(1).get(), a);
  table.Release(ha);
  table.Release(hb);
  EXPECT_EQ(a_gone, 0);
  table.Release(1);
  EXPECT_FALSE(table.IsAlias(1));
  EXPECT_EQ(a_gone, 1);
}
