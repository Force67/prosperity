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
