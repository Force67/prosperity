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
