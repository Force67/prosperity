#include <gtest/gtest.h>

#include "kern/util/object_table.h"

using krnl::objectTable;

// A freshly constructed table owns nothing, so every lookup or mutation against
// an unknown handle must fail instead of crashing or returning garbage.
TEST(ObjectTable, EmptyTableLookupsFail) {
  objectTable table;
  EXPECT_EQ(table.get(0x4), nullptr);
  EXPECT_EQ(table.get(0x1234), nullptr);
  EXPECT_FALSE(table.remove(0x4));
  EXPECT_FALSE(table.release(0x4));
  EXPECT_FALSE(table.keep(0x4));
}

// reset()/purge() on an empty table are no-ops and leave it usable.
TEST(ObjectTable, ResetAndPurgeAreSafeWhenEmpty) {
  objectTable table;
  table.reset();
  table.purge();
  EXPECT_EQ(table.get(0x4), nullptr);
}
