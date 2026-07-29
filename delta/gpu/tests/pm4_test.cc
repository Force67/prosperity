#include <gtest/gtest.h>

#include "gpu/ps4/pm4.h"

TEST(Pm4, IndexedShaderRegisterUsesLowOffsetBits) {
  EXPECT_EQ(gpu::IT_SET_SH_REG_INDEX, 0x9bu);
  EXPECT_EQ(gpu::Pm4SetRegAddress(gpu::kShRegBase, 0x40000258), 0x2e58u);
}
