#include "gpu/render/compute_hazard.h"

#include <gtest/gtest.h>

namespace gpu::render {
namespace {

TEST(ComputeHazard, AllowsIndependentReads) {
  EXPECT_FALSE(NeedsComputeBarrier({false, false}, {true, false}));
  EXPECT_FALSE(NeedsComputeBarrier({true, false}, {true, false}));
}

TEST(ComputeHazard, OrdersEveryWriteDependency) {
  EXPECT_TRUE(NeedsComputeBarrier({false, true}, {true, false}));
  EXPECT_TRUE(NeedsComputeBarrier({false, true}, {true, true}));
  EXPECT_TRUE(NeedsComputeBarrier({true, false}, {true, true}));
  EXPECT_TRUE(NeedsComputeBarrier({true, true}, {true, true}));
}

}  // namespace
}  // namespace gpu::render
