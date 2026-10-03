#include "gpu/ps5/label_publisher.h"
#include "gpu/render/renderer.h"

#include <gtest/gtest.h>

namespace {
u32 g_staged_result = 0;
u32 g_guest_result = 0;
}  // namespace

namespace gpu::render {
Renderer& DefaultRenderer() {
  static Renderer renderer;
  return renderer;
}
bool FlushCsWrites(Renderer&, const char*) {
  g_guest_result = g_staged_result;
  return true;
}
u64 GuestWritePublishBatch() { return 0; }
bool WaitBatch(u64) { return true; }
void SubmitForPublishedLabels() {}
void ReportBatchState() {}
}  // namespace gpu::render

TEST(AgcLabel, PublishesStagedOutputsBeforeCompletion) {
  g_staged_result = 42;
  g_guest_result = 0;
  bool completed = false;
  gpu::ps5::PublishLabel([&] {
    EXPECT_EQ(g_guest_result, g_staged_result);
    completed = true;
  });
  EXPECT_TRUE(completed);
}
