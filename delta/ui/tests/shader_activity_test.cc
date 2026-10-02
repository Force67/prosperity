#include <gtest/gtest.h>

#include "base/threading/thread.h"
#include "ui/shader_activity.h"

TEST(ShaderActivity, NestedScopesRemainActiveUntilBothFinish) {
  const auto before = ui::GetShaderActivity();
  {
    const ui::ShaderCompilation outer;
    const auto started = ui::GetShaderActivity();
    EXPECT_EQ(started.active, before.active + 1);
    EXPECT_GT(started.active_since_ns, 0u);
    {
      const ui::ShaderCompilation inner;
      EXPECT_EQ(ui::GetShaderActivity().active, before.active + 2);
      EXPECT_EQ(ui::GetShaderActivity().active_since_ns,
                started.active_since_ns);
    }
    EXPECT_EQ(ui::GetShaderActivity().active, before.active + 1);
  }
  EXPECT_EQ(ui::GetShaderActivity().active, before.active);
}

TEST(ShaderActivity, WorkerPublishesSlowCompletion) {
  const auto before = ui::GetShaderActivity();
  base::Thread worker(
      "shader_test",
      [] {
        const ui::ShaderCompilation compilation;
        base::SleepForMicroseconds(50'000);
      },
      true);
  worker.Join();
  const auto after = ui::GetShaderActivity();
  EXPECT_EQ(after.active, before.active);
  EXPECT_GT(after.last_slow_compile_ns, before.last_slow_compile_ns);
}
