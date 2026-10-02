#include <gtest/gtest.h>

#include "base/containers/array.h"
#include "ui/home_background.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include "ui/home_screen.h"
#include "ui/overlay.h"
#endif

TEST(HomeBackground, NormalStylesAreEqualAndAnniversaryIsOneIn256) {
  base::Array<u32, 5> counts{};
  for (u32 draw = 0; draw < 768; ++draw) {
    const auto style = ui::ChooseHomeBackground(draw);
    ASSERT_GE(style, 1u);
    ASSERT_LE(style, 4u);
    ++counts[style];
  }
  EXPECT_EQ(counts[1], 255u);
  EXPECT_EQ(counts[2], 255u);
  EXPECT_EQ(counts[3], 255u);
  EXPECT_EQ(counts[4], 3u);
}

#if defined(__linux__) && !defined(__ANDROID__)
TEST(HomeBackground, RandomChoiceStaysFixedThroughoutTheMenuSession) {
  const base::Vector<ui::HomeGame> games;
  ui::BeginHomeScreen(games, true, true);
  const auto selected = ui::HomeScreenBackground();
  EXPECT_TRUE(selected.visible);
  EXPECT_GE(selected.style, 1u);
  EXPECT_LE(selected.style, 4u);
  for (u32 frame = 0; frame < 100; ++frame)
    EXPECT_EQ(ui::HomeScreenBackground().style, selected.style);
  EXPECT_TRUE(ui::EndHomeScreen().empty());
  EXPECT_FALSE(ui::HomeScreenBackground().visible);
  ui::OverlayShutdownImGui();
}

#endif
