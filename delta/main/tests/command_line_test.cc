#include <gtest/gtest.h>

#include "main/command_line.h"

TEST(CommandLine, NoGameLeavesStartupToTheHomeScreen) {
  char program[] = "ps4delta";
  char* args[] = {program};
  const auto command = cli::Parse(1, args);
  EXPECT_FALSE(command.exit);
  EXPECT_TRUE(command.game.empty());
}

TEST(CommandLine, OptionsCanBeUsedWithoutAGame) {
  char program[] = "ps4delta";
  char option[] = "+DELTA_GPU_BACKEND=opengl";
  char* args[] = {program, option};
  const auto command = cli::Parse(2, args);
  EXPECT_FALSE(command.exit);
  EXPECT_TRUE(command.game.empty());
  EXPECT_FALSE(command.options.empty());
}

TEST(CommandLine, GamePathAndGuestArgumentsStillBootDirectly) {
  char program[] = "ps4delta";
  char path[] = "/games/my game.pkg";
  char divider[] = "--";
  char guest[] = "-guest-option";
  char* args[] = {program, path, divider, guest};
  const auto command = cli::Parse(4, args);
  EXPECT_FALSE(command.exit);
  EXPECT_EQ(command.game, path);
  ASSERT_EQ(command.guest_args.size(), 1u);
  EXPECT_EQ(command.guest_args.front(), guest);
}

TEST(CommandLine, HeadlessStartupRequiresAGame) {
  char program[] = "ps4delta";
  char option[] = "--headless";
  char* args[] = {program, option};
  const auto command = cli::Parse(2, args);
  EXPECT_TRUE(command.exit);
  EXPECT_EQ(command.exit_code, 2);
}
