#include <gtest/gtest.h>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

#include "base/environment_variables.h"
#include "base/filesystem/file.h"
#include "io/file.h"
#include "main/recent_games.h"

namespace {
void RemoveDirectory(const base::String& path) {
  DIR* directory = ::opendir(path.c_str());
  if (!directory)
    return;
  while (const auto* entry = ::readdir(directory)) {
    if (base::String(entry->d_name) == "." ||
        base::String(entry->d_name) == "..")
      continue;
    const auto child = path + "/" + entry->d_name;
    struct stat info{};
    if (::lstat(child.c_str(), &info) == 0 && S_ISDIR(info.st_mode))
      RemoveDirectory(child);
    else
      ::unlink(child.c_str());
  }
  ::closedir(directory);
  ::rmdir(path.c_str());
}

class RecentGamesTest : public testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/prosperity-recents-XXXXXX";
    const char* directory = ::mkdtemp(path);
    ASSERT_NE(directory, nullptr);
    root_ = directory;
    had_data_home_ =
        base::GetEnvironmentVariable(u8"XDG_DATA_HOME", old_data_home_);
    ASSERT_EQ(::setenv("XDG_DATA_HOME", root_.c_str(), 1), 0);
  }
  void TearDown() override {
    if (had_data_home_)
      ::setenv("XDG_DATA_HOME",
               reinterpret_cast<const char*>(old_data_home_.c_str()), 1);
    else
      ::unsetenv("XDG_DATA_HOME");
    RemoveDirectory(root_);
  }
  base::String AddGame(const char* name) {
    const auto path = root_ + "/" + name;
    ::mkdir(path.c_str(), 0700);
    return path;
  }
  base::String root_;
  base::StringU8 old_data_home_;
  bool had_data_home_ = false;
};

TEST_F(RecentGamesTest, StoresArtworkAndMovesReplayedTitleToFront) {
  const auto first = AddGame("first game");
  const auto second = AddGame("second game");
  const base::Vector<u8> icon{1, 2, 3};
  const base::Vector<u8> art{4, 5, 6};
  cli::RememberGame(first, "Name\twith\nseparators", "CUSA00001", false, icon,
                    art);
  cli::RememberGame(second, "Second", "PPSA00002", true, {}, {});
  cli::RememberGame(first, "Name\twith\nseparators", "CUSA00001", false, {},
                    {});
  const auto games = cli::ReadRecentGames();
  ASSERT_EQ(games.size(), 2u);
  EXPECT_EQ(games[0].path, first);
  EXPECT_EQ(games[0].name, "Name\twith\nseparators");
  EXPECT_EQ(games[0].icon, icon);
  EXPECT_EQ(games[0].artwork, art);
  EXPECT_FALSE(games[0].is_ps5);
  EXPECT_TRUE(games[1].is_ps5);
}

TEST_F(RecentGamesTest, LimitsHistoryAndMarksMissingGamesUnavailable) {
  for (int i = 0; i < 14; ++i) {
    base::String name("game");
    name += static_cast<char>('A' + i);
    const auto path = AddGame(name.c_str());
    cli::RememberGame(path, name, "", false, {}, {});
  }
  auto games = cli::ReadRecentGames();
  ASSERT_EQ(games.size(), 12u);
  EXPECT_EQ(games.front().name, "gameN");
  ASSERT_EQ(::rmdir(games.front().path.c_str()), 0);
  games = cli::ReadRecentGames();
  EXPECT_FALSE(games.front().available);
  EXPECT_TRUE(games.back().available);
}

TEST_F(RecentGamesTest, SkipsCorruptHistory) {
  ::mkdir((root_ + "/prosperity").c_str(), 0700);
  const auto path = root_ + "/prosperity/recent-games";
  base::File file(base::Path(base::StringRefU8(
                      reinterpret_cast<const char8_t*>(path.c_str()))),
                  base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
  ASSERT_TRUE(file.IsValid());
  constexpr char text[] = "bad\nzz\t00\t00\t5\n";
  ASSERT_EQ(file.WriteAtCurrentPos(text, sizeof(text) - 1), sizeof(text) - 1);
  file.Close();
  EXPECT_TRUE(cli::ReadRecentGames().empty());
}
}  // namespace
