#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include <gtest/gtest.h>

#include "kern/vfs.h"
#include "options/options.h"
#include "runtime/vprx/ps4/lib_sce_save_data/lib_sce_save_data.h"

TEST(SaveData, ReportsUsedBlocksFromFiles) {
  char root[] = "/tmp/prosperity-save-XXXXXX";
  ASSERT_NE(mkdtemp(root), nullptr);
  setenv("DELTA_SAVEDATA_DIR", root, 1);
  base::InitOptionsFromEnv();
  kern::vfs::SetTitleId("TESTSAVE");

  struct {
    u64 user_id = 1;
    const char* name = "world";
    u64 blocks = 60000;
    u32 mode = 6;
  } mount;
  u64 result[8]{};
  ASSERT_EQ(sceSaveDataMount2(&mount, result), 0);
  const auto path = kern::vfs::ResolveWritable(
      (std::string(reinterpret_cast<char*>(result)) + "/level.dat").c_str());
  FILE* file = std::fopen(path.c_str(), "wb");
  ASSERT_NE(file, nullptr);
  ASSERT_EQ(std::fseek(file, 32768, SEEK_SET), 0);
  ASSERT_NE(std::fputc(0, file), EOF);
  std::fclose(file);

  u64 info[6]{};
  ASSERT_EQ(sceSaveDataGetMountInfo(result, info), 0);
  EXPECT_EQ(info[0] - info[1], 2u);

  struct {
    u32 hits = 0;
    u32 pad = 0;
    void* names = nullptr;
    u32 capacity = 1;
    u32 count = 0;
    void* params = nullptr;
    void* infos = nullptr;
    u8 reserve[16]{};
  } search;
  search.infos = info;
  ASSERT_EQ(sceSaveDataDirNameSearch(nullptr, &search), 0);
  ASSERT_EQ(search.count, 1u);
  EXPECT_EQ(info[0], 2u);
  EXPECT_EQ(info[0] - info[1], 2u);

  EXPECT_EQ(sceSaveDataUmount(result), 0);
  std::filesystem::remove_all(root);
}
