#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

#include "base/containers/vector.h"
#include "base/filesystem/file.h"
#include "base/option.h"
#include "base/option_file.h"
#include "main/firmware_config.h"

namespace {
base::Option<const char*> kPs4Modules{"DELTA_PS4_MODULES", nullptr};
base::Option<const char*> kPs5Modules{"DELTA_PS5_MODULES", nullptr};

class FirmwareConfigTest : public testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/prosperity-firmware-XXXXXX";
    ASSERT_NE(::mkdtemp(path), nullptr);
    root_ = path;
    ASSERT_TRUE(base::SetOptionValue("DELTA_PS4_MODULES", root_));
    kPs5Modules.Reset();
  }
  void TearDown() override {
    for (const auto& path : files_)
      ::unlink(path.c_str());
    for (const auto& path : directories_)
      ::rmdir(path.c_str());
    ::rmdir(root_.c_str());
    kPs4Modules.Reset();
    kPs5Modules.Reset();
  }
  base::String AddDirectory(const char* name) {
    const auto path = root_ + "/" + name;
    ::mkdir(path.c_str(), 0700);
    directories_.push_back(path);
    return path;
  }
  void AddModule(const base::String& directory,
                 const char* name,
                 bool valid = true) {
    const auto path = directory + "/" + name;
    files_.push_back(path);
    base::File file(base::Path(base::StringRefU8(
                        reinterpret_cast<const char8_t*>(path.c_str()))),
                    base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
    ASSERT_TRUE(file.IsValid());
    u8 header[20]{};
    header[0] = 0x7f;
    header[1] = 'E';
    header[2] = 'L';
    header[3] = 'F';
    header[4] = 2;
    header[5] = 1;
    header[18] = valid ? 62 : 183;
    ASSERT_EQ(file.WriteAtCurrentPos(reinterpret_cast<const char*>(header),
                                     sizeof(header)),
              sizeof(header));
  }
  base::String root_;
  base::Vector<base::String> files_;
  base::Vector<base::String> directories_;
};

TEST_F(FirmwareConfigTest, EmptyFoldersAndUnsetPs5AreNotReady) {
  EXPECT_FALSE(cli::FirmwareModulesReady(false));
  EXPECT_FALSE(cli::FirmwareModulesReady(true));
}

TEST_F(FirmwareConfigTest, RequiresBothPs4ModulesWithCorrectArchitecture) {
  AddModule(root_, "libkernel.sprx");
  EXPECT_FALSE(cli::FirmwareModulesReady(false));
  AddModule(root_, "libSceLibcInternal.sprx", false);
  EXPECT_FALSE(cli::FirmwareModulesReady(false));
  AddModule(root_, "libSceLibcInternal.sprx");
  EXPECT_TRUE(cli::FirmwareModulesReady(false));
}

TEST_F(FirmwareConfigTest, Ps5ModulesCanSpanDirectoriesAndUseNativeSuffix) {
  const auto common = AddDirectory("common");
  const auto priv = AddDirectory("priv");
  AddModule(common, "libkernel.native.sprx");
  AddModule(priv, "libSceLibcInternal.sprx");
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", common + ":" + priv));
  EXPECT_TRUE(cli::FirmwareModulesReady(true));
  EXPECT_FALSE(cli::FirmwareModulesReady(false));
}

TEST_F(FirmwareConfigTest,
       InvalidOverrideDoesNotReportConfiguredFirmwareReady) {
  AddModule(root_, "libkernel.sprx");
  AddModule(root_, "libSceLibcInternal.sprx");
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", root_));
  EXPECT_TRUE(cli::FirmwareModulesReady(true));
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", root_ + "/missing"));
  EXPECT_FALSE(cli::FirmwareModulesReady(true));
}
}  // namespace
