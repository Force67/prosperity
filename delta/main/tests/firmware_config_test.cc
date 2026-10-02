#include <gtest/gtest.h>

#include <elf.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include "base/containers/vector.h"
#include "base/filesystem/file.h"
#include "base/option.h"
#include "base/option_file.h"
#include "main/firmware_config.h"
#include "main/game_firmware.h"

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
  void WriteFile(const base::String& path, const void* data, mem_size size) {
    files_.push_back(path);
    base::File file(base::Path(base::StringRefU8(
                        reinterpret_cast<const char8_t*>(path.c_str()))),
                    base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
    ASSERT_TRUE(file.IsValid());
    ASSERT_EQ(file.WriteAtCurrentPos(static_cast<const char*>(data), size),
              size);
  }
  void AddKernel(const base::String& directory,
                 u32 version,
                 const char* name = "libkernel.sprx") {
    Elf64_Ehdr header{};
    std::memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_machine = EM_X86_64;
    header.e_phoff = sizeof(header);
    header.e_phentsize = sizeof(Elf64_Phdr);
    header.e_phnum = 1;
    Elf64_Phdr segment{};
    segment.p_type = 0x61000002;
    segment.p_offset = sizeof(header) + sizeof(segment);
    segment.p_filesz = 32;
    base::Vector<u8> bytes(segment.p_offset + segment.p_filesz);
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + header.e_phoff, &segment, sizeof(segment));
    std::memcpy(bytes.data() + segment.p_offset + 0x14, &version,
                sizeof(version));
    WriteFile(directory + "/" + name, bytes.data(), bytes.size());
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

TEST_F(FirmwareConfigTest, ReadsFirmwareRatherThanSdkFromSelectedKernel) {
  AddKernel(root_, 0x08400005);
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", root_));
  EXPECT_EQ(cli::Ps5FirmwareVersion(), 0x08400005u);
  AddKernel(root_, 0x10010001, "libkernel.native.sprx");
  EXPECT_EQ(cli::Ps5FirmwareVersion(), 0x10010001u);
  const auto older = AddDirectory("older");
  AddKernel(older, 0x01140001);
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", older + ":" + root_));
  EXPECT_EQ(cli::Ps5FirmwareVersion(), 0x01140001u);
}

TEST_F(FirmwareConfigTest, InvalidKernelHasUnknownFirmwareVersion) {
  AddModule(root_, "libkernel.sprx");
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", root_));
  EXPECT_EQ(cli::Ps5FirmwareVersion(), 0u);
}

TEST_F(FirmwareConfigTest, BlocksNewerGameAndAllowsMatchingOrNewerFirmware) {
  const auto game = AddDirectory("game");
  ::mkdir((game + "/sce_sys").c_str(), 0700);
  directories_.insert(directories_.begin(), game + "/sce_sys");
  WriteFile(game + "/eboot.bin", "elf", 3);
  constexpr char json[] =
      "{\"requiredSystemSoftwareVersion\":\"0x1001000000000000\","
      "\"sdkVersion\":\"0x0900000000000000\"}";
  WriteFile(game + "/sce_sys/param.json", json, sizeof(json) - 1);
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", root_));
  AddKernel(root_, 0x08400005);
  EXPECT_EQ(cli::CheckGameFirmware(game),
            "Install PS5 firmware 10.01 or later to play this game.");
  EXPECT_EQ(cli::CheckGameFirmware(game + "/eboot.bin"),
            cli::CheckGameFirmware(game));
  AddKernel(root_, 0x10010000);
  EXPECT_TRUE(cli::CheckGameFirmware(game).empty());
  AddKernel(root_, 0x13600007);
  EXPECT_TRUE(cli::CheckGameFirmware(game).empty());
  kPs5Modules.Reset();
  EXPECT_FALSE(cli::CheckGameFirmware(game).empty());
}

TEST_F(FirmwareConfigTest, MissingRequirementDoesNotBlockGame) {
  const auto game = AddDirectory("game");
  ::mkdir((game + "/sce_sys").c_str(), 0700);
  directories_.insert(directories_.begin(), game + "/sce_sys");
  WriteFile(game + "/eboot.bin", "elf", 3);
  EXPECT_TRUE(cli::CheckGameFirmware(game).empty());
  constexpr char json[] = "{\"sdkVersion\":\"0x0900000000000000\"}";
  WriteFile(game + "/sce_sys/param.json", json, sizeof(json) - 1);
  EXPECT_TRUE(cli::CheckGameFirmware(game).empty());
}

TEST_F(FirmwareConfigTest, DemonsSoulsArchiveRequiresFirmware1001) {
  const base::String game =
      "/home/vince/Documents/dumps/PS5/PPSA01342 (1.05).rar";
  if (::access(game.c_str(), R_OK) != 0)
    GTEST_SKIP() << "test archive not present";
  AddKernel(root_, 0x08400005);
  ASSERT_TRUE(base::SetOptionValue("DELTA_PS5_MODULES", root_));
  EXPECT_EQ(cli::CheckGameFirmware(game),
            "Install PS5 firmware 10.01 or later to play this game.");
}
}  // namespace
