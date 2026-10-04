#include <gtest/gtest.h>

#include <unistd.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <vector>
#include "base/option_file.h"

#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
constexpr i32 kParameter = static_cast<i32>(0x80d90002u);
constexpr i32 kBusy = static_cast<i32>(0x80d90003u);
constexpr i32 kNotFound = static_cast<i32>(0x80d90005u);
class AppContentTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::ResetSessionResources();
    char path[] = "/tmp/prosperity-appcontent-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    root = path;
    std::filesystem::create_directory(root / "sce_sys");
    std::filesystem::create_directory(root / "download");
    kern::vfs::Mount("/app0", root.c_str());
    kern::vfs::MountWritable("/download0", (root / "download").c_str());
    kern::vfs::SetTitleId("PPSA03311");
    base::FindOption("DELTA_APPCONTENT_DIR")->SetFromString(root.c_str());
    Client();
  }
  void TearDown() override {
    guest::ResetSessionResources();
    base::FindOption("DELTA_APPCONTENT_DIR")->Reset();
    std::filesystem::remove_all(root);
  }
  void Client() {
    const char* name = "SceAppContent";
    const u64 request = reinterpret_cast<u64>(name);
    ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, const_cast<u64*>(&request),
                                      sizeof(request)),
              0);
  }
  struct Buffer {
    const void* data;
    u64 size;
  };
  i32 Invoke(u32 method,
             std::initializer_list<Buffer> inputs = {},
             std::initializer_list<Buffer> outputs = {}) {
    std::array<u64, 4> in{};
    std::array<u64, 3> out{};
    size_t index = 0;
    for (auto buffer : inputs) {
      in[index++] = reinterpret_cast<u64>(buffer.data);
      in[index++] = buffer.size;
    }
    index = 0;
    for (auto buffer : outputs) {
      out[index++] = reinterpret_cast<u64>(buffer.data);
      out[index++] = buffer.size;
      out[index++] = 1;
    }
    i32 result = -1;
    u32 manager_result = ~0u;
    struct Request {
      u32 method, num_in, num_out, pad;
      const u64* in;
      const u64* out;
      i32* result;
    } request{method,
              static_cast<u32>(inputs.size()),
              static_cast<u32>(outputs.size()),
              0,
              in.data(),
              out.data(),
              &result};
    EXPECT_EQ(kern::ipmi::ManagerCall(800, kid, &manager_result, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(manager_result, 0u);
    return result;
  }
  i32 Mount(u32 option = 0, u32 method = 0x2000a) {
    point.fill(0x55);
    return Invoke(method, {{&option, 4}}, {{point.data(), 16}});
  }
  i32 Unmount() { return Invoke(0x2000b, {{point.data(), 16}}); }
  i32 Param(u32 id, i32& value) {
    return Invoke(0x20001, {{&id, 4}}, {{&value, 4}});
  }
  std::filesystem::path root;
  std::array<char, 17> point;
  u32 kid = 0;
};

TEST_F(AppContentTest, InitializationUsesSonyBufferSizesAndPreservesCanary) {
  const u32 sdk = 0x1000000;
  std::array<u8, 32> init{};
  std::array<u8, 41> boot;
  boot.fill(0x55);
  ASSERT_EQ(
      Invoke(0x20000, {{&sdk, 4}, {init.data(), 32}}, {{boot.data(), 40}}), 0);
  for (size_t i = 0; i < 40; ++i)
    EXPECT_EQ(boot[i], 0);
  EXPECT_EQ(boot[40], 0x55);
  boot.fill(0x55);
  init[0] = 1;
  EXPECT_EQ(
      Invoke(0x20000, {{&sdk, 4}, {init.data(), 32}}, {{boot.data(), 40}}),
      kParameter);
  EXPECT_EQ(boot[0], 0x55);
  init[0] = 0;
  EXPECT_EQ(
      Invoke(0x20000, {{&sdk, 4}, {init.data(), 31}}, {{boot.data(), 40}}),
      kParameter);
  EXPECT_EQ(
      Invoke(0x20000, {{&sdk, 4}, {init.data(), 32}}, {{boot.data(), 39}}),
      kParameter);
}

TEST_F(AppContentTest, ParametersReadCurrentTitleAndRejectUnknownIds) {
  std::ofstream(root / "sce_sys/param.json")
      << R"({"nested":{"userDefinedParam1":99},"userDefinedParam1":42,
             "userDefinedParam2":-7,"userDefinedParam3":4294967295})";
  i32 value = 99;
  ASSERT_EQ(Param(0, value), 0);
  EXPECT_EQ(value, 3);
  ASSERT_EQ(Param(1, value), 0);
  EXPECT_EQ(value, 42);
  ASSERT_EQ(Param(2, value), 0);
  EXPECT_EQ(value, -7);
  ASSERT_EQ(Param(3, value), 0);
  EXPECT_EQ(value, -1);
  ASSERT_EQ(Param(4, value), 0);
  EXPECT_EQ(value, 0);
  value = 99;
  EXPECT_EQ(Param(5, value), kParameter);
  EXPECT_EQ(value, 99);
  std::ofstream(root / "sce_sys/param.json") << R"({"userDefinedParam1":17})";
  ASSERT_EQ(Param(1, value), 0);
  EXPECT_EQ(value, 17);
}

TEST_F(AppContentTest, InvalidNumericMetadataDoesNotBecomeAnIntegerPrefix) {
  for (const char* value :
       {"1.5", "17e2", "\"23\"", "4294967296", "-2147483649"}) {
    std::ofstream(root / "sce_sys/param.json")
        << "{\"userDefinedParam1\":" << value << "}";
    i32 result = 99;
    ASSERT_EQ(Param(1, result), 0);
    EXPECT_EQ(result, 0);
  }
}

TEST_F(AppContentTest, Ps4ParametersReadSfoIntegersAndPreserveSignedBits) {
  std::vector<u8> data(84);
  auto write = [&](size_t offset, u32 value) {
    std::memcpy(data.data() + offset, &value, sizeof(value));
  };
  write(0, 0x46535000);
  write(4, 0x101);
  write(8, 84);
  write(16, 4);
  for (u32 i = 0; i < 4; ++i) {
    const auto key = "USER_DEFINED_PARAM_" + std::to_string(i + 1);
    const u16 offset = data.size() - 84;
    std::memcpy(data.data() + 20 + i * 16, &offset, sizeof(offset));
    data[22 + i * 16] = 4;
    data[23 + i * 16] = 4;
    write(24 + i * 16, 4);
    write(28 + i * 16, 4);
    write(32 + i * 16, i * 4);
    data.insert(data.end(), key.begin(), key.end());
    data.push_back(0);
  }
  const size_t start = data.size();
  write(12, start);
  data.resize(start + 16);
  const std::array<i32, 4> values{42, -7, -1, 17};
  std::memcpy(data.data() + start, values.data(), sizeof(values));
  auto save = [&] {
    std::ofstream file(root / "sce_sys/param.sfo", std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
  };
  save();
  for (u32 i = 0; i < 4; ++i) {
    i32 value = 0;
    ASSERT_EQ(Param(i + 1, value), 0);
    EXPECT_EQ(value, values[i]);
  }
  write(start, 99);
  save();
  i32 value = 0;
  ASSERT_EQ(Param(1, value), 0);
  EXPECT_EQ(value, 99);
}

TEST_F(AppContentTest, MissingVirtualMetadataReturnsZeroWithoutAHostFault) {
  struct Missing : kern::vfs::VirtualProvider {
    base::UniquePointer<kern::vfs::VirtualFile> Open(const char*) override {
      return nullptr;
    }
    bool Stat(const char*, i64&) override { return false; }
  };
  kern::vfs::MountVirtual("/app0", base::MakeShared<Missing>());
  i32 value = 99;
  ASSERT_EQ(Param(1, value), 0);
  EXPECT_EQ(value, 0);
}

TEST_F(AppContentTest, MountUnmountAndFormatManageWritableStorage) {
  ASSERT_EQ(Mount(), 0);
  EXPECT_STREQ(point.data(), "/temp0");
  EXPECT_EQ(point[16], 0x55);
  const auto path = kern::vfs::ResolveWritable("/temp0/world");
  ASSERT_FALSE(path.empty());
  std::ofstream(path.c_str()) << "world";
  const auto saved_point = point;
  EXPECT_EQ(Mount(), kBusy);
  point = saved_point;
  ASSERT_EQ(Unmount(), 0);
  EXPECT_TRUE(kern::vfs::ResolveWritable("/temp0/world").empty());
  ASSERT_EQ(Mount(), 0);
  EXPECT_TRUE(std::filesystem::exists(path.c_str()));
  ASSERT_EQ(Unmount(), 0);
  ASSERT_EQ(Mount(1), 0);
  EXPECT_FALSE(std::filesystem::exists(path.c_str()));
}

TEST_F(AppContentTest, FormatMountedStorageKeepsTheMountAndOnlyClearsItsArea) {
  ASSERT_EQ(Mount(), 0);
  const auto path = kern::vfs::ResolveWritable("/temp0/data");
  std::ofstream(path.c_str()) << "data";
  std::ofstream(root / "download/keep") << "keep";
  ASSERT_EQ(Invoke(0x2000c, {{point.data(), 16}}), 0);
  EXPECT_FALSE(std::filesystem::exists(path.c_str()));
  EXPECT_FALSE(kern::vfs::ResolveWritable("/temp0/data").empty());
  EXPECT_TRUE(std::filesystem::exists(root / "download/keep"));
}

TEST_F(AppContentTest, FreeSpaceUsesHostCapacityAndValidatesStorageKind) {
  ASSERT_EQ(Mount(), 0);
  u32 kind = 1;
  std::array<u64, 2> space{0, 0x5555};
  ASSERT_EQ(
      Invoke(0x2000e, {{&kind, 4}, {point.data(), 16}}, {{space.data(), 8}}),
      0);
  const auto host_space = std::filesystem::space(root).available / 1024;
  EXPECT_GT(space[0], 0u);
  EXPECT_NEAR(static_cast<double>(space[0]), static_cast<double>(host_space),
              1024);
  EXPECT_EQ(space[1], 0x5555u);
  kind = 0;
  const char download[16] = "/download0";
  ASSERT_EQ(Invoke(0x2000e, {{&kind, 4}, {download, 16}}, {{space.data(), 8}}),
            0);
  EXPECT_EQ(
      Invoke(0x2000e, {{&kind, 4}, {point.data(), 16}}, {{space.data(), 8}}),
      kParameter);
  kind = 1;
  ASSERT_EQ(Unmount(), 0);
  EXPECT_EQ(
      Invoke(0x2000e, {{&kind, 4}, {point.data(), 16}}, {{space.data(), 8}}),
      kNotFound);
}

TEST_F(AppContentTest, InvalidRequestsCannotMountOrUnmapOtherStorage) {
  EXPECT_EQ(Mount(2), kParameter);
  EXPECT_TRUE(kern::vfs::ResolveWritable("/temp0").empty());
  ASSERT_EQ(Mount(), 0);
  const char other[16] = "/download0";
  EXPECT_EQ(Invoke(0x2000b, {{other, 16}}), kNotFound);
  EXPECT_FALSE(kern::vfs::ResolveWritable("/download0").empty());
  point.fill('x');
  EXPECT_EQ(Unmount(), kParameter);
  EXPECT_EQ(
      Invoke(0x2000a, {{reinterpret_cast<void*>(1), 4}}, {{point.data(), 16}}),
      kParameter);
}

TEST_F(AppContentTest, StorageFailureDoesNotPublishMountOrLatchMethodTable) {
  const auto blocked = root / "blocked";
  std::ofstream(blocked) << "keep";
  base::FindOption("DELTA_APPCONTENT_DIR")->SetFromString(blocked.c_str());
  EXPECT_EQ(Mount(0, 0x2000b), static_cast<i32>(0x80d9000au));
  EXPECT_EQ(point[0], 0x55);
  EXPECT_TRUE(kern::vfs::ResolveWritable("/temp0").empty());
  base::FindOption("DELTA_APPCONTENT_DIR")->SetFromString(root.c_str());
  ASSERT_EQ(Mount(), 0);
  EXPECT_TRUE(std::filesystem::exists(blocked));
}

TEST_F(AppContentTest, InvalidTitleCannotFormatOutsideItsStorageArea) {
  std::ofstream(root / "keep") << "keep";
  kern::vfs::SetTitleId("../OTHER");
  EXPECT_EQ(Mount(1), kParameter);
  EXPECT_TRUE(kern::vfs::ResolveWritable("/temp0").empty());
  EXPECT_TRUE(std::filesystem::exists(root / "keep"));
}

TEST_F(AppContentTest, ResetRetiresMountAndDoesNotCachePreviousTitleMetadata) {
  ASSERT_EQ(Mount(), 0);
  guest::ResetSessionResources();
  EXPECT_TRUE(kern::vfs::ResolveWritable("/temp0").empty());
  kern::vfs::SetTitleId("PPSA01234");
  Client();
  ASSERT_EQ(Mount(), 0);
  EXPECT_EQ(kern::vfs::ResolveWritable("/temp0"),
            (root / "PPSA01234/temp0").string());
}

TEST_F(AppContentTest, NewerWireTableKeepsUnmountDistinctFromFormat) {
  ASSERT_EQ(Mount(0, 0x2000b), 0);
  const auto path = kern::vfs::ResolveWritable("/temp0/world");
  std::ofstream(path.c_str()) << "world";
  u32 kind = 1;
  u64 space = 0;
  ASSERT_EQ(Invoke(0x2000f, {{&kind, 4}, {point.data(), 16}}, {{&space, 8}}),
            0);
  EXPECT_GT(space, 0u);
  ASSERT_EQ(Invoke(0x2000c, {{point.data(), 16}}), 0);
  EXPECT_TRUE(std::filesystem::exists(path.c_str()));
  EXPECT_TRUE(kern::vfs::ResolveWritable("/temp0").empty());
  ASSERT_EQ(Mount(0, 0x2000b), 0);
  ASSERT_EQ(Invoke(0x2000d, {{point.data(), 16}}), 0);
  EXPECT_FALSE(std::filesystem::exists(path.c_str()));
  EXPECT_FALSE(kern::vfs::ResolveWritable("/temp0").empty());
}

TEST_F(AppContentTest,
       ClientsKeepIndependentMethodTablesAndDestroyRetiresThem) {
  ASSERT_EQ(Mount(), 0);
  ASSERT_EQ(Unmount(), 0);
  const u32 old_client = kid;
  Client();
  ASSERT_EQ(Mount(0, 0x2000b), 0);
  const auto path = kern::vfs::ResolveWritable("/temp0/world");
  std::ofstream(path.c_str()) << "world";
  const u32 modern_client = kid;
  kid = old_client;
  ASSERT_EQ(Unmount(), 0);
  EXPECT_TRUE(std::filesystem::exists(path.c_str()));
  kid = modern_client;
  ASSERT_EQ(Mount(0, 0x2000b), 0);
  u32 result = 99;
  ASSERT_EQ(kern::ipmi::ManagerCall(3, kid, &result, nullptr, 0), 0);
  EXPECT_EQ(result, 0u);
  Client();
  EXPECT_EQ(Invoke(0x2000c, {{point.data(), 16}}), kNotFound);
  EXPECT_TRUE(std::filesystem::exists(path.c_str()));
}
}  // namespace
