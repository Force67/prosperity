#include <gtest/gtest.h>

#include <unistd.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>

#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
class PlayGoTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::ResetSessionResources();
    char path[] = "/tmp/prosperity-playgo-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    root = path;
    std::filesystem::create_directory(root / "sce_sys");
    kern::vfs::Mount("/app0", root.c_str());
    CreateClient();
  }
  void TearDown() override {
    kern::vfs::Unmount("/app0");
    guest::ResetSessionResources();
    std::filesystem::remove_all(root);
    base::OptionBase::VisitAll([](const base::OptionBase* option) {
      if (std::strcmp(option->name(), "DELTA_PLAYGO_CHUNKS") == 0)
        const_cast<base::OptionBase*>(option)->Reset();
    });
  }
  void CreateClient() {
    const char* name = "ScePlayGo";
    u64 request = reinterpret_cast<u64>(name);
    ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)),
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
    std::array<u64, 6> out{};
    u32 i = 0;
    for (const auto& buffer : inputs) {
      in[i++] = reinterpret_cast<u64>(buffer.data);
      in[i++] = buffer.size;
    }
    i = 0;
    for (const auto& buffer : outputs) {
      out[i++] = reinterpret_cast<u64>(buffer.data);
      out[i++] = buffer.size;
      out[i++] = 1;
    }
    i32 result = -1;
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
    u32 manager_result = ~0u;
    EXPECT_EQ(kern::ipmi::ManagerCall(800, kid, &manager_result, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(manager_result, 0u);
    return result;
  }
  u32 Open() {
    u32 handle = 0;
    EXPECT_EQ(Invoke(0x30000, {}, {{&handle, sizeof(handle)}}), 0);
    EXPECT_NE(handle, 0u);
    return handle;
  }
  u32 Count(u32 handle) {
    u32 count = 0;
    EXPECT_EQ(
        Invoke(0x3000f, {{&handle, sizeof(handle)}}, {{&count, sizeof(count)}}),
        0);
    return count;
  }
  void WriteCount(u16 count) {
    std::array<u8, 16> header{'p', 'g', 'd', 0};
    header[10] = count;
    header[11] = count >> 8;
    std::ofstream file(root / "sce_sys/playgo-chunk.dat", std::ios::binary);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
  }
  void Reset() {
    guest::ResetSessionResources();
    kern::vfs::Mount("/app0", root.c_str());
    CreateClient();
  }
  u32 kid = 0;
  std::filesystem::path root;
};

TEST_F(PlayGoTest, MetadataEnumerationAndRequestedLoci) {
  WriteCount(4);
  const u32 handle = Open();
  ASSERT_EQ(Count(handle), 4u);
  u32 request[] = {handle, 2};
  u16 ids[3] = {77, 77, 77};
  u32 count = ~0u;
  ASSERT_EQ(Invoke(0x3000e, {{request, sizeof(request)}},
                   {{ids, 4}, {&count, sizeof(count)}}),
            0);
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 0);
  EXPECT_EQ(ids[1], 1);
  EXPECT_EQ(ids[2], 77);
  u16 chunks[] = {3, 0};
  u8 loci[] = {77, 77, 77};
  ASSERT_EQ(
      Invoke(0x30008, {{request, sizeof(request)}, {chunks, sizeof(chunks)}},
             {{loci, 2}}),
      0);
  EXPECT_EQ(loci[0], 3);
  EXPECT_EQ(loci[1], 3);
  EXPECT_EQ(loci[2], 77);
  chunks[1] = 4;
  loci[0] = loci[1] = 77;
  EXPECT_EQ(
      Invoke(0x30008, {{request, sizeof(request)}, {chunks, sizeof(chunks)}},
             {{loci, 2}}),
      static_cast<i32>(0x80b2000cu));
  EXPECT_EQ(loci[0], 77);
  EXPECT_EQ(loci[1], 77);
}

TEST_F(PlayGoTest, InstalledProgressEtaAndEmptyTodo) {
  WriteCount(2);
  const u32 handle = Open();
  u32 request[] = {handle, 2};
  u16 ids[] = {0, 1};
  u64 progress[] = {77, 77, 77};
  ASSERT_EQ(Invoke(0x3000d, {{request, sizeof(request)}, {ids, sizeof(ids)}},
                   {{progress, 16}}),
            0);
  EXPECT_EQ(progress[0], progress[1]);
  EXPECT_GT(progress[1], 0u);
  EXPECT_EQ(progress[2], 77u);
  i64 eta = -1;
  ASSERT_EQ(Invoke(0x3000c, {{request, sizeof(request)}, {ids, sizeof(ids)}},
                   {{&eta, sizeof(eta)}}),
            0);
  EXPECT_EQ(eta, 0);
  u32 todo[2] = {77, 77}, count = 77;
  ASSERT_EQ(Invoke(0x3000a, {{request, sizeof(request)}},
                   {{todo, sizeof(todo)}, {&count, sizeof(count)}}),
            0);
  EXPECT_EQ(count, 0u);
  EXPECT_EQ(todo[0], 0u);
  EXPECT_EQ(todo[1], 0u);
  const u8 entries[] = {0, 0, 3, 0, 1, 0, 3, 0};
  EXPECT_EQ(
      Invoke(0x30009, {{request, sizeof(request)}, {entries, sizeof(entries)}}),
      0);
  struct Prefetch {
    u32 handle, count;
    u8 locus, pad[3];
  } prefetch{handle, 2, 3, {}};
  EXPECT_EQ(
      Invoke(0x3000b, {{&prefetch, sizeof(prefetch)}, {ids, sizeof(ids)}}), 0);
}

TEST_F(PlayGoTest, SettingsRoundTripAndSessionReset) {
  WriteCount(3);
  u32 handle = Open(), speed = 77;
  u64 mask = 0;
  EXPECT_EQ(Invoke(0x30003, {{&handle, 4}}, {{&speed, 4}}), 0);
  EXPECT_EQ(speed, 2u);
  EXPECT_EQ(Invoke(0x30005, {{&handle, 4}}, {{&mask, 8}}), 0);
  EXPECT_EQ(mask, ~u64(0));
  u32 set_speed[] = {handle, 0};
  EXPECT_EQ(Invoke(0x30002, {{set_speed, 8}}), 0);
  EXPECT_EQ(Invoke(0x30003, {{&handle, 4}}, {{&speed, 4}}), 0);
  EXPECT_EQ(speed, 0u);
  set_speed[1] = 3;
  EXPECT_EQ(Invoke(0x30002, {{set_speed, 8}}), static_cast<i32>(0x80b2000du));
  EXPECT_EQ(Invoke(0x30003, {{&handle, 4}}, {{&speed, 4}}), 0);
  EXPECT_EQ(speed, 0u);
  struct Language {
    u32 handle, pad;
    u64 mask;
  } language{handle, 0, 0x8000000000000042ull};
  EXPECT_EQ(Invoke(0x30004, {{&language, sizeof(language)}}), 0);
  EXPECT_EQ(Invoke(0x30005, {{&handle, 4}}, {{&mask, 8}}), 0);
  EXPECT_EQ(mask, language.mask);
  EXPECT_EQ(Count(handle), 3u);
  WriteCount(7);
  Reset();
  handle = Open();
  EXPECT_EQ(Count(handle), 7u);
  EXPECT_EQ(Invoke(0x30003, {{&handle, 4}}, {{&speed, 4}}), 0);
  EXPECT_EQ(speed, 2u);
  EXPECT_EQ(Invoke(0x30005, {{&handle, 4}}, {{&mask, 8}}), 0);
  EXPECT_EQ(mask, ~u64(0));
}

TEST_F(PlayGoTest, CloseInvalidatesOnlyItsHandle) {
  const u32 first = Open(), second = Open();
  EXPECT_NE(first, second);
  EXPECT_EQ(Invoke(0x30001, {{&first, 4}}), 0);
  u32 count = 77;
  EXPECT_EQ(Invoke(0x3000f, {{&first, 4}}, {{&count, 4}}),
            static_cast<i32>(0x80b20009u));
  EXPECT_EQ(count, 77u);
  EXPECT_GT(Count(second), 0u);
}

TEST_F(PlayGoTest, MalformedRequestsLeaveOutputsAndSettingsUntouched) {
  WriteCount(2);
  const u32 handle = Open();
  u32 request[] = {handle, 2};
  u16 ids[] = {0, 1};
  u8 loci[2] = {77, 77};
  for (u64 size : {0u, 2u}) {
    EXPECT_EQ(Invoke(0x30008, {{request, 8}, {ids, size}}, {{loci, 2}}),
              static_cast<i32>(0x80b20004u));
    EXPECT_EQ(loci[0], 77);
  }
  EXPECT_EQ(Invoke(0x30008, {{request, 8}, {ids, 4}}, {{loci, 1}}),
            static_cast<i32>(0x80b20004u));
  EXPECT_EQ(loci[0], 77);
  EXPECT_EQ(Invoke(0x30008, {{request, 8}, {reinterpret_cast<void*>(1), 4}},
                   {{loci, 2}}),
            static_cast<i32>(0x80b20004u));
  EXPECT_EQ(loci[0], 77);
  ids[0] = ids[1] = 77;
  u32 count = 77;
  EXPECT_EQ(Invoke(0x3000e, {{request, 8}}, {{ids, 4}, {&count, 3}}),
            static_cast<i32>(0x80b20004u));
  EXPECT_EQ(ids[0], 77);
  EXPECT_EQ(ids[1], 77);
  EXPECT_EQ(count, 77u);
  u32 speed = 77;
  EXPECT_EQ(Invoke(0x30002, {{request, 4}}), static_cast<i32>(0x80b20004u));
  EXPECT_EQ(Invoke(0x30003, {{&handle, 4}}, {{&speed, 4}}), 0);
  EXPECT_EQ(speed, 2u);
  request[1] = ~0u;
  EXPECT_EQ(Invoke(0x30008, {{request, 8}, {ids, 4}}, {{loci, 2}}),
            static_cast<i32>(0x80b20004u));
  EXPECT_EQ(loci[0], 77);
}

TEST_F(PlayGoTest, Ps5MetadataAndLargeCountsStayBounded) {
  std::ofstream(root / "sce_sys/playgo-scenario.json")
      << R"({"chunks":["0-15","17","4294967296"]})";
  EXPECT_EQ(Count(Open()), 18u);
  std::filesystem::remove(root / "sce_sys/playgo-scenario.json");
  std::ofstream(root / "playgo-chunkdefs.xml")
      << R"(<chunks><chunk id="31"/></chunks>)";
  Reset();
  EXPECT_EQ(Count(Open()), 32u);
  WriteCount(65535);
  Reset();
  const u32 handle = Open();
  EXPECT_EQ(Count(handle), 1000u);
  u32 request[] = {handle, ~0u};
  std::array<u16, 1001> ids;
  ids.fill(77);
  u32 count = 77;
  EXPECT_EQ(Invoke(0x3000e, {{request, 8}}, {{ids.data(), 2000}, {&count, 4}}),
            0);
  EXPECT_EQ(count, 1000u);
  EXPECT_EQ(ids[999], 999);
  EXPECT_EQ(ids[1000], 77);
}
TEST_F(PlayGoTest, OversizedOverrideCannotOverflowEnumeration) {
  base::OptionBase::VisitAll([](const base::OptionBase* option) {
    if (std::strcmp(option->name(), "DELTA_PLAYGO_CHUNKS") == 0)
      ASSERT_TRUE(
          const_cast<base::OptionBase*>(option)->SetFromString("4294967295"));
  });
  const u32 handle = Open();
  EXPECT_EQ(Count(handle), 1000u);
  u32 request[] = {handle, ~0u};
  std::array<u16, 1001> ids;
  ids.fill(77);
  u32 count = 77;
  EXPECT_EQ(Invoke(0x3000e, {{request, 8}}, {{ids.data(), 2000}, {&count, 4}}),
            0);
  EXPECT_EQ(count, 1000u);
  EXPECT_EQ(ids[999], 999);
  EXPECT_EQ(ids[1000], 77);
}
}  // namespace
