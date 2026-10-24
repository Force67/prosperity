// APR: resolve a file, then run command buffers encoded as libSceAmpr encodes
// them and collect their results through apr_wait.

#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "guest_abi.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/sys_signal.h"
#include "kern/process.h"
#include "kern/vfs.h"

namespace {
constexpr int kEsrch = 3;
constexpr int kEnoent = 2;

struct Buffer {
  base::Vector<u32> dw;

  void ReadFile(u32 id, void* dst, u32 len, u64 off) {
    const u64 d = reinterpret_cast<u64>(dst);
    const u32 n = off >> 32 ? 6 : 5;
    dw.push_back(0x28 | (n - 1) << 8 | static_cast<u32>(off & 0x3ffff) << 12);
    dw.push_back(len - 1);
    dw.push_back(id);
    dw.push_back(static_cast<u32>(d));
    dw.push_back((static_cast<u32>(off) & 0xfffc0000) | ((d >> 32) & 0xffff));
    if (n == 6)
      dw.push_back(static_cast<u32>(off >> 32));
  }
  void Gather(u32 len, u64 off) {
    const u32 n = off >= 0x40000 ? 3 : 2;
    dw.push_back(0x29 | (n - 1) << 8 | static_cast<u32>(off & 0x3ffff) << 12);
    dw.push_back(len - 1);
    if (n == 3)
      dw.push_back(static_cast<u32>(off >> 18));
  }
  void Scatter(void* dst, u32 len) {
    const u64 d = reinterpret_cast<u64>(dst);
    dw.push_back(0x22a | static_cast<u32>((d >> 20) & 0xffff000));
    dw.push_back(len - 1);
    dw.push_back(static_cast<u32>(d));
  }
  // source 0 writes `v`, 2 writes counter `v`.
  void WriteAddress(u64* addr, u64 v, u32 source) {
    const u64 a = reinterpret_cast<u64>(addr);
    const u32 n = v < 4 ? 2 : 3;
    dw.push_back(0x75 | (n - 1) << 8 | static_cast<u32>(v & 3) << 12 |
                 (n - 2) << 14 | static_cast<u32>((a >> 16) & 0xffff0000));
    dw.push_back((static_cast<u32>(a) & ~7u) | source);
    if (n == 3)
      dw.push_back(static_cast<u32>(v >> 2));
  }
  void WriteCounter(u8 idx, u32 v) {  // a 4-byte store, v < 0x1000
    dw.push_back(0x06 | (v & 0xfff) << 12 | u32(idx) << 24);
  }
  void WaitOnCounter(u8 idx, u8 ref) {  // equal, 4 bytes
    dw.push_back(0x02 | u32(ref) << 16 | u32(idx) << 24);
  }
  void WaitOnAddress(u64* addr, u32 ref) {  // equal
    const u64 a = reinterpret_cast<u64>(addr);
    dw.push_back(0x01 | 2 << 8 | static_cast<u32>((a >> 16) & 0xffff0000));
    dw.push_back(static_cast<u32>(a));
    dw.push_back(ref);
  }
  void Nop() { dw.push_back(0x5452000f); }
};

struct Result {
  int result;
  u32 error_offset;
};

class Apr : public ::testing::Test {
 protected:
  void SetUp() override {
    char tmpl[] = "/tmp/apr_testXXXXXX";
    ASSERT_NE(::mkdtemp(tmpl), nullptr);
    dir_ = tmpl;
    FILE* f = std::fopen((dir_ + "/data.bin").c_str(), "wb");
    ASSERT_NE(f, nullptr);
    for (u32 i = 0; i < 0x1000; i++)
      std::fputc(static_cast<int>(i & 0xff), f);
    std::fclose(f);
    kern::vfs::Mount("/apr-test", dir_.c_str());
    kern::SignalThreadStart(1);
  }
  void TearDown() override {
    kern::SignalThreadExit();
    kern::vfs::Unmount("/apr-test");
    std::remove((dir_ + "/data.bin").c_str());
    ::rmdir(dir_.c_str());
  }

  u32 ResolveData() {
    const char* paths[] = {"/data.bin"};
    u32 id = 0xffffffff;
    u64 size = 0;
    EXPECT_EQ(kern::Ps5AprResolve("/apr-test", paths, 1, &id, &size, nullptr,
                                  nullptr, nullptr),
              0);
    EXPECT_EQ(size, 0x1000u);
    return id;
  }

  Result Run(const Buffer& b) {
    Result r{-1, 0};
    u32 id = 0;
    EXPECT_EQ(kern::Ps5AprSubmit(b.dw.data(), b.dw.size(), 3, 1,
                                 reinterpret_cast<u32*>(&r), &id),
              0);
    EXPECT_EQ(kern::Ps5AprWait(&id, 1), 0);
    EXPECT_EQ(kern::Ps5AprWait(&id, 1), -kEsrch);  // collected once
    return r;
  }

  base::String dir_;
  kern::Process process_;
};

TEST_F(Apr, ResolveReportsEachPath) {
  const char* paths[] = {"/data.bin", "/missing.bin"};
  u32 ids[2] = {};
  u64 sizes[2] = {};
  int results[2] = {};
  u32 count = 0;
  ASSERT_EQ(kern::Ps5AprResolve("/apr-test", paths, 2, ids, sizes, nullptr,
                                results, &count),
            0);
  EXPECT_EQ(count, 1u);
  EXPECT_EQ(results[0], 0);
  EXPECT_EQ(results[1], static_cast<int>(0x80020000u | kEnoent));
  EXPECT_EQ(ids[1], 0xffffffffu);
  EXPECT_EQ(sizes[0], 0x1000u);

  u32 error_index = 9;
  EXPECT_EQ(kern::Ps5AprResolve("/apr-test", paths, 2, ids, sizes, &error_index,
                                nullptr, nullptr),
            -kEnoent);
  EXPECT_EQ(error_index, 1u);

  alignas(8) u8 st[0x78];
  EXPECT_EQ(kern::Ps5AprStat(&ids[0], 1, st), 0);
  u64 st_size = 0;
  std::memcpy(&st_size, st + 0x48, 8);
  EXPECT_EQ(st_size, 0x1000u);
  const u32 bad = 0x123456;
  EXPECT_EQ(kern::Ps5AprStat(&bad, 1, st), -kEnoent);
}

TEST_F(Apr, ReadsGatherAndScatter) {
  const u32 id = ResolveData();
  u8 a[0x30] = {}, b[0x10] = {};
  Buffer cb;
  cb.ReadFile(id, a, 0x10, 0x100);
  cb.Gather(0x10, 0x200);  // continues in `a`
  cb.Gather(0x10, 0x300);
  cb.Scatter(b, 0x10);  // continues in the file after 0x300
  cb.Nop();
  const Result r = Run(cb);
  EXPECT_EQ(r.result, 0);
  for (u32 i = 0; i < 0x10; i++) {
    EXPECT_EQ(a[i], static_cast<u8>(i));
    EXPECT_EQ(a[0x10 + i], static_cast<u8>(i));
    EXPECT_EQ(a[0x20 + i], static_cast<u8>(i));
    EXPECT_EQ(b[i], static_cast<u8>(0x10 + i));
  }
}

TEST_F(Apr, CountersAndAddresses) {
  alignas(8) u64 flag = 0, copy = 0, ready = 7;
  Buffer cb;
  cb.WaitOnAddress(&ready, 7);
  cb.WriteCounter(5, 0x42);
  cb.WaitOnCounter(5, 0x42);
  cb.WriteAddress(&copy, 5, 2);
  cb.WriteAddress(&flag, 0xabcdef, 0);
  EXPECT_EQ(Run(cb).result, 0);
  EXPECT_EQ(copy, 0x42u);
  EXPECT_EQ(flag, 0xabcdefu);
}

TEST_F(Apr, ReportsTheFailingCommand) {
  u8 a[0x10];
  Buffer cb;
  cb.Nop();
  cb.ReadFile(0x777, a, 0x10, 0);  // never resolved
  const Result r = Run(cb);
  EXPECT_EQ(static_cast<u32>(r.result), 0x81912039u);
  EXPECT_EQ(r.error_offset, 4u);

  Buffer gs;
  gs.Gather(0x10, 0);  // no read before it
  EXPECT_EQ(static_cast<u32>(Run(gs).result), 0x8191203bu);
}
}  // namespace

namespace {
// aio_init: SceKernelAioParam is checked, applied once, and a second start of
// a game's scheduler is EBUSY.
TEST(AioInit, ChecksParamsAndStartsOnce) {
  kern::Process process;
  u32 param[15] = {32, 32, 1, 0x100000, 0x100000,  //
                   32, 32, 1, 0x100000, 0x100000,  //
                   32, 32, 2, 0,        0};        // enableSplit 2
  EXPECT_EQ(kern::Ps5AioInit(param, sizeof(param), 1, 0), -22);
  EXPECT_EQ(kern::Ps5AioInit(param, sizeof(param) - 4, 0, 0), -22);
  EXPECT_EQ(kern::Ps5AioInit(param, sizeof(param), 0, 0), -22);
  param[12] = 0;
  param[0] = 129;  // over SCE_KERNEL_AIO_SCHED_WINDOW_MAX
  EXPECT_EQ(kern::Ps5AioInit(param, sizeof(param), 0, 0), -22);
  param[0] = 32;
  EXPECT_EQ(kern::Ps5AioInit(param, sizeof(param), 0, 0), 0);
  EXPECT_EQ(kern::Ps5AioInit(nullptr, 0, 0, 0), -16);  // EBUSY
}
}  // namespace
