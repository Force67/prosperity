#include <array>
#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak.h>
#endif

#include "guest/session.h"
#include "kern/lv2/dispatch.h"
#include "kern/lv2/sys_vfs.h"
#include "kern/process.h"
#include "kern/ps4/dev/pipe_dev.h"

namespace {
class PipeDeviceTest : public testing::Test {
 protected:
  void SetUp() override {
    guest::BeginSession();
    kern::CreatePipe(objects, reader, writer);
    read_ref = objects.Get(reader);
    write_ref = objects.Get(writer);
    read = static_cast<kern::PipeDevice*>(read_ref.get());
    write = static_cast<kern::PipeDevice*>(write_ref.get());
  }
  void TearDown() override { guest::BeginSession(); }
  kern::ObjectTable objects;
  u32 reader, writer;
  kern::ObjectRef<kern::Object> read_ref, write_ref;
  kern::PipeDevice *read, *write;
};

TEST_F(PipeDeviceTest, DataReadinessAndDirectionalAccess) {
  const std::array<u8, 4> input{1, 3, 7, 11};
  std::array<u8, 8> output{};
  EXPECT_EQ(read->Poll(1), 0);
  EXPECT_EQ(write->Poll(4), 4);
  EXPECT_EQ(read->Write(input.data(), input.size()), -kern::SysError::eBADF);
  EXPECT_EQ(write->Read(output.data(), output.size()), -kern::SysError::eBADF);
  ASSERT_EQ(write->Write(input.data(), input.size()), 4);
  EXPECT_EQ(read->Poll(1), 1);
  ASSERT_EQ(read->Read(output.data(), output.size()), 4);
  EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
  EXPECT_EQ(read->Poll(1), 0);
  EXPECT_EQ(read->Read(nullptr, 0), 0);
  EXPECT_EQ(write->Write(nullptr, 0), 0);
  EXPECT_EQ(write->Write(nullptr, 1), -kern::SysError::eFAULT);
  EXPECT_EQ(read->Read(nullptr, 1), -kern::SysError::eFAULT);
}

TEST_F(PipeDeviceTest,
       NonblockingWritesAreAtomicAndSpaceBecomesReadyAfterRead) {
  read->SetFlags(4);
  write->SetFlags(4);
  EXPECT_EQ(read->GetFlags(), 4u);
  EXPECT_EQ(write->GetFlags(), 5u);
  u8 byte = 0;
  EXPECT_EQ(read->Read(&byte, 1), -kern::SysError::eAGAIN);
  std::vector<u8> full(16384, 0x5a);
  ASSERT_EQ(write->Write(full.data(), full.size()), 16384);
  EXPECT_EQ(write->Poll(4), 0);
  EXPECT_EQ(write->Write(&byte, 1), -kern::SysError::eAGAIN);
  ASSERT_EQ(read->Read(&byte, 1), 1);
  EXPECT_EQ(byte, 0x5a);
  const u8 pair[] = {1, 2};
  EXPECT_EQ(write->Write(pair, 2), -kern::SysError::eAGAIN);
  ASSERT_EQ(read->Read(&byte, 1), 1);
  ASSERT_EQ(write->Write(pair, 2), 2);
  ASSERT_EQ(read->Read(full.data(), full.size()), 16384);
  EXPECT_EQ(full[16382], 1);
  EXPECT_EQ(full[16383], 2);
}

TEST_F(PipeDeviceTest, ClosingWriterPreservesBufferedBytesThenReturnsEof) {
  u8 byte = 42;
  ASSERT_EQ(write->Write(&byte, 1), 1);
  write_ref.Reset();
  ASSERT_TRUE(objects.Release(writer));
  EXPECT_EQ(read->Poll(1), 0x11);
  ASSERT_EQ(read->Read(&byte, 1), 1);
  EXPECT_EQ(byte, 42);
  EXPECT_EQ(read->Read(&byte, 1), 0);
  EXPECT_EQ(read->Poll(1), 0x10);
}

TEST_F(PipeDeviceTest, ClosingReaderBreaksWrites) {
  read_ref.Reset();
  ASSERT_TRUE(objects.Release(reader));
  u8 byte = 42;
  EXPECT_EQ(write->Write(&byte, 1), -kern::SysError::ePIPE);
  EXPECT_EQ(write->Poll(4) & 0x10, 0x10);
}

TEST_F(PipeDeviceTest, BlockingReaderWakesForDataAndSessionStop) {
  std::atomic<bool> entered{false};
  i64 result = 0;
  u8 byte = 0;
  std::thread worker([&] {
    entered = true;
    result = read->Read(&byte, 1);
  });
  while (!entered)
    std::this_thread::yield();
  const u8 value = 123;
  EXPECT_EQ(write->Write(&value, 1), 1);
  worker.join();
  EXPECT_EQ(result, 1);
  EXPECT_EQ(byte, value);
  worker = std::thread([&] { result = read->Read(&byte, 1); });
  guest::RequestStop();
  worker.join();
  EXPECT_EQ(result, static_cast<i64>(-kern::SysError::eINTR));
}

TEST_F(PipeDeviceTest, BlockedWriterWakesWhenReaderCloses) {
  std::vector<u8> full(16384, 0x5a);
  ASSERT_EQ(write->Write(full.data(), full.size()), 16384);
  i64 result = 0;
  std::thread worker([&] { result = write->Write(full.data(), 1); });
  read_ref.Reset();
  EXPECT_TRUE(objects.Release(reader));
  worker.join();
  EXPECT_EQ(result, static_cast<i64>(-kern::SysError::ePIPE));
}

TEST(PipeSyscall, FcntlPollReadWriteAndCloseUseGuestDescriptors) {
  guest::BeginSession();
  kern::Process process;
  const u64 pair = kern::sys_pipe();
  const u32 reader = static_cast<u32>(pair), writer = pair >> 32;
  ASSERT_NE(reader, writer);
  ASSERT_EQ(kern::sys_fcntl(reader, 4, 4), 0);
  EXPECT_EQ(kern::sys_fcntl(reader, 3, 0), 4);
  struct PollFd {
    i32 fd;
    u16 events, returned;
  };
  PollFd entries[] = {{static_cast<i32>(reader), 1, 0xa5a5},
                      {static_cast<i32>(writer), 4, 0xa5a5},
                      {-1, 1, 0xa5a5}};
  ASSERT_EQ(kern::sys_poll(entries, 3, 0), 1);
  EXPECT_EQ(entries[0].returned, 0);
  EXPECT_EQ(entries[1].returned, 4);
  EXPECT_EQ(entries[2].returned, 0);
  u8 byte = 77;
  ASSERT_EQ(kern::sys_write(writer, &byte, 1), 1);
  ASSERT_EQ(kern::sys_poll(entries, 3, 0), 2);
  EXPECT_EQ(entries[0].returned, 1);
  ASSERT_EQ(kern::sys_read(reader, &byte, 1), 1);
  EXPECT_EQ(byte, 77);
  EXPECT_EQ(kern::sys_poll(entries, 1, 2), 0);
  EXPECT_EQ(kern::sys_poll(nullptr, 1, 0), -kern::SysError::eFAULT);
  EXPECT_EQ(kern::sys_poll(nullptr, 65537, 0), -kern::SysError::eINVAL);
  ASSERT_EQ(kern::sys_close(writer), 0);
  ASSERT_EQ(kern::sys_poll(entries, 3, 0), 2);
  EXPECT_EQ(entries[0].returned, 0x10);
  EXPECT_EQ(entries[1].returned, 0x20);
  ASSERT_EQ(kern::sys_close(reader), 0);
  EXPECT_EQ(kern::sys_poll(entries, 1, 0), 1);
  EXPECT_EQ(entries[0].returned, 0x20);
}

#if defined(DELTA_BACKEND_NATIVE)
bool g_pipe_error = false;
u64 PS4ABI PipeResult() {
  return g_pipe_error ? -kern::SysError::eMFILE : 17 | (u64(29) << 32);
}

TEST(PipeSyscall, BridgePreservesWrapperPointerAndReturnsBothDescriptors) {
  for (bool failure : {false, true}) {
    g_pipe_error = failure;
    const auto target =
        kern::Lv2Trampoline(reinterpret_cast<void*>(&PipeResult), 42);
    Xbyak::CodeGenerator wrapper;
    wrapper.sub(wrapper.rsp, 8);
    wrapper.mov(wrapper.rax, target);
    wrapper.call(wrapper.rax);
    Xbyak::Label error;
    wrapper.jc(error);
    wrapper.mov(wrapper.dword[wrapper.rdi], wrapper.eax);
    wrapper.mov(wrapper.dword[wrapper.rdi + 4], wrapper.edx);
    wrapper.xor_(wrapper.eax, wrapper.eax);
    wrapper.L(error);
    wrapper.add(wrapper.rsp, 8);
    wrapper.ret();
    u32 descriptors[] = {0xa5a5a5a5, 0xa5a5a5a5};
    auto invoke = wrapper.getCode<int (*)(u32*)>();
    if (failure) {
      EXPECT_EQ(invoke(descriptors), 24);
      EXPECT_EQ(descriptors[0], 0xa5a5a5a5);
      EXPECT_EQ(descriptors[1], 0xa5a5a5a5);
    } else {
      ASSERT_EQ(invoke(descriptors), 0);
      EXPECT_EQ(descriptors[0], 17u);
      EXPECT_EQ(descriptors[1], 29u);
    }
  }
}
#endif
}  // namespace
