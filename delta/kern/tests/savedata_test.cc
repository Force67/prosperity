#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <thread>
#include <tuple>
#include <vector>

#include "base/option_file.h"
#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_procid.h"
#include "kern/lv2/sys_semaphore.h"
#include "kern/lv2/sys_vfs.h"
#include "kern/process.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/vfs.h"
#include "options/options.h"

namespace {
class SaveDataTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::BeginSession();
    char path[] = "/tmp/prosperity-savedata-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    root = path;
    base::FindOption("DELTA_SAVEDATA_DIR")->SetFromString(root.c_str());
    kern::vfs::SetTitleId("CUSA00792");
    const char* name = "SceSaveData";
    u64 request = reinterpret_cast<u64>(name);
    ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)),
              0);
    EXPECT_EQ(Initialize(), 0);
    u32 command = 60, user = 1, mode = 6;
    u64 blocks = 1024;
    std::memcpy(header.data() + 4, &command, 4);
    std::memcpy(data.data() + 0x400, &user, 4);
    std::memcpy(data.data() + 0x414, "TESTSAVE", 9);
    std::memcpy(data.data() + 0x488, &blocks, 8);
    std::memcpy(data.data() + 0x490, &mode, 4);
    std::memcpy(data.data() + 0x494, &mode, 4);
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    process.GetObjTable().Reset();
    guest::ResetSessionResources();
    kern::vfs::SetTitleId("");
    base::FindOption("DELTA_SAVEDATA_DIR")->Reset();
    std::filesystem::remove_all(root);
    guest::BeginSession();
  }
  struct Buffer {
    const void* data;
    u64 size;
  };
  i32 Invoke(u32 method,
             std::initializer_list<Buffer> inputs = {},
             std::initializer_list<Buffer> outputs = {}) {
    std::array<u64, 8> in{};
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
  i32 Initialize(u32 command = 56) {
    u64 input[] = {command, 4, command == 56 ? 4u : 140u};
    u32 payload = 0;
    std::array<u8, 140> output;
    output.fill(0xa5);
    u64 written = ~u64(0);
    return Invoke(0x12340002, {{input, 24}, {&payload, 4}},
                  {{output.data(), input[2]}, {&written, 8}});
  }
  i32 Submit(u64 size = 0x4a0) {
    u32 output = ~0u;
    return Invoke(0x12340000,
                  {{&size, 8}, {header.data(), 128}, {data.data(), size}},
                  {{&output, 4}});
  }
  i32 Collect(u64 capacity = 32, u64 size = 32) {
    return Invoke(0x12340001, {{&size, 8}, {header.data(), 132}},
                  {{reply.data(), capacity}, {&written, 8}});
  }
  i32 MemorySetup(u64 size = 4096, u64 icon = 0, u64 param = 0, u32 count = 1) {
    std::array<u8, 0x558> payload{};
    const u32 command = 33, user = 1;
    const u64 input_size = payload.size(), hint = 0x880000000;
    std::memcpy(header.data() + 4, &command, 4);
    std::memcpy(payload.data(), &user, 4);
    std::memcpy(payload.data() + 8, &size, 8);
    std::memcpy(payload.data() + 16, &icon, 8);
    std::memcpy(payload.data() + 24, &param, 8);
    std::memcpy(payload.data() + 40, &hint, 8);
    std::memcpy(payload.data() + 48, &count, 4);
    if (param)
      std::memset(payload.data() + 0x34, 0xa5, 0x524);
    u32 submitted = ~0u;
    if (const auto result = Invoke(0x12340000,
                                   {{&input_size, 8},
                                    {header.data(), 128},
                                    {payload.data(), payload.size()}},
                                   {{&submitted, 4}}))
      return result;
    return Collect();
  }
  i32 MemoryClear(u32 user = 1) {
    const u64 input[] = {34, 4, 0};
    u64 received = ~u64(0);
    const auto result = Invoke(0x12340002, {{input, 24}, {&user, 4}},
                               {{nullptr, 0}, {&received, 8}});
    if (!result)
      EXPECT_EQ(received, 0u);
    return result;
  }
  std::array<u8, 0x590> SlotPayload(const char* dir, u32 slot) {
    std::array<u8, 0x590> payload{};
    const u32 user = 1, count = 2;
    const u64 size = 4096, hint = 0x880000000;
    std::memcpy(payload.data(), &user, 4);
    std::memcpy(payload.data() + 8, &size, 8);
    std::memcpy(payload.data() + 40, &hint, 8);
    std::memcpy(payload.data() + 48, &count, 4);
    std::snprintf(reinterpret_cast<char*>(payload.data() + 0x558), 16, "%s",
                  dir);
    payload[0x568] = 1;
    std::memcpy(payload.data() + 0x56c, &slot, 4);
    return payload;
  }
  i32 SlotSetup(const std::array<u8, 0x590>& payload) {
    const u32 command = 78;
    const u64 size = payload.size();
    std::memcpy(header.data() + 4, &command, 4);
    u32 submitted = ~0u;
    if (const auto result =
            Invoke(0x12340000,
                   {{&size, 8}, {header.data(), 128}, {payload.data(), size}},
                   {{&submitted, 4}}))
      return result;
    return Collect();
  }
  i32 SlotClear(const char* dir, u32 user = 1) {
    const u64 input[] = {81, 60, 0};
    std::array<u8, 60> payload{};
    std::memcpy(payload.data(), &user, 4);
    std::snprintf(reinterpret_cast<char*>(payload.data() + 4), 16, "%s", dir);
    u64 received = ~u64(0);
    const auto result = Invoke(0x12340002, {{input, 24}, {payload.data(), 60}},
                               {{nullptr, 0}, {&received, 8}});
    if (!result)
      EXPECT_EQ(received, 0u);
    return result;
  }
  i32 SlotConfirm(const char* dir,
                  u32 slot,
                  u32 app = 0,
                  u32 user = 1,
                  u64 size = 4096) {
    const u32 command = 80;
    std::memcpy(header.data() + 4, &command, 4);
    std::fill(data.begin(), data.begin() + 72, 0);
    std::memcpy(data.data(), &app, 4);
    std::memcpy(data.data() + 4, &user, 4);
    std::memcpy(data.data() + 8, &size, 8);
    std::snprintf(reinterpret_cast<char*>(data.data() + 16), 16, "%s", dir);
    data[32] = 1;
    std::memcpy(data.data() + 36, &slot, 4);
    if (const auto result = Submit(72))
      return result;
    return Collect();
  }
  i32 MemorySubscribe(const char* dir, bool add = true) {
    const u32 command = add ? 51 : 52;
    const u32 pid = kern::sys_getpid(), user = 1, flags = 4;
    const u64 size = 4096;
    std::memcpy(header.data() + 4, &command, 4);
    std::fill(data.begin(), data.begin() + 96, 0);
    std::memcpy(data.data(), &pid, 4);
    std::memcpy(data.data() + (add ? 8 : 4), &user, 4);
    std::snprintf(reinterpret_cast<char*>(data.data() + (add ? 28 : 24)), 32,
                  "sce_%s", dir);
    if (add) {
      std::memcpy(data.data() + 72, &flags, 4);
      std::memcpy(data.data() + 80, &size, 8);
    }
    if (const auto result = Submit(add ? 96 : 56))
      return result;
    return Collect();
  }
  std::string MemoryEventName() {
    char name[64];
    std::snprintf(name, sizeof(name), "SceSaveDataMemoryRUI%08d",
                  kern::sys_getpid());
    return name;
  }
  i32 MemoryConfirm(u32 app = 0, u32 user = 1, u64 size = 4096) {
    const u32 command = 47;
    std::memcpy(header.data() + 4, &command, 4);
    std::memcpy(data.data(), &app, 4);
    std::memcpy(data.data() + 4, &user, 4);
    std::memcpy(data.data() + 8, &size, 8);
    if (const auto result = Submit(16))
      return result;
    return Collect();
  }
  i32 MemoryCreate(u64 bytes = 0x300000, u32 user = 1, u32 reserved = 0) {
    const u32 command = 35;
    std::memcpy(header.data() + 4, &command, 4);
    std::memcpy(data.data(), &user, 4);
    std::memcpy(data.data() + 4, &reserved, 4);
    std::memcpy(data.data() + 8, &bytes, 8);
    if (const auto result = Submit(16))
      return result;
    return Collect();
  }
  i32 MemoryAcknowledgement(i32& result, u64& received, u64 capacity = 4) {
    const u64 request[] = {0, reinterpret_cast<u64>(&result),
                           reinterpret_cast<u64>(&received), capacity};
    u32 manager_result = ~0u;
    const auto status = kern::ipmi::ManagerCall(
        594, kid, &manager_result, const_cast<u64*>(request), sizeof(request));
    EXPECT_EQ(manager_result, status ? ~0u : 0);
    return status;
  }
  i32 MemorySync(u32 user = 1,
                 u32 mode = 0,
                 const char* dir = "sdmemory",
                 bool acknowledge = true) {
    const u64 input[] = {42, 84, 0};
    std::array<u8, 84> payload{};
    std::memcpy(payload.data(), &user, 4);
    std::snprintf(reinterpret_cast<char*>(payload.data() + 4), 16, "%s", dir);
    std::memcpy(payload.data() + 20, &mode, 4);
    u64 received = ~u64(0);
    const auto result = Invoke(0x12340002, {{input, 24}, {payload.data(), 84}},
                               {{nullptr, 0}, {&received, 8}});
    EXPECT_EQ(received, result ? ~u64(0) : 0);
    if (!result && mode == 1 && acknowledge) {
      i32 acknowledgement = -1;
      received = ~u64(0);
      EXPECT_EQ(MemoryAcknowledgement(acknowledgement, received), 0);
      EXPECT_EQ(acknowledgement, 0);
      EXPECT_EQ(received, 4u);
    }
    return result;
  }
  i32 MemoryResult(std::array<u8, 60>& output,
                   u64& received,
                   u64 capacity = 60,
                   u32 payload = 0) {
    const u64 input[] = {43, 4, 60};
    return Invoke(0x12340002, {{input, 24}, {&payload, 4}},
                  {{output.data(), capacity}, {&received, 8}});
  }
  i32 WaitMemoryResult(std::array<u8, 60>& output, u64& received) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    i32 result;
    do {
      result = MemoryResult(output, received);
      if (static_cast<u32>(result) != 0x809f0018u)
        return result;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return result;
  }
  void Mode(u32 value) {
    std::memcpy(data.data() + 0x494, &value, 4);
    if (value == 1)
      value = 9;
    if (value == 4)
      value = 6;
    if (value == 32 || value == 34)
      value = 38;
    std::memcpy(data.data() + 0x490, &value, 4);
  }
  i32 Param(u32 type,
            std::vector<u8>& output,
            const void* value = nullptr,
            u32 length = 0,
            u64 capacity = 0) {
    const bool write = value != nullptr;
    const u64 input_size = write ? 0x1c818 : 148;
    const u64 output_size = write ? 128 : 0x1c804;
    std::vector<u8> payload(input_size);
    std::memcpy(payload.data(), reply.data(), 16);
    std::memcpy(payload.data() + 16, &type, 4);
    if (write) {
      std::memcpy(payload.data() + 20, value, length);
      std::memcpy(payload.data() + 0x1c814, &length, 4);
    }
    std::array<u8, 132> param_header{};
    const u32 command = write ? 9 : 10;
    std::memcpy(param_header.data() + 4, &command, 4);
    u32 submitted = ~0u;
    const auto result = Invoke(0x12340000,
                               {{&input_size, 8},
                                {param_header.data(), 128},
                                {payload.data(), input_size}},
                               {{&submitted, 4}});
    if (result)
      return result;
    output.assign(output_size + 16, 0xa5);
    u64 received = ~u64(0);
    return Invoke(
        0x12340001, {{&output_size, 8}, {param_header.data(), 132}},
        {{output.data(), capacity ? capacity : output_size}, {&received, 8}});
  }
  i32 Icon(std::vector<u8>& output,
           const void* value = nullptr,
           u32 length = 0,
           u64 capacity = 0) {
    const bool write = value != nullptr;
    const u64 input_size = write ? 0x1c814 : 144;
    const u64 output_size = write ? 128 : 0x1c884;
    std::vector<u8> payload(input_size);
    std::memcpy(payload.data(), reply.data(), 16);
    if (write) {
      if (length <= 0x1c800)
        std::memcpy(payload.data() + 16, value, length);
      std::memcpy(payload.data() + 0x1c810, &length, 4);
    }
    std::array<u8, 132> icon_header{};
    const u32 command = write ? 7 : 8;
    std::memcpy(icon_header.data() + 4, &command, 4);
    u32 submitted = ~0u;
    const auto result = Invoke(0x12340000,
                               {{&input_size, 8},
                                {icon_header.data(), 128},
                                {payload.data(), input_size}},
                               {{&submitted, 4}});
    if (result)
      return result;
    output.assign(output_size + 16, 0xa5);
    u64 received = ~u64(0);
    return Invoke(
        0x12340001, {{&output_size, 8}, {icon_header.data(), 132}},
        {{output.data(), capacity ? capacity : output_size}, {&received, 8}});
  }
  i32 Delete() {
    const u64 input_size = 0x434, output_size = 128;
    std::array<u8, 132> delete_header{};
    const u32 command = 4;
    std::memcpy(delete_header.data() + 4, &command, 4);
    u32 submitted = ~0u;
    const auto result = Invoke(0x12340000,
                               {{&input_size, 8},
                                {delete_header.data(), 128},
                                {data.data(), input_size}},
                               {{&submitted, 4}});
    if (result)
      return result;
    std::array<u8, 144> output;
    output.fill(0xa5);
    u64 received = ~u64(0);
    const auto collected =
        Invoke(0x12340001, {{&output_size, 8}, {delete_header.data(), 132}},
               {{output.data(), output_size}, {&received, 8}});
    for (size_t i = 128; i < output.size(); ++i)
      EXPECT_EQ(output[i], 0xa5);
    if (!collected)
      EXPECT_EQ(received, 128u);
    return collected;
  }
  std::vector<u8> BackupPayload(u32 command, const char* dir = "TESTSAVE") {
    std::vector<u8> payload(command == 36 ? 136 : command == 37 ? 56 : 60);
    const u32 user = 1;
    std::memcpy(payload.data(), &user, 4);
    if (command == 46)
      std::memcpy(payload.data() + 4, &user, 4);
    std::snprintf(
        reinterpret_cast<char*>(payload.data() + (command == 46 ? 24 : 20)), 32,
        "%s", dir);
    return payload;
  }
  i32 BackupSubmit(u32 command, const std::vector<u8>& payload) {
    std::array<u8, 128> request{};
    std::memcpy(request.data() + 4, &command, 4);
    const u64 size = payload.size();
    u32 submitted = ~0u;
    return Invoke(0x12340000,
                  {{&size, 8},
                   {request.data(), request.size()},
                   {payload.data(), payload.size()}},
                  {{&submitted, 4}});
  }
  i32 BackupCollect(u32 command, std::vector<u8>& output, u64 capacity = 0) {
    std::array<u8, 132> request{};
    std::memcpy(request.data() + 4, &command, 4);
    const u64 size = command == 36 ? 36 : command == 37 ? 0x1cd40 : 32;
    output.assign(size + 16, 0xa5);
    written = ~u64(0);
    const auto result =
        Invoke(0x12340001, {{&size, 8}, {request.data(), request.size()}},
               {{output.data(), capacity ? capacity : size}, {&written, 8}});
    EXPECT_EQ(written, result ? ~u64(0) : size);
    EXPECT_TRUE(std::all_of(output.begin() + size, output.end(),
                            [](u8 value) { return value == 0xa5; }));
    if (result)
      EXPECT_TRUE(std::all_of(output.begin(), output.end(),
                              [](u8 value) { return value == 0xa5; }));
    return result;
  }
  i32 BackupCommand(u32 command,
                    const std::vector<u8>& payload,
                    std::vector<u8>& output) {
    if (const auto result = BackupSubmit(command, payload))
      return result;
    return BackupCollect(command, output);
  }
  i32 Search(std::vector<u8>& output,
             const char* filter = "",
             u32 key = 0,
             u32 order = 0,
             u32 offset = 0) {
    const u64 input_size = 192, output_size = 0x11308;
    std::array<u8, 192> payload{};
    const u32 user = 1;
    std::memcpy(payload.data(), &user, 4);
    std::memcpy(payload.data() + 20, filter, std::strlen(filter) + 1);
    std::memcpy(payload.data() + 52, &key, 4);
    std::memcpy(payload.data() + 56, &order, 4);
    std::memcpy(payload.data() + 60, &offset, 4);
    std::array<u8, 132> search_header{};
    const u32 command = 6;
    std::memcpy(search_header.data() + 4, &command, 4);
    u32 submitted = ~0u;
    const auto result = Invoke(0x12340000,
                               {{&input_size, 8},
                                {search_header.data(), 128},
                                {payload.data(), input_size}},
                               {{&submitted, 4}});
    if (result)
      return result;
    output.assign(output_size + 16, 0xa5);
    u64 received = ~u64(0);
    const auto collected =
        Invoke(0x12340001, {{&output_size, 8}, {search_header.data(), 132}},
               {{output.data(), output_size}, {&received, 8}});
    for (size_t i = output_size; i < output.size(); ++i)
      EXPECT_EQ(output[i], 0xa5);
    if (!collected)
      EXPECT_EQ(received, output_size);
    return collected;
  }
  i32 MountInfo(std::vector<u8>& output) {
    const u64 input_size = 16, output_size = 176;
    std::array<u8, 132> info_header{};
    const u32 command = 3;
    std::memcpy(info_header.data() + 4, &command, 4);
    u32 submitted = ~0u;
    const auto result = Invoke(0x12340000,
                               {{&input_size, 8},
                                {info_header.data(), 128},
                                {reply.data(), input_size}},
                               {{&submitted, 4}});
    if (result)
      return result;
    output.assign(output_size + 16, 0xa5);
    u64 received = ~u64(0);
    const auto collected =
        Invoke(0x12340001, {{&output_size, 8}, {info_header.data(), 132}},
               {{output.data(), output_size}, {&received, 8}});
    for (size_t i = output_size; i < output.size(); ++i)
      EXPECT_EQ(output[i], 0xa5);
    if (!collected)
      EXPECT_EQ(received, output_size);
    return collected;
  }
  std::filesystem::path root;
  i32 SmallMount() {
    const u64 blocks = 96;
    std::memcpy(data.data() + 0x488, &blocks, 8);
    if (const auto result = Submit())
      return result;
    return Collect();
  }
  i32 TransactionResource(u32 command, u32 value, u32* resource = nullptr) {
    const u64 output_size = command == 88 ? 68 : 64;
    const u64 request[] = {command, 68, output_size};
    std::array<u8, 68> payload{};
    std::memcpy(payload.data(), &value, sizeof(value));
    std::array<u8, 80> output;
    output.fill(0xa5);
    u64 received = ~u64(0);
    const auto result =
        Invoke(0x12340002,
               {{request, sizeof(request)}, {payload.data(), payload.size()}},
               {{output.data(), output_size}, {&received, 8}});
    if (!result) {
      EXPECT_EQ(received, output_size);
      if (resource)
        std::memcpy(resource, output.data(), sizeof(*resource));
      for (size_t i = 4; i < output_size; ++i)
        EXPECT_EQ(output[i], 0u);
      for (size_t i = output_size; i < output.size(); ++i)
        EXPECT_EQ(output[i], 0xa5);
    }
    return result;
  }
  i32 CommitTransaction(u32 resource, u32 mode = 0, u8 reserved = 0) {
    data.fill(0);
    const u32 command = 91;
    std::memcpy(header.data() + 4, &command, 4);
    std::memcpy(data.data(), &resource, 4);
    std::memcpy(data.data() + 4, &mode, 4);
    data[8] = reserved;
    if (const auto result = Submit(72))
      return result;
    reply.fill(0xa5);
    const auto result = Collect(64, 64);
    if (result)
      return result;
    EXPECT_EQ(written, 64u);
    for (size_t i = 0; i < 64; ++i)
      EXPECT_EQ(reply[i], 0u);
    for (size_t i = 64; i < reply.size(); ++i)
      EXPECT_EQ(reply[i], 0xa5);
    return result;
  }
  i32 Unmount(u32 backup = 0) {
    const u64 input_size = 20, output_size = 132;
    std::array<u8, 132> unmount_header{};
    std::array<u8, 20> payload{};
    std::memcpy(payload.data(), reply.data(), 16);
    std::memcpy(payload.data() + 16, &backup, 4);
    const u32 command = 2;
    std::memcpy(unmount_header.data() + 4, &command, 4);
    u32 submitted = ~0u;
    if (const auto result = Invoke(0x12340000,
                                   {{&input_size, 8},
                                    {unmount_header.data(), 128},
                                    {payload.data(), input_size}},
                                   {{&submitted, 4}}))
      return result;
    std::array<u8, 148> output;
    output.fill(0xa5);
    u64 received = ~u64(0);
    const auto result =
        Invoke(0x12340001, {{&output_size, 8}, {unmount_header.data(), 132}},
               {{output.data(), output_size}, {&received, 8}});
    for (size_t i = output_size; i < output.size(); ++i)
      EXPECT_EQ(output[i], 0xa5);
    if (!result)
      EXPECT_EQ(received, output_size);
    return result;
  }
  u64 FreeBlocks() {
    std::vector<u8> output;
    const auto result = MountInfo(output);
    EXPECT_EQ(result, 0);
    if (result || output.size() < 16)
      return ~u64(0);
    u64 free;
    std::memcpy(&free, output.data() + 8, 8);
    return free;
  }
  kern::Process process;
  u32 kid = 0;
  std::array<u8, 132> header{};
  std::array<u8, 0x4a0> data{};
  std::array<u8, 148> reply{};
  u64 written = 0;
};

TEST_F(SaveDataTest, Ps5TransactionCommitPreservesNativeSaveWrites) {
  u32 resource = ~0u;
  ASSERT_EQ(TransactionResource(88, 0xc0000, &resource), 0);
  ASSERT_EQ(SmallMount(), 0);
  const int writer =
      kern::sys_open("/savedata0/USR-DATA", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(writer, 0);
  const char payload[] = "options profile";
  ASSERT_EQ(kern::sys_write(writer, payload, sizeof(payload)), sizeof(payload));
  ASSERT_EQ(kern::sys_close(writer), 0);
  ASSERT_EQ(CommitTransaction(resource), 0);
  ASSERT_EQ(CommitTransaction(resource, 1), 0);
  std::ifstream saved(root / "CUSA00792" / "TESTSAVE" / "USR-DATA",
                      std::ios::binary);
  ASSERT_TRUE(saved);
  std::array<char, sizeof(payload)> loaded{};
  saved.read(loaded.data(), loaded.size());
  EXPECT_EQ(std::memcmp(loaded.data(), payload, sizeof(payload)), 0);
  ASSERT_EQ(TransactionResource(89, resource), 0);
  EXPECT_EQ(static_cast<u32>(CommitTransaction(resource)), 0x809f0000u);
}

TEST_F(SaveDataTest, Ps5TransactionResourcesAreDistinctAndOwnedByClient) {
  u32 first, second;
  ASSERT_EQ(TransactionResource(88, 0xc0000, &first), 0);
  ASSERT_EQ(TransactionResource(88, 0xc0000, &second), 0);
  ASSERT_NE(first, second);
  const u32 owner = kid;
  const char* name = "SceSaveData";
  u64 request = reinterpret_cast<u64>(name);
  ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)), 0);
  ASSERT_EQ(Initialize(), 0);
  EXPECT_EQ(static_cast<u32>(CommitTransaction(first)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(TransactionResource(89, first)), 0x809f0000u);
  kid = owner;
  ASSERT_EQ(CommitTransaction(first), 0);
  ASSERT_EQ(TransactionResource(89, first), 0);
  EXPECT_EQ(static_cast<u32>(TransactionResource(89, first)), 0x809f0000u);
  ASSERT_EQ(CommitTransaction(second), 0);
}

TEST_F(SaveDataTest, Ps5TransactionCommitRejectsInvalidModeAndReservedBytes) {
  u32 resource;
  ASSERT_EQ(TransactionResource(88, 0xc0000, &resource), 0);
  EXPECT_EQ(static_cast<u32>(CommitTransaction(resource, 2)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(CommitTransaction(resource, 0, 1)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(CommitTransaction(~0u)), 0x809f0000u);
  ASSERT_EQ(CommitTransaction(resource), 0);
}

TEST_F(SaveDataTest, MemorySetupProvisionsNativeBuffersControlAndSemaphore) {
  ASSERT_EQ(MemoryClear(), 0);
  ASSERT_EQ(MemorySetup(4096, 128, 0x530, 2), 0);
  for (const char* name :
       {"/SceSaveData0000000000000001_0", "/SceSaveData0000000000000001_1"}) {
    const int fd = kern::sys_shm_open(name, 2, 0777);
    ASSERT_GE(fd, 0);
    const size_t total = 4096 + 64 + 128 + 0x530;
    EXPECT_EQ(kern::ShmFstatSize(fd), total);
    auto* view = kern::sys_mmap(nullptr, 0x4000, 3, 1, fd, 0);
    ASSERT_FALSE(kern::IsErrnoPtr(view));
    u64 capacity;
    std::memcpy(&capacity, view + 4096, 8);
    EXPECT_EQ(capacity, 128u);
    for (size_t i = 0; i < 4096; ++i)
      EXPECT_EQ(view[i], 0);
    for (size_t i = 4096 + 8; i < total - 0x530; ++i)
      EXPECT_EQ(view[i], 0);
    for (size_t i = total - 0x530; i < total - 12; ++i)
      EXPECT_EQ(view[i], 0xa5);
    for (size_t i = total - 12; i < total; ++i)
      EXPECT_EQ(view[i], 0);
    EXPECT_EQ(kern::sys_close(fd), 0);
  }
  std::array<u8, 64> control;
  control.fill(0xa5);
  ASSERT_EQ(kern::ShmRead("/SceSaveDataI0000000000000001", 0, control.data(),
                          control.size()),
            0);
  for (u8 value : control)
    EXPECT_EQ(value, 0);
  EXPECT_EQ(
      kern::ShmRead("/SceSaveDataI0000000000000001", 64, control.data(), 1),
      -kern::SysError::eINVAL);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  EXPECT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  EXPECT_EQ(kern::sys_osem_trywait(sem, 1), -kern::SysError::eBUSY);
  EXPECT_EQ(kern::sys_osem_post(sem, 1), 0);
  EXPECT_EQ(kern::sys_osem_post(sem, 1), -kern::SysError::eINVAL);
  ASSERT_EQ(MemoryClear(), 0);
  EXPECT_EQ(kern::sys_osem_trywait(sem, 1), -kern::SysError::eSRCH);
  EXPECT_EQ(
      kern::ShmRead("/SceSaveDataI0000000000000001", 0, control.data(), 1),
      -kern::SysError::eNOENT);
}

TEST_F(SaveDataTest, MissingMemoryResourcesAreNotCreatedByOpen) {
  for (const char* name :
       {"/SceSaveData0000000000000001_0", "/SceSaveDataI0000000000000001",
        "/SceSaveData0000000000000001_0_0", "/SceSaveDataI0000000000000001_0"})
    EXPECT_EQ(kern::sys_shm_open(name, 2, 0777), -kern::SysError::eNOENT);
  for (const char* name :
       {"SceSaveData0000000000000001", "SceSaveData0000000000000001_0"})
    EXPECT_EQ(kern::sys_osem_open(name), -kern::SysError::eNOENT);
}

TEST_F(SaveDataTest, MemorySlotCollectionCapacityFailureCanBeRetried) {
  const auto payload = SlotPayload("sdmemory", 0);
  u64 size = payload.size();
  const u32 command = 78;
  std::memcpy(header.data() + 4, &command, 4);
  u32 submitted = 117;
  EXPECT_EQ(static_cast<u32>(Invoke(
                0x12340000,
                {{&size, 8}, {header.data(), 128}, {payload.data(), size - 1}},
                {{&submitted, 4}})),
            0x809f0000u);
  EXPECT_EQ(submitted, 117u);
  ASSERT_EQ(Invoke(0x12340000,
                   {{&size, 8}, {header.data(), 128}, {payload.data(), size}},
                   {{&submitted, 4}}),
            0);
  written = 117;
  reply.fill(0xa5);
  EXPECT_EQ(static_cast<u32>(Collect(31)), 0x809f0000u);
  EXPECT_EQ(written, 117u);
  EXPECT_EQ(reply[0], 0xa5);
  EXPECT_EQ(kern::sys_shm_open("/SceSaveData0000000000000001_0_0", 2, 0777),
            -kern::SysError::eNOENT);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(written, 32u);
  u8 value = 117;
  ASSERT_EQ(kern::ShmRead("/SceSaveData0000000000000001_0_0", 0, &value, 1), 0);
  EXPECT_EQ(value, 0);
}

TEST_F(SaveDataTest, MemorySlotsHaveIndependentNativeBuffersAndSemaphores) {
  ASSERT_EQ(MemorySetup(), 0);
  for (u32 slot = 0; slot < 4; ++slot) {
    char dir[16], name[64], semaphore[64];
    std::snprintf(dir, sizeof(dir), slot ? "sdmemory%u" : "sdmemory", slot);
    ASSERT_EQ(SlotClear(dir), 0);
    ASSERT_EQ(SlotSetup(SlotPayload(dir, slot)), 0);
    EXPECT_EQ(written, 32u);
    for (u32 buffer = 0; buffer < 2; ++buffer) {
      std::snprintf(name, sizeof(name), "/SceSaveData0000000000000001_%u_%u",
                    slot, buffer);
      const int fd = kern::sys_shm_open(name, 2, 0777);
      ASSERT_GE(fd, 0);
      EXPECT_EQ(kern::ShmFstatSize(fd), 4096u);
      ASSERT_EQ(kern::sys_close(fd), 0);
      std::array<u8, 4096> bytes;
      bytes.fill(0xa5);
      ASSERT_EQ(kern::ShmRead(name, 0, bytes.data(), bytes.size()), 0);
      EXPECT_EQ(bytes, (std::array<u8, 4096>{}));
      bytes.fill(slot + 1);
      ASSERT_EQ(kern::ShmWrite(name, 0, bytes.data(), bytes.size()), 0);
    }
    std::snprintf(name, sizeof(name), "/SceSaveDataI0000000000000001_%u", slot);
    std::array<u8, 64> control;
    ASSERT_EQ(kern::ShmRead(name, 0, control.data(), control.size()), 0);
    EXPECT_EQ(control, (std::array<u8, 64>{}));
    std::snprintf(semaphore, sizeof(semaphore),
                  "SceSaveData0000000000000001_%u", slot);
    const int sem = kern::sys_osem_open(semaphore);
    ASSERT_GE(sem, 0);
    ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
    EXPECT_EQ(kern::sys_osem_trywait(sem, 1), -kern::SysError::eBUSY);
    ASSERT_EQ(kern::sys_osem_post(sem, 1), 0);
  }
  ASSERT_EQ(SlotClear("sdmemory2"), 0);
  EXPECT_EQ(kern::sys_osem_open("SceSaveData0000000000000001_2"),
            -kern::SysError::eNOENT);
  for (u32 slot : {0u, 1u, 3u}) {
    char name[64];
    std::snprintf(name, sizeof(name), "/SceSaveData0000000000000001_%u_0",
                  slot);
    u8 value = 0;
    ASSERT_EQ(kern::ShmRead(name, 0, &value, 1), 0);
    EXPECT_EQ(value, slot + 1);
  }
  ASSERT_EQ(MemoryConfirm(), 0);
  ASSERT_EQ(MemoryClear(), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory2", 2)), 0);
  ASSERT_EQ(Initialize(63), 0);
  for (u32 slot = 0; slot < 4; ++slot) {
    char name[64];
    std::snprintf(name, sizeof(name), "/SceSaveDataI0000000000000001_%u", slot);
    u8 value = 0;
    EXPECT_EQ(kern::ShmRead(name, 0, &value, 1), -kern::SysError::eNOENT);
  }
}

TEST_F(SaveDataTest, MemorySlotClearPreservesLiveViewsAndReusesPhysicalIndex) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory3", 0)), 0);
  const char* name = "/SceSaveData0000000000000001_0_0";
  const int fd = kern::sys_shm_open(name, 2, 0777);
  ASSERT_GE(fd, 0);
  auto* view = kern::sys_mmap(nullptr, 0x4000, 3, 1, fd, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(view));
  view[0] = 117;
  ASSERT_EQ(SlotClear("sdmemory3"), 0);
  ASSERT_EQ(SlotClear("sdmemory3"), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory1", 0)), 0);
  u8 value = 99;
  ASSERT_EQ(kern::ShmRead(name, 0, &value, 1), 0);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(view[0], 117);
  EXPECT_EQ(kern::sys_close(fd), 0);
  EXPECT_EQ(kern::sys_munmap(view, 0x4000), 0);
}

TEST_F(SaveDataTest, MemorySlotValidationPreservesExistingResources) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 2)), 0);
  const char* name = "/SceSaveData0000000000000001_2_0";
  const u8 marker = 117;
  ASSERT_EQ(kern::ShmWrite(name, 0, &marker, 1), 0);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory", 1))),
            0x809f0003u);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory1", 2))),
            0x809f0003u);
  for (const char* dir : {"sdmemory0", "sdmemory4", "../sdmemory", ""}) {
    EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload(dir, 0))), 0x809f0000u);
    EXPECT_EQ(static_cast<u32>(SlotClear(dir)), 0x809f0000u);
  }
  EXPECT_EQ(static_cast<u32>(SlotClear("sdmemory", 2)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory1", 4))),
            0x809f0000u);
  for (size_t offset : {0x568u, 0x569u, 0x56bu, 0x570u, 0x58fu}) {
    auto payload = SlotPayload("sdmemory1", 0);
    payload[offset] = 2;
    EXPECT_EQ(static_cast<u32>(SlotSetup(payload)), 0x809f0000u);
  }
  auto payload = SlotPayload("sdmemory1", 0);
  payload[32] = 1;
  EXPECT_EQ(SlotSetup(payload), -static_cast<i32>(kern::SysError::eNOSYS));
  u8 value = 0;
  ASSERT_EQ(kern::ShmRead(name, 0, &value, 1), 0);
  EXPECT_EQ(value, marker);
}

TEST_F(SaveDataTest, MemorySlotCollisionRollsBackOnlyNewResources) {
  const char* collision = "/SceSaveData0000000000000001_0_1";
  ASSERT_EQ(kern::ShmCreate(collision, 8192), 0);
  const u8 marker = 117;
  ASSERT_EQ(kern::ShmWrite(collision, 0, &marker, 1), 0);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory2", 0))),
            0x809f0003u);
  u8 value = 0;
  EXPECT_EQ(kern::ShmRead("/SceSaveData0000000000000001_0_0", 0, &value, 1),
            -kern::SysError::eNOENT);
  ASSERT_EQ(kern::ShmRead(collision, 0, &value, 1), 0);
  EXPECT_EQ(value, marker);
  ASSERT_EQ(kern::sys_shm_unlink(collision), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory2", 0)), 0);
}

TEST_F(SaveDataTest,
       MemorySlotCleanupRejectsMalformedAndUninitializedRequests) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  std::array<u8, 60> payload{};
  payload[0] = 1;
  std::memcpy(payload.data() + 4, "sdmemory", 9);
  const u64 input[] = {81, 60, 0};
  u64 received = 117;
  for (size_t length : {0u, 59u})
    EXPECT_EQ(static_cast<u32>(Invoke(0x12340002,
                                      {{input, 24}, {payload.data(), length}},
                                      {{nullptr, 0}, {&received, 8}})),
              0x809f0000u);
  payload[59] = 1;
  EXPECT_EQ(
      static_cast<u32>(Invoke(0x12340002, {{input, 24}, {payload.data(), 60}},
                              {{nullptr, 0}, {&received, 8}})),
      0x809f0000u);
  EXPECT_EQ(received, 117u);
  u8 value = 0;
  ASSERT_EQ(kern::ShmRead("/SceSaveDataI0000000000000001_0", 0, &value, 1), 0);
  ASSERT_EQ(Initialize(63), 0);
  EXPECT_EQ(static_cast<u32>(SlotClear("sdmemory")), 0x809f0001u);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory", 0))),
            0x809f0001u);
}

TEST_F(SaveDataTest, MemorySlotSemaphoreCollisionPreservesExistingObject) {
  const char* name = "SceSaveData0000000000000001_0";
  const int sem = kern::sys_osem_create(name, 1, 1, 1);
  ASSERT_GE(sem, 0);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory", 0))),
            0x809f0003u);
  EXPECT_EQ(kern::sys_osem_open(name), sem);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  ASSERT_EQ(kern::sys_osem_post(sem, 1), 0);
  u8 value = 0;
  for (const char* buffer :
       {"/SceSaveData0000000000000001_0_0", "/SceSaveData0000000000000001_0_1",
        "/SceSaveDataI0000000000000001_0"})
    EXPECT_EQ(kern::ShmRead(buffer, 0, &value, 1), -kern::SysError::eNOENT);
  ASSERT_EQ(kern::sys_osem_delete(sem), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
}

TEST_F(SaveDataTest, MemorySlotConfirmationMatchesLogicalAndPhysicalSlots) {
  ASSERT_EQ(MemorySetup(), 0);
  EXPECT_EQ(static_cast<u32>(SlotConfirm("sdmemory", 0)), 0x809f0000u);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory3", 0)), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 1)), 0);
  ASSERT_EQ(SlotConfirm("sdmemory3", 0), 0);
  EXPECT_EQ(written, 32u);
  ASSERT_EQ(SlotConfirm("sdmemory", 1), 0);
  for (const auto& [dir, slot, app, user, size] : {
           std::tuple{"sdmemory3", 3u, 0u, 1u, 4096ull},
           std::tuple{"sdmemory1", 0u, 0u, 1u, 4096ull},
           std::tuple{"sdmemory", 0u, 0u, 1u, 4096ull},
           std::tuple{"sdmemory3", 0u, 1u, 1u, 4096ull},
           std::tuple{"sdmemory3", 0u, 0u, 2u, 4096ull},
           std::tuple{"sdmemory3", 0u, 0u, 1u, 0ull},
           std::tuple{"sdmemory3", 0u, 0u, 1u, 8192ull},
       }) {
    reply.fill(0xa5);
    written = 117;
    EXPECT_EQ(static_cast<u32>(SlotConfirm(dir, slot, app, user, size)),
              0x809f0000u);
    EXPECT_EQ(written, 117u);
    EXPECT_EQ(reply[0], 0xa5);
  }
  ASSERT_EQ(MemoryConfirm(), 0);
  ASSERT_EQ(SlotClear("sdmemory3"), 0);
  EXPECT_EQ(static_cast<u32>(SlotConfirm("sdmemory3", 0)), 0x809f0000u);
  ASSERT_EQ(SlotConfirm("sdmemory", 1), 0);
}

TEST_F(SaveDataTest, MemorySlotConfirmationPreservesGuestDataAndPermit) {
  auto payload = SlotPayload("sdmemory2", 1);
  const u64 icon = 128, param = 0x530;
  std::memcpy(payload.data() + 16, &icon, 8);
  std::memcpy(payload.data() + 24, &param, 8);
  ASSERT_EQ(SlotSetup(payload), 0);
  const char* control_name = "/SceSaveDataI0000000000000001_1";
  const char* buffer_name = "/SceSaveData0000000000000001_1_1";
  std::array<u8, 64> control;
  control.fill(117);
  std::array<u8, 4096 + 64 + 128 + 0x530> bytes;
  bytes.fill(99);
  ASSERT_EQ(kern::ShmWrite(control_name, 0, control.data(), control.size()), 0);
  ASSERT_EQ(kern::ShmWrite(buffer_name, 0, bytes.data(), bytes.size()), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001_1");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  for (u32 cycle = 0; cycle < 100; ++cycle)
    ASSERT_EQ(SlotConfirm("sdmemory2", 1), 0);
  EXPECT_EQ(kern::sys_osem_trywait(sem, 1), -kern::SysError::eBUSY);
  ASSERT_EQ(kern::sys_osem_post(sem, 1), 0);
  std::array<u8, 64> actual_control;
  decltype(bytes) actual_bytes;
  ASSERT_EQ(kern::ShmRead(control_name, 0, actual_control.data(),
                          actual_control.size()),
            0);
  ASSERT_EQ(
      kern::ShmRead(buffer_name, 0, actual_bytes.data(), actual_bytes.size()),
      0);
  EXPECT_EQ(actual_control, control);
  EXPECT_EQ(actual_bytes, bytes);
}

TEST_F(SaveDataTest, MemorySlotConfirmationRejectsMissingOrTruncatedResources) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  for (const auto& [name, required] : {
           std::pair{"/SceSaveData0000000000000001_0_0", 4096u},
           std::pair{"/SceSaveData0000000000000001_0_1", 4096u},
           std::pair{"/SceSaveDataI0000000000000001_0", 64u},
       }) {
    const int fd = kern::sys_shm_open(name, 2, 0777);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(kern::sys_ftruncate(fd, required - 1), 0);
    EXPECT_EQ(static_cast<u32>(SlotConfirm("sdmemory", 0)), 0x809f000bu);
    ASSERT_EQ(kern::sys_ftruncate(fd, required), 0);
    ASSERT_EQ(SlotConfirm("sdmemory", 0), 0);
    ASSERT_EQ(kern::sys_close(fd), 0);
  }
  ASSERT_EQ(kern::sys_shm_unlink("/SceSaveData0000000000000001_0_1"), 0);
  EXPECT_EQ(static_cast<u32>(SlotConfirm("sdmemory", 0)), 0x809f000bu);
  ASSERT_EQ(SlotClear("sdmemory"), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001_0");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_delete(sem), 0);
  EXPECT_EQ(static_cast<u32>(SlotConfirm("sdmemory", 0)), 0x809f000bu);
}

TEST_F(SaveDataTest, MemorySlotConfirmationRejectsMalformedReservedFields) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  for (size_t offset : {32u, 33u, 35u, 40u, 71u}) {
    ASSERT_EQ(SlotConfirm("sdmemory", 0), 0);
    data[offset] = 2;
    ASSERT_EQ(Submit(72), 0);
    EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
  }
  ASSERT_EQ(SlotConfirm("sdmemory", 0), 0);
  std::memset(data.data() + 16, 'x', 16);
  ASSERT_EQ(Submit(72), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
  ASSERT_EQ(SlotConfirm("sdmemory", 0), 0);
}

TEST_F(SaveDataTest,
       MemorySlotConfirmationCapacityFailurePreservesPendingReply) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  ASSERT_EQ(SlotConfirm("sdmemory", 0), 0);
  ASSERT_EQ(Submit(72), 0);
  reply.fill(0xa5);
  written = 117;
  EXPECT_EQ(static_cast<u32>(Collect(31)), 0x809f0000u);
  EXPECT_EQ(reply[0], 0xa5);
  EXPECT_EQ(written, 117u);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(written, 32u);
  for (size_t i = 0; i < 32; ++i)
    EXPECT_EQ(reply[i], 0);
  EXPECT_EQ(reply[32], 0xa5);
}

TEST_F(SaveDataTest, MemoryCreateUsesSelectedSlotDirectoryAndPreservesOthers) {
  for (u32 slot = 0; slot < 4; ++slot) {
    char dir[16];
    std::snprintf(dir, sizeof(dir), slot ? "sdmemory%u" : "sdmemory", slot);
    ASSERT_EQ(SlotSetup(SlotPayload(dir, 3 - slot)), 0);
    ASSERT_EQ(SlotConfirm(dir, 3 - slot), 0);
    ASSERT_EQ(MemoryCreate(), 0);
    const auto path = root / "CUSA00792" / (std::string("sce_") + dir);
    ASSERT_TRUE(std::filesystem::is_directory(path));
    u64 blocks = 0;
    std::ifstream quota(path / ".sce_quota.bin", std::ios::binary);
    ASSERT_TRUE(quota.read(reinterpret_cast<char*>(&blocks), sizeof(blocks)));
    EXPECT_EQ(blocks, 96u);
    std::ofstream(path / "memory.dat", std::ios::binary).put(slot + 1);
    EXPECT_EQ(static_cast<u32>(MemoryCreate()), 0x809f0007u);
  }
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  for (u32 slot = 0; slot < 4; ++slot) {
    const auto dir =
        slot ? "sce_sdmemory" + std::to_string(slot) : "sce_sdmemory";
    std::ifstream file(root / "CUSA00792" / dir / "memory.dat",
                       std::ios::binary);
    EXPECT_EQ(file.get(), slot + 1);
  }
}

TEST_F(SaveDataTest, FailedSlotSetupDoesNotRedirectMemoryDirectoryCreation) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory2", 0)), 0);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory3", 0))),
            0x809f0003u);
  ASSERT_EQ(MemoryCreate(), 0);
  EXPECT_TRUE(
      std::filesystem::is_directory(root / "CUSA00792" / "sce_sdmemory2"));
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "sce_sdmemory3"));
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  EXPECT_TRUE(
      std::filesystem::is_directory(root / "CUSA00792" / "sce_sdmemory"));
}

TEST_F(SaveDataTest, ClearingSelectedSlotRetiresItsDirectoryCreationContext) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory3", 0)), 0);
  ASSERT_EQ(SlotClear("sdmemory3"), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  EXPECT_TRUE(
      std::filesystem::is_directory(root / "CUSA00792" / "sce_sdmemory"));
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "sce_sdmemory3"));
}

TEST_F(SaveDataTest, MemoryMigrationUsesLogicalSlotsAndPreservesSources) {
  const auto legacy = root / "CUSA00792" / "sce_sdmemory";
  std::filesystem::create_directories(legacy);
  for (u32 slot = 0; slot < 4; ++slot) {
    const auto source = legacy / ("memory" + std::to_string(slot) + ".bin");
    std::vector<u8> bytes(slot == 2 ? 8192 : slot == 1 ? 16 : 4096, slot + 1);
    std::ofstream(source, std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    const auto dir = slot ? "sdmemory" + std::to_string(slot) : "sdmemory";
    ASSERT_EQ(SlotSetup(SlotPayload(dir.c_str(), 3 - slot)), 0);
    const auto target = root / "CUSA00792" / ("sce_" + dir) / "memory.dat";
    std::vector<u8> migrated(std::max<size_t>(4096, bytes.size()));
    std::ifstream file(target, std::ios::binary);
    ASSERT_TRUE(
        file.read(reinterpret_cast<char*>(migrated.data()), migrated.size()));
    bytes.resize(migrated.size());
    EXPECT_EQ(migrated, bytes);
    EXPECT_EQ(std::filesystem::file_size(source), slot == 2   ? 8192u
                                                  : slot == 1 ? 16u
                                                              : 4096u);
    ASSERT_EQ(SlotClear(dir.c_str()), 0);
    std::ofstream(source, std::ios::binary | std::ios::trunc) << "changed";
    ASSERT_EQ(SlotSetup(SlotPayload(dir.c_str(), slot)), 0);
    EXPECT_EQ(std::filesystem::file_size(target), migrated.size());
    std::vector<u8> again(migrated.size());
    std::ifstream read(target, std::ios::binary);
    ASSERT_TRUE(read.read(reinterpret_cast<char*>(again.data()), again.size()));
    EXPECT_EQ(again, migrated);
    ASSERT_EQ(SlotClear(dir.c_str()), 0);
  }
}

TEST_F(SaveDataTest,
       MemoryMigrationAllowsMetadataAllocationsWithoutInventingFiles) {
  const auto source_dir = root / "CUSA00792" / "sce_sdmemory";
  std::filesystem::create_directories(source_dir);
  for (u32 slot = 0; slot < 4; ++slot) {
    const auto source = source_dir / ("memory" + std::to_string(slot) + ".bin");
    std::ofstream(source, std::ios::binary) << "previous memory";
    const auto dir = slot ? "sdmemory" + std::to_string(slot) : "sdmemory";
    auto payload = SlotPayload(dir.c_str(), 3 - slot);
    const u64 icon = slot & 1 ? 0x1c800 : 0;
    const u64 param = slot & 2 ? 0x530 : 0;
    std::memcpy(payload.data() + 16, &icon, 8);
    std::memcpy(payload.data() + 24, &param, 8);
    ASSERT_EQ(SlotSetup(payload), 0);
    const auto target_dir = root / "CUSA00792" / ("sce_" + dir);
    std::array<u8, 4096> memory{};
    std::ifstream file(target_dir / "memory.dat", std::ios::binary);
    ASSERT_TRUE(
        file.read(reinterpret_cast<char*>(memory.data()), memory.size()));
    EXPECT_EQ(std::memcmp(memory.data(), "previous memory", 15), 0);
    EXPECT_TRUE(std::all_of(memory.begin() + 15, memory.end(),
                            [](u8 value) { return value == 0; }));
    EXPECT_EQ(std::filesystem::file_size(source), 15u);
    EXPECT_FALSE(std::filesystem::exists(target_dir / ".sce_param.bin"));
    EXPECT_FALSE(std::filesystem::exists(target_dir / ".sce_icon.bin"));
    EXPECT_EQ(SlotConfirm(dir.c_str(), 3 - slot), 0);
  }
}

TEST_F(SaveDataTest, MemoryMigrationPreservesExistingParametersAndIcon) {
  const auto source_dir = root / "CUSA00792" / "sce_sdmemory";
  const auto target_dir = root / "CUSA00792" / "sce_sdmemory2";
  std::filesystem::create_directories(source_dir);
  std::filesystem::create_directories(target_dir);
  std::ofstream(source_dir / "memory2.bin", std::ios::binary) << "memory";
  std::array<u8, 0x530> params;
  params.fill(0x71);
  std::ofstream(target_dir / ".sce_param.bin", std::ios::binary)
      .write(reinterpret_cast<const char*>(params.data()), params.size());
  std::ofstream(target_dir / ".sce_icon.bin", std::ios::binary) << "icon";
  auto payload = SlotPayload("sdmemory2", 1);
  const u64 icon = 0x1c800, param = params.size();
  std::memcpy(payload.data() + 16, &icon, 8);
  std::memcpy(payload.data() + 24, &param, 8);
  ASSERT_EQ(SlotSetup(payload), 0);
  std::array<u8, 0x530> stored{};
  std::ifstream file(target_dir / ".sce_param.bin", std::ios::binary);
  ASSERT_TRUE(file.read(reinterpret_cast<char*>(stored.data()), stored.size()));
  EXPECT_EQ(stored, params);
  EXPECT_EQ(std::filesystem::file_size(target_dir / ".sce_icon.bin"), 4u);
  EXPECT_EQ(std::filesystem::file_size(target_dir / "memory.dat"), 4096u);
  EXPECT_EQ(std::filesystem::file_size(source_dir / "memory2.bin"), 6u);
}

TEST_F(SaveDataTest, LegacyMemoryMigrationLoadsOnlySlotZero) {
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::filesystem::create_directories(path);
  std::ofstream(path / "memory0.bin", std::ios::binary) << "legacy";
  std::ofstream(path / "memory1.bin", std::ios::binary) << "other";
  ASSERT_EQ(MemorySetup(), 0);
  std::array<u8, 4096> bytes{};
  std::ifstream file(path / "memory.dat", std::ios::binary);
  ASSERT_TRUE(file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()));
  EXPECT_EQ(std::memcmp(bytes.data(), "legacy", 6), 0);
  EXPECT_TRUE(std::all_of(bytes.begin() + 6, bytes.end(),
                          [](u8 value) { return value == 0; }));
  EXPECT_EQ(std::filesystem::file_size(path / "memory0.bin"), 6u);
}

TEST_F(SaveDataTest, MemoryMigrationRejectsUnsafeSourcesAndRollsBackResources) {
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::filesystem::create_directories(path);
  const auto source = path / "memory2.bin";
  std::ofstream(root / "outside") << "outside";
  std::filesystem::create_symlink(root / "outside", source);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory2", 1))),
            0x809f0000u);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "sce_sdmemory2"));
  EXPECT_EQ(kern::sys_osem_open("SceSaveData0000000000000001_1"),
            -kern::SysError::eNOENT);
  std::filesystem::remove(source);
  std::ofstream(source).close();
  std::filesystem::resize_file(source, 0x2000001);
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory2", 1))),
            0x809f0000u);
  std::filesystem::resize_file(source, 4096);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory2", 1)), 0);
}

TEST_F(SaveDataTest, MemoryMigrationCapacityFailurePreservesDiskAndRetries) {
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  const auto target = root / "CUSA00792" / "sce_sdmemory1";
  std::filesystem::create_directories(path);
  std::filesystem::create_directories(target);
  std::ofstream(path / "memory1.bin", std::ios::binary) << "source";
  u64 blocks = 96;
  std::ofstream(target / "filler").close();
  std::filesystem::resize_file(target / "filler", 96 * 0x8000);
  std::ofstream(target / ".sce_quota.bin", std::ios::binary)
      .write(reinterpret_cast<const char*>(&blocks), sizeof(blocks));
  EXPECT_EQ(static_cast<u32>(SlotSetup(SlotPayload("sdmemory1", 0))),
            0x809f000au);
  EXPECT_FALSE(std::filesystem::exists(target / "memory.dat"));
  EXPECT_FALSE(std::filesystem::exists(target / "memory.dat.tmp"));
  EXPECT_EQ(std::filesystem::file_size(path / "memory1.bin"), 6u);
  blocks = 97;
  std::ofstream(target / ".sce_quota.bin", std::ios::binary)
      .write(reinterpret_cast<const char*>(&blocks), sizeof(blocks));
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory1", 0)), 0);
  EXPECT_EQ(std::filesystem::file_size(target / "memory.dat"), 4096u);
}

TEST_F(SaveDataTest, MemorySubscriptionsOwnAnEmptyFlagUntilLastUnsubscribe) {
  const auto name = MemoryEventName();
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 1)), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory3", 0)), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  const int flag = kern::sys_evf_open(name.c_str());
  ASSERT_GE(flag, 0);
  u64 result = 117;
  EXPECT_EQ(kern::sys_evf_trywait(flag, 0x1000000100010000, 0x22, &result),
            -kern::SysError::eBUSY);
  EXPECT_EQ(result, 0u);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory3"), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), flag);
  ASSERT_EQ(kern::sys_evf_set(flag, 64), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory", false), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), flag);
  ASSERT_EQ(MemorySubscribe("sdmemory3", false), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
  EXPECT_EQ(kern::sys_evf_trywait(flag, 1, 2, &result), -kern::SysError::eSRCH);
  ASSERT_EQ(MemorySubscribe("sdmemory3", false), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory3"), 0);
  const int fresh = kern::sys_evf_open(name.c_str());
  ASSERT_GE(fresh, 0);
  EXPECT_EQ(kern::sys_evf_trywait(fresh, ~u64(0), 2, &result),
            -kern::SysError::eBUSY);
}

TEST_F(SaveDataTest, MemorySubscriptionCollisionPreservesOtherClientOwnership) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  const auto name = MemoryEventName();
  const int flag = kern::sys_evf_open(name.c_str());
  ASSERT_GE(flag, 0);
  const u32 owner = kid;
  const char* service = "SceSaveData";
  u64 request = reinterpret_cast<u64>(service);
  ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)), 0);
  ASSERT_EQ(Initialize(), 0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory1", 1)), 0);
  EXPECT_EQ(static_cast<u32>(MemorySubscribe("sdmemory1")), 0x809f0003u);
  ASSERT_EQ(MemorySubscribe("sdmemory1", false), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), flag);
  u32 output = 0;
  ASSERT_EQ(kern::ipmi::ManagerCall(3, kid, &output, nullptr, 0), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), flag);
  ASSERT_EQ(kern::ipmi::ManagerCall(3, owner, &output, nullptr, 0), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
  u64 result = 0;
  EXPECT_EQ(kern::sys_evf_trywait(flag, 1, 2, &result), -kern::SysError::eSRCH);
}

TEST_F(SaveDataTest, MemorySubscriptionRefreshPreservesExistingFlagBits) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  const auto name = MemoryEventName();
  const int flag = kern::sys_evf_open(name.c_str());
  ASSERT_GE(flag, 0);
  ASSERT_EQ(kern::sys_evf_set(flag, 64), 0);
  for (u64 mask : {0ull, 1ull << 32, ~0ull}) {
    std::memcpy(data.data() + 64, &mask, 8);
    ASSERT_EQ(Submit(96), 0);
    ASSERT_EQ(Collect(), 0);
    EXPECT_EQ(kern::sys_evf_open(name.c_str()), flag);
    u64 result = 0;
    EXPECT_EQ(kern::sys_evf_trywait(flag, 64, 2, &result), 0);
    EXPECT_EQ(result, 64u);
  }
  ASSERT_EQ(MemorySubscribe("sdmemory", false), 0);
}

TEST_F(SaveDataTest, MemorySubscriptionsValidateRequestsAndPreserveOutput) {
  EXPECT_EQ(static_cast<u32>(MemorySubscribe("sdmemory")), 0x809f0000u);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory2", 0)), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory2"), 0);
  const auto valid = data;
  for (size_t offset : {0u, 4u, 8u, 60u, 63u, 76u, 80u, 88u}) {
    data = valid;
    data[offset] ^= 2;
    ASSERT_EQ(Submit(96), 0);
    written = 117;
    reply.fill(0xa5);
    EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
    EXPECT_EQ(written, 117u);
    EXPECT_EQ(reply[0], 0xa5);
  }
  data = valid;
  data[72] = 8;
  ASSERT_EQ(Submit(96), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
  data = valid;
  data[12] = 'x';
  ASSERT_EQ(Submit(96), 0);
  EXPECT_EQ(Collect(), -static_cast<i32>(kern::SysError::eNOSYS));
  EXPECT_EQ(static_cast<u32>(MemorySubscribe("sdmemory4")), 0x809f0000u);
  const auto name = MemoryEventName();
  EXPECT_GE(kern::sys_evf_open(name.c_str()), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory2", false), 0);
}

TEST_F(SaveDataTest, MemorySubscriptionsRetireOnSlotCleanupAndTermination) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  const auto name = MemoryEventName();
  ASSERT_GE(kern::sys_evf_open(name.c_str()), 0);
  ASSERT_EQ(MemoryClear(), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory1", 0)), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory1"), 0);
  ASSERT_EQ(SlotClear("sdmemory1"), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 1)), 0);
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  ASSERT_EQ(MemoryClear(), 0);
  EXPECT_GE(kern::sys_evf_open(name.c_str()), 0);
  ASSERT_EQ(Initialize(63), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
}

TEST_F(SaveDataTest, MemorySubscriptionCapacityFailureIsRetryable) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  const auto payload = data;
  ASSERT_EQ(MemorySubscribe("sdmemory", false), 0);
  data = payload;
  const u32 command = 51;
  std::memcpy(header.data() + 4, &command, 4);
  ASSERT_EQ(Submit(96), 0);
  reply.fill(0xa5);
  written = 117;
  EXPECT_EQ(static_cast<u32>(Collect(31)), 0x809f0000u);
  EXPECT_EQ(written, 117u);
  EXPECT_EQ(reply[0], 0xa5);
  const auto name = MemoryEventName();
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(written, 32u);
  EXPECT_EQ(reply[32], 0xa5);
  EXPECT_GE(kern::sys_evf_open(name.c_str()), 0);
}

TEST_F(SaveDataTest,
       MemorySubscriptionPreservesReplacementFlagAfterGuestClose) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  const auto name = MemoryEventName();
  const int old = kern::sys_evf_open(name.c_str());
  ASSERT_GE(old, 0);
  ASSERT_EQ(kern::sys_evf_close(old), 0);
  const int replacement = kern::sys_evf_create(name.c_str(), 0x102, 64);
  ASSERT_GE(replacement, 0);
  EXPECT_EQ(static_cast<u32>(MemorySubscribe("sdmemory")), 0x809f000bu);
  ASSERT_EQ(MemorySubscribe("sdmemory", false), 0);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), replacement);
  u64 result = 0;
  EXPECT_EQ(kern::sys_evf_trywait(replacement, 64, 2, &result), 0);
  EXPECT_EQ(result, 64u);
  EXPECT_EQ(static_cast<u32>(MemorySubscribe("sdmemory")), 0x809f0003u);
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), replacement);
  ASSERT_EQ(kern::sys_evf_delete(replacement), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
}

TEST_F(SaveDataTest, MemorySubscriptionRejectsTruncatedBacking) {
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 0)), 0);
  const int fd = kern::sys_shm_open("/SceSaveDataI0000000000000001_0", 2, 0777);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 63), 0);
  EXPECT_EQ(static_cast<u32>(MemorySubscribe("sdmemory")), 0x809f000bu);
  const auto name = MemoryEventName();
  EXPECT_EQ(kern::sys_evf_open(name.c_str()), -kern::SysError::eNOENT);
  ASSERT_EQ(kern::sys_ftruncate(fd, 64), 0);
  ASSERT_EQ(MemorySubscribe("sdmemory"), 0);
  ASSERT_EQ(kern::sys_close(fd), 0);
}

TEST_F(SaveDataTest, MemoryCreateProvisionsPersistentDirectoryAndQuota) {
  ASSERT_EQ(MemoryCreate(), 0);
  EXPECT_EQ(written, 32u);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  ASSERT_TRUE(std::filesystem::is_directory(path));
  u64 blocks = 0;
  std::ifstream quota(path / ".sce_quota.bin", std::ios::binary);
  ASSERT_TRUE(quota.read(reinterpret_cast<char*>(&blocks), sizeof(blocks)));
  EXPECT_EQ(blocks, 96u);
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  const u32 command = 1;
  data.fill(0);
  const u32 user = 1;
  std::memcpy(header.data() + 4, &command, 4);
  std::memcpy(data.data() + 0x400, &user, 4);
  std::memcpy(data.data() + 0x414, "sce_sdmemory", 13);
  Mode(2);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(reply[24], 0);
  EXPECT_EQ(FreeBlocks(), 96u);
  ASSERT_EQ(Unmount(), 0);
}

TEST_F(SaveDataTest, MemoryCreatePreservesExistingAndLegacySaves) {
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::ofstream(path / "memory.dat", std::ios::binary) << "memory";
  reply.fill(0xa5);
  written = ~u64(0);
  EXPECT_EQ(static_cast<u32>(MemoryCreate(0x400000)), 0x809f0007u);
  EXPECT_EQ(written, ~u64(0));
  for (u8 value : reply)
    EXPECT_EQ(value, 0xa5);
  EXPECT_EQ(std::filesystem::file_size(path / "memory.dat"), 6u);
  u64 blocks = 0;
  std::ifstream quota(path / ".sce_quota.bin", std::ios::binary);
  ASSERT_TRUE(quota.read(reinterpret_cast<char*>(&blocks), sizeof(blocks)));
  EXPECT_EQ(blocks, 96u);
  const auto legacy = root / "sce_sdmemory";
  std::filesystem::rename(path, legacy);
  EXPECT_EQ(static_cast<u32>(MemoryCreate()), 0x809f0007u);
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_EQ(std::filesystem::file_size(legacy / "memory.dat"), 6u);
}

TEST_F(SaveDataTest, MemoryCreateValidatesRequestBeforeCreatingStorage) {
  for (const auto& [bytes, user, reserved] :
       {std::tuple<u64, u32, u32>{0, 1, 0},
        {0x300000, 2, 0},
        {0x300000, 1, 1}}) {
    EXPECT_EQ(static_cast<u32>(MemoryCreate(bytes, user, reserved)),
              0x809f0000u);
    EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "sce_sdmemory"));
  }
  const u32 command = 35, user = 1;
  const u64 bytes = 0x300001;
  data.fill(0);
  std::memcpy(header.data() + 4, &command, 4);
  std::memcpy(data.data(), &user, 4);
  std::memcpy(data.data() + 8, &bytes, 8);
  ASSERT_EQ(Submit(16), 0);
  EXPECT_EQ(static_cast<u32>(Collect(31)), 0x809f0000u);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "sce_sdmemory"));
  ASSERT_EQ(Collect(), 0);
  u64 blocks = 0;
  std::ifstream quota(root / "CUSA00792" / "sce_sdmemory" / ".sce_quota.bin",
                      std::ios::binary);
  ASSERT_TRUE(quota.read(reinterpret_cast<char*>(&blocks), sizeof(blocks)));
  EXPECT_EQ(blocks, 97u);
}

TEST_F(SaveDataTest, MemoryCreateRejectsSymlinkAndNonDirectoryTargets) {
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::filesystem::create_directories(path.parent_path());
  std::filesystem::create_directory(root / "outside");
  std::filesystem::create_directory_symlink(root / "outside", path);
  EXPECT_EQ(static_cast<u32>(MemoryCreate()), 0x809f0000u);
  EXPECT_TRUE(std::filesystem::is_empty(root / "outside"));
  std::filesystem::remove(path);
  std::ofstream(path) << "marker";
  EXPECT_EQ(static_cast<u32>(MemoryCreate()), 0x809f000bu);
  EXPECT_EQ(std::filesystem::file_size(path), 6u);
}

TEST_F(SaveDataTest, SlotSyncPersistsLogicalDirectoriesAndSelectedMetadata) {
  for (u32 slot = 0; slot < 4; ++slot) {
    const auto dir = slot ? "sdmemory" + std::to_string(slot) : "sdmemory";
    const u32 physical = 3 - slot;
    auto payload = SlotPayload(dir.c_str(), physical);
    const u64 icon = 32, param = 0x530;
    std::memcpy(payload.data() + 16, &icon, 8);
    std::memcpy(payload.data() + 24, &param, 8);
    ASSERT_EQ(SlotSetup(payload), 0);
    ASSERT_EQ(MemoryCreate(), 0);
    std::vector<u8> bytes(4096 + 64 + icon + param, slot + 31);
    const u64 length = slot + 1;
    std::memcpy(bytes.data() + 4096, &icon, 8);
    std::memcpy(bytes.data() + 4104, &length, 8);
    const auto name =
        "/SceSaveData0000000000000001_" + std::to_string(physical) + "_1";
    ASSERT_EQ(kern::ShmWrite(name.c_str(), 0, bytes.data(), bytes.size()), 0);
    const auto control =
        "/SceSaveDataI0000000000000001_" + std::to_string(physical);
    const u32 selected = 1;
    ASSERT_EQ(kern::ShmWrite(control.c_str(), 0, &selected, 4), 0);
  }
  for (u32 slot = 0; slot < 4; ++slot) {
    const auto dir = slot ? "sdmemory" + std::to_string(slot) : "sdmemory";
    ASSERT_EQ(MemorySync(1, 0, dir.c_str()), 0);
    const auto path = root / "CUSA00792" / ("sce_" + dir);
    for (const auto& [name, size] :
         {std::pair{"memory.dat", size_t(4096)},
          std::pair{".sce_icon.bin", size_t(slot + 1)},
          std::pair{".sce_param.bin", size_t(0x530)}}) {
      EXPECT_EQ(std::filesystem::file_size(path / name), size);
      std::vector<u8> bytes(size);
      std::ifstream file(path / name, std::ios::binary);
      ASSERT_TRUE(file.read(reinterpret_cast<char*>(bytes.data()), size));
      EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(),
                              [slot](u8 value) { return value == slot + 31; }));
    }
  }
  ASSERT_EQ(Initialize(63), 0);
  for (const char* dir : {"sdmemory", "sdmemory1", "sdmemory2", "sdmemory3"})
    EXPECT_EQ(
        std::filesystem::file_size(root / "CUSA00792" /
                                   (std::string("sce_") + dir) / "memory.dat"),
        4096u);
}

TEST_F(SaveDataTest, SlotSyncPrefersModernSlotAndReturnsToLegacyAfterCleanup) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  std::array<u8, 4096> bytes;
  bytes.fill(17);
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_0", 0, bytes.data(),
                           bytes.size()),
            0);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory", 2)), 0);
  bytes.fill(31);
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_2_0", 0, bytes.data(),
                           bytes.size()),
            0);
  const auto path = root / "CUSA00792" / "sce_sdmemory" / "memory.dat";
  ASSERT_EQ(MemorySync(), 0);
  std::array<u8, 4096> loaded{};
  std::ifstream modern(path, std::ios::binary);
  ASSERT_TRUE(
      modern.read(reinterpret_cast<char*>(loaded.data()), loaded.size()));
  EXPECT_EQ(loaded, bytes);
  ASSERT_EQ(SlotClear("sdmemory"), 0);
  ASSERT_EQ(MemorySync(), 0);
  std::ifstream legacy(path, std::ios::binary);
  ASSERT_TRUE(
      legacy.read(reinterpret_cast<char*>(loaded.data()), loaded.size()));
  bytes.fill(17);
  EXPECT_EQ(loaded, bytes);
}

TEST_F(SaveDataTest,
       SlotSyncRejectsMissingAndTruncatedBackingWithoutDiskWrites) {
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 0, "sdmemory2")), 0x809f0001u);
  ASSERT_EQ(SlotSetup(SlotPayload("sdmemory2", 1)), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 0, "sdmemory2")), 0x809f0008u);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory2" / "memory.dat";
  std::ofstream(path) << "previous";
  const int fd =
      kern::sys_shm_open("/SceSaveData0000000000000001_1_0", 2, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 4095), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 0, "sdmemory2")), 0x809f000bu);
  EXPECT_EQ(std::filesystem::file_size(path), 8u);
  ASSERT_EQ(kern::sys_ftruncate(fd, 4096), 0);
  ASSERT_EQ(MemorySync(1, 0, "sdmemory2"), 0);
  ASSERT_EQ(SlotClear("sdmemory2"), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 0, "sdmemory2")), 0x809f0001u);
  EXPECT_EQ(std::filesystem::file_size(path), 4096u);
}

TEST_F(SaveDataTest, MemorySyncPersistsSelectedBufferAcrossTermination) {
  ASSERT_EQ(MemorySetup(4096, 0, 0, 2), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  std::array<u8, 4096> memory;
  for (size_t i = 0; i < memory.size(); ++i)
    memory[i] = i;
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_1", 0, memory.data(),
                           memory.size()),
            0);
  const u32 selected = 1;
  ASSERT_EQ(kern::ShmWrite("/SceSaveDataI0000000000000001", 0, &selected, 4),
            0);
  ASSERT_EQ(MemorySync(), 0);
  ASSERT_EQ(Initialize(63), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory" / "memory.dat";
  EXPECT_EQ(std::filesystem::file_size(path), memory.size());
  std::array<u8, 4096> loaded{};
  std::ifstream file(path, std::ios::binary);
  ASSERT_TRUE(file.read(reinterpret_cast<char*>(loaded.data()), loaded.size()));
  EXPECT_EQ(loaded, memory);
  EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}

TEST_F(SaveDataTest, MemorySyncPersistsParametersFromSelectedBuffer) {
  ASSERT_EQ(MemorySetup(4096, 0, 0x530, 2), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  std::array<u8, 4096 + 0x530> snapshot{};
  for (size_t i = 0; i < snapshot.size(); ++i)
    snapshot[i] = i * 17 + 31;
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_1", 0, snapshot.data(),
                           snapshot.size()),
            0);
  const u32 index = 1;
  ASSERT_EQ(kern::ShmWrite("/SceSaveDataI0000000000000001", 0, &index, 4), 0);
  ASSERT_EQ(MemorySync(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  EXPECT_EQ(std::filesystem::file_size(path / "memory.dat"), 4096u);
  EXPECT_EQ(std::filesystem::file_size(path / ".sce_param.bin"), 0x530u);
  std::array<u8, 4096 + 0x530> loaded{};
  std::ifstream memory(path / "memory.dat", std::ios::binary);
  ASSERT_TRUE(memory.read(reinterpret_cast<char*>(loaded.data()), 4096));
  std::ifstream param(path / ".sce_param.bin", std::ios::binary);
  ASSERT_TRUE(param.read(reinterpret_cast<char*>(loaded.data() + 4096), 0x530));
  EXPECT_EQ(loaded, snapshot);
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  EXPECT_EQ(std::filesystem::file_size(path / ".sce_param.bin"), 0x530u);
}

TEST_F(SaveDataTest, MemorySyncWithoutParametersPreservesExistingMetadata) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::ofstream(path / ".sce_param.bin", std::ios::binary) << "existing";
  ASSERT_EQ(MemorySync(), 0);
  EXPECT_EQ(std::filesystem::file_size(path / ".sce_param.bin"), 8u);
}

TEST_F(SaveDataTest, MemorySyncReportsParameterStorageFailureAndReleasesToken) {
  ASSERT_EQ(MemorySetup(4096, 0, 0x530), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::filesystem::create_directory(path / ".sce_param.bin");
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f000bu);
  EXPECT_EQ(std::filesystem::file_size(path / "memory.dat"), 4096u);
  EXPECT_TRUE(std::filesystem::is_directory(path / ".sce_param.bin"));
  EXPECT_FALSE(std::filesystem::exists(path / ".sce_param.bin.tmp"));
  const int semaphore = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(semaphore, 0);
  ASSERT_EQ(kern::sys_osem_trywait(semaphore, 1), 0);
  ASSERT_EQ(kern::sys_osem_post(semaphore, 1), 0);
  std::filesystem::remove(path / ".sce_param.bin");
  ASSERT_EQ(MemorySync(), 0);
}

TEST_F(SaveDataTest, MemorySyncPersistsIconsAndParametersFromSelectedBuffer) {
  ASSERT_EQ(MemorySetup(4096, 0x1c800, 0x530, 2), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const size_t param_offset = 4096 + 64 + 0x1c800;
  std::vector<u8> snapshot(param_offset + 0x530);
  for (size_t i = 0; i < snapshot.size(); ++i)
    snapshot[i] = i * 13 + 17;
  const u64 capacity = 0x1c800;
  std::memcpy(snapshot.data() + 4096, &capacity, 8);
  std::memcpy(snapshot.data() + 4104, &capacity, 8);
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_1", 0, snapshot.data(),
                           snapshot.size()),
            0);
  const u32 index = 1;
  ASSERT_EQ(kern::ShmWrite("/SceSaveDataI0000000000000001", 0, &index, 4), 0);
  ASSERT_EQ(MemorySync(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  for (const auto& [name, offset, size] :
       {std::tuple{"memory.dat", size_t(0), size_t(4096)},
        std::tuple{".sce_icon.bin", size_t(4160), size_t(0x1c800)},
        std::tuple{".sce_param.bin", param_offset, size_t(0x530)}}) {
    EXPECT_EQ(std::filesystem::file_size(path / name), size);
    std::vector<u8> loaded(size);
    std::ifstream file(path / name, std::ios::binary);
    ASSERT_TRUE(file.read(reinterpret_cast<char*>(loaded.data()), size));
    EXPECT_EQ(std::memcmp(loaded.data(), snapshot.data() + offset, size), 0);
  }
  const u64 smaller = 17;
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_1", 4104, &smaller, 8),
            0);
  ASSERT_EQ(MemorySync(), 0);
  EXPECT_EQ(std::filesystem::file_size(path / ".sce_icon.bin"), smaller);
}

TEST_F(SaveDataTest, MemorySyncEmptyIconAllocationPreservesDiskIcon) {
  ASSERT_EQ(MemorySetup(4096, 128), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::ofstream(path / ".sce_icon.bin", std::ios::binary) << "previous";
  ASSERT_EQ(MemorySync(), 0);
  EXPECT_EQ(std::filesystem::file_size(path / ".sce_icon.bin"), 8u);
  EXPECT_FALSE(std::filesystem::exists(path / ".sce_param.bin"));
}

TEST_F(SaveDataTest, MemorySyncCorruptIconHeaderPreservesAllFiles) {
  ASSERT_EQ(MemorySetup(4096, 128, 0x530), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  for (const char* name : {"memory.dat", ".sce_icon.bin", ".sce_param.bin"})
    std::ofstream(path / name, std::ios::binary) << "previous";
  for (const auto& [capacity, length] :
       {std::pair<u64, u64>{127, 16}, {129, 16}, {128, 129}}) {
    const u64 header[] = {capacity, length};
    ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_0", 4096, header,
                             sizeof(header)),
              0);
    EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f000bu);
    for (const char* name : {"memory.dat", ".sce_icon.bin", ".sce_param.bin"})
      EXPECT_EQ(std::filesystem::file_size(path / name), 8u);
  }
}

TEST_F(SaveDataTest, MemorySyncIconStorageFailureReleasesTokenAndRetries) {
  ASSERT_EQ(MemorySetup(4096, 128), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const u64 length = 16;
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_0", 4104, &length, 8),
            0);
  const auto path = root / "CUSA00792" / "sce_sdmemory";
  std::filesystem::create_directory(path / ".sce_icon.bin");
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f000bu);
  EXPECT_TRUE(std::filesystem::is_directory(path / ".sce_icon.bin"));
  EXPECT_FALSE(std::filesystem::exists(path / ".sce_icon.bin.tmp"));
  const int semaphore = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(semaphore, 0);
  ASSERT_EQ(kern::sys_osem_trywait(semaphore, 1), 0);
  ASSERT_EQ(kern::sys_osem_post(semaphore, 1), 0);
  std::filesystem::remove(path / ".sce_icon.bin");
  ASSERT_EQ(MemorySync(), 0);
  EXPECT_EQ(std::filesystem::file_size(path / ".sce_icon.bin"), length);
}

TEST_F(SaveDataTest,
       AsyncMemorySyncAcknowledgesBeforeGuestWriterReleasesPermit) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  std::atomic<bool> accepted{false};
  i32 result = -1;
  std::thread caller([&] {
    result = MemorySync(1, 1);
    accepted = true;
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!accepted && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const bool nonblocking = accepted;
  std::array<u8, 60> output;
  output.fill(0xa5);
  const auto previous = output;
  u64 received = ~u64(0);
  if (nonblocking) {
    EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0018u);
    EXPECT_EQ(received, ~u64(0));
    EXPECT_EQ(output, previous);
    EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f0003u);
  }
  const std::array<u8, 4096> memory = [] {
    std::array<u8, 4096> value;
    value.fill(0x71);
    return value;
  }();
  EXPECT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_0", 0, memory.data(),
                           memory.size()),
            0);
  EXPECT_EQ(kern::sys_osem_post(sem, 1), 0);
  caller.join();
  ASSERT_TRUE(nonblocking);
  ASSERT_EQ(result, 0);
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  std::array<u8, 4096> stored{};
  std::ifstream file(root / "CUSA00792" / "sce_sdmemory" / "memory.dat",
                     std::ios::binary);
  ASSERT_TRUE(file.read(reinterpret_cast<char*>(stored.data()), stored.size()));
  EXPECT_EQ(stored, memory);
}

TEST_F(SaveDataTest, AsyncMemorySyncPreventsMountAndDeleteWhilePending) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  EXPECT_EQ(MemorySync(1, 1), 0);
  std::memset(data.data() + 0x414, 0, 32);
  std::memcpy(data.data() + 0x414, "sce_sdmemory", 13);
  Mode(1);
  const u32 command = 60;
  std::memcpy(header.data() + 4, &command, 4);
  EXPECT_EQ(Submit(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0003u);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  EXPECT_EQ(kern::sys_osem_post(sem, 1), 0);
  std::array<u8, 60> output{};
  u64 received = ~u64(0);
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  EXPECT_EQ(Delete(), 0);
}

TEST_F(SaveDataTest, AsyncMemorySyncTerminationWaitsForAcceptedWrite) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  EXPECT_EQ(MemorySync(1, 1), 0);
  std::atomic<bool> terminated{false};
  i32 result = -1;
  std::thread terminator([&] {
    result = Initialize(63);
    terminated = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(terminated);
  EXPECT_EQ(kern::sys_osem_post(sem, 1), 0);
  terminator.join();
  EXPECT_EQ(result, 0);
  EXPECT_EQ(std::filesystem::file_size(root / "CUSA00792" / "sce_sdmemory" /
                                       "memory.dat"),
            4096u);
  ASSERT_EQ(Initialize(), 0);
  std::array<u8, 60> output{};
  u64 received = ~u64(0);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0018u);
}

TEST_F(SaveDataTest, AsyncMemorySyncSessionStopCancelsSemaphoreWaitAndJoins) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  EXPECT_EQ(MemorySync(1, 1), 0);
  guest::RequestStop();
  guest::JoinThreads();
  EXPECT_EQ(guest::LiveThreads(), 0u);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "sce_sdmemory" /
                                       "memory.dat"));
  std::array<u8, 60> output{};
  u64 received = ~u64(0);
  ASSERT_EQ(MemoryResult(output, received), 0);
  u32 result = 0;
  std::memcpy(&result, output.data() + 4, 4);
  EXPECT_EQ(result, 0x809f000bu);
  EXPECT_EQ(Initialize(63), 0);
}

TEST_F(SaveDataTest, AsyncMemoryAcknowledgementIsSeparateAndRetryable) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  i32 result = -1;
  u64 received = ~u64(0);
  EXPECT_EQ(MemoryAcknowledgement(result, received), -1);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(received, ~u64(0));
  ASSERT_EQ(MemorySync(1, 1, "sdmemory", false), 0);
  EXPECT_EQ(MemoryAcknowledgement(result, received, 3), -1);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(received, ~u64(0));
  const u32 owner = kid;
  const char* name = "SceSaveData";
  u64 request = reinterpret_cast<u64>(name);
  ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)), 0);
  ASSERT_EQ(Initialize(), 0);
  EXPECT_EQ(MemoryAcknowledgement(result, received), -1);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(received, ~u64(0));
  kid = owner;
  ASSERT_EQ(MemoryAcknowledgement(result, received), 0);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(received, 4u);
  result = -1;
  received = ~u64(0);
  EXPECT_EQ(MemoryAcknowledgement(result, received), -1);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(received, ~u64(0));
  std::array<u8, 60> output{};
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  EXPECT_EQ(received, 60u);
}

TEST_F(SaveDataTest, MemoryResultsPreserveOutputsWhenEmptyOrUninitialized) {
  std::array<u8, 60> output;
  output.fill(0xa5);
  const auto previous = output;
  u64 received = ~u64(0);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0018u);
  EXPECT_EQ(output, previous);
  EXPECT_EQ(received, ~u64(0));
  ASSERT_EQ(Initialize(63), 0);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0001u);
  EXPECT_EQ(output, previous);
  EXPECT_EQ(received, ~u64(0));
}

TEST_F(SaveDataTest, AsyncMemorySyncQueuesSelectedSlotResultsInOrder) {
  for (u32 slot = 0; slot < 4; ++slot) {
    const auto dir = slot ? "sdmemory" + std::to_string(slot) : "sdmemory";
    ASSERT_EQ(SlotSetup(SlotPayload(dir.c_str(), 3 - slot)), 0);
    ASSERT_EQ(MemoryCreate(), 0);
    ASSERT_EQ(MemorySync(1, 1, dir.c_str()), 0);
  }
  for (u32 slot = 0; slot < 4; ++slot) {
    const auto dir = slot ? "sdmemory" + std::to_string(slot) : "sdmemory";
    std::array<u8, 60> output{}, expected{};
    const u32 type = 3, user = 1;
    std::memcpy(expected.data(), &type, 4);
    std::memcpy(expected.data() + 8, &user, 4);
    std::memcpy(expected.data() + 28, dir.c_str(), dir.size() + 1);
    u64 received = ~u64(0);
    ASSERT_EQ(WaitMemoryResult(output, received), 0);
    EXPECT_EQ(received, 60u);
    EXPECT_EQ(output, expected);
    EXPECT_EQ(std::filesystem::file_size(root / "CUSA00792" / ("sce_" + dir) /
                                         "memory.dat"),
              4096u);
  }
  ASSERT_EQ(MemorySync(), 0);
  std::array<u8, 60> output{};
  u64 received = ~u64(0);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0018u);
}

TEST_F(SaveDataTest, AsyncMemorySyncReturnsStorageFailureThroughEvent) {
  ASSERT_EQ(MemorySetup(0x10000), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto dir = root / "CUSA00792" / "sce_sdmemory";
  std::ofstream(dir / "memory.dat") << "previous";
  std::filesystem::resize_file(dir / "memory.dat", 0x8000);
  std::ofstream(dir / "filler");
  std::filesystem::resize_file(dir / "filler", 95 * 0x8000);
  ASSERT_EQ(MemorySync(1, 1), 0);
  EXPECT_EQ(std::filesystem::file_size(dir / "memory.dat"), 0x8000u);
  std::array<u8, 60> output{};
  u64 received = ~u64(0);
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  ASSERT_EQ(kern::sys_osem_post(sem, 1), 0);
  u32 result = 0;
  std::memcpy(&result, output.data() + 4, 4);
  EXPECT_EQ(result, 0x809f000au);
  std::filesystem::remove(dir / "filler");
  ASSERT_EQ(MemorySync(1, 1), 0);
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  std::memcpy(&result, output.data() + 4, 4);
  EXPECT_EQ(result, 0u);
  EXPECT_EQ(std::filesystem::file_size(dir / "memory.dat"), 0x10000u);
}

TEST_F(SaveDataTest, MemoryResultInvalidRequestsPreserveQueuedCompletion) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  ASSERT_EQ(MemorySync(1, 1), 0);
  std::array<u8, 60> output;
  output.fill(0xa5);
  const auto previous = output;
  u64 received = ~u64(0);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received, 59)), 0x809f0000u);
  EXPECT_EQ(output, previous);
  EXPECT_EQ(received, ~u64(0));
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received, 60, 1)),
            0x809f0000u);
  EXPECT_EQ(output, previous);
  EXPECT_EQ(received, ~u64(0));
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  EXPECT_EQ(received, 60u);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0018u);
}

TEST_F(SaveDataTest, AsyncMemorySyncQueueLimitPrecedesStorageAndTermClearsIt) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  for (u32 i = 0; i < 32; ++i)
    ASSERT_EQ(MemorySync(1, 1), 0);
  ASSERT_EQ(MemoryClear(), 0);
  ASSERT_EQ(MemorySetup(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory" / "memory.dat";
  std::ofstream(path) << "previous";
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 1)), 0x809f0003u);
  EXPECT_EQ(std::filesystem::file_size(path), 8u);
  std::array<u8, 60> output{};
  u64 received = ~u64(0);
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  ASSERT_EQ(MemorySync(1, 1), 0);
  ASSERT_EQ(MemoryClear(), 0);
  EXPECT_EQ(std::filesystem::file_size(path), 4096u);
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0018u);
}

TEST_F(SaveDataTest,
       AsyncMemorySyncRejectsSetupErrorsAndReportsSnapshotFailure) {
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 1)), 0x809f0001u);
  ASSERT_EQ(MemorySetup(), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 1)), 0x809f0008u);
  ASSERT_EQ(MemoryCreate(), 0);
  const u32 invalid = 1;
  ASSERT_EQ(kern::ShmWrite("/SceSaveDataI0000000000000001", 0, &invalid, 4), 0);
  ASSERT_EQ(MemorySync(1, 1), 0);
  std::array<u8, 60> output{};
  u64 received = ~u64(0);
  ASSERT_EQ(WaitMemoryResult(output, received), 0);
  u32 result = 0;
  std::memcpy(&result, output.data() + 4, 4);
  EXPECT_EQ(result, 0x809f000bu);
  EXPECT_EQ(static_cast<u32>(MemoryResult(output, received)), 0x809f0018u);
}

TEST_F(SaveDataTest, MemorySyncRejectsInvalidAndUnsupportedRequests) {
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f0001u);
  ASSERT_EQ(MemorySetup(), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f0008u);
  ASSERT_EQ(MemoryCreate(), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync(2)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 0, "other")), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 2)), 0x809f0000u);
  const auto path = root / "CUSA00792" / "sce_sdmemory" / "memory.dat";
  EXPECT_FALSE(std::filesystem::exists(path));
  ASSERT_EQ(MemoryClear(), 0);
  ASSERT_EQ(MemorySetup(4096, 128), 0);
  EXPECT_EQ(MemorySync(), 0);
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(path.parent_path() / ".sce_icon.bin"));
}

TEST_F(SaveDataTest,
       MemorySyncInvalidDescriptorsAndReservedFieldsPreserveStorage) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory" / "memory.dat";
  std::ofstream(path) << "previous";
  u64 input[] = {42, 84, 0};
  std::array<u8, 84> payload{};
  const u32 user = 1;
  std::memcpy(payload.data(), &user, 4);
  std::memcpy(payload.data() + 4, "sdmemory", 9);
  u64 received = ~u64(0);
  const auto request = [&](u64 payload_size, u64 output_size) {
    EXPECT_EQ(static_cast<u32>(Invoke(
                  0x12340002, {{input, 24}, {payload.data(), payload_size}},
                  {{nullptr, 0}, {&received, output_size}})),
              0x809f0000u);
    EXPECT_EQ(received, ~u64(0));
    EXPECT_EQ(std::filesystem::file_size(path), 8u);
  };
  request(83, 8);
  request(84, 7);
  input[1] = 83;
  request(84, 8);
  input[1] = 84;
  input[2] = 1;
  request(84, 8);
  input[2] = 0;
  payload[83] = 1;
  request(84, 8);
}

TEST_F(SaveDataTest,
       MemorySyncFailureReleasesSemaphoreAndPreservesPreviousFile) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory" / "memory.dat";
  std::ofstream(path) << "previous";
  const u32 invalid = 1;
  ASSERT_EQ(kern::ShmWrite("/SceSaveDataI0000000000000001", 0, &invalid, 4), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f000bu);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  ASSERT_EQ(kern::sys_osem_post(sem, 1), 0);
  EXPECT_EQ(std::filesystem::file_size(path), 8u);
  const u32 selected = 0;
  ASSERT_EQ(kern::ShmWrite("/SceSaveDataI0000000000000001", 0, &selected, 4),
            0);
  const int fd = kern::sys_shm_open("/SceSaveData0000000000000001_0", 2, 0777);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 4095), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f000bu);
  EXPECT_EQ(std::filesystem::file_size(path), 8u);
  ASSERT_EQ(kern::sys_ftruncate(fd, 4096), 0);
  ASSERT_EQ(kern::sys_close(fd), 0);
  ASSERT_EQ(MemorySync(), 0);
  EXPECT_EQ(std::filesystem::file_size(path), 4096u);
}

TEST_F(SaveDataTest, MemorySyncQuotaFailurePreservesPreviousFile) {
  ASSERT_EQ(MemorySetup(0x10000), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto dir = root / "CUSA00792" / "sce_sdmemory";
  std::ofstream(dir / "memory.dat") << "previous";
  std::filesystem::resize_file(dir / "memory.dat", 0x8000);
  std::ofstream(dir / "filler");
  std::filesystem::resize_file(dir / "filler", 95 * 0x8000);
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f000au);
  EXPECT_EQ(std::filesystem::file_size(dir / "memory.dat"), 0x8000u);
  char previous[8];
  std::ifstream file(dir / "memory.dat");
  ASSERT_TRUE(file.read(previous, sizeof(previous)));
  EXPECT_EQ(std::memcmp(previous, "previous", sizeof(previous)), 0);
  EXPECT_FALSE(std::filesystem::exists(dir / "memory.dat.tmp"));
  std::filesystem::remove(dir / "filler");
  ASSERT_EQ(MemorySync(), 0);
  EXPECT_EQ(std::filesystem::file_size(dir / "memory.dat"), 0x10000u);
}

TEST_F(SaveDataTest, MemorySyncWaitsForGuestWritesBeforeTakingSnapshot) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  std::atomic<bool> started = false, completed = false;
  i32 result = -1;
  std::thread sync([&] {
    started = true;
    result = MemorySync();
    completed = true;
  });
  while (!started)
    std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(completed);
  std::array<u8, 4096> memory;
  memory.fill(117);
  EXPECT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_0", 0, memory.data(),
                           memory.size()),
            0);
  EXPECT_EQ(kern::sys_osem_post(sem, 1), 0);
  sync.join();
  ASSERT_EQ(result, 0);
  std::array<u8, 4096> loaded{};
  std::ifstream file(root / "CUSA00792" / "sce_sdmemory" / "memory.dat",
                     std::ios::binary);
  ASSERT_TRUE(file.read(reinterpret_cast<char*>(loaded.data()), loaded.size()));
  EXPECT_EQ(loaded, memory);
  EXPECT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  EXPECT_EQ(kern::sys_osem_post(sem, 1), 0);
}

TEST_F(SaveDataTest, MemorySyncPreservesMountedSaveUntilUnmount) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  const auto path = root / "CUSA00792" / "sce_sdmemory" / "memory.dat";
  std::ofstream(path) << "previous";
  const u32 command = 1;
  std::memcpy(header.data() + 4, &command, 4);
  std::memset(data.data() + 0x414, 0, 32);
  std::memcpy(data.data() + 0x414, "sce_sdmemory", 13);
  Mode(1);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync()), 0x809f0003u);
  EXPECT_EQ(std::filesystem::file_size(path), 8u);
  ASSERT_EQ(Unmount(), 0);
  ASSERT_EQ(MemorySync(), 0);
  EXPECT_EQ(std::filesystem::file_size(path), 4096u);
}

TEST_F(SaveDataTest,
       MemoryConfirmationValidatesConfigurationAndPreservesOutput) {
  const auto rejected = [&](u32 app, u32 user, u64 size) {
    reply.fill(0xa5);
    written = ~u64(0);
    EXPECT_EQ(static_cast<u32>(MemoryConfirm(app, user, size)), 0x809f0000u);
    EXPECT_EQ(written, ~u64(0));
    for (u8 value : reply)
      EXPECT_EQ(value, 0xa5);
  };
  rejected(0, 1, 4096);
  ASSERT_EQ(MemorySetup(), 0);
  rejected(1, 1, 4096);
  rejected(0, 2, 4096);
  rejected(0, 1, 0);
  rejected(0, 1, 8192);
  ASSERT_EQ(MemoryConfirm(), 0);
  ASSERT_EQ(MemoryClear(), 0);
  rejected(0, 1, 4096);
  ASSERT_EQ(MemorySetup(8192), 0);
  rejected(0, 1, 4096);
  ASSERT_EQ(MemoryConfirm(0, 1, 8192), 0);
}

TEST_F(SaveDataTest, MemoryConfirmationPreservesBuffersControlAndSemaphore) {
  ASSERT_EQ(MemorySetup(4096, 128, 0x530, 2), 0);
  const char* buffer = "/SceSaveData0000000000000001_1";
  const char* control = "/SceSaveDataI0000000000000001";
  const u8 marker = 97;
  const u32 index = 1;
  ASSERT_EQ(kern::ShmWrite(buffer, 0, &marker, 1), 0);
  ASSERT_EQ(kern::ShmWrite(control, 0, &index, sizeof(index)), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_trywait(sem, 1), 0);
  for (int i = 0; i < 100; ++i) {
    reply.fill(0xa5);
    ASSERT_EQ(MemoryConfirm(), 0);
    EXPECT_EQ(written, 32u);
    for (size_t j = 0; j < reply.size(); ++j)
      EXPECT_EQ(reply[j], j < 32 ? 0 : 0xa5);
  }
  EXPECT_EQ(kern::sys_osem_trywait(sem, 1), -kern::SysError::eBUSY);
  ASSERT_EQ(kern::sys_osem_post(sem, 1), 0);
  u8 loaded = 0;
  ASSERT_EQ(kern::ShmRead(buffer, 0, &loaded, 1), 0);
  EXPECT_EQ(loaded, marker);
  u32 selected = 0;
  ASSERT_EQ(kern::ShmRead(control, 0, &selected, sizeof(selected)), 0);
  EXPECT_EQ(selected, index);
}

TEST_F(SaveDataTest, MemoryConfirmationRejectsMissingResources) {
  ASSERT_EQ(MemorySetup(4096, 0, 0, 2), 0);
  ASSERT_EQ(kern::sys_shm_unlink("/SceSaveData0000000000000001_1"), 0);
  EXPECT_EQ(static_cast<u32>(MemoryConfirm()), 0x809f000bu);
  ASSERT_EQ(MemoryClear(), 0);
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(kern::sys_shm_unlink("/SceSaveDataI0000000000000001"), 0);
  EXPECT_EQ(static_cast<u32>(MemoryConfirm()), 0x809f000bu);
  ASSERT_EQ(MemoryClear(), 0);
  ASSERT_EQ(MemorySetup(), 0);
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_close(sem), 0);
  EXPECT_EQ(static_cast<u32>(MemoryConfirm()), 0x809f000bu);
}

TEST_F(SaveDataTest, MemoryConfirmationRejectsTruncatedBacking) {
  ASSERT_EQ(MemorySetup(4096, 128, 0x530, 2), 0);
  for (const auto& [name, total] :
       {std::pair<const char*, u64>{"/SceSaveData0000000000000001_1",
                                    4096 + 64 + 128 + 0x530},
        {"/SceSaveDataI0000000000000001", 64}}) {
    const int fd = kern::sys_shm_open(name, 2, 0777);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(kern::sys_ftruncate(fd, total - 1), 0);
    EXPECT_EQ(static_cast<u32>(MemoryConfirm()), 0x809f000bu);
    ASSERT_EQ(kern::sys_ftruncate(fd, total), 0);
    ASSERT_EQ(MemoryConfirm(), 0);
    ASSERT_EQ(kern::sys_close(fd), 0);
  }
}

TEST_F(SaveDataTest, InvalidMemorySetupAndClearPreserveExistingResources) {
  for (const auto& [size, icon, param, count] :
       {std::tuple<u64, u64, u64, u32>{0, 0, 0, 1},
        {0x2000001, 0, 0, 1},
        {4096, 0x1c801, 0, 1},
        {4096, 0, 1, 1},
        {4096, 0, 0, 0},
        {4096, 0, 0, 3}}) {
    reply.fill(0xa5);
    written = ~u64(0);
    EXPECT_EQ(static_cast<u32>(MemorySetup(size, icon, param, count)),
              0x809f0000u);
    EXPECT_EQ(written, ~u64(0));
    for (u8 value : reply)
      EXPECT_EQ(value, 0xa5);
  }
  ASSERT_EQ(MemorySetup(), 0);
  const u8 marker = 71;
  ASSERT_EQ(kern::ShmWrite("/SceSaveData0000000000000001_0", 0, &marker, 1), 0);
  EXPECT_EQ(static_cast<u32>(MemorySetup()), 0x809f0003u);
  EXPECT_EQ(static_cast<u32>(MemoryClear(2)), 0x809f0000u);
  u8 loaded = 0;
  ASSERT_EQ(kern::ShmRead("/SceSaveData0000000000000001_0", 0, &loaded, 1), 0);
  EXPECT_EQ(loaded, marker);
}

TEST_F(SaveDataTest, MemorySetupCollisionRollsBackOnlyNewResources) {
  const char* second = "/SceSaveData0000000000000001_1";
  ASSERT_EQ(kern::ShmCreate(second, 4), 0);
  const u8 marker = 81;
  ASSERT_EQ(kern::ShmWrite(second, 0, &marker, 1), 0);
  EXPECT_EQ(static_cast<u32>(MemorySetup(4096, 0, 0, 2)), 0x809f0003u);
  u8 value = 0;
  EXPECT_EQ(kern::ShmRead("/SceSaveData0000000000000001_0", 0, &value, 1),
            -kern::SysError::eNOENT);
  ASSERT_EQ(kern::ShmRead(second, 0, &value, 1), 0);
  EXPECT_EQ(value, marker);
  ASSERT_EQ(kern::sys_shm_unlink(second), 0);
  ASSERT_EQ(MemorySetup(4096, 0, 0, 2), 0);
}

TEST_F(SaveDataTest, MemoryClearPreservesLiveMappingsAndRecreatesFreshBacking) {
  ASSERT_EQ(MemorySetup(), 0);
  const int fd = kern::sys_shm_open("/SceSaveData0000000000000001_0", 2, 0777);
  ASSERT_GE(fd, 0);
  auto* view = kern::sys_mmap(nullptr, 0x4000, 3, 1, fd, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(view));
  view[0] = 93;
  const int sem = kern::sys_osem_open("SceSaveData0000000000000001");
  ASSERT_GE(sem, 0);
  ASSERT_EQ(kern::sys_osem_close(sem), 0);
  ASSERT_EQ(MemoryClear(), 0);
  EXPECT_EQ(view[0], 93);
  ASSERT_EQ(MemorySetup(), 0);
  u8 value = 0xa5;
  ASSERT_EQ(kern::ShmRead("/SceSaveData0000000000000001_0", 0, &value, 1), 0);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(view[0], 93);
  ASSERT_EQ(Initialize(63), 0);
  EXPECT_EQ(kern::ShmRead("/SceSaveData0000000000000001_0", 0, &value, 1),
            -kern::SysError::eNOENT);
  EXPECT_EQ(view[0], 93);
  ASSERT_EQ(kern::sys_close(fd), 0);
}

TEST_F(SaveDataTest, MemoryCleanupPreservesReplacementSemaphore) {
  ASSERT_EQ(MemorySetup(), 0);
  const char* name = "SceSaveData0000000000000001";
  const int old = kern::sys_osem_open(name);
  ASSERT_GE(old, 0);
  const int replacement = kern::sys_osem_create(name, 0, 1, 1);
  ASSERT_GE(replacement, 0);
  ASSERT_NE(replacement, old);
  ASSERT_EQ(MemoryClear(), 0);
  EXPECT_EQ(kern::sys_osem_open(name), replacement);
  EXPECT_EQ(kern::sys_osem_trywait(replacement, 1), 0);
  EXPECT_EQ(kern::sys_osem_post(replacement, 1), 0);
  EXPECT_EQ(kern::sys_osem_delete(replacement), 0);
}

TEST_F(SaveDataTest, LegacyMountUsesPersistentStorageAndQuota) {
  const u32 command = 1;
  const u64 blocks = 96;
  std::memcpy(header.data() + 4, &command, 4);
  std::memcpy(data.data() + 0x488, &blocks, 8);
  Mode(34);
  reply.fill(0xa5);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(reply.data()), "/savedata0");
  EXPECT_EQ(reply[24], 1u);
  EXPECT_EQ(written, 32u);
  for (size_t i = 32; i < reply.size(); ++i)
    EXPECT_EQ(reply[i], 0xa5);
  const int writer =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(writer, 0);
  const char payload[] = "native memory save backing";
  ASSERT_EQ(kern::sys_write(writer, payload, sizeof(payload)), sizeof(payload));
  EXPECT_EQ(FreeBlocks(), 95u);
  ASSERT_EQ(kern::sys_close(writer), 0);
  ASSERT_EQ(Unmount(), 0);
  Mode(1);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(reply[24], 0u);
  EXPECT_EQ(FreeBlocks(), 95u);
  const int reader = kern::sys_open("/savedata0/data", kern::O_RDONLY, 0);
  ASSERT_GE(reader, 0);
  std::array<char, sizeof(payload)> loaded{};
  ASSERT_EQ(kern::sys_read(reader, loaded.data(), loaded.size()),
            loaded.size());
  EXPECT_EQ(std::memcmp(loaded.data(), payload, sizeof(payload)), 0);
  EXPECT_EQ(kern::sys_truncate("/savedata0/data", 0), -kern::SysError::eROFS);
  ASSERT_EQ(kern::sys_close(reader), 0);
  ASSERT_EQ(Unmount(), 0);
}

TEST_F(SaveDataTest, LegacyMountFailurePreservesOutputAndStorage) {
  const u32 command = 1;
  std::memcpy(header.data() + 4, &command, 4);
  Mode(1);
  reply.fill(0xa5);
  written = ~u64(0);
  ASSERT_EQ(Submit(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0008u);
  EXPECT_EQ(written, ~u64(0));
  for (u8 value : reply)
    EXPECT_EQ(value, 0xa5);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "TESTSAVE"));
}

TEST_F(SaveDataTest, MountCreatesWritableStorageAndReportsCreation) {
  reply.fill(0xa5);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(reply.data()), "/savedata0");
  EXPECT_EQ(reply[24], 1u);
  EXPECT_EQ(written, 32u);
  for (size_t i = 32; i < reply.size(); ++i)
    EXPECT_EQ(reply[i], 0xa5);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  EXPECT_EQ(kern::vfs::ResolveWritable("/savedata0/file"),
            (host / "file").string().c_str());
  std::ofstream(host / "file") << "persisted save";
  ASSERT_EQ(Initialize(63), 0);
  EXPECT_TRUE(kern::vfs::Resolve("/savedata0/file").empty());
  ASSERT_EQ(Initialize(), 0);
  Mode(2);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(reply[24], 0u);
  auto file = kern::vfs::OpenRead("/savedata0/file");
  ASSERT_TRUE(file.IsOpen());
  char contents[15]{};
  ASSERT_EQ(file.Read(contents, 14), 14u);
  EXPECT_STREQ(contents, "persisted save");
}

TEST_F(SaveDataTest, UnmountReleasesStorageAndReturnsMountMode) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::memcpy(data.data(), reply.data(), 16);
  u32 command = 2;
  std::memcpy(header.data() + 4, &command, 4);
  ASSERT_EQ(Submit(20), 0);
  EXPECT_FALSE(kern::vfs::Resolve("/savedata0/file").empty());
  EXPECT_EQ(static_cast<u32>(Collect(131, 132)), 0x809f0000u);
  ASSERT_EQ(Collect(132, 132), 0);
  EXPECT_TRUE(kern::vfs::Resolve("/savedata0/file").empty());
  u32 mode;
  std::memcpy(&mode, reply.data(), 4);
  EXPECT_EQ(mode, 6u);
  EXPECT_EQ(written, 132u);
  ASSERT_EQ(Submit(20), 0);
  EXPECT_EQ(static_cast<u32>(Collect(132, 132)), 0x809f0004u);
  EXPECT_TRUE(std::filesystem::is_directory(root / "CUSA00792" / "TESTSAVE"));
}

TEST_F(SaveDataTest, NativeMountSlotsAreBoundedAndReusable) {
  Mode(34);
  for (u32 i = 0; i < 16; ++i) {
    ASSERT_EQ(Submit(), 0);
    ASSERT_EQ(Collect(), 0);
    const auto point = "/savedata" + std::to_string(i);
    EXPECT_STREQ(reinterpret_cast<char*>(reply.data()), point.c_str());
  }
  ASSERT_EQ(Submit(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f000bu);
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(reply.data()), "/savedata0");
}

TEST_F(SaveDataTest, ReadOnlyLegacyMountAndMissingSave) {
  Mode(1);
  ASSERT_EQ(Submit(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0008u);
  const auto host = root / "TESTSAVE";
  std::filesystem::create_directory(host);
  std::ofstream(host / "file") << "legacy";
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_FALSE(kern::vfs::Resolve("/savedata0/file").empty());
  EXPECT_TRUE(kern::vfs::ResolveWritable("/savedata0/file").empty());
  EXPECT_EQ(kern::sys_open("/savedata0/file", 2, 0),
            -static_cast<i32>(kern::SysError::eROFS));
  EXPECT_EQ(kern::sys_open("/savedata0/newfile", 0x602, 0600),
            -static_cast<i32>(kern::SysError::eROFS));
  EXPECT_FALSE(std::filesystem::exists(host / "newfile"));
  EXPECT_EQ(reply[24], 0u);
  Mode(6);
  ASSERT_EQ(Submit(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0007u);
}

TEST_F(SaveDataTest, BadCollectionDoesNotConsumeRequestOrCreateFiles) {
  ASSERT_EQ(Submit(), 0);
  reply.fill(0xa5);
  written = 81;
  EXPECT_EQ(static_cast<u32>(Collect(31)), 0x809f0000u);
  EXPECT_EQ(written, 81u);
  EXPECT_EQ(reply[0], 0xa5);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792"));
  header[16] = 1;
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
  header[16] = 0;
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
}

TEST_F(SaveDataTest, BackupMissingAndRestoreFailureNotificationAreRealResults) {
  std::vector<u8> output;
  EXPECT_EQ(static_cast<u32>(BackupCommand(37, BackupPayload(37), output)),
            0x809f0008u);
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, BackupPayload(36), output)),
            0x809f0008u);
  auto notification = BackupPayload(46);
  const u32 result = 0x809f0008u;
  std::memcpy(notification.data() + 56, &result, 4);
  ASSERT_EQ(BackupCommand(46, notification, output), 0);
  std::array<u8, 60> event{};
  u64 received = ~u64(0);
  ASSERT_EQ(MemoryResult(event, received), 0);
  std::array<u8, 60> expected{};
  const u32 type = 1, user = 1;
  std::memcpy(expected.data(), &type, 4);
  std::memcpy(expected.data() + 4, &result, 4);
  std::memcpy(expected.data() + 8, &user, 4);
  std::memcpy(expected.data() + 28, "TESTSAVE", 9);
  EXPECT_EQ(event, expected);
  EXPECT_EQ(static_cast<u32>(BackupCommand(46, notification, output)),
            0x809f0000u);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792"));
}

TEST_F(SaveDataTest, BackupUnmountCheckAndRestorePreserveWholeSave) {
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  const auto backup = root / "CUSA00792" / "TESTSAVE.backup";
  std::filesystem::create_directories(host / "nested" / "empty");
  std::ofstream(host / "nested" / "data", std::ios::binary) << "original";
  std::array<u8, 1328> params;
  params.fill(0x71);
  std::ofstream(host / ".sce_param.bin", std::ios::binary)
      .write(reinterpret_cast<const char*>(params.data()), params.size());
  std::ofstream(host / ".sce_icon.bin", std::ios::binary) << "icon";
  ASSERT_EQ(Unmount(1), 0);
  EXPECT_TRUE(kern::vfs::Resolve("/savedata0/file").empty());
  EXPECT_EQ(std::filesystem::file_size(backup / "nested" / "data"), 8u);
  EXPECT_TRUE(std::filesystem::is_directory(backup / "nested" / "empty"));
  auto payload = BackupPayload(37);
  payload[52] = payload[53] = 1;
  std::vector<u8> output;
  ASSERT_EQ(BackupCommand(37, payload, output), 0);
  EXPECT_EQ(std::memcmp(output.data(), params.data(), params.size()), 0);
  EXPECT_EQ(std::memcmp(output.data() + 0x530, "icon", 4), 0);
  u32 length = 0;
  std::memcpy(&length, output.data() + 0x1cd30, 4);
  EXPECT_EQ(length, 4u);
  std::ofstream(host / "nested" / "data", std::ios::binary) << "changed";
  std::ofstream(host / "added", std::ios::binary) << "new";
  std::filesystem::remove(host / ".sce_icon.bin");
  ASSERT_EQ(BackupCommand(36, BackupPayload(36), output), 0);
  char stored[8];
  std::ifstream file(host / "nested" / "data", std::ios::binary);
  ASSERT_TRUE(file.read(stored, sizeof(stored)));
  EXPECT_EQ(std::memcmp(stored, "original", sizeof(stored)), 0);
  EXPECT_FALSE(std::filesystem::exists(host / "added"));
  EXPECT_EQ(std::filesystem::file_size(host / ".sce_icon.bin"), 4u);
  EXPECT_EQ(std::filesystem::file_size(backup / "nested" / "data"), 8u);
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".old"));
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".tmp"));
  ASSERT_EQ(BackupCommand(46, BackupPayload(46), output), 0);
  ASSERT_EQ(Search(output), 0);
  u32 count = 0;
  std::memcpy(&count, output.data() + 4, 4);
  EXPECT_EQ(count, 1u);
}

TEST_F(SaveDataTest, BackupReplacementFailurePreservesPreviousBackupAndMount) {
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::ofstream(host / "file") << "original";
  ASSERT_EQ(Unmount(1), 0);
  Mode(2);
  ASSERT_EQ(SmallMount(), 0);
  std::ofstream(host / "file") << "replacement";
  const auto staging = host.string() + ".backup.tmp";
  std::ofstream(staging) << "foreign";
  EXPECT_EQ(static_cast<u32>(Unmount(1)), 0x809f0003u);
  EXPECT_FALSE(kern::vfs::Resolve("/savedata0/file").empty());
  EXPECT_EQ(std::filesystem::file_size(host.string() + ".backup/file"), 8u);
  EXPECT_EQ(std::filesystem::file_size(staging), 7u);
  std::filesystem::remove(staging);
  ASSERT_EQ(Unmount(1), 0);
  EXPECT_EQ(std::filesystem::file_size(host.string() + ".backup/file"), 11u);
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".backup.old"));
}

TEST_F(SaveDataTest, BackupRejectsSymlinksWithoutChangingOutsideFiles) {
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::ofstream(root / "outside") << "outside";
  std::filesystem::create_symlink(root / "outside", host / "unsafe");
  EXPECT_EQ(static_cast<u32>(Unmount(1)), 0x809f0000u);
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".backup"));
  EXPECT_EQ(std::filesystem::file_size(root / "outside"), 7u);
  std::filesystem::remove(host / "unsafe");
  std::filesystem::create_directory_symlink(root, host.string() + ".backup");
  EXPECT_EQ(static_cast<u32>(Unmount(1)), 0x809f0000u);
  EXPECT_EQ(std::filesystem::file_size(root / "outside"), 7u);
}

TEST_F(SaveDataTest, BackupAndRestoreRejectLiveFilesAndMountedSave) {
  ASSERT_EQ(SmallMount(), 0);
  const int fd = kern::sys_open("/savedata0/file", 0x202, 0600);
  ASSERT_GE(fd, 0);
  EXPECT_EQ(static_cast<u32>(Unmount(1)), 0x809f0003u);
  ASSERT_EQ(kern::sys_close(fd), 0);
  ASSERT_EQ(Unmount(1), 0);
  Mode(2);
  ASSERT_EQ(SmallMount(), 0);
  std::vector<u8> output;
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, BackupPayload(36), output)),
            0x809f0003u);
  const int retained = kern::sys_open("/savedata0/file", 0, 0);
  ASSERT_GE(retained, 0);
  ASSERT_EQ(Unmount(), 0);
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, BackupPayload(36), output)),
            0x809f0003u);
  ASSERT_EQ(kern::sys_close(retained), 0);
  ASSERT_EQ(BackupCommand(36, BackupPayload(36), output), 0);
}

TEST_F(SaveDataTest, BackupCollectionCapacityRetriesBeforeFilesystemEffects) {
  ASSERT_EQ(SmallMount(), 0);
  ASSERT_EQ(Unmount(1), 0);
  std::vector<u8> output;
  ASSERT_EQ(BackupSubmit(37, BackupPayload(37)), 0);
  EXPECT_EQ(static_cast<u32>(BackupCollect(37, output, 0x1cd3f)), 0x809f0000u);
  ASSERT_EQ(BackupCollect(37, output), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::ofstream(host / "added") << "new";
  ASSERT_EQ(BackupSubmit(36, BackupPayload(36)), 0);
  EXPECT_EQ(static_cast<u32>(BackupCollect(36, output, 35)), 0x809f0000u);
  EXPECT_TRUE(std::filesystem::exists(host / "added"));
  ASSERT_EQ(BackupCollect(36, output), 0);
  EXPECT_FALSE(std::filesystem::exists(host / "added"));
}

TEST_F(SaveDataTest, BackupCopyFailurePreservesPreviousSnapshotAndCurrentSave) {
  if (geteuid() == 0)
    GTEST_SKIP() << "Root bypasses the file-permission failure.";
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  const auto backup = std::filesystem::path(host.string() + ".backup");
  std::ofstream(host / "file") << "original";
  ASSERT_EQ(Unmount(1), 0);
  Mode(2);
  ASSERT_EQ(SmallMount(), 0);
  std::ofstream(host / "file") << "changed";
  std::ofstream(host / "unreadable") << "private";
  std::filesystem::permissions(host / "unreadable",
                               std::filesystem::perms::none);
  EXPECT_EQ(static_cast<u32>(Unmount(1)), 0x809f000bu);
  EXPECT_EQ(std::filesystem::file_size(backup / "file"), 8u);
  EXPECT_FALSE(std::filesystem::exists(backup / "unreadable"));
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".backup.tmp"));
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".backup.old"));
  EXPECT_FALSE(kern::vfs::Resolve("/savedata0/file").empty());
  std::filesystem::permissions(host / "unreadable",
                               std::filesystem::perms::owner_all);
  ASSERT_EQ(Unmount(), 0);
  std::ofstream(backup / "unreadable") << "private";
  std::filesystem::permissions(backup / "unreadable",
                               std::filesystem::perms::none);
  std::vector<u8> output;
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, BackupPayload(36), output)),
            0x809f000bu);
  EXPECT_EQ(std::filesystem::file_size(host / "file"), 7u);
  EXPECT_TRUE(std::filesystem::exists(host / "unreadable"));
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".tmp"));
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".old"));
  std::filesystem::permissions(backup / "unreadable",
                               std::filesystem::perms::owner_all);
}

TEST_F(SaveDataTest,
       RestoreResultsReserveCompletionCapacityBeforeMemoryWrites) {
  ASSERT_EQ(MemorySetup(), 0);
  ASSERT_EQ(MemoryCreate(), 0);
  std::vector<u8> output;
  for (u32 i = 0; i < 32; ++i) {
    const auto dir = "RESTORE" + std::to_string(i);
    EXPECT_EQ(static_cast<u32>(
                  BackupCommand(36, BackupPayload(36, dir.c_str()), output)),
              0x809f0008u);
  }
  EXPECT_EQ(
      static_cast<u32>(BackupCommand(36, BackupPayload(36, "NEXT"), output)),
      0x809f0003u);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 1)), 0x809f0003u);
  auto notification = BackupPayload(46, "RESTORE0");
  const u32 result = 0x809f0008u;
  std::memcpy(notification.data() + 56, &result, 4);
  ASSERT_EQ(BackupCommand(46, notification, output), 0);
  EXPECT_EQ(static_cast<u32>(MemorySync(1, 1)), 0x809f0003u);
  std::array<u8, 60> event{};
  u64 received = ~u64(0);
  ASSERT_EQ(MemoryResult(event, received), 0);
  ASSERT_EQ(MemorySync(1, 1), 0);
  ASSERT_EQ(WaitMemoryResult(event, received), 0);
}

TEST_F(SaveDataTest, LegacyBackupCanRestoreAfterPrimaryDirectoryDisappears) {
  const auto legacy = root / "TESTSAVE";
  std::filesystem::create_directory(legacy);
  std::ofstream(legacy / "file") << "legacy";
  Mode(2);
  ASSERT_EQ(SmallMount(), 0);
  ASSERT_EQ(Unmount(1), 0);
  std::filesystem::remove_all(legacy);
  std::vector<u8> output;
  ASSERT_EQ(BackupCommand(37, BackupPayload(37), output), 0);
  ASSERT_EQ(BackupCommand(36, BackupPayload(36), output), 0);
  EXPECT_EQ(std::filesystem::file_size(legacy / "file"), 6u);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "TESTSAVE"));
  EXPECT_EQ(std::filesystem::file_size(root / "TESTSAVE.backup" / "file"), 6u);
}

TEST_F(SaveDataTest, BackupValidationPreservesStorageAndRejectsForgedResults) {
  ASSERT_EQ(SmallMount(), 0);
  EXPECT_EQ(static_cast<u32>(Unmount(2)), 0x809f0000u);
  ASSERT_EQ(Unmount(1), 0);
  std::vector<u8> output;
  auto check = BackupPayload(37);
  for (size_t offset : {size_t(52), size_t(53), size_t(54), size_t(55)}) {
    check[offset] = 2;
    EXPECT_EQ(static_cast<u32>(BackupCommand(37, check, output)), 0x809f0000u);
    check[offset] = 0;
  }
  auto restore = BackupPayload(36);
  restore[135] = 1;
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, restore, output)), 0x809f0000u);
  auto notification = BackupPayload(46);
  EXPECT_EQ(static_cast<u32>(BackupCommand(46, notification, output)),
            0x809f0000u);
  const u32 result = 0x809f0000u;
  std::memcpy(notification.data() + 56, &result, 4);
  ASSERT_EQ(BackupCommand(46, notification, output), 0);
  EXPECT_TRUE(std::filesystem::is_directory(root / "CUSA00792" / "TESTSAVE"));
}

TEST_F(SaveDataTest, DestroyClientReleasesMounts) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  ASSERT_FALSE(kern::vfs::Resolve("/savedata0/file").empty());
  u32 output = 0;
  EXPECT_EQ(kern::ipmi::ManagerCall(3, kid, &output, nullptr, 0), 0);
  EXPECT_TRUE(kern::vfs::Resolve("/savedata0/file").empty());
}

TEST_F(SaveDataTest, InvalidNamesModesUsersAndSymlinksNeverMount) {
  for (const char* name : {"../escape", "", "sub/dir"}) {
    std::memset(data.data() + 0x414, 0, 32);
    std::memcpy(data.data() + 0x414, name, std::strlen(name));
    ASSERT_EQ(Submit(), 0);
    EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
  }
  std::memcpy(data.data() + 0x414, "TESTSAVE", 9);
  for (u32 mode : {0u, 3u, 5u, 0x26u, 0x80000006u}) {
    Mode(mode);
    ASSERT_EQ(Submit(), 0);
    EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
  }
  Mode(6);
  u32 user = 2;
  std::memcpy(data.data() + 0x400, &user, 4);
  ASSERT_EQ(Submit(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
  user = 1;
  std::memcpy(data.data() + 0x400, &user, 4);
  std::filesystem::create_directory_symlink(root, root / "TESTSAVE");
  ASSERT_EQ(Submit(), 0);
  EXPECT_EQ(static_cast<u32>(Collect()), 0x809f0000u);
}

TEST_F(SaveDataTest, UnsupportedCommandsAndUninitializedClientsReturnErrors) {
  u32 command = 61;
  std::memcpy(header.data() + 4, &command, 4);
  EXPECT_EQ(Submit(), -static_cast<i32>(kern::SysError::eNOSYS));
  EXPECT_EQ(Invoke(0x1234ffff), -static_cast<i32>(kern::SysError::eNOSYS));
  ASSERT_EQ(Initialize(63), 0);
  EXPECT_EQ(static_cast<u32>(Submit()), 0x809f0001u);
}
TEST_F(SaveDataTest, ParamFieldsPersistWithoutOverwritingOtherFields) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  const std::array<u32, 6> sizes = {1328, 128, 128, 1024, 4, 8};
  for (u32 type = 1; type < sizes.size(); ++type) {
    std::vector<u8> value(sizes[type], static_cast<u8>(type + 16));
    ASSERT_EQ(Param(type, output, value.data(), value.size()), 0);
    ASSERT_EQ(Param(type, output), 0);
    EXPECT_EQ(std::memcmp(output.data(), value.data(), value.size()), 0);
    u32 length;
    std::memcpy(&length, output.data() + 0x1c800, 4);
    EXPECT_EQ(length, value.size());
    for (size_t i = 0x1c804; i < output.size(); ++i)
      EXPECT_EQ(output[i], 0xa5);
  }
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  Mode(2);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  ASSERT_EQ(Param(0, output), 0);
  const std::array<u32, 6> offsets = {0, 0, 128, 256, 1280, 1288};
  for (u32 type = 1; type < sizes.size(); ++type)
    for (u32 i = 0; i < sizes[type]; ++i)
      EXPECT_EQ(output[offsets[type] + i], type + 16);
  EXPECT_EQ(std::filesystem::file_size(root / "CUSA00792" / "TESTSAVE" /
                                       ".sce_param.bin"),
            1328u);
}

TEST_F(SaveDataTest, LegacyParamsReadAndReadOnlyWritesFail) {
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::filesystem::create_directories(host);
  std::array<u8, 1328> old{};
  std::memcpy(old.data(), "Legacy title", 13);
  std::ofstream file(host / ".sce_param.bin", std::ios::binary);
  file.write(reinterpret_cast<const char*>(old.data()), old.size());
  file.close();
  Mode(1);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  ASSERT_EQ(Param(1, output), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data()), "Legacy title");
  const char text[] = "Replacement";
  EXPECT_EQ(Param(1, output, text, sizeof(text)),
            -static_cast<i32>(kern::SysError::eROFS));
  ASSERT_EQ(Param(1, output), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data()), "Legacy title");
}

TEST_F(SaveDataTest, InvalidParamRequestsDoNotCreateMetadata) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  std::array<u8, 1330> value{};
  EXPECT_EQ(static_cast<u32>(Param(99, output)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(Param(1, output, value.data(), 129)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(Param(1, output, value.data(), 0)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(Param(1, output, value.data(), 128, 127)),
            0x809f0000u);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "TESTSAVE" /
                                       ".sce_param.bin"));
}

TEST_F(SaveDataTest, IconsPersistAtMaximumSizeAndCanBeReplaced) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output, value(0x1c800);
  for (size_t i = 0; i < value.size(); ++i)
    value[i] = static_cast<u8>(i);
  ASSERT_EQ(Icon(output, value.data(), value.size()), 0);
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  Mode(2);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  ASSERT_EQ(Icon(output), 0);
  EXPECT_EQ(std::memcmp(output.data(), value.data(), value.size()), 0);
  u32 length;
  std::memcpy(&length, output.data() + 0x1c800, 4);
  EXPECT_EQ(length, value.size());
  for (size_t i = 0x1c884; i < output.size(); ++i)
    EXPECT_EQ(output[i], 0xa5);
  ASSERT_EQ(Icon(output, value.data(), 24), 0);
  ASSERT_EQ(Icon(output), 0);
  EXPECT_EQ(std::memcmp(output.data(), value.data(), 24), 0);
  EXPECT_EQ(output[24], 0u);
  std::memcpy(&length, output.data() + 0x1c800, 4);
  EXPECT_EQ(length, 24u);
  EXPECT_EQ(std::filesystem::file_size(root / "CUSA00792" / "TESTSAVE" /
                                       ".sce_icon.bin"),
            24u);
}

TEST_F(SaveDataTest, InvalidIconRequestsDoNotWriteAndMissingIconsFail) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  const u8 value = 42;
  EXPECT_EQ(static_cast<u32>(Icon(output)), 0x809f000eu);
  EXPECT_EQ(static_cast<u32>(Icon(output, &value, 0)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(Icon(output, &value, 0x1c801)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(Icon(output, &value, 1, 127)), 0x809f0000u);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "TESTSAVE" /
                                       ".sce_icon.bin"));
}

TEST_F(SaveDataTest, LegacyIconsReadAndReadOnlyWritesFail) {
  const auto host = root / "TESTSAVE";
  std::filesystem::create_directories(host);
  std::ofstream(host / ".sce_icon.bin", std::ios::binary) << "Legacy icon";
  Mode(1);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  ASSERT_EQ(Icon(output), 0);
  EXPECT_EQ(std::memcmp(output.data(), "Legacy icon", 11), 0);
  const char value[] = "Replacement";
  EXPECT_EQ(Icon(output, value, sizeof(value)),
            -static_cast<i32>(kern::SysError::eROFS));
  ASSERT_EQ(Icon(output), 0);
  EXPECT_EQ(std::memcmp(output.data(), "Legacy icon", 11), 0);
}

TEST_F(SaveDataTest, IconStorageFailuresPreserveExistingData) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  const auto icon = host / ".sce_icon.bin";
  std::ofstream(icon, std::ios::binary) << "Existing icon";
  std::filesystem::create_directory(host / ".sce_icon.bin.tmp");
  std::vector<u8> output;
  const char value[] = "Replacement";
  EXPECT_EQ(static_cast<u32>(Icon(output, value, sizeof(value))), 0x809f000bu);
  ASSERT_EQ(Icon(output), 0);
  EXPECT_EQ(std::memcmp(output.data(), "Existing icon", 13), 0);
  std::filesystem::resize_file(icon, 0x1c801);
  EXPECT_EQ(static_cast<u32>(Icon(output)), 0x809f0000u);
}

TEST_F(SaveDataTest, DeleteRetiresBackupAndCannotRestoreDeletedContents) {
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  const auto backup = root / "CUSA00792" / "TESTSAVE.backup";
  std::filesystem::create_directory(host / "nested");
  std::ofstream(host / "nested" / "data") << "deleted contents";
  ASSERT_EQ(Unmount(1), 0);
  const auto outside = root / "outside";
  std::filesystem::create_directory(outside);
  std::ofstream(outside / "data") << "preserved";
  std::filesystem::create_directory_symlink(outside, backup / "outside");
  ASSERT_EQ(Delete(), 0);
  EXPECT_FALSE(std::filesystem::exists(host));
  EXPECT_FALSE(std::filesystem::exists(backup));
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".delete"));
  EXPECT_EQ(std::filesystem::file_size(outside / "data"), 9u);
  std::vector<u8> output;
  EXPECT_EQ(static_cast<u32>(BackupCommand(37, BackupPayload(37), output)),
            0x809f0008u);
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, BackupPayload(36), output)),
            0x809f0008u);
  ASSERT_EQ(SmallMount(), 0);
  EXPECT_FALSE(std::filesystem::exists(host / "nested"));
  ASSERT_EQ(Unmount(), 0);
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, BackupPayload(36), output)),
            0x809f0008u);
}

TEST_F(SaveDataTest, DeletePreservesBackupWhileMountedOrFilesAreOpen) {
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  const auto backup = root / "CUSA00792" / "TESTSAVE.backup";
  std::ofstream(host / "data") << "preserved";
  ASSERT_EQ(Unmount(1), 0);
  Mode(2);
  ASSERT_EQ(SmallMount(), 0);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  EXPECT_EQ(std::filesystem::file_size(host / "data"), 9u);
  EXPECT_EQ(std::filesystem::file_size(backup / "data"), 9u);
  const int fd = kern::sys_open("/savedata0/data", kern::O_RDONLY, 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(Unmount(), 0);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  EXPECT_TRUE(std::filesystem::exists(backup));
  ASSERT_EQ(kern::sys_close(fd), 0);
  kern::vfs::MountWritable("/backup_probe", backup.string().c_str(), true);
  const int snapshot = kern::sys_open("/backup_probe/data", kern::O_RDONLY, 0);
  ASSERT_GE(snapshot, 0);
  kern::vfs::Unmount("/backup_probe");
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  EXPECT_TRUE(std::filesystem::exists(host));
  EXPECT_TRUE(std::filesystem::exists(backup));
  ASSERT_EQ(kern::sys_close(snapshot), 0);
  ASSERT_EQ(Delete(), 0);
  EXPECT_FALSE(std::filesystem::exists(host));
  EXPECT_FALSE(std::filesystem::exists(backup));
}

TEST_F(SaveDataTest, DeletePreservesForeignStagingAndUnsafeBackupRoots) {
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  const auto backup = root / "CUSA00792" / "TESTSAVE.backup";
  std::ofstream(host / "data") << "preserved";
  ASSERT_EQ(Unmount(1), 0);
  for (const auto* suffix : {".delete", ".tmp", ".old", ".backup.tmp",
                             ".backup.old"}) {
    const auto staging = host.string() + suffix;
    std::ofstream(staging) << "foreign";
    EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
    EXPECT_EQ(std::filesystem::file_size(staging), 7u);
    EXPECT_EQ(std::filesystem::file_size(host / "data"), 9u);
    EXPECT_EQ(std::filesystem::file_size(backup / "data"), 9u);
    ASSERT_TRUE(std::filesystem::remove(staging));
  }
  std::filesystem::remove_all(backup);
  const auto outside = root / "outside";
  std::filesystem::create_directory(outside);
  std::ofstream(outside / "data") << "preserved";
  std::filesystem::create_directory_symlink(outside, backup);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f000bu);
  EXPECT_TRUE(std::filesystem::is_symlink(backup));
  EXPECT_EQ(std::filesystem::file_size(outside / "data"), 9u);
  EXPECT_EQ(std::filesystem::file_size(host / "data"), 9u);
  ASSERT_TRUE(std::filesystem::remove(backup));
  ASSERT_EQ(Delete(), 0);
}

TEST_F(SaveDataTest, DeleteFailureBeforeRetirementPreservesSaveAndBackup) {
  if (geteuid() == 0)
    GTEST_SKIP() << "Requires host permission enforcement";
  ASSERT_EQ(SmallMount(), 0);
  const auto parent = root / "CUSA00792";
  const auto host = parent / "TESTSAVE", backup = parent / "TESTSAVE.backup";
  std::ofstream(host / "data") << "preserved";
  ASSERT_EQ(Unmount(1), 0);
  ASSERT_EQ(chmod(parent.c_str(), 0500), 0);
  const auto result = Delete();
  ASSERT_EQ(chmod(parent.c_str(), 0700), 0);
  EXPECT_EQ(static_cast<u32>(result), 0x809f000bu);
  EXPECT_EQ(std::filesystem::file_size(host / "data"), 9u);
  EXPECT_EQ(std::filesystem::file_size(backup / "data"), 9u);
  EXPECT_FALSE(std::filesystem::exists(host.string() + ".delete"));
  ASSERT_EQ(Delete(), 0);
}

TEST_F(SaveDataTest, DeleteCleanupFailureRetainsOnlyHiddenRetiredStorage) {
  if (geteuid() == 0)
    GTEST_SKIP() << "Requires host permission enforcement";
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  const auto backup = root / "CUSA00792" / "TESTSAVE.backup";
  std::filesystem::create_directory(host / "nested");
  std::ofstream(host / "nested" / "data") << "retired contents";
  ASSERT_EQ(Unmount(1), 0);
  ASSERT_EQ(chmod((host / "nested").c_str(), 0000), 0);
  const auto result = Delete();
  const auto retired = std::filesystem::path(host.string() + ".delete");
  ASSERT_EQ(chmod((retired / "save" / "nested").c_str(), 0700), 0);
  EXPECT_EQ(result, 0);
  EXPECT_FALSE(std::filesystem::exists(host));
  EXPECT_FALSE(std::filesystem::exists(backup));
  EXPECT_TRUE(std::filesystem::exists(retired));
  std::vector<u8> output;
  EXPECT_EQ(static_cast<u32>(BackupCommand(37, BackupPayload(37), output)),
            0x809f0008u);
  EXPECT_EQ(static_cast<u32>(BackupCommand(36, BackupPayload(36), output)),
            0x809f0008u);
  ASSERT_EQ(Search(output), 0);
  u16 count;
  std::memcpy(&count, output.data(), sizeof(count));
  EXPECT_EQ(count, 0u);
}

TEST_F(SaveDataTest, DeleteRejectsMountedSavesAcrossClients) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::ofstream(host / "file") << "Persisted save";
  const u32 owner = kid;
  const char* name = "SceSaveData";
  u64 request = reinterpret_cast<u64>(name);
  ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)), 0);
  ASSERT_EQ(Initialize(), 0);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  EXPECT_TRUE(std::filesystem::exists(host / "file"));
  const u32 other = kid;
  kid = owner;
  ASSERT_EQ(Initialize(63), 0);
  kid = other;
  ASSERT_EQ(Delete(), 0);
  EXPECT_FALSE(std::filesystem::exists(host));
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0008u);
}

TEST_F(SaveDataTest, DeleteRejectsOpenFilesAfterClientTermination) {
  ASSERT_EQ(SmallMount(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  const char contents[] = "Persisted save";
  ASSERT_EQ(kern::sys_write(fd, contents, sizeof(contents)), sizeof(contents));
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  EXPECT_TRUE(std::filesystem::exists(host / "data"));
  std::array<char, sizeof(contents)> read{};
  ASSERT_EQ(kern::sys_pread(fd, read.data(), read.size(), 0), read.size());
  EXPECT_STREQ(read.data(), contents);
  ASSERT_EQ(kern::sys_close(fd), 0);
  ASSERT_EQ(Delete(), 0);
  EXPECT_FALSE(std::filesystem::exists(host));
  ASSERT_EQ(SmallMount(), 0);
  EXPECT_EQ(FreeBlocks(), 96u);
  EXPECT_FALSE(std::filesystem::exists(host / "data"));
}

TEST_F(SaveDataTest,
       DeleteRetainsSaveForInFlightReferenceAfterDescriptorClose) {
  ASSERT_EQ(SmallMount(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 96 * 32768), 0);
  auto pending = process.GetObjTable().Get(fd);
  ASSERT_TRUE(pending);
  ASSERT_EQ(kern::sys_unlink("/savedata0/data"), 0);
  ASSERT_EQ(kern::sys_close(fd), 0);
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  pending = nullptr;
  ASSERT_EQ(Delete(), 0);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "TESTSAVE"));
}

TEST_F(SaveDataTest, ReadOnlyReferenceSurvivesUnmountAndBlocksDelete) {
  ASSERT_EQ(SmallMount(), 0);
  const int writer =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(writer, 0);
  ASSERT_EQ(kern::sys_ftruncate(writer, 96 * 32768), 0);
  ASSERT_EQ(kern::sys_close(writer), 0);
  ASSERT_EQ(Unmount(), 0);
  Mode(1);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const int reader = kern::sys_open("/savedata0/data", kern::O_RDONLY, 0600);
  ASSERT_GE(reader, 0);
  ASSERT_EQ(Unmount(), 0);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0003u);
  Mode(2);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  ASSERT_EQ(kern::sys_close(reader), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  ASSERT_EQ(Unmount(), 0);
  ASSERT_EQ(Delete(), 0);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "TESTSAVE"));
}

TEST_F(SaveDataTest, DeleteHandlesLegacySavesWithoutFollowingSymlinks) {
  const auto legacy = root / "TESTSAVE";
  const auto outside = root / "unrelated";
  std::filesystem::create_directories(legacy / "nested");
  std::filesystem::create_directory(outside);
  std::ofstream(legacy / "nested" / "file") << "Save data";
  std::ofstream(outside / "file") << "Keep this";
  std::filesystem::create_directory_symlink(outside, legacy / "link");
  ASSERT_EQ(Delete(), 0);
  EXPECT_FALSE(std::filesystem::exists(legacy));
  EXPECT_TRUE(std::filesystem::exists(outside / "file"));
  std::filesystem::create_directory_symlink(outside, legacy);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0000u);
  EXPECT_TRUE(std::filesystem::exists(outside / "file"));
}

TEST_F(SaveDataTest, DeleteWaitsForValidReplyBufferBeforeRemovingFiles) {
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::filesystem::create_directories(host);
  std::ofstream(host / "file") << "Save data";
  const u32 command = 4;
  std::memcpy(header.data() + 4, &command, 4);
  ASSERT_EQ(Submit(0x434), 0);
  EXPECT_EQ(static_cast<u32>(Collect(127, 128)), 0x809f0000u);
  EXPECT_TRUE(std::filesystem::exists(host / "file"));
  ASSERT_EQ(Collect(128, 128), 0);
  EXPECT_FALSE(std::filesystem::exists(host));
}

TEST_F(SaveDataTest, InvalidDeleteRequestsPreserveSaves) {
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::filesystem::create_directories(host);
  for (const char* name : {"../TESTSAVE", "", "sub/dir"}) {
    std::memset(data.data() + 0x414, 0, 32);
    std::memcpy(data.data() + 0x414, name, std::strlen(name));
    EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0000u);
    EXPECT_TRUE(std::filesystem::exists(host));
  }
  std::memcpy(data.data() + 0x414, "TESTSAVE", 9);
  const u32 user = 2;
  std::memcpy(data.data() + 0x400, &user, 4);
  EXPECT_EQ(static_cast<u32>(Delete()), 0x809f0000u);
  EXPECT_TRUE(std::filesystem::exists(host));
}

TEST_F(SaveDataTest, SearchPaginatesAndReturnsMetadataAndStorageInfo) {
  const auto title = root / "CUSA00792";
  for (u32 i = 0; i < 55; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "SAVE%03u", i);
    const auto host = title / name;
    std::filesystem::create_directories(host / "nested");
    std::array<u8, 1328> param{};
    std::memcpy(param.data(), name, std::strlen(name) + 1);
    const u32 user = 55 - i;
    const i64 mtime = i;
    std::memcpy(param.data() + 1280, &user, 4);
    std::memcpy(param.data() + 1288, &mtime, 8);
    std::ofstream file(host / ".sce_param.bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(param.data()), param.size());
    std::ofstream(host / "nested" / "data") << "Payload";
  }
  std::vector<u8> output;
  ASSERT_EQ(Search(output), 0);
  u32 total, count;
  std::memcpy(&total, output.data(), 4);
  std::memcpy(&count, output.data() + 4, 4);
  EXPECT_EQ(total, 55u);
  EXPECT_EQ(count, 50u);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SAVE000");
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8 + 49 * 32), "SAVE049");
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 0x648), "SAVE000");
  u64 blocks;
  std::memcpy(&blocks, output.data() + 0x109a8, 8);
  EXPECT_EQ(blocks, 60000u);
  std::memcpy(&blocks, output.data() + 0x109b0, 8);
  EXPECT_EQ(blocks, 59998u);
  ASSERT_EQ(Search(output, "", 0, 0, 50), 0);
  std::memcpy(&count, output.data() + 4, 4);
  EXPECT_EQ(count, 5u);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SAVE050");
  ASSERT_EQ(Search(output, "", 1), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SAVE054");
  ASSERT_EQ(Search(output, "", 3, 1), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SAVE054");
  ASSERT_EQ(Search(output, "", 0, 0, UINT32_MAX), 0);
  std::memcpy(&total, output.data(), 4);
  std::memcpy(&count, output.data() + 4, 4);
  EXPECT_EQ(total, 55u);
  EXPECT_EQ(count, 0u);
}

TEST_F(SaveDataTest, SearchFiltersAndExcludesInternalAndLinkedDirectories) {
  const auto title = root / "CUSA00792";
  for (const char* name :
       {"SLOT1", "SLOT2", "OTHER", "_internal", "sce_system"})
    std::filesystem::create_directories(title / name);
  std::filesystem::create_directory_symlink(title / "OTHER", title / "LINK");
  std::ofstream(title / "FILE") << "Not a save";
  std::vector<u8> output;
  ASSERT_EQ(Search(output, "SLOT_"), 0);
  u32 total;
  std::memcpy(&total, output.data(), 4);
  EXPECT_EQ(total, 2u);
  ASSERT_EQ(Search(output, "%OT%", 0, 1), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SLOT2");
  ASSERT_EQ(Search(output, "SLOT1"), 0);
  std::memcpy(&total, output.data(), 4);
  EXPECT_EQ(total, 1u);
  ASSERT_EQ(Search(output, "MISSING"), 0);
  std::memcpy(&total, output.data(), 4);
  EXPECT_EQ(total, 0u);
  EXPECT_EQ(static_cast<u32>(Search(output, "../%")), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(Search(output, "", 6)), 0x809f0000u);
  EXPECT_EQ(static_cast<u32>(Search(output, "", 0, 2)), 0x809f0000u);
  EXPECT_EQ(Search(output, "", 4), -static_cast<i32>(kern::SysError::eNOSYS));
}

TEST_F(SaveDataTest, SearchReturnsNoHitsWhenTitleHasNoSaves) {
  std::vector<u8> output;
  ASSERT_EQ(Search(output), 0);
  const std::array<u8, 8> empty{};
  EXPECT_EQ(std::memcmp(output.data(), empty.data(), empty.size()), 0);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792"));
}

TEST_F(SaveDataTest, MountQuotaPersistsAndTracksAllocatedFileBlocks) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  const auto check = [&](u64 expected_free) {
    ASSERT_EQ(MountInfo(output), 0);
    u64 total, free;
    std::memcpy(&total, output.data(), 8);
    std::memcpy(&free, output.data() + 8, 8);
    EXPECT_EQ(total, 1024u);
    EXPECT_EQ(free, expected_free);
  };
  check(1024);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::ofstream(host / "data") << "Payload";
  std::filesystem::resize_file(host / "data", 32769);
  check(1022);
  std::filesystem::create_directory(host / "nested");
  std::ofstream(host / "nested" / "file") << "Payload";
  std::ofstream(root / "outside") << "Ignore linked data";
  std::filesystem::create_symlink(root / "outside", host / "link");
  check(1021);
  std::filesystem::resize_file(host / "data", 1);
  check(1022);
  ASSERT_EQ(Initialize(63), 0);
  ASSERT_EQ(Initialize(), 0);
  Mode(2);
  const u64 requested = 2048;
  std::memcpy(data.data() + 0x488, &requested, 8);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  check(1022);
  std::filesystem::resize_file(host / "data", 32768ull * 2000);
  check(0);
}

TEST_F(SaveDataTest, LegacyQuotaAndMalformedQuotaFiles) {
  const auto host = root / "TESTSAVE";
  std::filesystem::create_directory(host);
  Mode(1);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  ASSERT_EQ(MountInfo(output), 0);
  u64 total;
  std::memcpy(&total, output.data(), 8);
  EXPECT_EQ(total, 60000u);
  EXPECT_FALSE(std::filesystem::exists(host / ".sce_quota.bin"));
  std::ofstream(host / ".sce_quota.bin") << "Bad quota";
  EXPECT_EQ(static_cast<u32>(MountInfo(output)), 0x809f000bu);
  std::memset(reply.data(), 0, 16);
  std::memcpy(reply.data(), "/savedata15", 12);
  EXPECT_EQ(static_cast<u32>(MountInfo(output)), 0x809f0004u);
}

TEST_F(SaveDataTest, SearchSortsByCapacityAndFreeBlocks) {
  const auto title = root / "CUSA00792";
  for (const auto& [name, blocks, used] :
       {std::tuple{"SAVEA", 96ull, 1ull}, std::tuple{"SAVEB", 128ull, 40ull}}) {
    const auto host = title / name;
    std::filesystem::create_directories(host);
    std::ofstream quota(host / ".sce_quota.bin", std::ios::binary);
    quota.write(reinterpret_cast<const char*>(&blocks), 8);
    std::ofstream(host / "data") << "Payload";
    std::filesystem::resize_file(host / "data", used * 32768);
  }
  std::vector<u8> output;
  ASSERT_EQ(Search(output, "", 2), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SAVEA");
  ASSERT_EQ(Search(output, "", 5), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SAVEB");
  u64 total, free;
  std::memcpy(&total, output.data() + 0x109a8, 8);
  std::memcpy(&free, output.data() + 0x109b0, 8);
  EXPECT_EQ(total, 128u);
  EXPECT_EQ(free, 88u);
  ASSERT_EQ(Search(output, "", 5, 1), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data() + 8), "SAVEA");
}

TEST_F(SaveDataTest, TruncateOpenRequiresExistingFileUnlessCreateIsSet) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE" / "data";
  for (u32 access : {kern::O_WRONLY, kern::O_RDWR}) {
    EXPECT_EQ(kern::sys_open("/savedata0/data", access | kern::O_TRUNC, 0600),
              -static_cast<i32>(kern::SysError::eNOENT));
    EXPECT_FALSE(std::filesystem::exists(host));
    const int fd = kern::sys_open(
        "/savedata0/data", access | kern::O_CREAT | kern::O_TRUNC, 0600);
    ASSERT_GE(fd, 0);
    kern::SceKernelStat stat{};
    ASSERT_EQ(kern::sys_fstat(fd, &stat), 0);
    EXPECT_EQ(stat.st_size, 0);
    ASSERT_EQ(kern::sys_close(fd), 0);
    ASSERT_EQ(kern::sys_unlink("/savedata0/data"), 0);
  }
}

TEST_F(SaveDataTest, CreatePreservesExistingDataAndTruncateUpdatesLiveFiles) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const int original =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(original, 0);
  const std::array<u8, 16> payload{0x37, 0x42};
  ASSERT_EQ(kern::sys_write(original, payload.data(), payload.size()),
            payload.size());
  const auto host = root / "CUSA00792" / "TESTSAVE" / "data";
  struct stat before{};
  ASSERT_EQ(stat(host.c_str(), &before), 0);
  for (u32 access : {kern::O_WRONLY, kern::O_RDWR}) {
    const int fd = kern::sys_open("/savedata0/data", access | kern::O_CREAT,
                                  0600);
    ASSERT_GE(fd, 0);
    std::array<u8, 16> read{};
    ASSERT_EQ(kern::sys_pread(original, read.data(), read.size(), 0),
              read.size());
    EXPECT_EQ(read, payload);
    ASSERT_EQ(kern::sys_close(fd), 0);
  }
  std::vector<u8> output;
  ASSERT_EQ(MountInfo(output), 0);
  u64 free_before;
  std::memcpy(&free_before, output.data() + 8, 8);
  const int truncated =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_TRUNC, 0600);
  ASSERT_GE(truncated, 0);
  struct stat after{};
  ASSERT_EQ(stat(host.c_str(), &after), 0);
  EXPECT_EQ(after.st_ino, before.st_ino);
  kern::SceKernelStat guest_stat{};
  ASSERT_EQ(kern::sys_fstat(original, &guest_stat), 0);
  EXPECT_EQ(guest_stat.st_size, 0);
  EXPECT_EQ(kern::sys_lseek(original, 0, 1), payload.size());
  EXPECT_EQ(kern::sys_lseek(truncated, 0, 1), 0);
  ASSERT_EQ(MountInfo(output), 0);
  u64 free_after;
  std::memcpy(&free_after, output.data() + 8, 8);
  EXPECT_EQ(free_after, free_before + 1);
  ASSERT_EQ(kern::sys_write(truncated, payload.data(), payload.size()),
            payload.size());
  std::array<u8, 16> read{};
  ASSERT_EQ(kern::sys_pread(original, read.data(), read.size(), 0),
            read.size());
  EXPECT_EQ(read, payload);
  ASSERT_EQ(kern::sys_close(truncated), 0);
  ASSERT_EQ(kern::sys_close(original), 0);
}

TEST_F(SaveDataTest, GuestWritesStopAtQuotaAndTruncationReleasesSpace) {
  const u64 blocks = 96, bytes = blocks * 32768;
  std::memcpy(data.data() + 0x488, &blocks, 8);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> payload(bytes + 16, 0x37), output;
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  EXPECT_EQ(kern::sys_write(fd, payload.data(), payload.size()), bytes);
  EXPECT_EQ(kern::sys_write(fd, payload.data(), 1),
            -static_cast<i32>(kern::SysError::eNOSPC));
  EXPECT_EQ(kern::sys_write(fd, payload.data(), 0), 0);
  EXPECT_EQ(kern::sys_pwrite(fd, payload.data(), 16, 0), 16);
  EXPECT_EQ(kern::sys_lseek(fd, 0, 1), bytes);
  EXPECT_EQ(kern::sys_pwrite(fd, payload.data(), 1, -1),
            -static_cast<i32>(kern::SysError::eINVAL));
  EXPECT_EQ(kern::sys_pwrite(fd, payload.data(), 1, bytes),
            -static_cast<i32>(kern::SysError::eNOSPC));
  kern::SceKernelStat stat{};
  ASSERT_EQ(kern::sys_fstat(fd, &stat), 0);
  EXPECT_EQ(stat.st_size, bytes);
  EXPECT_EQ(static_cast<u32>(Param(1, output, payload.data(), 128)),
            0x809f000au);
  EXPECT_EQ(static_cast<u32>(Icon(output, payload.data(), 24)), 0x809f000au);
  ASSERT_EQ(MountInfo(output), 0);
  u64 free;
  std::memcpy(&free, output.data() + 8, 8);
  EXPECT_EQ(free, 0u);
  const int second =
      kern::sys_open("/savedata0/second", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(second, 0);
  EXPECT_EQ(kern::sys_write(second, payload.data(), 1),
            -static_cast<i32>(kern::SysError::eNOSPC));
  ASSERT_EQ(kern::sys_close(fd), 0);
  const int truncated =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_TRUNC, 0600);
  ASSERT_GE(truncated, 0);
  EXPECT_EQ(kern::sys_write(second, payload.data(), 32769), 32769);
  ASSERT_EQ(MountInfo(output), 0);
  std::memcpy(&free, output.data() + 8, 8);
  EXPECT_EQ(free, 94u);
}

TEST_F(SaveDataTest, FileTruncateChecksQuotaAndPreservesCursor) {
  const u64 blocks = 96, bytes = blocks * 32768;
  std::memcpy(data.data() + 0x488, &blocks, 8);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  const int sibling = kern::sys_open("/savedata0/data", kern::O_RDWR, 0600);
  ASSERT_GE(sibling, 0);
  const int reader = kern::sys_open("/savedata0/data", kern::O_RDONLY, 0600);
  ASSERT_GE(reader, 0);
  ASSERT_EQ(kern::sys_lseek(fd, 123, 0), 123);
  EXPECT_EQ(kern::sys_ftruncate(fd, bytes + 1),
            -static_cast<i32>(kern::SysError::eNOSPC));
  EXPECT_EQ(kern::sys_ftruncate(fd, -1),
            -static_cast<i32>(kern::SysError::eINVAL));
  EXPECT_EQ(kern::sys_ftruncate(reader, 1),
            -static_cast<i32>(kern::SysError::eINVAL));
  kern::SceKernelStat stat{};
  ASSERT_EQ(kern::sys_fstat(fd, &stat), 0);
  EXPECT_EQ(stat.st_size, 0);
  EXPECT_EQ(kern::sys_lseek(fd, 0, 1), 123);
  ASSERT_EQ(kern::sys_ftruncate(fd, bytes), 0);
  ASSERT_EQ(kern::sys_fstat(sibling, &stat), 0);
  EXPECT_EQ(stat.st_size, bytes);
  EXPECT_EQ(kern::sys_lseek(fd, 0, 1), 123);
  std::array<u8, 16> zeros;
  zeros.fill(0xff);
  ASSERT_EQ(kern::sys_pread(reader, zeros.data(), zeros.size(), bytes - 16),
            zeros.size());
  EXPECT_EQ(zeros, (std::array<u8, 16>{}));
  std::vector<u8> output;
  ASSERT_EQ(MountInfo(output), 0);
  u64 free;
  std::memcpy(&free, output.data() + 8, 8);
  EXPECT_EQ(free, 0u);
  ASSERT_EQ(kern::sys_ftruncate(sibling, 32768), 0);
  ASSERT_EQ(kern::sys_fstat(fd, &stat), 0);
  EXPECT_EQ(stat.st_size, 32768);
  ASSERT_EQ(MountInfo(output), 0);
  std::memcpy(&free, output.data() + 8, 8);
  EXPECT_EQ(free, 95u);
  const int second =
      kern::sys_open("/savedata0/second", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(second, 0);
  ASSERT_EQ(kern::sys_ftruncate(second, 95 * 32768), 0);
  EXPECT_EQ(kern::sys_ftruncate(fd, 32769),
            -static_cast<i32>(kern::SysError::eNOSPC));
  EXPECT_EQ(kern::sys_pwrite(fd, zeros.data(), 1, 32768),
            -static_cast<i32>(kern::SysError::eNOSPC));
  ASSERT_EQ(kern::sys_lseek(fd, bytes, 0), bytes);
  ASSERT_EQ(kern::sys_ftruncate(fd, 0), 0);
  EXPECT_EQ(kern::sys_lseek(fd, 0, 1), bytes);
  ASSERT_EQ(kern::sys_fstat(sibling, &stat), 0);
  EXPECT_EQ(stat.st_size, 0);
  ASSERT_EQ(MountInfo(output), 0);
  std::memcpy(&free, output.data() + 8, 8);
  EXPECT_EQ(free, 1u);
}

TEST_F(SaveDataTest, PhysicalSizeIncludesBufferedWritesAndExternalTruncation) {
  const auto path = (root / "buffered").string();
  io::File file(path.c_str(), io::FileMode::kCreate);
  ASSERT_TRUE(file.IsOpen());
  const std::array<u8, 128> payload{};
  ASSERT_NE(file.Write(payload.data(), payload.size()), 0);
  EXPECT_EQ(file.GetSize(), payload.size());
  EXPECT_EQ(file.Tell(), payload.size());
  ASSERT_EQ(truncate(path.c_str(), 16), 0);
  EXPECT_EQ(file.GetSize(), 16u);
  EXPECT_EQ(file.Tell(), payload.size());
}

TEST_F(SaveDataTest, PathTruncateChecksQuotaAndLeavesOpenCursorUnchanged) {
  const u64 blocks = 96, bytes = blocks * 32768;
  std::memcpy(data.data() + 0x488, &blocks, 8);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  const std::array<u8, 16> payload{0x37};
  ASSERT_EQ(kern::sys_write(fd, payload.data(), payload.size()),
            payload.size());
  EXPECT_EQ(kern::sys_truncate("/savedata0/data", bytes + 1),
            -static_cast<i32>(kern::SysError::eNOSPC));
  EXPECT_EQ(kern::sys_truncate("/savedata0/data", -1),
            -static_cast<i32>(kern::SysError::eINVAL));
  std::array<u8, 16> read{};
  ASSERT_EQ(kern::sys_pread(fd, read.data(), read.size(), 0), read.size());
  EXPECT_EQ(read, payload);
  ASSERT_EQ(kern::sys_truncate("/savedata0/data", bytes), 0);
  EXPECT_EQ(kern::sys_lseek(fd, 0, 1), payload.size());
  kern::SceKernelStat stat{};
  ASSERT_EQ(kern::sys_fstat(fd, &stat), 0);
  EXPECT_EQ(stat.st_size, bytes);
  std::vector<u8> output;
  ASSERT_EQ(MountInfo(output), 0);
  u64 free;
  std::memcpy(&free, output.data() + 8, 8);
  EXPECT_EQ(free, 0u);
  ASSERT_EQ(kern::sys_truncate("/savedata0/data", 0), 0);
  EXPECT_EQ(kern::sys_lseek(fd, 0, 1), payload.size());
  ASSERT_EQ(kern::sys_fstat(fd, &stat), 0);
  EXPECT_EQ(stat.st_size, 0);
  ASSERT_EQ(MountInfo(output), 0);
  std::memcpy(&free, output.data() + 8, 8);
  EXPECT_EQ(free, blocks);
  EXPECT_EQ(kern::sys_truncate("/savedata0/missing", 1),
            -static_cast<i32>(kern::SysError::eNOENT));
  EXPECT_EQ(kern::sys_truncate("/savedata0", 0),
            -static_cast<i32>(kern::SysError::eISDIR));
  ASSERT_TRUE(kern::vfs::MakeDir("/savedata0/directory"));
  EXPECT_EQ(kern::sys_truncate("/savedata0/directory", 0),
            -static_cast<i32>(kern::SysError::eISDIR));
  EXPECT_EQ(kern::sys_truncate("/savedata0/.sce_quota.bin", 0),
            -static_cast<i32>(kern::SysError::eACCES));
  const auto host = root / "CUSA00792" / "TESTSAVE";
  kern::vfs::Mount("/truncate_readonly", host.string().c_str());
  EXPECT_EQ(kern::sys_truncate("/truncate_readonly/data", 1),
            -static_cast<i32>(kern::SysError::eROFS));
  kern::vfs::Unmount("/truncate_readonly");
  std::ofstream(root / "outside") << "Keep this";
  std::filesystem::create_symlink(root / "outside", host / "outside");
  EXPECT_EQ(kern::sys_truncate("/savedata0/outside", 0),
            -static_cast<i32>(kern::SysError::eACCES));
  EXPECT_EQ(std::filesystem::file_size(root / "outside"), 9u);
}

TEST_F(SaveDataTest, ConcurrentPathTruncatesCannotReuseFreeBlocks) {
  const u64 blocks = 96, bytes = blocks * 32768;
  std::memcpy(data.data() + 0x488, &blocks, 8);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  for (const char* name : {"/savedata0/first", "/savedata0/second"})
    ASSERT_GE(kern::sys_open(name, kern::O_RDWR | kern::O_CREAT, 0600), 0);
  i32 a, b;
  std::thread first([&] { a = kern::sys_truncate("/savedata0/first", bytes); });
  std::thread second(
      [&] { b = kern::sys_truncate("/savedata0/second", bytes); });
  first.join();
  second.join();
  const i32 full = -static_cast<i32>(kern::SysError::eNOSPC);
  EXPECT_TRUE((a == 0 && b == full) || (b == 0 && a == full));
  const auto host = root / "CUSA00792" / "TESTSAVE";
  EXPECT_EQ(std::filesystem::file_size(host / "first") +
                std::filesystem::file_size(host / "second"),
            bytes);
}

TEST_F(SaveDataTest, RenamedDescriptorsUseTheirOwnAllocation) {
  ASSERT_EQ(SmallMount(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 96 * 32768), 0);
  ASSERT_EQ(kern::sys_rename("/savedata0/data", "/savedata0/renamed"), 0);
  const int replacement =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(replacement, 0);
  const u8 value = 0x37;
  EXPECT_EQ(kern::sys_pwrite(fd, &value, 1, 95 * 32768), 1);
  EXPECT_EQ(kern::sys_ftruncate(replacement, 1),
            -static_cast<i32>(kern::SysError::eNOSPC));
  ASSERT_EQ(kern::sys_ftruncate(fd, 95 * 32768), 0);
  EXPECT_EQ(FreeBlocks(), 1u);
  ASSERT_EQ(kern::sys_ftruncate(replacement, 1), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  EXPECT_EQ(kern::sys_pwrite(fd, &value, 1, 95 * 32768),
            -static_cast<i32>(kern::SysError::eNOSPC));
  ASSERT_EQ(kern::sys_unlink("/savedata0/renamed"), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  ASSERT_EQ(kern::sys_close(fd), 0);
  EXPECT_EQ(FreeBlocks(), 95u);
}

TEST_F(SaveDataTest, UnlinkedAllocationIsReleasedAfterLastDescriptorCloses) {
  ASSERT_EQ(SmallMount(), 0);
  const int writer =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(writer, 0);
  ASSERT_EQ(kern::sys_ftruncate(writer, 96 * 32768), 0);
  const int reader = kern::sys_open("/savedata0/data", kern::O_RDONLY, 0600);
  ASSERT_GE(reader, 0);
  ASSERT_EQ(kern::sys_unlink("/savedata0/data"), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  const int replacement =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(replacement, 0);
  EXPECT_EQ(kern::sys_ftruncate(replacement, 1),
            -static_cast<i32>(kern::SysError::eNOSPC));
  ASSERT_EQ(kern::sys_close(writer), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  EXPECT_FALSE(process.GetObjTable().Get(writer));
  kern::SceKernelStat stat{};
  ASSERT_EQ(kern::sys_fstat(reader, &stat), 0);
  EXPECT_EQ(stat.st_size, 96 * 32768);
  ASSERT_EQ(kern::sys_close(reader), 0);
  EXPECT_EQ(FreeBlocks(), 96u);
  ASSERT_EQ(kern::sys_ftruncate(replacement, 96 * 32768), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
}

TEST_F(SaveDataTest, RenameOverOpenFileKeepsBothInodesAllocated) {
  ASSERT_EQ(SmallMount(), 0);
  const int source =
      kern::sys_open("/savedata0/source", kern::O_RDWR | kern::O_CREAT, 0600);
  const int target =
      kern::sys_open("/savedata0/target", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(source, 0);
  ASSERT_GE(target, 0);
  ASSERT_EQ(kern::sys_ftruncate(source, 95 * 32768), 0);
  ASSERT_EQ(kern::sys_ftruncate(target, 32768), 0);
  const u8 source_value = 0x37, target_value = 0x42;
  ASSERT_EQ(kern::sys_pwrite(source, &source_value, 1, 0), 1);
  ASSERT_EQ(kern::sys_pwrite(target, &target_value, 1, 0), 1);
  ASSERT_EQ(kern::sys_rename("/savedata0/source", "/savedata0/target"), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  u8 value = 0;
  ASSERT_EQ(kern::sys_pread(target, &value, 1, 0), 1);
  EXPECT_EQ(value, target_value);
  const int reader = kern::sys_open("/savedata0/target", kern::O_RDONLY, 0600);
  ASSERT_GE(reader, 0);
  ASSERT_EQ(kern::sys_pread(reader, &value, 1, 0), 1);
  EXPECT_EQ(value, source_value);
  EXPECT_EQ(kern::sys_ftruncate(source, 96 * 32768),
            -static_cast<i32>(kern::SysError::eNOSPC));
  ASSERT_EQ(kern::sys_close(target), 0);
  EXPECT_EQ(FreeBlocks(), 1u);
  ASSERT_EQ(kern::sys_ftruncate(source, 96 * 32768), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
}

TEST_F(SaveDataTest, HardLinksAndOpenDescriptorsCountAllocationOnce) {
  ASSERT_EQ(SmallMount(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 32768), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::filesystem::create_hard_link(host / "data", host / "alias");
  const int alias = kern::sys_open("/savedata0/alias", kern::O_RDONLY, 0600);
  ASSERT_GE(alias, 0);
  EXPECT_EQ(FreeBlocks(), 95u);
  ASSERT_EQ(kern::sys_ftruncate(fd, 96 * 32768), 0);
  ASSERT_EQ(kern::sys_unlink("/savedata0/data"), 0);
  ASSERT_EQ(kern::sys_unlink("/savedata0/alias"), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  ASSERT_EQ(kern::sys_ftruncate(fd, 0), 0);
  EXPECT_EQ(FreeBlocks(), 96u);
  ASSERT_EQ(kern::sys_ftruncate(fd, 96 * 32768), 0);
  ASSERT_EQ(kern::sys_close(fd), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  ASSERT_EQ(kern::sys_close(alias), 0);
  EXPECT_EQ(FreeBlocks(), 96u);
}

TEST_F(SaveDataTest, MetadataReplacementReservesSpaceForOpenPreviousVersion) {
  ASSERT_EQ(SmallMount(), 0);
  std::vector<u8> output;
  const char initial[] = "Initial title", replacement[] = "Replacement title";
  ASSERT_EQ(Param(1, output, initial, sizeof(initial)), 0);
  const int reader =
      kern::sys_open("/savedata0/.sce_param.bin", kern::O_RDONLY, 0600);
  ASSERT_GE(reader, 0);
  const int data_fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(data_fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(data_fd, 95 * 32768), 0);
  EXPECT_EQ(
      static_cast<u32>(Param(1, output, replacement, sizeof(replacement))),
      0x809f000au);
  ASSERT_EQ(kern::sys_ftruncate(data_fd, 94 * 32768), 0);
  ASSERT_EQ(Param(1, output, replacement, sizeof(replacement)), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  std::array<char, 128> title{};
  ASSERT_EQ(kern::sys_read(reader, title.data(), title.size()), title.size());
  EXPECT_STREQ(title.data(), initial);
  ASSERT_EQ(Param(1, output), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data()), replacement);
  ASSERT_EQ(kern::sys_close(reader), 0);
  EXPECT_EQ(FreeBlocks(), 1u);
}

TEST_F(SaveDataTest, ReadOnlyQuotaMountRetainsUnlinkedAllocation) {
  ASSERT_EQ(SmallMount(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 96 * 32768), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  kern::vfs::Mount("/quota_readonly", host.string().c_str(), true);
  const int reader = kern::sys_open("/quota_readonly/data", kern::O_RDONLY, 0);
  EXPECT_GE(reader, 0);
  EXPECT_EQ(kern::sys_unlink("/quota_readonly/data"),
            -static_cast<i32>(kern::SysError::eROFS));
  EXPECT_EQ(kern::sys_rename("/quota_readonly/data", "/quota_readonly/renamed"),
            -static_cast<i32>(kern::SysError::eROFS));
  kern::vfs::Unmount("/quota_readonly");
  ASSERT_GE(reader, 0);
  ASSERT_EQ(kern::sys_unlink("/savedata0/data"), 0);
  ASSERT_EQ(kern::sys_close(fd), 0);
  EXPECT_EQ(FreeBlocks(), 0u);
  ASSERT_EQ(kern::sys_close(reader), 0);
  EXPECT_EQ(FreeBlocks(), 96u);
}

TEST_F(SaveDataTest, InFlightObjectReferenceRetainsUnlinkedAllocation) {
  ASSERT_EQ(SmallMount(), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 96 * 32768), 0);
  auto pending = process.GetObjTable().Get(fd);
  ASSERT_TRUE(pending);
  ASSERT_EQ(kern::sys_unlink("/savedata0/data"), 0);
  ASSERT_EQ(kern::sys_close(fd), 0);
  EXPECT_FALSE(process.GetObjTable().Get(fd));
  EXPECT_EQ(FreeBlocks(), 0u);
  pending = nullptr;
  EXPECT_EQ(FreeBlocks(), 96u);
}

TEST_F(SaveDataTest, SequentialReadSeesTruncationThroughSiblingDescriptor) {
  ASSERT_EQ(SmallMount(), 0);
  const int writer =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(writer, 0);
  const std::array<u8, 128> payload{0x37};
  ASSERT_EQ(kern::sys_write(writer, payload.data(), payload.size()),
            payload.size());
  const int reader = kern::sys_open("/savedata0/data", kern::O_RDONLY, 0600);
  ASSERT_GE(reader, 0);
  u8 value = 0;
  ASSERT_EQ(kern::sys_read(reader, &value, 1), 1);
  EXPECT_EQ(value, payload[0]);
  ASSERT_EQ(kern::sys_ftruncate(writer, 1), 0);
  value = 0xa5;
  EXPECT_EQ(kern::sys_read(reader, &value, 1), 0);
  EXPECT_EQ(value, 0xa5);
}

TEST_F(SaveDataTest, ConcurrentUnlinkedWritersCannotReuseFreeBlocks) {
  ASSERT_EQ(SmallMount(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  auto* first = new kern::FileDevice(process.GetObjTable());
  auto* second = new kern::FileDevice(process.GetObjTable());
  ASSERT_TRUE(first->OpenWritable((host / "first").string().c_str(), true,
                                  false, host.string().c_str()));
  ASSERT_TRUE(second->OpenWritable((host / "second").string().c_str(), true,
                                   false, host.string().c_str()));
  ASSERT_EQ(kern::sys_unlink("/savedata0/first"), 0);
  ASSERT_EQ(kern::sys_unlink("/savedata0/second"), 0);
  const i64 bytes = 96 * 32768;
  std::vector<u8> payload(bytes, 0x37);
  i64 a, b;
  std::thread one([&] { a = first->Write(payload.data(), payload.size()); });
  std::thread two([&] { b = second->Write(payload.data(), payload.size()); });
  one.join();
  two.join();
  const i32 full = -static_cast<i32>(kern::SysError::eNOSPC);
  EXPECT_TRUE((a == bytes && b == full) || (b == bytes && a == full));
  EXPECT_EQ(FreeBlocks(), 0u);
  ASSERT_EQ(kern::sys_close(first->handle()), 0);
  ASSERT_EQ(kern::sys_close(second->handle()), 0);
  EXPECT_EQ(FreeBlocks(), 96u);
}

TEST_F(SaveDataTest, ConcurrentFilesCannotAllocateTheSameFreeBlocks) {
  const u64 blocks = 96, bytes = blocks * 32768;
  std::memcpy(data.data() + 0x488, &blocks, 8);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  auto* first = new kern::FileDevice(process.GetObjTable());
  auto* second = new kern::FileDevice(process.GetObjTable());
  ASSERT_TRUE(first->OpenWritable((host / "first").string().c_str(), true,
                                  false, host.string().c_str()));
  ASSERT_TRUE(second->OpenWritable((host / "second").string().c_str(), true,
                                   false, host.string().c_str()));
  std::vector<u8> payload(bytes, 0x37);
  i64 a = 0, b = 0;
  std::thread one([&] { a = first->Write(payload.data(), payload.size()); });
  std::thread two([&] { b = second->Write(payload.data(), payload.size()); });
  one.join();
  two.join();
  EXPECT_TRUE((a == bytes && b == -static_cast<i32>(kern::SysError::eNOSPC)) ||
              (b == bytes && a == -static_cast<i32>(kern::SysError::eNOSPC)));
  EXPECT_EQ(std::filesystem::file_size(host / "first") +
                std::filesystem::file_size(host / "second"),
            bytes);
}

TEST_F(SaveDataTest, QuotaBookkeepingAndOutsidePathsCannotBeWritten) {
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  const auto host = root / "CUSA00792" / "TESTSAVE";
  std::ofstream(root / "outside") << "Keep this";
  std::filesystem::create_symlink(root / "outside", host / "link");
  for (const char* path :
       {"/savedata0/.sce_quota.bin", "/savedata0/.sce_quota.bin.tmp",
        "/savedata0/../../../outside", "/savedata0/link"})
    EXPECT_LT(kern::sys_open(path, kern::O_RDWR | kern::O_CREAT | kern::O_TRUNC,
                             0600),
              0);
  EXPECT_EQ(std::filesystem::file_size(root / "outside"), 9u);
  EXPECT_EQ(std::filesystem::file_size(host / ".sce_quota.bin"), 8u);
  EXPECT_EQ(kern::sys_unlink("/savedata0/.sce_quota.bin"),
            -static_cast<i32>(kern::SysError::eACCES));
  EXPECT_EQ(kern::sys_rename("/savedata0/.sce_quota.bin", "/savedata0/renamed"),
            -static_cast<i32>(kern::SysError::eACCES));
  EXPECT_EQ(std::filesystem::file_size(host / ".sce_quota.bin"), 8u);
}

TEST_F(SaveDataTest, ExistingMetadataCanBeReplacedAtFullQuota) {
  const u64 blocks = 96;
  std::memcpy(data.data() + 0x488, &blocks, 8);
  ASSERT_EQ(Submit(), 0);
  ASSERT_EQ(Collect(), 0);
  std::vector<u8> output;
  const char initial[] = "Initial title";
  ASSERT_EQ(Param(1, output, initial, sizeof(initial)), 0);
  const int fd =
      kern::sys_open("/savedata0/data", kern::O_RDWR | kern::O_CREAT, 0600);
  ASSERT_GE(fd, 0);
  std::vector<u8> payload(95 * 32768, 0x37);
  ASSERT_EQ(kern::sys_write(fd, payload.data(), payload.size()),
            payload.size());
  const char replacement[] = "Replacement title";
  ASSERT_EQ(Param(1, output, replacement, sizeof(replacement)), 0);
  ASSERT_EQ(Param(1, output), 0);
  EXPECT_STREQ(reinterpret_cast<char*>(output.data()), replacement);
  EXPECT_EQ(static_cast<u32>(Icon(output, payload.data(), 24)), 0x809f000au);
  EXPECT_FALSE(std::filesystem::exists(root / "CUSA00792" / "TESTSAVE" /
                                       ".sce_param.bin.tmp"));
}

}  // namespace
