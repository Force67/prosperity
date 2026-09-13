#include <chrono>
#include <array>
#include <cstring>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "gpu/ps5/cmd_processor.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/draw_state.h"
#include "gpu/ps4/pm4.h"

// Exercise the real packet walker with a backend that records dispatch state.
// Queue scheduling must work independently of a host graphics device.
namespace {
std::vector<u32> dispatch_values;
std::vector<std::array<u32, 3>> dispatch_groups;
std::vector<u32> draw_instances;
std::vector<u32> draw_index_types;
std::array<u8, 65536> gds{};
u64 pending_address = 0;
std::array<u32, 3> pending_groups{};
bool flush_succeeds = true;
bool accept_draw = false;
u64 presented = 0;
constexpr u64 draw_target = 0x500000000;
using Words = std::vector<u32>;

void Packet(Words& words, u32 op, std::initializer_list<u32> body) {
  words.push_back(0xc0000000u | ((body.size() - 1) << 16) | (op << 8));
  words.insert(words.end(), body);
}
u32 Lo(const void* p) { return static_cast<u32>(reinterpret_cast<u64>(p)); }
u32 Hi(const void* p) { return static_cast<u32>(reinterpret_cast<u64>(p) >> 32); }
void Wait(Words& w, const u32* label, u32 reference = 1, u32 mask = ~0u) {
  Packet(w, 0x3c, {0x13, Lo(label), Hi(label), reference, mask, 0x19});
}
void Add(Words& w, u32* label) {
  Packet(w, 0x1e, {79, Lo(label), Hi(label), 1, 0, 0, 0, 0});
}
Words Indirect(const Words& words) {
  Words out;
  Packet(out, 0x3f, {Lo(words.data()), Hi(words.data()),
                     static_cast<u32>(words.size())});
  return out;
}
u32 Submit(const Words& words, u32 queue) {
  return gpu::ps5::SubmitDcbRing(words.data(), words.size() * 4, queue);
}

TEST(AgcQueue, NestedWaitsResumeEachQueueWithoutReplayingWrites) {
  u32 ready_a = 0, ready_b = 0, count_a = 0, count_b = 0;
  Words a, b;
  Add(a, &count_a); Wait(a, &ready_a); Add(a, &count_a);
  Add(b, &count_b); Wait(b, &ready_b); Add(b, &count_b);
  const Words inner_a = Indirect(a), inner_b = Indirect(b);
  const Words ring_a = Indirect(inner_a), ring_b = Indirect(inner_b);
  EXPECT_EQ(Submit(ring_a, 101), 0u);
  EXPECT_EQ(Submit(ring_b, 102), 0u);
  EXPECT_EQ(count_a, 1u);
  EXPECT_EQ(count_b, 1u);
  ready_a = 1;
  EXPECT_EQ(Submit(ring_a, 101), ring_a.size());
  EXPECT_EQ(count_a, 2u);
  EXPECT_EQ(Submit(ring_b, 102), 0u);
  EXPECT_EQ(count_b, 1u);
  ready_b = 1;
  EXPECT_EQ(Submit(ring_b, 102), ring_b.size());
  EXPECT_EQ(count_b, 2u);
}

TEST(AgcQueue, RegistersSurviveAnotherQueueRunningWhileSuspended) {
  u32 ready = 0;
  Words a, b;
  Packet(a, 0x76, {gpu::ps5::mmCOMPUTE_USER_DATA_0 - gpu::kShRegBase, 11});
  Wait(a, &ready);
  Packet(a, 0x15, {1, 1, 1, 1});
  Packet(b, 0x76, {gpu::ps5::mmCOMPUTE_USER_DATA_0 - gpu::kShRegBase, 22});
  Packet(b, 0x15, {1, 1, 1, 1});
  const Words ring = Indirect(a);
  dispatch_values.clear();
  EXPECT_EQ(Submit(ring, 103), 0u);
  EXPECT_EQ(Submit(b, 104), b.size());
  ready = 1;
  EXPECT_EQ(Submit(ring, 103), ring.size());
  EXPECT_EQ(dispatch_values, (std::vector<u32>{22, 11}));
}

TEST(AgcQueue, ElapsedTimeDoesNotCompleteAWait) {
  u32 ready = 0, count = 0;
  Words w;
  Wait(w, &ready); Add(w, &count);
  EXPECT_EQ(Submit(w, 105), 0u);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_EQ(Submit(w, 105), 0u);
  EXPECT_EQ(count, 0u);
  ready = 1;
  EXPECT_EQ(Submit(w, 105), w.size());
  EXPECT_EQ(count, 1u);
}

TEST(AgcQueue, MaskAppliesToMemoryAndNotReference) {
  u32 ready = 1;
  Words w;
  Wait(w, &ready, 0x101, 0xff);
  EXPECT_EQ(Submit(w, 106), 0u);
}

TEST(AgcQueue, IndirectDispatchReadsCompletedGpuArguments) {
  std::array<u32, 3> args{};
  pending_address = reinterpret_cast<u64>(args.data());
  pending_groups = {3, 2, 1};
  dispatch_groups.clear();
  Words w;
  Packet(w, 0x16, {Lo(args.data()), Hi(args.data()), 1});
  EXPECT_EQ(Submit(w, 107), w.size());
  ASSERT_EQ(dispatch_groups.size(), 1u);
  EXPECT_EQ(dispatch_groups[0], pending_groups);
  EXPECT_EQ(pending_address, 0u);
}

TEST(AgcQueue, IndirectOffsetDispatchReadsCompletedGpuArguments) {
  std::array<u32, 4> args{};
  pending_address = reinterpret_cast<u64>(&args[1]);
  pending_groups = {4, 5, 6};
  dispatch_groups.clear();
  Words w;
  Packet(w, 0x11, {1, Lo(args.data()), Hi(args.data())});
  w[0] |= 2;  // SET_BASE shader type: compute
  Packet(w, 0x16, {4, 1});
  EXPECT_EQ(Submit(w, 108), w.size());
  ASSERT_EQ(dispatch_groups.size(), 1u);
  EXPECT_EQ(dispatch_groups[0], pending_groups);
  EXPECT_EQ(pending_address, 0u);
}

TEST(AgcQueue, FailedArgumentWritebackDoesNotDispatchStaleDimensions) {
  std::array<u32, 3> args{1, 1, 1};
  dispatch_groups.clear();
  flush_succeeds = false;
  Words w;
  Packet(w, 0x16, {Lo(args.data()), Hi(args.data()), 1});
  EXPECT_EQ(Submit(w, 109), w.size());
  flush_succeeds = true;
  EXPECT_TRUE(dispatch_groups.empty());
}

TEST(AgcQueue, PresentsTheRequestedBufferRatherThanTheLastDrawTarget) {
  accept_draw = true;
  presented = 0;
  Words w;
  Packet(w, 0x2d, {3, 0});
  EXPECT_EQ(Submit(w, 110), w.size());
  accept_draw = false;
  constexpr u64 requested = 0x510000000;
  gpu::ps5::EndFrame(requested);
  EXPECT_EQ(presented, requested);
}

TEST(AgcQueue, ZeroInstanceIndirectDrawDoesNotReusePreviousInstanceCount) {
  std::array<u32, 4> args{3, 0, 0, 0};
  draw_instances.clear();
  Words w;
  Packet(w, 0x11, {1, Lo(args.data()), Hi(args.data())});
  Packet(w, 0x2f, {7});  // NUM_INSTANCES
  Packet(w, 0x24, {0, 0, 0, 0});  // DRAW_INDIRECT
  EXPECT_EQ(Submit(w, 111), w.size());
  EXPECT_TRUE(draw_instances.empty());
  args[1] = 2;
  EXPECT_EQ(Submit(w, 111), w.size());
  EXPECT_EQ(draw_instances, (std::vector<u32>{2}));
}

TEST(AgcQueue, IndexTypePacketsAndRegisterWritesShareState) {
  draw_index_types.clear();
  Words w;
  Packet(w, 0x2a, {0});  // INDEX_TYPE: 16-bit
  Packet(w, 0x2d, {3, 0});
  Packet(w, 0x7a, {0x20000000u |
                       (gpu::ps5::mmVGT_INDEX_TYPE - gpu::kUConfigRegBase),
                   1});  // SET_UCONFIG_REG_INDEX: 32-bit
  Packet(w, 0x2d, {3, 0});
  Packet(w, 0x2a, {2});  // INDEX_TYPE: 8-bit
  Packet(w, 0x2d, {3, 0});
  EXPECT_EQ(Submit(w, 112), w.size());
  EXPECT_EQ(draw_index_types, (std::vector<u32>{0, 1, 2}));
}

TEST(AgcQueue, DmaDataCopiesAndClearsGdsCounters) {
  std::array<u32, 2> input{17, 29}, output{};
  Words w;
  Packet(w, 0x50, {1u << 20, Lo(input.data()), Hi(input.data()),
                    0xc70, 0, 8});
  Packet(w, 0x50, {1u << 29, 0xc70, 0, Lo(output.data()), Hi(output.data()), 8});
  EXPECT_EQ(Submit(w, 113), w.size());
  EXPECT_EQ(output, input);
  w.clear();
  Packet(w, 0x50, {0x46106000, 0, 0, 0xc70, 0, 4});
  Packet(w, 0x50, {0x24306000, 0xc70, 0, Lo(output.data()), Hi(output.data()), 8});
  EXPECT_EQ(Submit(w, 113), w.size());
  EXPECT_EQ(output, (std::array<u32, 2>{0, 29}));
}
}  // namespace

namespace gpu::ps5 {
bool BuildDrawInfo(const Regs&, const DrawPacket& packet, rhi::DrawInfo& draw) {
  draw_instances.push_back(packet.num_instances);
  draw_index_types.push_back(packet.index_type);
  draw.rt_base = draw_target;
  return accept_draw;
}
void DispatchCompute(rhi::Renderer&, const Regs& regs, const u32* body, u32) {
  dispatch_values.push_back(regs[mmCOMPUTE_USER_DATA_0]);
  dispatch_groups.push_back({body[0], body[1], body[2]});
}
}  // namespace gpu::ps5

namespace gpu::rhi {
struct BackendState {};
u64 g_ns_dcb = 0, g_ns_dcb_lock = 0;
u32 g_submit_queue = 0;
u32 g_dcb_n = 0;
Renderer& DefaultRenderer() { static Renderer renderer; return renderer; }
bool Init(Renderer& renderer) {
  static BackendState state;
  renderer.state = &state;
  return true;
}
void BeginFrame(Renderer&) {}
void EndFrame(Renderer&, u64 base) { presented = base; }
void Draw(Renderer&, const DrawInfo&) {}
bool FlushCsWrites(Renderer&) { return true; }
bool ReadGds(Renderer&, u32 offset, void* data, u32 bytes) {
  std::memcpy(data, gds.data() + offset, bytes);
  return true;
}
bool WriteGds(Renderer&, u32 offset, const void* data, u32 bytes) {
  std::memcpy(gds.data() + offset, data, bytes);
  return true;
}
bool FillGds(Renderer&, u32 offset, u32 bytes, u32 value) {
  for (u32 i = 0; i < bytes; i++)
    gds[offset + i] = static_cast<u8>(value >> ((i & 3u) * 8));
  return true;
}
bool FlushCsWritesRange(Renderer&, u64 base, u64 bytes, const char*) {
  if (!flush_succeeds)
    return false;
  if (pending_address == base && bytes == sizeof(pending_groups)) {
    std::memcpy(reinterpret_cast<void*>(base), pending_groups.data(), bytes);
    pending_address = 0;
  }
  return true;
}
void NoteMemoryFill(Renderer&, u64, u64, u32) {}
}  // namespace gpu::rhi

extern "C" bool prosperity_ps5_is_display_buffer(u64 base) {
  return base == draw_target;
}
extern "C" void prosperity_gpu_end_of_pipe() {}
extern "C" void prosperity_gpu_end_of_pipe_ctx(u64) {}
