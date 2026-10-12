// GPU timestamps for capacity-sized versus bounded indirect quad conversion.
// Usage: index_convert_bench [vulkan|opengl|d3d12]

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "base/containers/vector.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/render/backend.h"
#include "gpu/rhi/device.h"

int main(int argc, char** argv) {
  using namespace gpu;
  using namespace rhi;
  Backend backend = Backend::kVulkan;
  if (argc > 1) {
    bool found = false;
    for (Backend candidate : render::CompiledBackends())
      if (!std::strcmp(argv[1], BackendName(candidate))) {
        backend = candidate;
        found = true;
      }
    if (!found)
      return 1;
  }
  auto device = render::CreateBackendDevice(backend, {});
  if (!device || !device->caps().timestamps)
    return 1;
  base::Vector<Object*> objects;
  const auto own = [&]<typename T>(T* object) -> T* {
    if (object)
      objects.push_back(object);
    return object;
  };
  BindGroupLayoutDesc ld;
  ld.bindings = {{0, BindingType::kStorageBuffer, kStageCompute, true},
                 {1, BindingType::kStorageBuffer, kStageCompute}};
  ld.push = true;
  auto* set = own(device->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set};
  pl.push_constant_bytes = 16;
  pl.push_constant_stages = kStageCompute;
  ComputePipelineDesc cp;
  cp.layout = own(device->CreatePipelineLayout(pl));
  auto code = gcn::BuildIndexConversionShader(
      1, true, gcn::IndexConversionMode::kIndirect);
  cp.code = Code(code);
  auto* convert = own(device->CreateComputePipeline(cp));
  code = gcn::BuildIndexConversionShader(
      1, true, gcn::IndexConversionMode::kIndirectStrided);
  cp.code = Code(code);
  auto* strided = own(device->CreateComputePipeline(cp));
  constexpr u32 kSamples = 9, kRepeats = 16;
  auto* timestamps = own(device->CreateTimestampPool(kSamples * 2));
  auto* list = own(device->CreateCommandList());
  if (!set || !cp.layout || !convert || !strided || !timestamps || !list)
    return 1;
  std::puts("capacity,requested,capacity_us,bounded_us");
  for (u32 capacity : {64u, 1024u, 16384u, 65536u, 262144u, 1048576u}) {
    const u32 maximum = capacity / 4 * 6;
    BufferDesc bd;
    bd.size = u64(capacity) * 4 + 20;
    bd.usage = kBufferStorage | kBufferCopyDst;
    auto* input = own(device->CreateBuffer(bd));
    bd.size = u64(maximum) * 4 + 32;
    bd.usage = kBufferStorage | kBufferIndirect;
    auto* output = own(device->CreateBuffer(bd));
    if (!input || !output)
      return 1;
    for (u32 requested : {0u, 4u, capacity / 8, capacity}) {
      double elapsed[2]{};
      for (u32 strategy = 0; strategy < 2; strategy++) {
        list->Begin();
        list->ResetTimestamps(timestamps, 0, kSamples * 2);
        list->FillBuffer(input, 0, u64(capacity) * 4, 1);
        const u32 arguments[] = {requested, 1, 0, 0, 0};
        list->UpdateBuffer(input, u64(capacity) * 4, sizeof(arguments),
                           arguments);
        list->Barrier(kAccessCopyWrite,
                      kAccessComputeRead | kAccessComputeWrite);
        BindingWrite writes[2];
        writes[0].buffer = input;
        writes[0].range = input->desc().size;
        writes[1].binding = 1;
        writes[1].buffer = output;
        writes[1].range = output->desc().size;
        const u32 params[] = {capacity, 0, capacity, maximum};
        for (u32 sample = 0; sample < kSamples; sample++) {
          list->WriteTimestamp(timestamps, sample * 2, true);
          for (u32 repeat = 0; repeat < kRepeats; repeat++) {
            list->SetPipeline(strategy ? strided : convert);
            list->PushBindGroup(0, set, writes, 2);
            list->SetPushConstants(0, sizeof(params), params);
            list->Dispatch(
                strategy ? gcn::kIndexConversionGroups : (maximum + 63) / 64, 1,
                1);
            list->Barrier(kAccessComputeWrite,
                          kAccessComputeRead | kAccessComputeWrite);
          }
          list->WriteTimestamp(timestamps, sample * 2 + 1, false);
        }
        list->End();
        if (!device->Wait(device->Submit(list)))
          return 1;
        u64 ticks[kSamples * 2]{};
        if (!device->ReadTimestamps(timestamps, 0, kSamples * 2, ticks))
          return 1;
        double samples[kSamples - 1];
        for (u32 i = 1; i < kSamples; i++)
          samples[i - 1] = double(ticks[i * 2 + 1] - ticks[i * 2]) *
                           device->caps().timestamp_period_ns /
                           (1000 * kRepeats);
        std::sort(samples, samples + kSamples - 1);
        elapsed[strategy] = samples[(kSamples - 1) / 2];
      }
      std::printf("%u,%u", capacity, requested);
      for (double value : elapsed)
        std::printf(",%.3f", value);
      std::puts("");
    }
  }
  for (size_t i = objects.size(); i-- > 0;)
    device->Destroy(objects[i]);
  return 0;
}
