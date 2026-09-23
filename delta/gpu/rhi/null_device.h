#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * A device that records instead of rendering. Every operation appends one line
 * to a log, naming resources by their debug name (or by a stable id when they
 * have none), so what the renderer DECIDED can be asserted without a GPU:
 * which pass was opened, in what order the transitions ran, which group was
 * bound where, whether the draw was even issued.
 *
 * That is the whole point of it. The renderer's decisions have never been
 * testable, because reaching them meant standing up a real device first, and
 * the only observable was the frame that came out the far end.
 *
 * Mapped memory is real (a host allocation per buffer), so a test can write
 * upload data and read it back. Copies are recorded, not performed: this
 * device models the command stream, not the GPU.
 */

#include <memory>
#include <string>
#include <vector>

#include "base/arch.h"
#include "gpu/rhi/device.h"

namespace gpu::rhi {

class NullDevice : public Device {
 public:
  NullDevice();
  ~NullDevice() override;

  const char* backend_name() const override;  // NOLINT: accessor
  const DeviceCaps& caps() const override;    // NOLINT: accessor

  std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override;
  std::unique_ptr<Texture> CreateTexture(const TextureDesc& desc) override;
  std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) override;
  std::unique_ptr<ShaderModule> CreateShader(const ShaderDesc& desc) override;
  std::unique_ptr<BindGroupLayout> CreateBindGroupLayout(
      const BindGroupLayoutDesc& desc) override;
  std::unique_ptr<BindGroup> CreateBindGroup(
      const BindGroupDesc& desc) override;
  std::unique_ptr<Pipeline> CreateGraphicsPipeline(
      const GraphicsPipelineDesc& desc) override;
  std::unique_ptr<Pipeline> CreateComputePipeline(
      const ComputePipelineDesc& desc) override;
  std::unique_ptr<CommandList> CreateCommandList() override;

  u64 Submit(CommandList* list) override;
  void Wait(u64 submission) override;
  void WaitIdle() override;

  // The recorded command stream, one line per operation, in record order.
  // Lines from every command list this device made land in the same log.
  const std::vector<std::string>& log() const { return log_; }  // NOLINT
  // The log as one newline-terminated block, for a golden comparison.
  std::string LogText() const;
  void ClearLog();

  // Lines containing `needle`. A test asserting on one decision does not have
  // to spell the whole frame.
  std::vector<std::string> LogMatching(const char* needle) const;

  void Record(std::string line);
  // Stable label for a resource: its debug name, else "<kind><id>".
  std::string Label(const void* object) const;
  void Name(const void* object, const char* name, const char* kind);

 private:
  DeviceCaps caps_;
  std::vector<std::string> log_;
  std::vector<std::pair<const void*, std::string>> names_;
  u32 next_id_ = 0;
  u64 next_submission_ = 1;
};

}  // namespace gpu::rhi
