/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Handing a finished frame to the window. gfx::present blocks on the window
// swapchain and, on a software driver, rasterizes the blit on the CPU, so a
// dedicated thread owns the window and always shows the newest complete frame.

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace gpu::vk {

class LatestFramePresenter {
 public:
  LatestFramePresenter() = default;
  ~LatestFramePresenter();

  LatestFramePresenter(const LatestFramePresenter&) = delete;
  LatestFramePresenter& operator=(const LatestFramePresenter&) = delete;

  void Present(const uint8_t* pixels, uint32_t w, uint32_t h);
  void Present(std::vector<uint8_t>&& pixels, uint32_t w, uint32_t h);
  void Stop();

 private:
  void StartLocked();
  void Run();

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::vector<uint8_t> pending_pixels_;  // BGRA, tight pitch; latest wins
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  bool pending_ = false;
  bool stopping_ = false;
};

}  // namespace gpu::vk
