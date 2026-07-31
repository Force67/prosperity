// Opens the gfx window and presents an animated synthetic framebuffer, to
// exercise the window, Vulkan swapchain and present() path without the emulator.
// Close the window to exit.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gfx/gfx.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(uint32_t, kTestFrames, "DELTA_GFX_TEST_FRAMES", 0);
}  // namespace

int main() {
  const uint32_t W = 480, H = 270;  // a small framebuffer, scaled to the window
  // DELTA_GFX_TEST_FRAMES, if set, presents that many frames then exits;
  // otherwise runs until the window is closed.
  const uint32_t maxFrames = kTestFrames;

  if (!gfx::init("PS4Delta gfx test", 960, 540))
    return 1;

  std::vector<uint32_t> fb((size_t)W * H);
  uint32_t t = 0;
  while (gfx::pumpEvents()) {
    if (maxFrames && t >= maxFrames) {
      std::printf("[gfx_test] presented %u frames OK\n", t);
      break;
    }
    for (uint32_t y = 0; y < H; y++) {
      for (uint32_t x = 0; x < W; x++) {
        uint8_t r = (uint8_t)(x * 255 / W);
        uint8_t gch = (uint8_t)(y * 255 / H);
        uint8_t b = (uint8_t)(t & 0xff);
        // RGBA8, byte order R,G,B,A -> little-endian u32.
        fb[(size_t)y * W + x] =
            0xff000000u | ((uint32_t)b << 16) | ((uint32_t)gch << 8) | r;
      }
    }
    gfx::present(fb.data(), W, H, 0, gfx::PixelFormat::rgba8);
    t++;
  }
  gfx::shutdown();
  return 0;
}
