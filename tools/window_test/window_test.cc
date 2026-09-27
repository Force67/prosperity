// Opens the gfx window and presents an animated synthetic framebuffer, to
// exercise the window, Vulkan swapchain and present() path without the emulator.
// Close the window to exit.
#include "base/arch.h"
#include <cstdio>
#include <cstdlib>

#include "host/window.h"
#include <options/options.h>
#include <base/containers/vector.h>

namespace {
DELTA_OPTION(u32, kTestFrames, "DELTA_GFX_TEST_FRAMES", 0);
}  // namespace

int main() {
  const u32 W = 480, H = 270;  // a small framebuffer, scaled to the window
  // DELTA_GFX_TEST_FRAMES, if set, presents that many frames then exits;
  // otherwise runs until the window is closed.
  const u32 maxFrames = kTestFrames;

  if (!host::Init("PS4Delta gfx test", 960, 540))
    return 1;

  base::Vector<u32> fb((size_t)W * H);
  u32 t = 0;
  while (host::PumpEvents()) {
    if (maxFrames && t >= maxFrames) {
      std::printf("[window_test] presented %u frames OK\n", t);
      break;
    }
    for (u32 y = 0; y < H; y++) {
      for (u32 x = 0; x < W; x++) {
        u8 r = (u8)(x * 255 / W);
        u8 gch = (u8)(y * 255 / H);
        u8 b = (u8)(t & 0xff);
        // RGBA8, byte order R,G,B,A -> little-endian u32.
        fb[(size_t)y * W + x] =
            0xff000000u | ((u32)b << 16) | ((u32)gch << 8) | r;
      }
    }
    host::Present(fb.data(), W, H, 0, host::PixelFormat::kRgba8);
    t++;
  }
  host::Shutdown();
  return 0;
}
