# delta/host

The host machine as the emulator sees it: a window to present into, the input
it receives, and audio output. Namespace `host`.

| unit | hides |
|---|---|
| `window.h` | present, input, rumble; one variant per platform, picked by the build |
| `window_sdl.cc` | desktop: SDL3 window + Vulkan swapchain |
| `window_android.cc` | Android app: the NativeActivity surface and touch controls |
| `window_headless.cc` | Android adb runner: no window at all |
| `gameplay_state.cc` | the renderer's "a run is underway" latch the pad autoskip reads |
| `audio_output.h` | PCM playback ports (`audio_output_sdl.cc`, `_stub.cc` without SDL) |
| `ui` | menus and overlays are in [`delta/ui`](../ui/README.md) |
