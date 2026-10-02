# delta/ui

Desktop ImGui menus and overlays. Namespace `ui`.

| Unit | Purpose |
|---|---|
| `home_screen.h` | Recent games, artwork, PS4/PS5 chips, native file pickers |
| `overlay.h` | ImGui context, frame building, controls legend |
| `overlay_theme.h` | Palette, fonts, widget style, rounded panels |
| `overlay_log.h` | Live log capture and panel |
| `overlay_busy.h` | Animated shader compilation and frame busy toast |
| `shader_activity.h` | Thread-safe compiler activity tracking |
| `input_sdl.h` | SDL mouse, keyboard, and controller events for the UI |
| `overlay_vk.h` | ImGui Vulkan pipeline and font texture |

`host` owns the window, swapchain, and event pump. It forwards events and calls
this module to draw. `main/home_screen.cc` runs the startup menu and returns the
selected game path to the launcher. UI code does not depend on either module.

Rounded charcoal panels, white text, and blue focus highlights follow the PS5
menu style. Logs use a monospace font.

The toast reports slow shader translation, optimization, and pipeline builds.
A 150 ms gap between game frames shows "Frame busy". The async presenter redraws
the last GPU image while waiting. `DELTA_GPU_SYNCPRESENT=1` updates the UI only
when the game presents a frame.

Android builds only the compiler activity tracker. The Android app uses its
own touch controls.
