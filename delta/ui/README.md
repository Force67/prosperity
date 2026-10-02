# delta/ui

Desktop ImGui menus and overlays. Namespace `ui`.

**Open game** accepts packages, archives, standalone executables, and an
unpacked game's `eboot.bin`. **Open folder** selects its root instead. The
launcher mounts the whole game folder and saves that folder in recent games.

| Unit | Purpose |
|---|---|
| `home_screen.h` | Recent games, artwork, PS4/PS5 chips, native file pickers |
| `pause_menu.h` | Guest pause, Resume, and keyboard controls |
| `overlay.h` | ImGui context, frame building, controls legend |
| `overlay_theme.h` | Palette, fonts, widget style, rounded panels |
| `overlay_log.h` | Live log capture and panel |
| `overlay_busy.h` | Animated shader compilation and frame busy toast |
| `shader_activity.h` | Thread-safe compiler activity tracking |
| `input_sdl.h` | SDL mouse, keyboard, and controller events for the UI |
| `home_background_vk.h` | Procedural logo backgrounds when game artwork is missing |
| `overlay_vk.h` | ImGui Vulkan pipeline and font texture |

`host` owns the window, swapchain, and event pump. It forwards events and calls
this module to draw. `main/home_screen.cc` runs the startup menu and returns the
selected game path to the launcher. UI code does not depend on either module.

Selecting Play fades the menu out over 350 ms. Artwork or the animated fallback
stays visible with a loading indicator and a gentle artwork zoom during startup,
then fades into the first game frame over 550 ms. The same window stays open.

Rounded charcoal panels, white text, and blue focus highlights follow the PS5
menu style. Covers ease into a soft focus halo; titles reveal as you browse.
Keyboard hints use compact keycaps. Logs use a monospace font.

On Linux, press either **Ctrl** key during gameplay to pause guest execution
and audio. The last game frame stays behind an animated panel. Choose **Resume
game**, or press **Ctrl** or **Esc**, to continue. **Controls** shows the keyboard
mapping. Arrow keys and Enter navigate the menu; mouse input works too.
**Exit emulator** closes the application from the pause menu.
Pause becomes available after the first game frame.

The toast reports slow shader translation, optimization, and pipeline builds.
A 150 ms gap between game frames shows "Frame busy". The async presenter redraws
the last GPU image while waiting. `DELTA_GPU_SYNCPRESENT=1` updates the UI only
when the game presents a frame. The pause menu keeps rendering in either mode.

Android builds only the compiler activity tracker. The Android app uses its
own touch controls.

Each menu session chooses a fallback at random and keeps it while browsing.
Delta current, liquid glass, and signal formation share the normal rotation.
A gold anniversary scene appears with a 1-in-256 chance and reveals "EST 2019"
after six seconds. Artwork takes priority; an empty library uses the fallback too.

`+DELTA_UI_BACKGROUND=0` is the default random selection. For development,
`=1`, `=2`, and `=3` select the normal styles. The anniversary scene is available
only through the rare random selection.

Shader sources are in `shaders/home_background.vert` and `.frag`. Rebuild their
checked-in SPIR-V headers with `tools/gen_spv.sh` and `glslangValidator`:

```bash
tools/gen_spv.sh delta/ui/shaders/home_background.vert kHomeBackgroundVert delta/ui/shaders/home_background_vert.h
tools/gen_spv.sh delta/ui/shaders/home_background.frag kHomeBackgroundFrag delta/ui/shaders/home_background_frag.h
clang-format -i delta/ui/shaders/home_background_{vert,frag}.h
```
