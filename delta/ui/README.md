# delta/ui

Desktop ImGui menus and overlays. Namespace `ui`.

**Add game** accepts packages, archives, standalone executables, and an
unpacked game's `eboot.bin`. **Add folder** selects its root instead. Both add
the game and its artwork to the home menu without starting it. Select **Play**
to launch. Unpacked games are saved by their root folder.

| Unit | Purpose |
|---|---|
| `settings.h` | Lounge settings panel, API cards, preferences, and save/discard |
| `home_screen.h` | Recent games, artwork, PS4/PS5 chips, native file pickers |
| `pause_menu.h` | Guest pause, Resume, and keyboard controls |
| `overlay.h` | ImGui context, frame building, controls legend |
| `overlay_theme.h` | Palette, fonts, widget style, rounded panels |
| `overlay_log.h` | Live log capture and panel |
| `overlay_memory.h` | Compact memory dock and full allocation bucket tiles |
| `overlay_busy.h` | Animated shader compilation and frame busy toast |
| `shader_activity.h` | Thread-safe compiler activity tracking |
| `input_sdl.h` | SDL mouse, keyboard, and controller events for the UI |
| `mouse_look.h` | Optional mouse capture and right-stick camera input |
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

Mouse look is off by default. Enable it in **Settings > Controls**, or press
**F3** during play. **Ctrl > Controls** provides sensitivity (10–300%) and
invert-Y adjustments for the current session. Save defaults in home Settings.
Left click maps to R2, right click to L2. WASD, arrow keys, and controllers
remain available. Pausing, switching windows, and returning home release the
cursor and clear mouse input. Resume restores capture without forwarding a
held menu click. Movement emulates a controller stick, so each game's camera
speed and controller settings still apply.

Options: `DELTA_MOUSE_LOOK`, `DELTA_MOUSE_SENSITIVITY` (percent, default 100),
and `DELTA_MOUSE_INVERT_Y`.

**F4** cycles memory views during play: off, compact dock, full tiles. **Esc**
or controller Circle/B returns from full to compact. The guest keeps running;
the full view releases mouse capture and blocks guest input. Each tile shows
live bytes, backing, free bytes, allocation count, and up to 60 seconds of
growth. Blue, amber, and rose indicate used/backing occupancy, not device-wide
memory pressure. The background stays translucent.

Set `DELTA_MEMORY_OVERLAY=1` or `2` for the initial view. Allocation tracking
is controlled by CMake `DELTA_MEMORY_DEBUG`, on by default for desktop and
off for Android. See [tracking details](../../shared/memory_debug/README.md).
