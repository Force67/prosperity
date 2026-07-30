# Linux game window metadata
Status: implemented

## Context
Linux currently installs the old application icon and gives game windows a
generic title containing only the title ID. Running games also use the generic
application icon, which makes native games and emulated games hard to
distinguish in the taskbar.

## Decision
Install the supplied Prosperity logo as the Linux desktop icon. For PS4 PKGs,
read `TITLE` and `icon0.png` from the existing outer-PKG metadata entries. Use
the game title in the window title, with the title ID as a fallback. On Linux,
decode the game icon when the SDL window is created and draw a small Prosperity
triangle badge over its top-right corner before setting it as the window icon.

Extracted game directories use `sce_sys/param.sfo` and `sce_sys/icon0.png`.
When an `eboot.bin` is launched directly, metadata may also sit beside it.
PS5 package and app-directory boots use `sce_sys/icon0.png` when available.
Invalid, oversized, or missing artwork leaves the default SDL window icon
unchanged.

## Alternatives
- Use the game icon without a badge: rejected because it does not identify the
  taskbar entry as an emulated game.
- Load the installed desktop icon at runtime: rejected because development and
  relocatable installs do not have one reliable asset path.
- Add SDL_image: rejected because PNG-only decoding does not justify another
  runtime dependency.

## Consequences
Game taskbar entries carry their own artwork and remain visibly associated with
Prosperity. The desktop launcher keeps a stable application-only icon. The
Linux graphics target compiles the vendored PNG decoder.

## Acceptance
- Linux installs the supplied PNG as `ps4delta.png`.
- A PS4 PKG window title contains its `TITLE`, falling back to `TITLE_ID`.
- A valid PKG `icon0.png` becomes the Linux window icon with a Prosperity badge
  in its top-right corner.
- Extracted games find `param.sfo` and `icon0.png` beside `eboot.bin` or under
  its `sce_sys` directory.
- Missing or invalid metadata does not prevent the game window from opening.
- The desktop build succeeds.
