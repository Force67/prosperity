# Game profiles

One file per title, named after its title id: `CUSA00792.txt`, `PPSA03747.txt`.
The emulator reads `game_profiles/<title id>.txt` from next to the binary (or
from `DELTA_DATA_DIR`) once it has mounted the game and knows the id.

A profile is an options file (see [installation.md](../docs/installation.md)),
with one difference: its entries only fill in options that nothing else has
set, so a profile never overrides your environment, your `--options=` file or a
`+Name=Value` argument. `DELTA_PROFILE=<file>` uses a different one,
`DELTA_PROFILE=off` boots the title without one.

Two things belong in here:

* settings a title needs to run at its best, which is what ships to users;
* the knobs that reproduce where its boot currently gets to, commented out, so
  a debugging session can be picked up later without rediscovering them.

Keep the status line honest and dated: these files double as the record of what
each title does today.
