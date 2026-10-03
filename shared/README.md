# shared

Host-side foundations every other layer may use. Depends only on equilibrium
`base`. One include root: headers are spelled from here (`"io/file.h"`).

| unit | hides |
|---|---|
| `guest_abi.h` | the calling convention guest entry points use (`PS4ABI`) |
| `guest/` | pause and session workers, plus the 60 Hz display clock and paced flip completion (`DELTA_DISPLAY_TIMING`, enabled by default) |
| `elf_types.h`, `sce_types.h` | ELF and SCE/SELF on-disk layouts, spec spelling |
| `crypto/` | SHA-1 and HMAC-SHA-1 |
| `host_memory/` | host virtual memory: reserve, commit, protect, mapping identity |
| `io/` | host file streams and paths relative to the executable |
| `logger/` | the asynchronous log and its sinks (`LOG_*`) |
| `options/` | `DELTA_OPTION` knobs and saved UI preferences, followed by environment, files and arguments |
| `profile/` | Tracy zones, frame marks and plots (`DELTA_ZONE`), no-ops without `DELTA_TRACY` |
| `write_watch/` | lets any layer arm the kernel's guest write-watch probe |
