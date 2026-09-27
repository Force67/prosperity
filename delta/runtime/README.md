# delta/runtime

What runs next to the guest's own code: the HLE modules a title's imports can
bind to, and the lifter that rewrites guest code so it can run natively.

| unit | hides |
|---|---|
| `code_lift` | rewriting guest `syscall` and fs-relative accesses in place (x86-64 host) |
| `vprx/vprx` | the HLE export registry and the per-library HLE/LLE policy |
| `vprx/nid` | the NID codec: symbol name to its 11-character hashed name and back |
| `vprx/init_function.h` | static registration of each module's export table |
| `vprx/sys_params.h` | the system settings sceSystemServiceParamGetInt answers |
| `vprx/ps4/lib_sce_*/` | one PS4 HLE module each: `.h` API, `.cc` behaviour, `_exports.cc` NID table |
| `vprx/ps5/lib_sce_*.cc` | PS5 NID alias tables and the Prospero-only shims |
| `media/videodec` | H.264 decoding for libSceVideodec/AvPlayer (ffmpeg when present) |
