# gpu/ps5

The PS5 GPU is an RDNA2 part driven by the AGC command format, which is unrelated
to the PS4's Liverpool/GCN gen2 PM4 path in `gpu/ps4/`. This directory is
greenfield scaffolding: it will host an AGC command-buffer parser and an
RDNA2->SPIR-V shader recompiler, and must implement the same `gpu::` submit API
(`submitDcb`/`submitCcb`/`endFrame`, here under the `gpu::ps5` namespace) so the
runtime can dispatch to it on PS5 titles. Only unimplemented stubs exist today.
