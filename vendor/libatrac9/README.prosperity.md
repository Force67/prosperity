# LibAtrac9

Upstream: https://github.com/Thealexbarney/LibAtrac9 (commit efca2e3af35562a09a9bb6deed90e45b4b824dc4),
MIT licensed. Only the C decoder (`C/src`) is vendored.

Used by the /dev/ajm device (delta/kern/ps4/dev/ajm_dev.cc) to decode the
ATRAC9 streams submitted by guest audio jobs. The handle is
allocated with calloc and the decoder is plain C99, so it builds as a plain
static library with no further porting.
