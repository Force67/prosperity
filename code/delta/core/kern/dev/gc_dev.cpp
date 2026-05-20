
// Copyright (C) Force67 2019

#include <base.h>
#include "gc_dev.h"

namespace krnl {
gcDevice::gcDevice(proc *p) : device(p) {}

bool gcDevice::init(const char *, uint32_t, uint32_t) { return true; }

/* gc_ioctl */
int32_t gcDevice::ioctl(uint32_t cmd, void *data) {
  switch (cmd) {
  case 0xC00C8110: {

    struct argl {
      uint32_t unknown_0;
      uint32_t unknown_4;
      uint32_t unknown_8;
    };
    auto args = reinterpret_cast<argl *>(data);
    printf("gc ioctl(%x): %x, %x, %x\n", cmd, args->unknown_0, args->unknown_4,
           args->unknown_8);
    return 0;
  }
  case 0xC010810B: {
    struct argl {
      uint32_t cumask0;
      uint32_t cumask1;
      uint32_t cumask2;
      uint32_t cumask3;
    };

    /*idk what the proper value would be*/
    auto se0 = (uint16_t)1024 >> 6;
    auto se1 = (1024 >> 16) & 0x3FF;

    auto args = reinterpret_cast<argl *>(data);
    args->cumask0 = se0;
    args->cumask1 = se0;
    args->cumask2 = se1;
    args->cumask3 = se1;

    return 0;
  }
  case 0xC008811B: {
    struct argl {
      uint64_t unknown;
    };
    auto args = static_cast<argl *>(data);
    printf("gc ioctl(%x): %I64x\n", cmd, args->unknown);
    args->unknown = 0x1234FFFF00000000ull;
    return 0;
  }
  case 0xC0848119: {
    struct argl {
      uint32_t unknown_00;
      uint32_t unknown_04;
      uint32_t unknown_08;
      uint32_t unknown_0C;
      uint8_t unknown_10[112];
      uint32_t unknown_80;
    };
    auto args = static_cast<argl *>(data);
    printf("gc ioctl(%x): %x, %x, %x, %x, %x\n", cmd, args->unknown_00,
           args->unknown_04, args->unknown_08, args->unknown_0C,
           args->unknown_80);
    return 0;
  }
  }

  __debugbreak();

  return 0;
}

/*map to gfx memory*/
uint8_t *gcDevice::map(void *addr, size_t, uint32_t, uint32_t, size_t) {
  //__debugbreak();
  return reinterpret_cast<uint8_t *>(-1);
}
} // namespace krnl
