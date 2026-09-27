#pragma once

// Guest entry points (HLE exports, syscall handlers) use the System V calling
// convention the console's code was compiled for.
#define PS4ABI __attribute__((sysv_abi))
