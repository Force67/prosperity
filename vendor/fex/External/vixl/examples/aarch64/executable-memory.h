// Copyright 2020, VIXL authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//   * Neither the name of ARM Limited nor the names of its contributors may be
//     used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef VIXL_EXAMPLE_EXECUTABLE_MEMORY_H_
#define VIXL_EXAMPLE_EXECUTABLE_MEMORY_H_

extern "C" {
#include <stdint.h>
#ifndef VIXL_INCLUDE_SIMULATOR_AARCH64
#include <sys/mman.h>
#endif
}

#include <cstdio>
#include <string>

#include "aarch64/assembler-aarch64.h"
#include "aarch64/constants-aarch64.h"
#include "aarch64/cpu-aarch64.h"
#include "aarch64/macro-assembler-aarch64.h"

#ifndef VIXL_INCLUDE_SIMULATOR_AARCH64
class ExecutableMemory {
 public:
  ExecutableMemory(const vixl::byte* code_start, size_t size)
      : size_(size),
        buffer_(reinterpret_cast<vixl::byte*>(mmap(NULL,
                                                   size,
                                                   PROT_READ | PROT_WRITE,
                                                   MAP_SHARED | MAP_ANONYMOUS,
                                                   -1,
                                                   0))) {
    VIXL_CHECK(buffer_ != MAP_FAILED);
    memcpy(buffer_, code_start, size_);

    vixl::aarch64::CPU::EnsureIAndDCacheCoherency(buffer_, size_);
    int res = mprotect(buffer_, size_, PROT_READ | PROT_EXEC);
    VIXL_CHECK(res == 0);
  }
  ~ExecutableMemory() { munmap(buffer_, size_); }

  template <typename T>
  T GetEntryPoint(const vixl::aarch64::Label& entry_point) const {
    int64_t location = entry_point.GetLocation();
    return GetOffsetAddress<T>(location);
  }

 private:
  template <typename T>
  T GetOffsetAddress(int64_t offset) const {
    VIXL_ASSERT((offset >= 0) && (static_cast<size_t>(offset) <= size_));
    T function_address;
    vixl::byte* buffer_address = buffer_ + offset;

    VIXL_STATIC_ASSERT(sizeof(T) == sizeof(buffer_address));
    memcpy(&function_address, &buffer_address, sizeof(T));
    return function_address;
  }

  size_t size_;
  vixl::byte* buffer_;
};
#endif

#endif  // VIXL_EXAMPLE_EXECUTABLE_MEMORY_H_
