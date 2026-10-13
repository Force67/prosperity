#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"
#include "base/strings/xstring.h"

namespace gpu::gcn::spirv {

base::Vector<u32> Optimize(const base::Vector<u32>& spv);

bool Validate(const base::Vector<u32>& spv, base::String* err = nullptr);

bool Finalize(const base::Vector<u32>& spv,
              base::Vector<u32>* out,
              base::String* err = nullptr);

void Prefetch(const base::Vector<u32>& spv);
void ResetPrefetch();

void PrefetchThen(const base::Vector<u32>& spv,
                  base::Function<void(const base::Vector<u32>&)> then);

struct PrefetchScope {
  PrefetchScope();
  ~PrefetchScope();
};

}  // namespace gpu::gcn::spirv
