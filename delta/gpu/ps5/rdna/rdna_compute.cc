/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 -> GLCompute SPIR-V. See rdna_compute.h.
 */

#include "gpu/ps5/rdna/rdna_compute.h"

namespace gpu::rdna {

gpu::gcn::RecompiledCs RecompileCompute(const uint32_t*,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t,
                                        uint32_t) {
  return {};  // not implemented yet: the caller skips the dispatch
}

}  // namespace gpu::rdna
