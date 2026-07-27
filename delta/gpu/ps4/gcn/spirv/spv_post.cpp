/*
 * PS4Delta : PS4 emulation and research project
 *
 * SPIRV-Tools optimize + validate wrapper. See spv_post.h.
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include "gpu/ps4/gcn/spirv/spv_post.h"

#include <cstdio>

#include <spirv-tools/libspirv.h>
#include <spirv-tools/optimizer.hpp>

namespace gpu::gcn::spirv {

std::vector<uint32_t> Optimize(const std::vector<uint32_t>& spv) {
  spvtools::Optimizer opt(SPV_ENV_VULKAN_1_1);
  opt.SetMessageConsumer([](spv_message_level_t lvl, const char*,
                            const spv_position_t&, const char* msg) {
    if (lvl <= SPV_MSG_WARNING)
      std::fprintf(stderr, "[spv-opt] %s\n", msg);
  });
  // Legalization first: promotes the Private register-file variables to SSA
  // (mem2reg) so the performance passes can actually fold the naive load/store
  // stream the translator emits.
  opt.RegisterLegalizationPasses();
  opt.RegisterPerformancePasses();
  std::vector<uint32_t> out;
  if (!opt.Run(spv.data(), spv.size(), &out) || out.empty())
    return spv;  // keep the valid-but-unoptimized binary on failure
  return out;
}

bool Validate(const std::vector<uint32_t>& spv, std::string* err) {
  spv_context ctx = spvContextCreate(SPV_ENV_VULKAN_1_1);
  spv_diagnostic diag = nullptr;
  spv_const_binary_t bin{spv.data(), spv.size()};
  spv_result_t r = spvValidate(ctx, &bin, &diag);
  bool ok = r == SPV_SUCCESS;
  if (!ok && err && diag)
    *err = diag->error;
  spvDiagnosticDestroy(diag);
  spvContextDestroy(ctx);
  return ok;
}

}  // namespace gpu::gcn::spirv

#endif  // DELTA_HAVE_SPIRV_BACKEND
