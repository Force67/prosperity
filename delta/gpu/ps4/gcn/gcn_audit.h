#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Shader translation audit: records, per translated shader, the fate of every
 * instruction (translated / warned-unsupported / silently emitted nothing /
 * unreachable), aggregates the results across the run, and optionally dumps
 * per-shader artifacts. The point is finding unimplemented and
 * silently-wrong translations without staring at warn-once logs:
 *
 *  - DELTA_GPU_SHAUDIT=1|<path>: at exit, print a ranked report of every
 *    unsupported/approximated op (by shaders affected), every silently
 *    dropped instruction, and every declined shader, each with an example
 *    (stage, code hash, pc) to reproduce. With a path, also written there.
 *  - DELTA_GPU_SHDUMP=<dir>: per unique shader, write
 *      <stage>_<hash>.txt  annotated disassembly (per-instruction fate +
 *                          SPIR-V word count -- a non-trivial instruction
 *                          that emitted 0 words is a silent drop),
 *      <stage>_<hash>.gcn  raw bytecode,
 *      <stage>_<hash>.spv  the unoptimized SPIR-V, with OpLine markers
 *                          mapping every SPIR-V op back to its GCN pc
 *                          (spirv-dis shows them as "gcn:<pc>").
 *
 * Single-threaded like the rest of the recompiler (command-processor lock).
 * All hooks are no-ops unless one of the env vars is set.
 */

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/ps4/gcn/gcn_decode.h"

namespace gpu::gcn {

// Either audit or dump requested (cached env lookups).
bool ShaderDebugEnabled();
// DELTA_GPU_SHDUMP set (gates the OpLine source-mapping emission too).
bool ShaderDumpEnabled();

// Begin collecting for one stage translation ("vs" / "ps" / "cs").
// `program` must stay alive until AuditEnd.
void AuditBegin(const char* stage,
                const uint32_t* code,
                const Program& program);

// Bracket the emission of one instruction; `spirv_words` is how many function
// body words it produced.
void AuditInstBegin(uint32_t index, uint32_t pc);
void AuditInstEnd(uint32_t index, uint32_t spirv_words);

// Tag the current instruction as intentionally emitting nothing (e.g. a
// vertex fetch seeded through the vertex-input state) so it is not reported
// as a silent drop.
void AuditInstTag(const char* tag);

// Record an unsupported/approximated-op event (fed by WarnUnsupported), and
// attribute it to the instruction currently being emitted, if any.
void AuditNote(const char* what, uint32_t op);

// One header line for the dump (binding plan etc.).
void AuditPlan(const std::string& line);

// The stage translation was declined outright.
void AuditDecline(const char* reason);

// Finish: fold into the run-wide registry and (first time a shader is seen)
// write the dump files. `spirv` is the assembled unoptimized module, null if
// translation failed.
void AuditEnd(const std::vector<uint32_t>* spirv);

// Write the aggregate report (what the atexit hook prints to stderr).
void WriteAuditReport(std::FILE* f);

}  // namespace gpu::gcn
