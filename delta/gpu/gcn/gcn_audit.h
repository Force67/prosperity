#pragma once

#include <cstdio>
#include "base/arch.h"

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

bool ShaderDebugEnabled();
bool ShaderDumpEnabled();

void AuditBegin(const char* stage, const u32* code, const Program& program);

void AuditInstBegin(u32 index, u32 pc);
void AuditInstEnd(u32 index, u32 spirv_words);

void AuditInstTag(const char* tag);

void AuditNote(const char* what, u32 op);

void AuditPlan(const base::String& line);

void AuditDecline(const char* reason);

void AuditEnd(const base::Vector<u32>* spirv);

void WriteAuditReport(std::FILE* f);

}  // namespace gpu::gcn
