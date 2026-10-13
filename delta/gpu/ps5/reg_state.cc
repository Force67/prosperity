/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The four ways AGC programs a register. See reg_state.h.
 */

#include "gpu/ps5/reg_state.h"
#include "base/arch.h"

#include "options/options.h"

#include "base/math/value_bounds.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/guest_address.h"

namespace {
DELTA_OPTION(bool, kNoBlendState, "DELTA_AGC_NOBLENDSTATE", false);
}  // namespace

namespace gpu::ps5 {
namespace {

constexpr u32 kRegSpaceSize = 0x400;

u32 RegSpaceLimit(u32 base) {
  return base + kRegSpaceSize <= kRegFileSize ? base + kRegSpaceSize
                                              : kRegFileSize;
}

struct BlendObject {
  bool present = false;
  bool has_mask = false;
  bool applied = false;
};

bool EndsWithBlendArray(const u32* p, u32 num_pairs) {
  for (u32 k = 0; k < 15; k++) {
    const u32 tag = p[(num_pairs - 20 + k) * 2];
    if (!(tag & (1u << 28)) || (tag & ~kRegSelectorMask) != k + 1)
      return false;
  }
  return true;
}

BlendObject ReadBlendObject(Regs& regs,
                            u32 base,
                            u64 address,
                            const u32* p,
                            u32 num_pairs) {
  static u64 prev_addr = 0;
  static u32 prev_pairs = 0;

  BlendObject blend;
  const bool context = !kNoBlendState && base == kContextRegBase;
  const bool array_only = context && num_pairs >= 21 && num_pairs < 24 &&
                          prev_pairs && EndsWithBlendArray(p, num_pairs);
  blend.present =
      context && num_pairs >= 24 && EndsWithBlendArray(p, num_pairs);
  blend.has_mask = blend.present && (p[(num_pairs - 23) * 2] & (1u << 28)) != 0;
  if (blend.has_mask) {
    regs[mmCB_BLEND0_CONTROL] = (p[(num_pairs - 24) * 2] & (1u << 28))
                                    ? p[(num_pairs - 24) * 2 + 1]
                                    : 0;
    regs[mmCB_TARGET_MASK] = p[(num_pairs - 23) * 2 + 1];
    blend.applied = true;
  }
  if (array_only) {
    const u32 joined = prev_pairs + num_pairs;
    const u32* q = reinterpret_cast<const u32*>(prev_addr);
    if (joined >= 24 && joined - 24 < prev_pairs &&
        gpu::IsReadableRangeCached(prev_addr,
                                   static_cast<u64>(prev_pairs) * 2 * 4)) {
      blend.applied = (q[(joined - 23) * 2] & (1u << 28)) != 0;
      if (blend.applied) {
        regs[mmCB_BLEND0_CONTROL] =
            (q[(joined - 24) * 2] & (1u << 28)) ? q[(joined - 24) * 2 + 1] : 0;
        regs[mmCB_TARGET_MASK] = q[(joined - 23) * 2 + 1];
      }
    }
  }
  if (base == kContextRegBase) {
    prev_addr = address;
    prev_pairs = num_pairs;
  }
  return blend;
}

bool AnchorColorTarget(Regs& regs,
                       u32 base,
                       const u32* p,
                       u32 num_pairs,
                       bool blend_applied) {
  if (base != kContextRegBase || num_pairs < 15 || !(p[0] & (1u << 28)))
    return false;
  for (u32 k = 0; k + 14 < num_pairs; k++) {
    const u64 rt = static_cast<u64>(p[k * 2 + 1]) << 8;
    const u32 info = p[(k + 2) * 2 + 1];
    const u32 fmt = (info >> 2) & 0x1F;
    const u32 attrib2 = p[(k + 14) * 2 + 1];
    const u32 w = ((attrib2 >> 14) & 0x3FFF) + 1;
    const u32 h = (attrib2 & 0x3FFF) + 1;
    if (!fmt || fmt > 22 || !IsGpuAddress(rt) ||
        !gpu::IsReadableRangeCached(rt, 0x1000) || w < 16 || h < 16 ||
        w > 8192 || h > 8192)
      continue;
    regs[mmCB_COLOR0_BASE] = p[k * 2 + 1];
    regs[mmCB_COLOR0_INFO] = info;
    regs[mmCB_COLOR0_ATTRIB2] = attrib2;
    if (!blend_applied && !(regs[mmCB_TARGET_MASK] & 0xF))
      regs[mmCB_TARGET_MASK] |= 0xF;
    TraceRegBlockAnchor(k, rt, info, w, h);
    return true;
  }
  return false;
}

}  // namespace

void SetRegs(Regs& regs, u32 base, const u32* body, u32 count) {
  if (count < 1)
    return;
  const u32 first = base + (body[0] & ~kRegSelectorMask);
  const u32 limit = RegSpaceLimit(base);
  for (u32 i = 1; i < count; i++) {
    const u32 reg = first + (i - 1);
    if (reg < limit)
      regs[reg] = body[i];
    NoteRegisterWrite("SET_REG", reg, body[i]);
  }
}

void SetRegRun(Regs& regs, u32 first_reg, const u32* values, u32 count) {
  for (u32 i = 0; i < count; i++) {
    if (first_reg + i < kRegFileSize)
      regs[first_reg + i] = values[i];
    NoteRegisterWrite("type0", first_reg + i, values[i]);
  }
  TraceType0ShaderRegs(first_reg, values, count);
}

void LoadRegImage(Regs& regs, u32 base, const u32* body, u32 count) {
  if (count < 4)
    return;
  const u64 image = (static_cast<u64>(body[1] & 0xFFFF) << 32) | body[0];
  if (!IsGpuAddress(image))
    return;
  const u32 limit = RegSpaceLimit(base);
  u64 source_dwords = TraceRegImagePrefixDwords();
  for (u32 i = 2; i + 1 < count; i += 2) {
    const u32 off = body[i] & 0xFFFF;
    const u32 num = base::Min<u32>(body[i + 1] & 0xFFFF, 0x2000);
    if (base + off < limit)
      source_dwords = base::Max<u64>(
          source_dwords, off + base::Min<u32>(num, limit - base - off));
  }
  if (!source_dwords ||
      !gpu::IsReadableRangeCached(image, source_dwords * sizeof(u32)))
    return;
  const u32* src = reinterpret_cast<const u32*>(image);
  TraceRegImage(base, image, body, count);

  bool any_value = false;
  for (u32 i = 2; i + 1 < count && !any_value; i += 2) {
    const u32 off = body[i] & 0xFFFF;
    const u32 num = base::Min<u32>(body[i + 1] & 0xFFFF, 0x2000);
    for (u32 j = 0; j < num; j++)
      if (base + off + j < limit && src[off + j]) {
        any_value = true;
        break;
      }
  }
  if (!any_value)
    return;

  for (u32 i = 2; i + 1 < count; i += 2) {
    const u32 off = body[i] & 0xFFFF;
    const u32 num = base::Min<u32>(body[i + 1] & 0xFFFF, 0x2000);
    for (u32 j = 0; j < num; j++) {
      if (base + off + j >= limit)
        break;
      regs[base + off + j] = src[off + j];
      NoteRegisterWrite("LOAD_REG", base + off + j, src[off + j]);
    }
  }
}

void LoadRegBlock(Regs& regs, u32 base, const u32* body, u32 count) {
  if (count < 4) {
    NoteRegBlock(base, RegBlockOutcome::kShortPacket, 0, 0);
    return;
  }
  const u64 address =
      (static_cast<u64>(body[1] & 0xFFFF) << 32) | (body[0] & 0xFFFFFFFCu);
  if (!IsGuestAddress(address)) {
    NoteRegBlock(base, RegBlockOutcome::kBadAddress, address, 0);
    return;
  }
  const u32 num_pairs = body[3] & 0x3FFF;
  if (!num_pairs) {
    NoteRegBlock(base, RegBlockOutcome::kNoPairs, address, 0);
    return;
  }
  if (!gpu::IsReadableRangeCached(
          address, static_cast<u64>(num_pairs) * 2 * sizeof(u32))) {
    NoteRegBlock(base, RegBlockOutcome::kUnreadable, address, num_pairs);
    return;
  }
  NoteRegBlock(base, RegBlockOutcome::kApplied, address, num_pairs);
  TraceRegBlock(base, address, num_pairs, body[2]);

  const u32* p = reinterpret_cast<const u32*>(address);
  const u32 limit = RegSpaceLimit(base);
  const BlendObject blend = ReadBlendObject(regs, base, address, p, num_pairs);
  const bool bound_rt =
      AnchorColorTarget(regs, base, p, num_pairs, blend.applied);
  if (blend.present && !blend.has_mask && !bound_rt) {
    regs[mmCB_BLEND0_CONTROL] = 0;
    regs[mmCB_TARGET_MASK] = 0;
  }
  for (u32 i = 0; i < num_pairs; i++) {
    const u32 off = p[i * 2] & ~kRegSelectorMask;
    if (base == kContextRegBase && (p[i * 2] & (1u << 28)))
      continue;
    if (base + off < limit)
      regs[base + off] = p[i * 2 + 1];
    NoteRegisterWrite("SET_REG_INDIRECT", base + off, p[i * 2 + 1]);
    TraceColorBaseWrite(base + off, p[i * 2 + 1], i, num_pairs, address,
                        p[i * 2]);
  }
  if (base == kShRegBase)
    TraceShaderBind(address, regs[mmSPI_SHADER_PGM_LO_GS],
                    regs[mmSPI_SHADER_PGM_LO_PS]);
}

void WriteDataRegs(Regs& regs, const u32* body, u32 count) {
  const u32 first = body[1] & ~kRegSelectorMask;
  const bool one_addr = (body[0] >> 16) & 1;
  for (u32 i = 0; i < count - 3; i++) {
    const u32 reg = one_addr ? first : first + i;
    if (reg >= kRegFileSize)
      continue;
    regs[reg] = body[3 + i];
    NoteRegisterWrite("WRITE_DATA", reg, body[3 + i]);
  }
}

}  // namespace gpu::ps5
