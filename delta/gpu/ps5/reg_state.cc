/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The four ways AGC programs a register. See reg_state.h.
 */

#include "gpu/ps5/reg_state.h"
#include "base/arch.h"

#include <algorithm>

#include <utl/options.h>

#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/guest_address.h"

namespace {
DELTA_OPTION(bool, kNoBlendState, "DELTA_AGC_NOBLENDSTATE", false);
}  // namespace

namespace gpu::ps5 {
namespace {

// Registers per space, so a LOAD_*_REG range can never spill into the next one.
// A LOAD_SH_REG whose range ran long wrote zeros from its (empty) shadow image
// straight over CB_COLOR0_BASE at 0xA318, unbinding the render target that
// op 0x49 had just set -- every colour draw after it hit no target and the
// frame came out black.
constexpr u32 kRegSpaceSize = 0x400;

u32 RegSpaceLimit(u32 base) {
  return base + kRegSpaceSize <= kRegFileSize ? base + kRegSpaceSize
                                              : kRegFileSize;
}

// What the blend object at the end of a context state block said. `present` is
// whether one was recognised at all, `has_mask` whether it was long enough to
// reach the write-mask field, and `applied` whether a mask was written from it.
struct BlendObject {
  bool present = false;
  bool has_mask = false;
  bool applied = false;
};

// The blend object is recognised by its per-target array: the 15 entries at
// pairs-20..pairs-6 carry tags 1..0xF. Requiring that keeps a block which
// merely ENDS in some OTHER type-1 object from being read as blend state --
// doing so zeroed the write mask and silently deleted the draw.
bool EndsWithBlendArray(const u32* p, u32 num_pairs) {
  for (u32 k = 0; k < 15; k++) {
    const u32 tag = p[(num_pairs - 20 + k) * 2];
    if (!(tag & (1u << 28)) || (tag & ~kRegSelectorMask) != k + 1)
      return false;
  }
  return true;
}

// A type-1 block always ENDS with the blend object, and that object's fields sit
// at a constant distance from the block's end: over every shape this title
// submits (245/114/82/78 pairs, and a 21-entry partial) CB_TARGET_MASK is at
// pairs-23 and CB_BLEND0_CONTROL at pairs-24, followed by two spares, the
// per-target array and five more. The object is truncatable: a shorter tail
// stops before the mask field, which then means "writes nothing". That is how a
// clear quad says it must not touch colour -- Minecraft ends every frame with
// one, and taking it as a normal draw overwrote the composited UI with white.
BlendObject ReadBlendObject(Regs& regs,
                            u32 base,
                            u64 address,
                            const u32* p,
                            u32 num_pairs) {
  // The array can also arrive as a block of its OWN, the object's first four
  // entries having gone out with the draw's state in the block before it (57
  // pairs + 21, against 78 in one). The object is 25 entries either way, so
  // joining the two puts the mask back 23 from the end -- in the earlier block.
  // Read apart, the mask stays whatever the previous draw left, which is how
  // the quad that ends Minecraft's loading screen painted flat white over it.
  static u64 prev_addr = 0;
  static u32 prev_pairs = 0;

  BlendObject blend;
  const bool context = !kNoBlendState && base == kContextRegBase;
  const bool array_only = context && num_pairs >= 21 && num_pairs < 24 &&
                          prev_pairs && EndsWithBlendArray(p, num_pairs);
  blend.present = context && num_pairs >= 24 && EndsWithBlendArray(p, num_pairs);
  blend.has_mask =
      blend.present && (p[(num_pairs - 23) * 2] & (1u << 28)) != 0;
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
        gpu::IsReadableRange(prev_addr,
                             static_cast<u64>(prev_pairs) * 2 * 4)) {
      blend.applied = (q[(joined - 23) * 2] & (1u << 28)) != 0;
      if (blend.applied) {
        regs[mmCB_BLEND0_CONTROL] = (q[(joined - 24) * 2] & (1u << 28))
                                        ? q[(joined - 24) * 2 + 1]
                                        : 0;
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

// Type-1 context entries (offset dword bit 28 set) are NOT (offset, value)
// pairs: whole runs share one offset dword, so the flat reading collapses a
// CB_COLORn block onto CB_COLOR0_BASE and every draw ends up with no target.
// Both observed block shapes fit reg(pair) = B + 2*pair, differing only in B,
// and nothing in the packet or the tags carries B. Anchor it on evidence
// instead: a pair whose value is a mapped surface address (>>8) and whose
// pair+2 decodes as a CB_COLOR_INFO with a real FORMAT is a CB_COLOR0 bind.
// Both fields must check out, so a wrong anchor is rejected rather than binding
// garbage.
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
    // ATTRIB2 holds MIP0_WIDTH-1 / MIP0_HEIGHT-1, so it doubles as the check
    // that the anchor is real: a wrong k gives nonsense dimensions.
    const u32 attrib2 = p[(k + 14) * 2 + 1];
    const u32 w = ((attrib2 >> 14) & 0x3FFF) + 1;
    const u32 h = (attrib2 & 0x3FFF) + 1;
    if (!fmt || fmt > 22 || !IsGpuAddress(rt) ||
        !gpu::IsReadableRange(rt, 0x1000) || w < 16 || h < 16 || w > 8192 ||
        h > 8192)
      continue;
    regs[mmCB_COLOR0_BASE] = p[k * 2 + 1];
    regs[mmCB_COLOR0_INFO] = info;
    regs[mmCB_COLOR0_ATTRIB2] = attrib2;
    // A block carrying a base AND a valid colour format IS a colour-target
    // bind, so slot 0 is written. If the block brought no blend tail its write
    // mask is unknown, and a default 0 reads as "writes nothing" -- which would
    // drop the very draw this block set up.
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

void SetShRegsInline(Regs& regs, const u32* body, u32 count) {
  if (count < 2)
    return;
  const u32 first = kShRegBase + (body[0] & 0xFFFF);
  for (u32 i = 1; i < count; i++) {
    const u32 reg = first + (i - 1);
    if (reg < kRegFileSize)
      regs[reg] = body[i];
    NoteRegisterWrite("SET_SH_INLINE(0x93)", reg, body[i]);
  }
}

void LoadRegImage(Regs& regs, u32 base, const u32* body, u32 count) {
  if (count < 4)
    return;
  const u64 image = (static_cast<u64>(body[1] & 0xFFFF) << 32) | body[0];
  if (!IsGpuAddress(image))
    return;  // GPU aperture only
  const u32 limit = RegSpaceLimit(base);
  u64 source_dwords = TraceRegImagePrefixDwords();
  for (u32 i = 2; i + 1 < count; i += 2) {
    const u32 off = body[i] & 0xFFFF;
    const u32 num = std::min<u32>(body[i + 1] & 0xFFFF, 0x2000);
    if (base + off < limit)
      source_dwords = std::max<u64>(
          source_dwords, off + std::min<u32>(num, limit - base - off));
  }
  if (!source_dwords ||
      !gpu::IsReadableRange(image, source_dwords * sizeof(u32)))
    return;
  const u32* src = reinterpret_cast<const u32*>(image);
  TraceRegImage(base, image, body, count);

  // LOAD_*_REG restores a register shadow the command processor is supposed to
  // have SAVED into. We do not model the save side, so a shadow the title never
  // wrote reads as all zeros -- and applying it wipes live state: Skyrim binds
  // CB_COLOR0_BASE with SET_CONTEXT_REG_INDIRECT (0x9f) and the very next
  // LOAD_CONTEXT_REG zeroed it again, leaving every colour draw with no render
  // target. An all-zero shadow carries nothing to restore, so skip it.
  bool any_value = false;
  for (u32 i = 2; i + 1 < count && !any_value; i += 2) {
    const u32 off = body[i] & 0xFFFF;
    const u32 num = std::min<u32>(body[i + 1] & 0xFFFF, 0x2000);
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
    const u32 num = std::min<u32>(body[i + 1] & 0xFFFF, 0x2000);  // sanity cap
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
  // The whole guest map, not just the GPU aperture: a state block is CPU-built
  // data the GPU only reads, so a title is free to put it wherever it built it.
  // Demon's Souls submits 23% of its context blocks out of two places the
  // aperture test rejected -- the AGC system block at 0xfe0_xxxxxxx, which sits
  // just under the 64 GiB floor, and its own eboot data at 0x2014_xxxxxxxx,
  // far over the 1 TiB ceiling. Each one rejected is a whole draw's state
  // (render target, blend, write mask) left at whatever the last draw set. The
  // readability check below is the guard that matters; this one only rejects
  // null and the low pages.
  if (!IsGuestAddress(address)) {
    NoteRegBlock(base, RegBlockOutcome::kBadAddress, address, 0);
    return;
  }
  // body[3] is the count of (reg_offset, value) register PAIRS, not dwords:
  // each iteration reads two dwords (the gfx10.3 SET_*_REG_INDIRECT handlers
  // loop `i < (buffer[3] & 0x3fff)` advancing the pointer by 2). The old
  // dword-count reading wrote nothing for a single-register indirect
  // (num == 1), so VGT_PRIMITIVE_TYPE (set this way) never landed and prim
  // assembly died.
  const u32 num_pairs = body[3] & 0x3FFF;
  if (!num_pairs) {
    NoteRegBlock(base, RegBlockOutcome::kNoPairs, address, 0);
    return;
  }
  if (!gpu::IsReadableRange(address,
                            static_cast<u64>(num_pairs) * 2 * sizeof(u32))) {
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
  // A tail that stops before the mask field: the object says "no colour write".
  // A block whose tail is only the colour-target object instead (it binds a
  // base, so it is not a blend object at all) keeps the mask the anchor forced.
  if (blend.present && !blend.has_mask && !bound_rt) {
    regs[mmCB_BLEND0_CONTROL] = 0;
    regs[mmCB_TARGET_MASK] = 0;
  }
  for (u32 i = 0; i < num_pairs; i++) {
    const u32 off = p[i * 2] & ~kRegSelectorMask;  // strip the gfx10 selector
    if (base == kContextRegBase && (p[i * 2] & (1u << 28)))
      continue;  // handled by the anchor above; a flat write would undo it
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
  // WR_ONE_ADDR (control[16]) repeats one register instead of walking a range.
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
