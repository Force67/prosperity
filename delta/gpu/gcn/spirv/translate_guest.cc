/* PS4Delta: checked guest-address reads and writes for runtime compute. */
#ifdef DELTA_HAVE_SPIRV_BACKEND
#include "gpu/gcn/spirv/translator.h"

namespace gpu::gcn {
namespace {
Id Wide(Translator& t, Id value) {
  return t.m.Emit(spv::Op::OpUConvert, t.m.TypeInt(64, false), {value});
}
Id Pair(Translator& t, Id lo, Id hi) {
  const Id wide = t.m.TypeInt(64, false);
  return t.m.Emit(spv::Op::OpBitwiseOr, wide,
                  {Wide(t, lo), t.m.Emit(spv::Op::OpShiftLeftLogical, wide,
                                         {Wide(t, hi), t.U32(32)})});
}
}  // namespace

void DeclareGuestMemory(Translator& t, StageContext& sc, u32 binding) {
  t.m.PhysicalStorageBuffers();
  const Id wide = t.m.TypeInt(64, false);
  const Id array = t.m.TypeRuntimeArray(wide);
  t.m.Decorate(array, spv::Decoration::ArrayStride, {8});
  const Id block = t.m.TypeStruct({array});
  t.m.Decorate(block, spv::Decoration::Block);
  t.m.MemberDecorate(block, 0, spv::Decoration::Offset, {0});
  sc.cs_guest_table =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::StorageBuffer, block),
                   spv::StorageClass::StorageBuffer);
  t.m.Decorate(sc.cs_guest_table, spv::Decoration::DescriptorSet, {0});
  t.m.Decorate(sc.cs_guest_table, spv::Decoration::Binding, {binding});
  t.m.Decorate(sc.cs_guest_table, spv::Decoration::NonWritable);
  t.m.Name(sc.cs_guest_table, "guest_address_map");
  std::vector<Id> args;
  sc.cs_guest_translate =
      t.m.BeginFunction(wide, t.m.TypeFunction(wide, {wide, t.t_u, t.t_u}),
                        {wide, t.t_u, t.t_u}, &args);
  t.m.Name(sc.cs_guest_translate, "translate_guest_address");
  const Id function_ptr = t.m.TypePointer(spv::StorageClass::Function, t.t_u);
  const Id lo = t.m.Emit(spv::Op::OpVariable, function_ptr,
                         {static_cast<u32>(spv::StorageClass::Function)});
  const Id hi = t.m.Emit(spv::Op::OpVariable, function_ptr,
                         {static_cast<u32>(spv::StorageClass::Function)});
  const Id zero = t.U32(0), zero64 = Wide(t, zero);
  const Id count =
      t.Shr(t.m.Emit(spv::Op::OpArrayLength, t.t_u, {sc.cs_guest_table, 0}),
            t.U32(2));
  t.m.Store(lo, zero);
  t.m.Store(hi, count);
  const Id ptr = t.m.TypePointer(spv::StorageClass::StorageBuffer, wide);
  const auto field = [&](Id record, u32 member) {
    return t.m.Load(
        wide,
        t.m.AccessChain(ptr, sc.cs_guest_table,
                        {zero, t.Add(t.Mul(record, t.U32(4)), t.U32(member))}));
  };
  // upper_bound(address) over sorted, disjoint readable guest mappings.
  const Id header = t.m.NewBlock(), body = t.m.NewBlock();
  const Id next = t.m.NewBlock(), done = t.m.NewBlock();
  t.m.Branch(header);
  t.m.OpenBlock(header);
  const Id left = t.m.Load(t.t_u, lo), right = t.m.Load(t.t_u, hi);
  const Id more = t.Ult(left, right);
  t.m.LoopMerge(done, next);
  t.m.BranchConditional(more, body, done);
  t.m.OpenBlock(body);
  const Id mid = t.Add(left, t.Shr(t.Sub(right, left), t.U32(1)));
  const Id before =
      t.m.Emit(spv::Op::OpULessThanEqual, t.t_bool, {field(mid, 0), args[0]});
  t.m.Store(lo, t.SelectB(before, t.Add(mid, t.U32(1)), left));
  t.m.Store(hi, t.SelectB(before, right, mid));
  t.m.Branch(next);
  t.m.OpenBlock(next);
  t.m.Branch(header);
  t.m.OpenBlock(done);
  const Id upper = t.m.Load(t.t_u, lo);
  const Id record = t.SelectB(t.IsNonZero(upper), t.Sub(upper, t.U32(1)), zero);
  const Id base = field(record, 0), end = field(record, 1),
           host = field(record, 2);
  Id valid = t.LAnd(t.IsNonZero(upper), t.IsNonZero(args[1]));
  valid = t.LAnd(
      valid, t.m.Emit(spv::Op::OpULessThanEqual, t.t_bool, {base, args[0]}));
  valid = t.LAnd(valid,
                 t.m.Emit(spv::Op::OpULessThanEqual, t.t_bool, {args[0], end}));
  valid = t.LAnd(valid,
                 t.m.Emit(spv::Op::OpULessThanEqual, t.t_bool,
                          {Wide(t, args[1]),
                           t.m.Emit(spv::Op::OpISub, wide, {end, args[0]})}));
  const Id low_address = t.m.Emit(spv::Op::OpUConvert, t.t_u, {args[0]});
  valid = t.LAnd(valid, t.IsZero(t.And(low_address, t.U32(3))));
  const Id dirty = field(record, 3);
  const Id writing = t.IsNonZero(args[2]);
  const Id can_write =
      t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {dirty, zero64});
  valid = t.LAnd(valid, t.m.Emit(spv::Op::OpLogicalOr, t.t_bool,
                                 {t.IsZero(args[2]), can_write}));
  const Id mark = t.m.NewBlock(), marked = t.m.NewBlock();
  const Id mark_valid = t.LAnd(valid, writing);
  t.m.SelectionMerge(marked);
  t.m.BranchConditional(mark_valid, mark, marked);
  t.m.OpenBlock(mark);
  const Id offset = t.m.Emit(spv::Op::OpISub, wide, {args[0], base});
  const Id first = t.m.Emit(
      spv::Op::OpUConvert, t.t_u,
      {t.m.Emit(spv::Op::OpShiftRightLogical, wide, {offset, t.U32(2)})});
  const Id last = t.Add(first, t.Shr(t.Sub(args[1], t.U32(1)), t.U32(2)));
  const Id physical =
      t.m.TypePointer(spv::StorageClass::PhysicalStorageBuffer, t.t_u);
  const auto dirty_ptr = [&](Id byte) {
    return t.m.Emit(spv::Op::OpConvertUToPtr, physical,
                    {t.m.Emit(spv::Op::OpIAdd, wide, {dirty, Wide(t, byte)})});
  };
  const Id scope = t.U32(static_cast<u32>(spv::Scope::Device));
  t.m.Emit(spv::Op::OpAtomicUMin, t.t_u, {dirty_ptr(zero), scope, zero, first});
  t.m.Emit(spv::Op::OpAtomicUMax, t.t_u,
           {dirty_ptr(t.U32(4)), scope, zero, t.Add(last, t.U32(1))});
  // One bit per dword, so fallback writeback does not overwrite untouched
  // words. All callers access at most a 128-byte BVH node (read-only) or four
  // dwords. Store callers mark a single dword, including atomic byte/halfword
  // updates.
  const Id bits =
      dirty_ptr(t.Add(t.U32(8), t.Mul(t.Shr(first, t.U32(5)), t.U32(4))));
  t.m.Emit(spv::Op::OpAtomicOr, t.t_u,
           {bits, scope, zero, t.Shl(t.U32(1), t.And(first, t.U32(31)))});
  t.m.Branch(marked);
  t.m.OpenBlock(marked);
  const Id translated =
      t.m.Emit(spv::Op::OpIAdd, wide,
               {host, t.m.Emit(spv::Op::OpISub, wide, {args[0], base})});
  t.m.EmitVoid(spv::Op::OpReturnValue, {t.m.Emit(spv::Op::OpSelect, wide,
                                                 {valid, translated, zero64})});
  t.m.EndFunction();
}

Id CsGuestBase(Translator& t, StageContext& sc, u32 binding) {
  const auto [kind, sgpr] = sc.cs_runtime_resources.at(binding);
  const Id base = Pair(t, t.Sg(sgpr),
                       t.And(t.Sg(sgpr + 1), t.U32(kind == 3 ? 255 : 65535)));
  if (kind == 3)
    return t.m.Emit(spv::Op::OpShiftLeftLogical, t.m.TypeInt(64, false),
                    {base, t.U32(8)});
  return base;
}

Id CsGuestAddress(Translator& t,
                  StageContext& sc,
                  Id address,
                  Id bytes,
                  bool write) {
  return t.m.Emit(spv::Op::OpFunctionCall, t.m.TypeInt(64, false),
                  {sc.cs_guest_translate, address, bytes, t.U32(write)});
}

Id CsPhysicalLoad(Translator& t, Id address) {
  const Id pointer =
      t.m.TypePointer(spv::StorageClass::PhysicalStorageBuffer, t.t_u);
  const Id p = t.m.Emit(spv::Op::OpConvertUToPtr, pointer, {address});
  return t.m.Emit(spv::Op::OpLoad, t.t_u,
                  {p, static_cast<u32>(spv::MemoryAccessMask::Aligned), 4});
}

Id CsGuestLoad(Translator& t,
               StageContext& sc,
               u32 binding,
               Id dword_idx,
               Id required_end) {
  const Id wide = t.m.TypeInt(64, false);
  const Id offset = t.m.Emit(spv::Op::OpShiftLeftLogical, wide,
                             {Wide(t, dword_idx), t.U32(2)});
  const Id address = CsGuestAddress(
      t, sc,
      t.m.Emit(spv::Op::OpIAdd, wide, {CsGuestBase(t, sc, binding), offset}),
      t.U32(4));
  Id valid =
      t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {address, Wide(t, t.U32(0))});
  const auto [kind, sgpr] = sc.cs_runtime_resources.at(binding);
  if (kind == 0) {
    const Id stride = t.And(t.Shr(t.Sg(sgpr + 1), t.U32(16)), t.U32(0x3fff));
    const Id bytes =
        t.m.Emit(spv::Op::OpIMul, wide,
                 {Wide(t, t.UMax(stride, t.U32(1))), Wide(t, t.Sg(sgpr + 2))});
    valid = t.LAnd(
        valid, t.m.Emit(spv::Op::OpULessThanEqual, t.t_bool,
                        {required_end ? Wide(t, required_end)
                                      : t.m.Emit(spv::Op::OpIAdd, wide,
                                                 {offset, Wide(t, t.U32(4))}),
                         bytes}));
  }
  const Id from = t.m.CurrentBlock(), load = t.m.NewBlock(),
           done = t.m.NewBlock();
  t.m.SelectionMerge(done);
  t.m.BranchConditional(valid, load, done);
  t.m.OpenBlock(load);
  const Id value = CsPhysicalLoad(t, address);
  t.m.Branch(done);
  t.m.OpenBlock(done);
  return t.m.Emit(spv::Op::OpPhi, t.t_u, {value, load, t.U32(0), from});
}

namespace {
Id PhysicalPointer(Translator& t, Id address) {
  return t.m.Emit(
      spv::Op::OpConvertUToPtr,
      t.m.TypePointer(spv::StorageClass::PhysicalStorageBuffer, t.t_u),
      {address});
}
void PhysicalStore(Translator& t, Id address, Id value) {
  t.m.EmitVoid(spv::Op::OpStore,
               {PhysicalPointer(t, address), value,
                static_cast<u32>(spv::MemoryAccessMask::Aligned), 4});
}
Id NonNull(Translator& t, Id address) {
  return t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {address, Wide(t, t.U32(0))});
}
}  // namespace

void CsGuestStore(Translator& t,
                  StageContext& sc,
                  u32 binding,
                  Id dword_idx,
                  Id value) {
  const Id wide = t.m.TypeInt(64, false);
  const Id offset = t.m.Emit(spv::Op::OpShiftLeftLogical, wide,
                             {Wide(t, dword_idx), t.U32(2)});
  Id allowed = t.LaneActive(t.Exec());
  const auto [kind, sgpr] = sc.cs_runtime_resources.at(binding);
  if (kind == 0) {
    const Id stride = t.And(t.Shr(t.Sg(sgpr + 1), t.U32(16)), t.U32(0x3fff));
    const Id bytes =
        t.m.Emit(spv::Op::OpIMul, wide,
                 {Wide(t, t.UMax(stride, t.U32(1))), Wide(t, t.Sg(sgpr + 2))});
    allowed = t.LAnd(
        allowed,
        t.m.Emit(spv::Op::OpULessThanEqual, t.t_bool,
                 {t.m.Emit(spv::Op::OpIAdd, wide, {offset, Wide(t, t.U32(4))}),
                  bytes}));
  }
  const Id write = t.m.NewBlock(), done = t.m.NewBlock();
  t.m.SelectionMerge(done);
  t.m.BranchConditional(allowed, write, done);
  t.m.OpenBlock(write);
  const Id address = CsGuestAddress(
      t, sc,
      t.m.Emit(spv::Op::OpIAdd, wide, {CsGuestBase(t, sc, binding), offset}),
      t.U32(4), true);
  const Id store = t.m.NewBlock(), stored = t.m.NewBlock();
  const Id mapped = NonNull(t, address);
  t.m.SelectionMerge(stored);
  t.m.BranchConditional(mapped, store, stored);
  t.m.OpenBlock(store);
  PhysicalStore(t, address, value);
  t.m.Branch(stored);
  t.m.OpenBlock(stored);
  t.m.Branch(done);
  t.m.OpenBlock(done);
}

void EmitGuestGlobal(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1], op = (w >> 18) & 0x7f;
  const bool store = op >= 0x18;
  const u32 vaddr = w1 & 255, saddr = (w1 >> 16) & 127;
  const u32 data = store ? (w1 >> 8) & 255 : w1 >> 24;
  const i32 immediate = static_cast<i32>((w & 0xfff) << 20) >> 20;
  const Id wide = t.m.TypeInt(64, false);
  Id address = saddr == 125 ? Pair(t, t.Vg(vaddr), t.Vg(vaddr + 1))
                            : t.m.Emit(spv::Op::OpIAdd, wide,
                                       {Pair(t, t.Sg(saddr), t.Sg(saddr + 1)),
                                        Wide(t, t.Vg(vaddr))});
  address = t.m.Emit(spv::Op::OpIAdd, wide,
                     {address, Pair(t, t.U32(static_cast<u32>(immediate)),
                                    t.U32(immediate < 0 ? ~0u : 0))});
  const u32 minor = op & 15;
  const bool subword = minor < 12;
  const u32 count = subword       ? 1
                    : minor == 14 ? 4
                    : minor == 15 ? 3
                                  : minor - 11;
  const u32 bits = subword ? ((minor & 2) ? 16 : 8) : 32;
  if (data + count > 256 || (saddr == 125 && vaddr == 255)) {
    sc.cs_unsupported = true;
    return;
  }
  std::vector<Id> values;
  if (store)
    for (u32 i = 0; i < count; ++i)
      values.push_back(t.Vg(data + i));
  const Id active_done = store ? t.m.NewBlock() : 0;
  if (store) {
    const Id active = t.LaneActive(t.Exec()), body = t.m.NewBlock();
    t.m.SelectionMerge(active_done);
    t.m.BranchConditional(active, body, active_done);
    t.m.OpenBlock(body);
  }
  // Snapshot the full address and every result before assigning overlapping
  // VGPRs.
  for (u32 i = 0; i < count; ++i) {
    const Id guest =
        t.m.Emit(spv::Op::OpIAdd, wide, {address, Wide(t, t.U32(i * 4))});
    const Id low = t.m.Emit(spv::Op::OpUConvert, t.t_u, {guest});
    const Id aligned = subword
                           ? t.m.Emit(spv::Op::OpBitwiseAnd, wide,
                                      {guest, Pair(t, t.U32(~3u), t.U32(~0u))})
                           : guest;
    const Id checked = bits == 16 ? t.m.Emit(spv::Op::OpSelect, wide,
                                             {t.IsZero(t.And(low, t.U32(1))),
                                              aligned, Wide(t, t.U32(0))})
                                  : aligned;
    const Id phys = CsGuestAddress(t, sc, checked, t.U32(4), store);
    Id valid = NonNull(t, phys);
    if (bits == 16)
      valid = t.LAnd(valid, t.IsZero(t.And(low, t.U32(1))));
    const Id from = t.m.CurrentBlock(), access = t.m.NewBlock(),
             done = t.m.NewBlock();
    t.m.SelectionMerge(done);
    t.m.BranchConditional(valid, access, done);
    t.m.OpenBlock(access);
    Id result = 0;
    if (!store) {
      result = CsPhysicalLoad(t, phys);
      if (subword) {
        const Id shift = t.Mul(t.And(low, t.U32(3)), t.U32(8));
        result = t.And(t.Shr(result, shift), t.U32((1u << bits) - 1));
        if (minor & 1)
          result = t.Sar(t.Shl(result, t.U32(32 - bits)), t.U32(32 - bits));
      }
    } else if (!subword) {
      PhysicalStore(t, phys, values[i]);
    } else {
      // Multiple lanes may write distinct bytes of one dword. CAS preserves
      // their neighbours; a non-atomic read/modify/write loses those updates.
      const Id ptr = PhysicalPointer(t, phys);
      const Id scope = t.U32(static_cast<u32>(spv::Scope::Device)),
               zero = t.U32(0);
      const Id shift = t.Mul(t.And(low, t.U32(3)), t.U32(8));
      const Id mask = t.Shl(t.U32((1u << bits) - 1), shift);
      const Id insert = t.And(t.Shl(values[i], shift), mask);
      const Id initial =
          t.m.Emit(spv::Op::OpAtomicLoad, t.t_u, {ptr, scope, zero});
      const Id header = t.m.NewBlock(), retry = t.m.NewBlock(),
               complete = t.m.NewBlock();
      const Id returned = t.m.Alloc();
      t.m.Branch(header);
      t.m.OpenBlock(header);
      const Id expected =
          t.m.Emit(spv::Op::OpPhi, t.t_u, {initial, access, returned, retry});
      const Id replacement = t.Or(t.And(expected, t.Not(mask)), insert);
      t.m.EmitVoid(
          spv::Op::OpAtomicCompareExchange,
          {t.t_u, returned, ptr, scope, zero, zero, replacement, expected});
      const Id success = t.Eq(returned, expected);
      t.m.LoopMerge(complete, retry);
      t.m.BranchConditional(success, complete, retry);
      t.m.OpenBlock(retry);
      t.m.Branch(header);
      t.m.OpenBlock(complete);
    }
    const Id end = t.m.CurrentBlock();
    t.m.Branch(done);
    t.m.OpenBlock(done);
    if (!store)
      values.push_back(
          t.m.Emit(spv::Op::OpPhi, t.t_u, {result, end, t.U32(0), from}));
  }
  if (!store)
    for (u32 i = 0; i < count; ++i)
      t.SetVg(data + i, values[i]);
  else {
    t.m.Branch(active_done);
    t.m.OpenBlock(active_done);
  }
}
void CsGuestStoreMasked(Translator& t, StageContext& sc, Id guest, Id value, Id mask) {
  const Id address = CsGuestAddress(t, sc, guest, t.U32(4), true);
  const Id valid = NonNull(t, address);
  const Id access = t.m.NewBlock(), done = t.m.NewBlock();
  t.m.SelectionMerge(done);
  t.m.BranchConditional(valid, access, done);
  t.m.OpenBlock(access);
  const Id ptr = PhysicalPointer(t, address);
  const Id scope = t.U32(static_cast<u32>(spv::Scope::Device)), zero = t.U32(0);
  const Id initial = t.m.Emit(spv::Op::OpAtomicLoad, t.t_u, {ptr, scope, zero});
  const Id header = t.m.NewBlock(), retry = t.m.NewBlock(), complete = t.m.NewBlock();
  const Id returned = t.m.Alloc();
  t.m.Branch(header);
  t.m.OpenBlock(header);
  const Id expected = t.m.Emit(spv::Op::OpPhi, t.t_u, {initial, access, returned, retry});
  const Id replacement = t.Or(t.And(expected, t.Not(mask)), t.And(value, mask));
  t.m.EmitVoid(spv::Op::OpAtomicCompareExchange,
      {t.t_u, returned, ptr, scope, zero, zero, replacement, expected});
  const Id success = t.Eq(returned, expected);
  t.m.LoopMerge(complete, retry);
  t.m.BranchConditional(success, complete, retry);
  t.m.OpenBlock(retry);
  t.m.Branch(header);
  t.m.OpenBlock(complete);
  t.m.Branch(done);
  t.m.OpenBlock(done);
}

namespace {
struct LinearImage {
  Translator& t;
  u32 sgpr;
  Id field(u32 word, u32 shift, u32 mask) {
    return t.And(t.Shr(t.Sg(sgpr + word), t.U32(shift)), t.U32(mask));
  }
  Id any(Id value, std::initializer_list<u32> options) {
    Id result = t.m.ConstBool(false);
    for (u32 option : options)
      result = t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {result, t.Eq(value, t.U32(option))});
    return result;
  }
};
}

Id CsLinearImageEligible(Translator& t, const Inst& inst) {
  LinearImage im{t, ((inst.raw[1] >> 16) & 31) * 4};
  const Id format = im.field(1, 20, 511), type = im.field(3, 28, 15);
  Id valid = im.any(format, {5, 18, 20, 21, 27, 28, 60, 61, 62, 63, 69, 70, 75, 76});
  valid = t.LAnd(valid, im.any(type, {8, 9, 12, 13}));
  valid = t.LAnd(valid, t.IsZero(im.field(3, 12, 0x1fff))); // mip 0, SW_LINEAR
  valid = t.LAnd(valid, t.IsZero(im.field(5, 4, 15))); // one physical mip
  valid = t.LAnd(valid, t.IsZero(t.And(t.Sg(im.sgpr + 4), t.U32(0xe000c000))));
  const Id array_ok = t.m.Emit(spv::Op::OpLogicalOr, t.t_bool,
      {t.Ult(type, t.U32(12)), t.m.Emit(spv::Op::OpULessThanEqual, t.t_bool,
          {im.field(4, 16, 8191), im.field(4, 0, 8191)})});
  return t.LAnd(valid, array_ok);
}

void EmitCsLinearImage(Translator& t, const Inst& inst, StageContext& sc,
                       const Id* address) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const bool store = ((w >> 18) & 127) == 8;
  const u32 dmask = (w >> 8) & 15, vdata = (w1 >> 8) & 255;
  LinearImage im{t, ((w1 >> 16) & 31) * 4};
  const auto coord = [&](u32 i) { return address ? address[i] : t.Vg((w1 & 255) + i); };
  const Id format = im.field(1, 20, 511), type = im.field(3, 28, 15);
  const Id bits = t.SelectB(im.any(format, {5, 18, 60, 61}), t.U32(8),
      t.SelectB(im.any(format, {27, 28, 69, 70}), t.U32(16), t.U32(32)));
  const Id channels = t.SelectB(im.any(format, {5, 20, 21}), t.U32(1),
      t.SelectB(im.any(format, {18, 27, 28, 62, 63}), t.U32(2), t.U32(4)));
  const Id component_bytes = t.Shr(bits, t.U32(3));
  const Id texel_bytes = t.Mul(component_bytes, channels);
  const Id width = t.Add(t.Or(im.field(1, 30, 3), t.Shl(im.field(2, 0, 4095), t.U32(2))), t.U32(1));
  const Id height = t.Add(im.field(2, 14, 16383), t.U32(1));
  const Id one_d = im.any(type, {8, 12});
  const Id arrayed = im.any(type, {12, 13});
  const Id x = coord(0), y = t.SelectB(one_d, t.U32(0), coord(1));
  const Id layer = t.SelectB(arrayed,
      t.Add(im.field(4, 16, 8191), t.SelectB(one_d, coord(1), coord(2))), t.U32(0));
  const Id layers = t.SelectB(arrayed, t.Add(im.field(4, 0, 8191), t.U32(1)), t.U32(1));
  const Id rows = t.SelectB(one_d, t.U32(1), height);
  const Id pitch = t.And(t.Add(t.Mul(width, texel_bytes), t.U32(255)), t.U32(~255u));
  const Id wide = t.m.TypeInt(64, false);
  const Id base = t.m.Emit(spv::Op::OpShiftLeftLogical, wide,
      {Pair(t, t.Sg(im.sgpr), im.field(1, 0, 255)), t.U32(8)});
  // Compute in 64 bits: an array view can cover several GiB of decoder frames.
  const Id row = t.m.Emit(spv::Op::OpIAdd, wide,
      {t.m.Emit(spv::Op::OpIMul, wide, {Wide(t, layer), Wide(t, rows)}), Wide(t, y)});
  const Id offset = t.m.Emit(spv::Op::OpIAdd, wide,
      {t.m.Emit(spv::Op::OpIMul, wide, {row, Wide(t, pitch)}),
       Wide(t, t.Mul(x, texel_bytes))});
  const Id pixel = t.m.Emit(spv::Op::OpIAdd, wide, {base, offset});
  Id valid = t.LAnd(t.Ult(x, width), t.LAnd(t.Ult(y, rows), t.Ult(layer, layers)));
  valid = t.LAnd(valid, t.LaneActive(t.Exec()));
  // Keep sources intact when VDATA overlaps the coordinates.
  Id source[4] = {}, result[4] = {};
  u32 packed = 0;
  for (u32 c = 0; c < 4; ++c)
    if (dmask & (1u << c)) source[c] = t.Vg(vdata + packed++);
  const Id access = t.m.NewBlock(), done = t.m.NewBlock();
  const Id from = t.m.CurrentBlock();
  t.m.SelectionMerge(done);
  t.m.BranchConditional(valid, access, done);
  t.m.OpenBlock(access);
  const Id value_mask = t.Shr(t.U32(~0u), t.Sub(t.U32(32), bits));
  for (u32 c = 0; c < 4; ++c) {
    if (!(dmask & (1u << c))) continue;
    const Id selector = store ? t.U32(c + 4) : im.field(3, c * 3, 7);
    const Id channel = t.Sub(selector, t.U32(4));
    const Id channel_valid = t.Ult(channel, channels);
    const Id channel_from = t.m.CurrentBlock();
    const Id component = t.m.NewBlock(), component_done = t.m.NewBlock();
    t.m.SelectionMerge(component_done);
    t.m.BranchConditional(channel_valid, component, component_done);
    t.m.OpenBlock(component);
    const Id guest = t.m.Emit(spv::Op::OpIAdd, wide,
        {pixel, Wide(t, t.Mul(channel, component_bytes))});
    const Id low = t.m.Emit(spv::Op::OpUConvert, t.t_u, {guest});
    const Id aligned = t.m.Emit(spv::Op::OpBitwiseAnd, wide,
        {guest, Pair(t, t.U32(~3u), t.U32(~0u))});
    const Id shift = t.Mul(t.And(low, t.U32(3)), t.U32(8));
    Id value = 0;
    if (store) {
      CsGuestStoreMasked(t, sc, aligned, t.Shl(source[c], shift), t.Shl(value_mask, shift));
    } else {
      const Id physical = CsGuestAddress(t, sc, aligned, t.U32(4));
      const Id mapped = NonNull(t, physical), mapped_from = t.m.CurrentBlock();
      const Id read = t.m.NewBlock(), read_done = t.m.NewBlock();
      t.m.SelectionMerge(read_done);
      t.m.BranchConditional(mapped, read, read_done);
      t.m.OpenBlock(read);
      const Id raw = t.And(t.Shr(CsPhysicalLoad(t, physical), shift), value_mask);
      t.m.Branch(read_done);
      t.m.OpenBlock(read_done);
      value = t.m.Emit(spv::Op::OpPhi, t.t_u, {raw, read, t.U32(0), mapped_from});
      const Id sign_shift = t.Sub(t.U32(32), bits);
      value = t.SelectB(im.any(format, {21, 28, 61, 63, 70, 76}),
          t.Sar(t.Shl(value, sign_shift), sign_shift), value);
    }
    const Id component_end = t.m.CurrentBlock();
    t.m.Branch(component_done);
    t.m.OpenBlock(component_done);
    if (!store) {
      // Phi operands must be defined in their corresponding predecessor.
      // Use zero here; constant-one selectors are applied after the merge.
      result[c] = t.m.Emit(spv::Op::OpPhi, t.t_u,
          {value, component_end, t.U32(0), channel_from});
      const Id fallback = t.SelectB(t.Eq(selector, t.U32(1)), t.U32(1), t.U32(0));
      result[c] = t.SelectB(channel_valid, result[c], fallback);
    }
  }
  const Id access_end = t.m.CurrentBlock();
  t.m.Branch(done);
  t.m.OpenBlock(done);
  if (!store) {
    Id merged[4] = {};
    for (u32 c = 0; c < 4; ++c)
      if (dmask & (1u << c))
        merged[c] = t.m.Emit(spv::Op::OpPhi, t.t_u, {result[c], access_end, t.U32(0), from});
    packed = 0;
    for (u32 c = 0; c < 4; ++c)
      if (dmask & (1u << c)) t.SetVg(vdata + packed++, merged[c]);
  }
}
}  // namespace gpu::gcn
#endif
