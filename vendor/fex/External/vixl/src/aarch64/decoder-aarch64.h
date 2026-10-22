// Copyright 2019, VIXL authors
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

#ifndef VIXL_AARCH64_DECODER_AARCH64_H_
#define VIXL_AARCH64_DECODER_AARCH64_H_

#include <initializer_list>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../globals-vixl.h"

#include "instructions-aarch64.h"

// List macro containing all visitors needed by the decoder class.
#define VISITOR_LIST_THAT_RETURN(V) \
  V(SVEBroadcastBitmaskImm)         \
  V(Unallocated)                    \
  V(Unimplemented)

#define SIM_AUD_VISITOR_LIST_THAT_RETURN(V)                      \
  V(NEONScalar2RegMiscFP16)                                      \
  V(NEONScalar3SameFP16)                                         \
  V(NEONScalar3SameExtra)                                        \
  V(SVE32BitGatherLoadHalfwords_ScalarPlus32BitScaledOffsets)    \
  V(SVE32BitGatherLoadWords_ScalarPlus32BitScaledOffsets)        \
  V(SVE32BitGatherLoad_ScalarPlus32BitUnscaledOffsets)           \
  V(SVE32BitGatherPrefetch_ScalarPlus32BitScaledOffsets)         \
  V(SVE32BitScatterStore_ScalarPlus32BitScaledOffsets)           \
  V(SVE32BitScatterStore_ScalarPlus32BitUnscaledOffsets)         \
  V(SVE64BitGatherLoad_ScalarPlus32BitUnpackedScaledOffsets)     \
  V(SVE64BitGatherLoad_ScalarPlus64BitScaledOffsets)             \
  V(SVE64BitGatherLoad_ScalarPlus64BitUnscaledOffsets)           \
  V(SVE64BitGatherLoad_ScalarPlusUnpacked32BitUnscaledOffsets)   \
  V(SVE64BitScatterStore_ScalarPlus64BitScaledOffsets)           \
  V(SVE64BitScatterStore_ScalarPlus64BitUnscaledOffsets)         \
  V(SVE64BitScatterStore_ScalarPlusUnpacked32BitScaledOffsets)   \
  V(SVE64BitScatterStore_ScalarPlusUnpacked32BitUnscaledOffsets) \
  V(SVEBitwiseLogical_Predicated)                                \
  V(SVEBitwiseShiftByVector_Predicated)                          \
  V(SVEConditionallyBroadcastElementToVector)                    \
  V(SVEConditionallyExtractElementToSIMDFPScalar)                \
  V(SVEConstructivePrefix_Unpredicated)                          \
  V(SVEContiguousNonTemporalLoad_ScalarPlusScalar)               \
  V(SVEContiguousNonTemporalStore_ScalarPlusScalar)              \
  V(SVEContiguousStore_ScalarPlusScalar)                         \
  V(SVEExtractElementToSIMDFPScalarRegister)                     \
  V(SVEFFRInitialise)                                            \
  V(SVEFFRWriteFromPredicate)                                    \
  V(SVEFPConvertToInt)                                           \
  V(SVEInsertSIMDFPScalarRegister)                               \
  V(SVEIntAddSubtractVectors_Predicated)                         \
  V(SVEIntMinMaxDifference_Predicated)                           \
  V(SVEIntMinMaxImm_Unpredicated)                                \
  V(SVEIntMulImm_Unpredicated)                                   \
  V(SVEIntMulVectors_Predicated)                                 \
  V(SVELoadAndBroadcastQOWord_ScalarPlusScalar)                  \
  V(SVELoadMultipleStructures_ScalarPlusImm)                     \
  V(SVELoadMultipleStructures_ScalarPlusScalar)                  \
  V(SVEPartitionBreakCondition)                                  \
  V(SVEPermutePredicateElements)                                 \
  V(SVEPredicateFirstActive)                                     \
  V(SVEPredicateReadFromFFR_Unpredicated)                        \
  V(SVEPredicateTest)                                            \
  V(SVEPredicateZero)                                            \
  V(SVEPropagateBreakToNextPartition)                            \
  V(SVEStoreMultipleStructures_ScalarPlusImm)                    \
  V(SVEStoreMultipleStructures_ScalarPlusScalar)                 \
  V(SVETableLookup)                                              \
  V(SVEUnpackPredicateElements)                                  \
  V(SVEVectorSplice)                                             \
  V(SVEFPComplexMulAddIndex)                                     \
  V(SVEFPMulIndex)                                               \
  V(SVEFPMulAddIndex)                                            \
  V(SVEIncDecByPredicateCount)                                   \
  V(SVEIntArithmeticUnpredicated)                                \
  V(SVEIntCompareSignedImm)                                      \
  V(SVEIntCompareUnsignedImm)                                    \
  V(SVEIntCompareVectors)                                        \
  V(SVEIntMulAddPredicated)                                      \
  V(SVEMovprfx)                                                  \
  V(SVEPermuteVectorExtract)                                     \
  V(SVEPermuteVectorInterleaving)                                \
  V(SVEPredicateCount)                                           \
  V(SVEPredicateNextActive)                                      \
  V(SVEPredicateReadFromFFR_Predicated)                          \
  V(SVEPropagateBreak)                                           \
  V(SVEStackFrameAdjustment)                                     \
  V(SVEStackFrameSize)                                           \
  V(SVEContiguousLoad_ScalarPlusScalar)                          \
  V(RotateRightIntoFlags)                                        \
  V(EvaluateIntoFlags)                                           \
  V(ConditionalCompareRegister)                                  \
  V(ConditionalCompareImmediate)                                 \
  V(PCRelAddressing)                                             \
  V(UnconditionalBranch)                                         \
  V(DataProcessing1Source)                                       \
  V(CompareBranch)                                               \
  V(TestBranch)                                                  \
  V(LoadStoreRCpcUnscaledOffset)                                 \
  V(LoadStoreUnscaledOffset)                                     \
  V(LoadLiteral)                                                 \
  V(LoadStorePairNonTemporal)                                    \
  V(LoadStorePAC)                                                \
  V(FPCompare)                                                   \
  V(FPConditionalCompare)                                        \
  V(FPConditionalSelect)                                         \
  V(FPDataProcessing2Source)                                     \
  V(FPDataProcessing3Source)                                     \
  V(FPIntegerConvert)                                            \
  V(FPFixedPointConvert)                                         \
  V(Exception)                                                   \
  V(Crypto2RegSHA)                                               \
  V(Crypto3RegSHA)                                               \
  V(CryptoAES)                                                   \
  V(NEON2RegMiscFP16)                                            \
  V(SVEContiguousLoad_ScalarPlusImm)                             \
  V(SVEPredicateInitialize)                                      \
  V(SVEIndexGeneration)                                          \
  V(SVEAddressGeneration)                                        \
  V(SVELoadAndBroadcastQOWord_ScalarPlusImm)                     \
  V(SVEIntAddSubtractImm_Unpredicated)                           \
  V(SVEIntCompareScalarCountAndLimit)                            \
  V(FPImmediate)                                                 \
  V(FPDataProcessing1Source)                                     \
  V(NEONModifiedImmediate)                                       \
  V(NEONTable)                                                   \
  V(SVEIntReduction)                                             \
  V(SVEMulIndex)                                                 \
  V(NEON3SameFP16)                                               \
  V(NEONLoadStoreSingleStruct)                                   \
  V(NEONLoadStoreSingleStructPostIndex)                          \
  V(NEONLoadStoreMultiStructPostIndex)                           \
  V(LoadStorePreIndex)                                           \
  V(LoadStorePostIndex)                                          \
  V(LoadStoreUnsignedOffset)                                     \
  V(LoadStoreRegisterOffset)                                     \
  V(LoadStorePairPostIndex)                                      \
  V(LoadStorePairOffset)                                         \
  V(LoadStorePairPreIndex)                                       \
  V(SVEBitwiseShiftByWideElements_Predicated)                    \
  V(NEONScalarPairwise)                                          \
  V(SVEFPUnaryOp)                                                \
  V(SVEFPRoundToIntegralValue)                                   \
  V(SVEFPUnaryOpUnpredicated)                                    \
  V(SVEFPMulAdd)                                                 \
  V(SVEFPFastReduction)                                          \
  V(SVEFPComplexMulAdd)                                          \
  V(SVEFPComplexAddition)                                        \
  V(SVEFPCompareWithZero)                                        \
  V(SVEFPCompareVectors)                                         \
  V(SVEFPArithmeticUnpredicated)                                 \
  V(SVEFPAccumulatingReduction)                                  \
  V(SVEIntUnaryArithmeticPredicated)                             \
  V(SVEBitwiseShiftUnpredicated)                                 \
  V(SVEUnpackVectorElements)                                     \
  V(SVEIntConvertToFP)                                           \
  V(SVEReverseWithinElements)                                    \
  V(SVEReversePredicateElements)                                 \
  V(SVEReverseVectorElements)                                    \
  V(SVEInsertGeneralRegister)                                    \
  V(SVEFPTrigSelectCoefficient)                                  \
  V(SVEFPTrigMulAddCoefficient)                                  \
  V(SVEFPExponentialAccelerator)                                 \
  V(SVEFPArithmetic_Predicated)                                  \
  V(SVEFPArithmeticWithImm_Predicated)                           \
  V(SVEBitwiseLogicalWithImm_Unpredicated)                       \
  V(UnconditionalBranchToRegister)                               \
  V(NEONExtract)                                                 \
  V(ConditionalSelect)                                           \
  V(LogicalImmediate)                                            \
  V(AddSubImmediate)                                             \
  V(AddSubShifted)                                               \
  V(AddSubExtended)                                              \
  V(LogicalShifted)                                              \
  V(Extract)                                                     \
  V(DataProcessing2Source)                                       \
  V(DataProcessing3Source)                                       \
  V(ConditionalBranch)                                           \
  V(Bitfield)                                                    \
  V(AtomicMemory)                                                \
  V(SVEVectorSelect)                                             \
  V(SVEPredicateLogical)                                         \
  V(SVEIntDivideVectors_Predicated)                              \
  V(SVEBroadcastIntImm_Unpredicated)                             \
  V(SVEBroadcastFPImm_Unpredicated)                              \
  V(SVEBroadcastGeneralRegister)                                 \
  V(SVECompressActiveElements)                                   \
  V(SVEConditionallyTerminateScalars)                            \
  V(SVEConditionallyExtractElementToGeneralRegister)             \
  V(SVEBitwiseShiftByImm_Predicated)                             \
  V(SVECopyGeneralRegisterToVector_Predicated)                   \
  V(SVECopyIntImm_Predicated)                                    \
  V(SVECopySIMDFPScalarRegisterToVector_Predicated)              \
  V(SVECopyFPImm_Predicated)                                     \
  V(SVEExtractElementToGeneralRegister)                          \
  V(SVEIntMulAddUnpredicated)                                    \
  V(SVEContiguousStore_ScalarPlusImm)                            \
  V(SVEContiguousPrefetch_ScalarPlusScalar)                      \
  V(SVEContiguousPrefetch_ScalarPlusImm)                         \
  V(SVEContiguousNonFaultLoad_ScalarPlusImm)                     \
  V(SVEBitwiseLogicalUnpredicated)                               \
  V(SVE64BitGatherPrefetch_ScalarPlus64BitScaledOffsets)         \
  V(SVE64BitGatherPrefetch_ScalarPlusUnpacked32BitScaledOffsets) \
  V(SVE64BitGatherPrefetch_VectorPlusImm)                        \
  V(SVE32BitGatherPrefetch_VectorPlusImm)                        \
  V(SVELoadVectorRegister)                                       \
  V(SVEStoreVectorRegister)                                      \
  V(SVELoadPredicateRegister)                                    \
  V(SVEStorePredicateRegister)                                   \
  V(SVEFPConvertPrecision)                                       \
  V(SVE32BitGatherLoad_VectorPlusImm)                            \
  V(SVE32BitScatterStore_VectorPlusImm)                          \
  V(SVE64BitScatterStore_VectorPlusImm)                          \
  V(SVEContiguousFirstFaultLoad_ScalarPlusScalar)                \
  V(SVEContiguousNonTemporalLoad_ScalarPlusImm)                  \
  V(SVEContiguousNonTemporalStore_ScalarPlusImm)                 \
  V(SVELoadAndBroadcastElement)                                  \
  V(NEONPerm)                                                    \
  V(NEONLoadStoreMultiStruct)                                    \
  V(NEON3Same)                                                   \
  V(NEON3SameExtra)                                              \
  V(NEON2RegMisc)                                                \
  V(NEONByIndexedElement)                                        \
  V(SVE64BitGatherLoad_VectorPlusImm)                            \
  V(NEONShiftImmediate)                                          \
  V(NEONCopy)                                                    \
  V(NEONScalar2RegMisc)                                          \
  V(NEONScalar3Diff)                                             \
  V(NEONScalar3Same)                                             \
  V(NEONScalarCopy)                                              \
  V(NEONScalarByIndexedElement)                                  \
  V(NEONAcrossLanes)                                             \
  V(NEONScalarShiftImmediate)                                    \
  V(NEON3Different)                                              \
  V(MoveWideImmediate)                                           \
  V(SVEElementCount)                                             \
  V(SVEIncDecRegisterByElementCount)                             \
  V(SVEIncDecVectorByElementCount)                               \
  V(SVESaturatingIncDecVectorByElementCount)                     \
  V(SVESaturatingIncDecRegisterByElementCount)                   \
  V(LoadStoreExclusive)                                          \
  V(SVEBroadcastIndexElement)                                    \
  V(System)                                                      \
  V(AddSubWithCarry)

#define VISITOR_LIST_THAT_DONT_RETURN(V) V(Reserved)

#define VISITOR_LIST(V)       \
  VISITOR_LIST_THAT_RETURN(V) \
  VISITOR_LIST_THAT_DONT_RETURN(V)

#define SIM_AUD_VISITOR_LIST(V) \
  VISITOR_LIST(V)               \
  SIM_AUD_VISITOR_LIST_THAT_RETURN(V)

namespace vixl {
namespace aarch64 {

struct Metadata {
  std::string form;
  bool unallocated = false;
};

// The Visitor interface consists only of the Visit() method. User classes
// that inherit from this one must provide an implementation of the method.
// Information about the instruction encountered by the Decoder is available
// via the metadata pointer.
class DecoderVisitor {
 public:
  enum VisitorConstness { kConstVisitor, kNonConstVisitor };
  explicit DecoderVisitor(VisitorConstness constness = kConstVisitor)
      : constness_(constness) {}

  virtual ~DecoderVisitor() {}

  virtual void Visit(Metadata* metadata, const Instruction* instr) = 0;

  bool IsConstVisitor() const { return constness_ == kConstVisitor; }
  Instruction* MutableInstruction(const Instruction* instr) {
    VIXL_ASSERT(!IsConstVisitor());
    return const_cast<Instruction*>(instr);
  }

 private:
  const VisitorConstness constness_;
};

class Decoder;

typedef void (Decoder::*DecodeFnPtr)(const Instruction*);
typedef uint32_t (Instruction::*BitExtractFn)(void) const;

// A Visitor node maps the name of a visitor to the function that handles it.
struct VisitorNode {
  const char* name;
  const DecodeFnPtr visitor_fn;
};

// DecodePattern and DecodeMapping represent the input data to the decoder
// compilation stage. After compilation, the decoder is embodied in the graph
// of CompiledDecodeNodes pointer to by compiled_decoder_root_.

// A DecodePattern maps a pattern of set/unset/don't care (1, 0, x) bits to the
// hash of its handler name.
// Patterns are encoded as packed mask/value in a uint32_t:
//   pattern = (mask << 16) | value
// where each character contributes one bit in sample-order.
struct DecodePattern {
  uint32_t pattern;
  uint32_t handler;
};

// A DecodeMapping consists of the bits sampled in the instruction by that
// handler, and a mapping from the pattern that those sampled bits match to the
// corresponding hash of the name of a node.
struct DecodeMapping {
  static constexpr uint32_t GenerateSampledBitsMask(
      std::initializer_list<uint8_t> sampled_bits) {
    uint32_t mask = 0;
    for (uint8_t bit : sampled_bits) {
      mask |= 1U << bit;
    }
    return mask;
  }

  const uint32_t sampled_bits_mask;
  const std::vector<DecodePattern> mapping;
};

// For speed, before nodes can be used for decoding instructions, they must
// be compiled. This converts the mapping "bit pattern strings to decoder name
// string" stored in kDecodeMapping to an array look up for the pointer to the
// next node, stored in CompiledDecodeNodes. Compilation may also apply other
// optimisations for simple decode patterns.
class CompiledDecodeNode {
 public:
  // Constructor for decode node, containing a decode table and pointer to a
  // function that extracts the bits to be sampled.
  CompiledDecodeNode(BitExtractFn bit_extract_fn, size_t decode_table_size)
      : bit_extract_fn_(bit_extract_fn), decode_table_size_(decode_table_size) {
    decode_table_ = new CompiledDecodeNode*[decode_table_size_];
    memset(decode_table_, 0, decode_table_size_ * sizeof(decode_table_[0]));
  }

  // Constructor for wrappers around decoded instructions. These require no
  // decoding, so no bit extraction function or decode table is assigned.
  explicit CompiledDecodeNode(uint32_t hash)
      : bit_extract_fn_(NULL), decode_table_(NULL), hash_(hash) {}

  ~CompiledDecodeNode() VIXL_NEGATIVE_TESTING_ALLOW_EXCEPTION {
    // Free the decode table, if this is a compiled, non-leaf node.
    if (decode_table_ != NULL) {
      VIXL_ASSERT(!IsLeafNode());
      delete[] decode_table_;
    }
  }

  // Decode the instruction by either sampling the bits using the bit extract
  // function to find the next node, or, if we're at a leaf, calling the visitor
  // function.
  void Decode(const Instruction* instr, Decoder* decoder) const;

  // A leaf node represents a completely decoded instruction.
  bool IsLeafNode() const { return bit_extract_fn_ == NULL; }

  // Get a pointer to the next node required in the decode process, based on the
  // bits sampled by the current node.
  CompiledDecodeNode* GetNodeForBits(uint32_t bits) const {
    VIXL_ASSERT(bits < decode_table_size_);
    return decode_table_[bits];
  }

  // Set the next node in the decode process for the pattern of sampled bits in
  // the current node.
  void SetNodeForBits(uint32_t bits, CompiledDecodeNode* n) {
    VIXL_ASSERT(bits < decode_table_size_);
    VIXL_ASSERT(n != NULL);
    decode_table_[bits] = n;
  }

  void AddUnallocatedMaskValue(uint64_t mv) {
    VIXL_ASSERT(IsLeafNode());
    unallocated_masks_.push_back(mv);
  }

  bool IsUnallocated(const Instruction* instr) {
    for (uint64_t mv : unallocated_masks_) {
      uint32_t mask = mv >> 32;
      uint32_t value = mv & 0xffffffff;
      if (instr->Mask(mask) == value) {
        return true;
      }
    }
    return false;
  }

 private:
  BitExtractFn GetBitExtractFunction() const {
    VIXL_ASSERT(!IsLeafNode());
    VIXL_ASSERT(bit_extract_fn_ != NULL);
    return bit_extract_fn_;
  }

  uint32_t GetHash() const {
    VIXL_ASSERT(IsLeafNode());
    return hash_;
  }

  // Pointer to an instantiated template function for extracting the bits
  // sampled by this node. Set to NULL for leaf nodes.
  const BitExtractFn bit_extract_fn_;

  // Mapping table from instruction bits to next decode stage.
  CompiledDecodeNode** decode_table_;

  // If this is a leaf node, there is no decode table, so reuse the size storage
  // for the hash of the leaf node's instruction form.
  union {
    const size_t decode_table_size_;
    const uint32_t hash_;
  };

  std::vector<uint64_t> unallocated_masks_;
};

// The instruction decoder is constructed from a graph of decode nodes. At each
// node, a number of bits are sampled from the instruction being decoded. The
// resulting value is used to look up the next node in the graph, which then
// samples other bits, and moves to other decode nodes. Eventually, a leaf
// node is reached, and the corresponding visitor function is called, which
// handles the instruction.
class Decoder {
 public:
  Decoder() {
    std::lock_guard<std::mutex> guard(decoder_mtx_);

    if (compiled_decoder_root_ == NULL) {
      VIXL_ASSERT(hash_to_form_ == NULL);
      hash_to_form_ = GetHashToFormMap();

      ConstructDecodeGraph();
      PopulateGraphUnallocatedFields();
    } else {
      VIXL_ASSERT(hash_to_form_ != NULL);
    }
  }

  // Top-level wrappers around the actual decoding function.
  void Decode(const Instruction* instr);
  void Decode(Instruction* instr);

  // Decode all instructions from start (inclusive) to end (exclusive).
  template <typename T>
  void Decode(T start, T end) {
    for (T instr = start; instr < end; instr = instr->GetNextInstruction()) {
      Decode(instr);
    }
  }

  // Register a new visitor class with the decoder.
  // Decode() will call the corresponding visitor method from all registered
  // visitor classes when decoding reaches the leaf node of the instruction
  // decode tree.
  // Visitors are called in order.
  // A visitor can be registered multiple times.
  //
  //   d.AppendVisitor(V1);
  //   d.AppendVisitor(V2);
  //   d.PrependVisitor(V2);
  //   d.AppendVisitor(V3);
  //
  //   d.Decode(i);
  //
  // will call in order visitor methods in V2, V1, V2, V3.
  void AppendVisitor(DecoderVisitor* visitor);
  void PrependVisitor(DecoderVisitor* visitor);
  // These helpers register `new_visitor` before or after the first instance of
  // `registered_visiter` in the list.
  // So if
  //   V1, V2, V1, V2
  // are registered in this order in the decoder, calls to
  //   d.InsertVisitorAfter(V3, V1);
  //   d.InsertVisitorBefore(V4, V2);
  // will yield the order
  //   V1, V3, V4, V2, V1, V2
  //
  // For more complex modifications of the order of registered visitors, one can
  // directly access and modify the list of visitors via the `visitors()'
  // accessor.
  void InsertVisitorBefore(DecoderVisitor* new_visitor,
                           DecoderVisitor* registered_visitor);
  void InsertVisitorAfter(DecoderVisitor* new_visitor,
                          DecoderVisitor* registered_visitor);

  // Remove all instances of a previously registered visitor class from the list
  // of visitors stored by the decoder.
  void RemoveVisitor(DecoderVisitor* visitor);

  void VisitNamedInstruction(const Instruction* instr, uint32_t form_hash);

  std::list<DecoderVisitor*>* visitors() { return &visitors_; }

  // Compile the representation of the decoding tree stored in kDecodeMapping
  // to a tree of CompiledDecodeNodes. Returns a pointer to the node at the
  // root of the compiled tree (or subtree).
  CompiledDecodeNode* Compile(uint32_t hash);

  bool NodeIsCompiled(uint32_t hash) { return compiled_nodes_.count(hash) > 0; }

  bool IsLeafNode(uint32_t hash) { return hash_to_form_->count(hash) > 0; }

  // Extract mask and value from a packed (mask << 16) | value pattern.
  using MaskValuePair = std::pair<Instr, Instr>;
  MaskValuePair GenerateMaskValuePair(uint32_t pattern) const;

  // Get a pointer to an instruction method that extracts the instruction bits
  // specified by the mask argument, and returns those sampled bits as a
  // contiguous sequence, suitable for indexing an array.
  // For example, a mask of 0b1010 returns a function that, given an instruction
  // 0bXYZW, will return 0bXZ.
  BitExtractFn GetBitExtractFunction(uint32_t mask) {
    return GetBitExtractFunctionHelper(mask, 0);
  }

  // Get a pointer to an Instruction method that applies a mask to the
  // instruction bits, and tests if the result is equal to value. The returned
  // function gives a 1 result if (inst & mask == value), 0 otherwise.
  BitExtractFn GetBitExtractFunction(uint32_t mask, uint32_t value) {
    return GetBitExtractFunctionHelper(value, mask);
  }

 private:
  // Decodes an instruction and calls the visitor functions registered with the
  // Decoder class.
  void DecodeInstruction(const Instruction* instr);

  // Visitors are registered in a list.
  std::list<DecoderVisitor*> visitors_;
  Metadata metadata_;

  // Compile the dynamically generated decode graph based on the static
  // information in kDecodeMapping.
  void ConstructDecodeGraph();

  // Root node for the compiled decoder graph, stored here to avoid a map lookup
  // for every instruction decoded.
  inline static CompiledDecodeNode* compiled_decoder_root_ = NULL;
  inline static std::mutex decoder_mtx_;

  // Map of node name hashes to CompiledDecodeNodes.
  inline static std::unordered_map<uint32_t, CompiledDecodeNode*>
      compiled_nodes_;

  static void PopulateGraphUnallocatedFields();

  // Map from hash of instruction form to its string.
  using HashToFormMap = std::unordered_map<uint32_t, std::string>;
  inline static const HashToFormMap* hash_to_form_ = NULL;

  static const HashToFormMap* GetHashToFormMap();

  // Helper function that returns a bit extracting function. If y is zero,
  // x is a bit extraction mask. Otherwise, y is the mask, and x is the value
  // to match after masking.
  BitExtractFn GetBitExtractFunctionHelper(uint32_t x, uint32_t y);

  // Try to compile a more optimised decode operation for the mapping. Returns
  // a compiled node if successful, NULL otherwise.
  CompiledDecodeNode* TryCompileOptimisedDecodeTable(const DecodeMapping& d);
};

}  // namespace aarch64
}  // namespace vixl

#endif  // VIXL_AARCH64_DECODER_AARCH64_H_
