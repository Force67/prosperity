Immediate Range Limits in VIXL
==============================

VIXL's macro assembler tries to increase the range of branches and literal loads
automatically for you, but applications must still be aware of these extended
limits, and stay within them, in order to ensure valid code is generated.

In debug builds, assertions prevent exceeding these limits at run time. In
release builds, for performance reasons, the application is responsible for
staying within the limits.

You should decide what corrections should be applied in your application if it
exceeds these limits.

Terms
-----

**Bind** assigning an address to a label such that the instructions that refer
to the label can be assigned PC-relative offsets.

**Forward** a forward branch or load literal will refer to a location that will
be bound later in code generation, ie. at a higher address.

**Backward** a backward branch or load literal refers to a location that has
already been bound earlier in code generation, ie. at a lower address.

**Instruction range** the range of values that can be encoded in the instruction
to be generated. Outside the instruction range, additional instructions may be
generated to increase the range, branching further than would be possible in
one instruction, for example.

**Veneer** a sequence of additional instructions produced to increase the
instruction range.

**Adjusted PC** the PC including its architecturally-defined offset. In AArch32
T32, this is the current PC plus four bytes. In AArch64, there is no adjustment;
Adjusted PC is equal to PC.

AArch64
-------

### Branches

All instructions and targets must be aligned to the instruction size, four
bytes.

#### Unconditional immediate branches (`B`)

* Unconditional immediate branches have an instruction range of -134,217,728 to
+134,217,724 bytes from the current PC.
* No veneers are applied to unconditional immediate branches to extend their
instruction range.
* Callers can use the function `IsValidImmPCOffset(UncondBranchType, offset)` to
check `offset` (in units of instruction) is within the instruction range.

#### Conditional branches (`B.cond`) and compare-and-branch (`CBZ`, `CBNZ`)

* Conditional branch and compare-and-branch instructions have the same
instruction range.
* The instruction range is -1,048,576 to +1,048,574 bytes from the current PC.
* Veneers are applied to extend the range to -134,217,724 to +135,266,298 bytes
from the current PC.
  * Unconditional branch range minus one instruction backwards.
  * Unconditional branch range plus conditional branch range forwards.
* Callers can use the functions `IsValidImmPCOffset(CondBranchType, offset)` and
`IsValidImmPCOffset(CompareBranchType, offset)` to check `offset` (in units of
instruction) is within the instruction range.

#### Test-and-branch (`TBZ`, `TBNZ`)

* Test-and-branch instructions have an instruction range of -32,768 to 32,764
bytes from the current PC.
* Veneers are applied to extend the range to -134,217,728 to +135,299,062 bytes
from the current PC.
  * Unconditional branch range minus one instruction backwards.
  * Unconditional branch range plus test-and-branch range forwards.
* Callers can use the function `IsValidImmPCOffset(TestBranchType, offset)` to
check `offset` (in units of instruction) is within the instruction range.

### Literals

#### Compute PC-relative address (`ADR`)

* Compute PC-relative address instructions have an instruction range of
-1,048,576 to +1,048,575 bytes from the current PC.
* No veneers are applied to extend the instruction range.
* Callers can use `IsInt21(offset)` to check `offset` (in bytes) is within the
instruction range.

#### Load from PC-relative address (`LDR`)

* Load from PC-relative address instructions have an instruction range of
-1,048,576 to +1,048,572 bytes from the current PC. The offset must be four-byte
aligned.
* Automatically-placed literals (eg. those created by `Ldr(reg, literal_value)`)
will be emitted into code such that they are in range of the instructions that
refer to them.
* Veneers are not applied to manually-placed literals, ie. those created by
`Literal<T> x(value)` and emitted by `place()`.
* Callers can use `IsInt19(offset)` to check `offset` (in units of instruction)
is within the instruction range.

AArch32
-------

Limits stated in this section relate to the T32 instruction encodings only.

### Branches

#### Unconditional immediate branches (`B`)

* Unconditional immediate branches have an instruction range of -16,777,216 to
+16,777,214 bytes from the current adjusted PC.
* Veneers are applied to forward branches to extend them to an unlimited range.
* No veneers are applied to backward branches.

#### Conditional immediate branches (`B`)

* Conditional immediate branches have an instruction range of -1,048,576 to
+1,048,574 bytes from the current adjusted PC.
* Veneers are applied to forward branches to extend them to an unlimited range.
* Veneers are applied to backward branches to extend the range to that of
unconditional immediate branches, -16,777,216 bytes from the current adjusted
PC.

#### Compare and branch (`CBZ`, `CBNZ`)

* Compare and branch has an instruction range of 0 to +126 bytes from the
current adjusted PC.
* Veneers are applied to forward branches to extend them to an unlimited range.
* Veneers are applied to backward branches to extend the range to that of
unconditional immediate branches, -16,777,216 bytes from the current adjusted
PC.

### Literals

#### Compute/load PC-relative address (`ADR`, `LDR`)

* Compute and load PC-relative address instructions have the same instruction
range.
* The instruction range is -4,095 to +4,095 bytes from the current adjusted PC.
The PC is aligned down to a four-byte boundary before the offset is added.
* Automatically-placed literals (ie. those created by `Literal<T> x(value)`)
will be emitted into code such that they are in range of the instructions that
refer to them.
* Veneers are not applied to manually-placed literals, ie. those created by
`Literal<T> x(value, RawLiteral::kManuallyPlaced)` and emitted by `Place()`.

