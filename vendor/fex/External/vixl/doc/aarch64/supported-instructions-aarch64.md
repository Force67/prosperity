VIXL Supported Instruction List
===============================

This is a list of the AArch64 instructions supported by the VIXL assembler,
disassembler and simulator. The simulator may not support all floating point
operations to the precision required by AArch64 - please check the simulator
source code for details.

#### AAch64 integer instructions ####
[a](#integer-a) [b](#integer-b) [c](#integer-c) [d](#integer-d) [e](#integer-e) [h](#integer-h) [i](#integer-i) [l](#integer-l) [m](#integer-m) [n](#integer-n) [o](#integer-o) [p](#integer-p) [r](#integer-r) [s](#integer-s) [t](#integer-t) [u](#integer-u) [x](#integer-x)

#### AArch64 floating point and NEON instructions ####
[a](#float-a) [b](#float-b) [c](#float-c) [d](#float-d) [e](#float-e) [f](#float-f) [i](#float-i) [l](#float-l) [m](#float-m) [n](#float-n) [o](#float-o) [p](#float-p) [r](#float-r) [s](#float-s) [t](#float-t) [u](#float-u) [x](#float-x) [z](#float-z)

#### AArch64 Scalable Vector Extension (SVE) instructions ####
[a](#sve-a) [b](#sve-b) [c](#sve-c) [d](#sve-d) [e](#sve-e) [f](#sve-f) [h](#sve-h) [i](#sve-i) [l](#sve-l) [m](#sve-m) [n](#sve-n) [o](#sve-o) [p](#sve-p) [r](#sve-r) [s](#sve-s) [t](#sve-t) [u](#sve-u) [w](#sve-w) [x](#sve-x) [z](#sve-z)

#### Additional or pseudo instructions ####
[b](#pseudo-b) [d](#pseudo-d) [p](#pseudo-p)

___

AArch64 integer instructions
----------------------------

<a id="integer-a">

### ADC ###

Add with carry bit.

    void adc(const Register& rd, const Register& rn, const Operand& operand)


### ADCS ###

Add with carry bit and update status flags.

    void adcs(const Register& rd, const Register& rn, const Operand& operand)


### ADD ###

Add.

    void add(const Register& rd, const Register& rn, const Operand& operand)


### ADDS ###

Add and update status flags.

    void adds(const Register& rd, const Register& rn, const Operand& operand)


### ADR ###

Calculate the address of a PC offset.

    void adr(const Register& xd, int64_t imm21)


### ADR ###

Calculate the address of a label.

    void adr(const Register& xd, Label* label)


### ADRP ###

Calculate the page address of a PC offset.

    void adrp(const Register& xd, int64_t imm21)


### ADRP ###

Calculate the page address of a label.

    void adrp(const Register& xd, Label* label)


### AND ###

Bitwise and (A & B).

    void and_(const Register& rd, const Register& rn, const Operand& operand)


### ANDS ###

Bitwise and (A & B) and update status flags.

    void ands(const Register& rd, const Register& rn, const Operand& operand)


### ASR ###

Arithmetic shift right.

    void asr(const Register& rd, const Register& rn, unsigned shift)


### ASRV ###

Arithmetic shift right by variable.

    void asrv(const Register& rd, const Register& rn, const Register& rm)


### AUTDA ###

Authenticate Data address, using key A _(Armv8.3)_.

    void autda(const Register& xd, const Register& xn)


### AUTDB ###

Authenticate Data address, using key B _(Armv8.3)_.

    void autdb(const Register& xd, const Register& xn)


### AUTDZA ###

Authenticate Data address, using key A and a modifier of zero _(Armv8.3)_.

    void autdza(const Register& xd)


### AUTDZB ###

Authenticate Data address, using key B and a modifier of zero _(Armv8.3)_.

    void autdzb(const Register& xd)


### AUTIA ###

Authenticate Instruction address, using key A _(Armv8.3)_.

    void autia(const Register& xd, const Register& xn)


### AUTIA1716 ###

Authenticate Instruction address, using key A, with address in x17 and modifier in x16 _(Armv8.3)_.

    void autia1716()


### AUTIASP ###

Authenticate Instruction address, using key A, with address in LR and modifier in SP _(Armv8.3)_.

    void autiasp()


### AUTIAZ ###

Authenticate Instruction address, using key A, with address in LR and a modifier of zero _(Armv8.3)_.

    void autiaz()


### AUTIB ###

Authenticate Instruction address, using key B _(Armv8.3)_.

    void autib(const Register& xd, const Register& xn)


### AUTIB1716 ###

Authenticate Instruction address, using key B, with address in x17 and modifier in x16 _(Armv8.3)_.

    void autib1716()


### AUTIBSP ###

Authenticate Instruction address, using key B, with address in LR and modifier in SP _(Armv8.3)_.

    void autibsp()


### AUTIBZ ###

Authenticate Instruction address, using key B, with address in LR and a modifier of zero _(Armv8.3)_.

    void autibz()


### AUTIZA ###

Authenticate Instruction address, using key A and a modifier of zero _(Armv8.3)_.

    void autiza(const Register& xd)


### AUTIZB ###

Authenticate Instruction address, using key B and a modifier of zero _(Armv8.3)_.

    void autizb(const Register& xd)


### AXFLAG ###

Convert floating-point condition flags from Arm format to alternative format _(Armv8.5)_.

    void axflag()


<a id="integer-b">

### B ###

Conditional branch to PC offset.

    void b(int64_t imm19, Condition cond)


### B ###

Conditional branch to label.

    void b(Label* label, Condition cond)


### B ###

Unconditional branch to PC offset.

    void b(int64_t imm26)


### B ###

Unconditional branch to label.

    void b(Label* label)


### BFC ###

Bitfield clear _(Armv8.2)_.

    void bfc(const Register& rd, unsigned lsb, unsigned width)


### BFI ###

Bitfield insert.

    void bfi(const Register& rd,
             const Register& rn,
             unsigned lsb,
             unsigned width)


### BFM ###

Bitfield move.

    void bfm(const Register& rd,
             const Register& rn,
             unsigned immr,
             unsigned imms)


### BFXIL ###

Bitfield extract and insert low.

    void bfxil(const Register& rd,
               const Register& rn,
               unsigned lsb,
               unsigned width)


### BIC ###

Bit clear (A & ~B).

    void bic(const Register& rd, const Register& rn, const Operand& operand)


### BICS ###

Bit clear (A & ~B) and update status flags.

    void bics(const Register& rd, const Register& rn, const Operand& operand)


### BL ###

Branch with link to PC offset.

    void bl(int64_t imm26)


### BL ###

Branch with link to label.

    void bl(Label* label)


### BLR ###

Branch with link to register.

    void blr(const Register& xn)


### BLRAA ###

Branch with link to register, with pointer authentication. Using key A _(Armv8.3)_.

    void blraa(const Register& xn, const Register& xm)


### BLRAAZ ###

Branch with link to register, with pointer authentication. Using key A and a modifier of zero _(Armv8.3)_.

    void blraaz(const Register& xn)


### BLRAB ###

Branch with link to register, with pointer authentication. Using key B _(Armv8.3)_.

    void blrab(const Register& xn, const Register& xm)


### BLRABZ ###

Branch with link to register, with pointer authentication. Using key B and a modifier of zero _(Armv8.3)_.

    void blrabz(const Register& xn)


### BR ###

Branch to register.

    void br(const Register& xn)


### BRAA ###

Branch to register, with pointer authentication. Using key A _(Armv8.3)_.

    void braa(const Register& xn, const Register& xm)


### BRAAZ ###

Branch to register, with pointer authentication. Using key A and a modifier of zero _(Armv8.3)_.

    void braaz(const Register& xn)


### BRAB ###

Branch to register, with pointer authentication. Using key B _(Armv8.3)_.

    void brab(const Register& xn, const Register& xm)


### BRABZ ###

Branch to register, with pointer authentication. Using key B and a modifier of zero _(Armv8.3)_.

    void brabz(const Register& xn)


### BRK ###

Monitor debug-mode breakpoint.

    void brk(int code)


### BTI ###

Branch target identification.

    void bti(BranchTargetIdentifier id)


<a id="integer-c">

### CAS ###

Compare and Swap word or doubleword in memory _(Armv8.1)_.

    void cas(const Register& rs, const Register& rt, const MemOperand& src)


### CASA ###

Compare and Swap word or doubleword in memory _(Armv8.1)_.

    void casa(const Register& rs, const Register& rt, const MemOperand& src)


### CASAB ###

Compare and Swap byte in memory _(Armv8.1)_.

    void casab(const Register& rs, const Register& rt, const MemOperand& src)


### CASAH ###

Compare and Swap halfword in memory _(Armv8.1)_.

    void casah(const Register& rs, const Register& rt, const MemOperand& src)


### CASAL ###

Compare and Swap word or doubleword in memory _(Armv8.1)_.

    void casal(const Register& rs, const Register& rt, const MemOperand& src)


### CASALB ###

Compare and Swap byte in memory _(Armv8.1)_.

    void casalb(const Register& rs, const Register& rt, const MemOperand& src)


### CASALH ###

Compare and Swap halfword in memory _(Armv8.1)_.

    void casalh(const Register& rs, const Register& rt, const MemOperand& src)


### CASB ###

Compare and Swap byte in memory _(Armv8.1)_.

    void casb(const Register& rs, const Register& rt, const MemOperand& src)


### CASH ###

Compare and Swap halfword in memory _(Armv8.1)_.

    void cash(const Register& rs, const Register& rt, const MemOperand& src)


### CASL ###

Compare and Swap word or doubleword in memory _(Armv8.1)_.

    void casl(const Register& rs, const Register& rt, const MemOperand& src)


### CASLB ###

Compare and Swap byte in memory _(Armv8.1)_.

    void caslb(const Register& rs, const Register& rt, const MemOperand& src)


### CASLH ###

Compare and Swap halfword in memory _(Armv8.1)_.

    void caslh(const Register& rs, const Register& rt, const MemOperand& src)


### CASP ###

Compare and Swap Pair of words or doublewords in memory _(Armv8.1)_.

    void casp(const Register& rs,
              const Register& rs2,
              const Register& rt,
              const Register& rt2,
              const MemOperand& src)


### CASPA ###

Compare and Swap Pair of words or doublewords in memory _(Armv8.1)_.

    void caspa(const Register& rs,
               const Register& rs2,
               const Register& rt,
               const Register& rt2,
               const MemOperand& src)


### CASPAL ###

Compare and Swap Pair of words or doublewords in memory _(Armv8.1)_.

    void caspal(const Register& rs,
                const Register& rs2,
                const Register& rt,
                const Register& rt2,
                const MemOperand& src)


### CASPL ###

Compare and Swap Pair of words or doublewords in memory _(Armv8.1)_.

    void caspl(const Register& rs,
               const Register& rs2,
               const Register& rt,
               const Register& rt2,
               const MemOperand& src)


### CBNZ ###

Compare and branch to PC offset if not zero.

    void cbnz(const Register& rt, int64_t imm19)


### CBNZ ###

Compare and branch to label if not zero.

    void cbnz(const Register& rt, Label* label)


### CBZ ###

Compare and branch to PC offset if zero.

    void cbz(const Register& rt, int64_t imm19)


### CBZ ###

Compare and branch to label if zero.

    void cbz(const Register& rt, Label* label)


### CCMN ###

Conditional compare negative.

    void ccmn(const Register& rn,
              const Operand& operand,
              StatusFlags nzcv,
              Condition cond)


### CCMP ###

Conditional compare.

    void ccmp(const Register& rn,
              const Operand& operand,
              StatusFlags nzcv,
              Condition cond)


### CFINV ###

Invert carry flag _(Armv8.4)_.

    void cfinv()


### CINC ###

Conditional increment: rd = cond ? rn + 1 : rn.

    void cinc(const Register& rd, const Register& rn, Condition cond)


### CINV ###

Conditional invert: rd = cond ? ~rn : rn.

    void cinv(const Register& rd, const Register& rn, Condition cond)


### CLREX ###

Clear exclusive monitor.

    void clrex(int imm4 = 0xf)


### CLS ###

Count leading sign bits.

    void cls(const Register& rd, const Register& rn)


### CLZ ###

Count leading zeroes.

    void clz(const Register& rd, const Register& rn)


### CMN ###

Compare negative.

    void cmn(const Register& rn, const Operand& operand)


### CMP ###

Compare.

    void cmp(const Register& rn, const Operand& operand)


### CNEG ###

Conditional negate: rd = cond ? -rn : rn.

    void cneg(const Register& rd, const Register& rn, Condition cond)


### CRC32B ###

CRC-32 checksum from byte.

    void crc32b(const Register& wd, const Register& wn, const Register& wm)


### CRC32CB ###

CRC-32 C checksum from byte.

    void crc32cb(const Register& wd, const Register& wn, const Register& wm)


### CRC32CH ###

CRC-32 C checksum from half-word.

    void crc32ch(const Register& wd, const Register& wn, const Register& wm)


### CRC32CW ###

CRC-32 C checksum from word.

    void crc32cw(const Register& wd, const Register& wn, const Register& wm)


### CRC32CX ###

CRC-32C checksum from double word.

    void crc32cx(const Register& wd, const Register& wn, const Register& xm)


### CRC32H ###

CRC-32 checksum from half-word.

    void crc32h(const Register& wd, const Register& wn, const Register& wm)


### CRC32W ###

CRC-32 checksum from word.

    void crc32w(const Register& wd, const Register& wn, const Register& wm)


### CRC32X ###

CRC-32 checksum from double word.

    void crc32x(const Register& wd, const Register& wn, const Register& xm)


### CSDB ###

Conditional speculation dependency barrier.

    void csdb()


### CSEL ###

Conditional select: rd = cond ? rn : rm.

    void csel(const Register& rd,
              const Register& rn,
              const Register& rm,
              Condition cond)


### CSET ###

Conditional set: rd = cond ? 1 : 0.

    void cset(const Register& rd, Condition cond)


### CSETM ###

Conditional set mask: rd = cond ? -1 : 0.

    void csetm(const Register& rd, Condition cond)


### CSINC ###

Conditional select increment: rd = cond ? rn : rm + 1.

    void csinc(const Register& rd,
               const Register& rn,
               const Register& rm,
               Condition cond)


### CSINV ###

Conditional select inversion: rd = cond ? rn : ~rm.

    void csinv(const Register& rd,
               const Register& rn,
               const Register& rm,
               Condition cond)


### CSNEG ###

Conditional select negation: rd = cond ? rn : -rm.

    void csneg(const Register& rd,
               const Register& rn,
               const Register& rm,
               Condition cond)


<a id="integer-d">

### DC ###

System data cache operation.

    void dc(DataCacheOp op, const Register& rt)


### DMB ###

Data memory barrier.

    void dmb(BarrierDomain domain, BarrierType type)


### DSB ###

Data synchronization barrier.

    void dsb(BarrierDomain domain, BarrierType type)


<a id="integer-e">

### EON ###

Bitwise enor/xnor (A ^ ~B).

    void eon(const Register& rd, const Register& rn, const Operand& operand)


### EOR ###

Bitwise eor/xor (A ^ B).

    void eor(const Register& rd, const Register& rn, const Operand& operand)


### ESB ###

Error synchronization barrier.

    void esb()


### EXTR ###

Extract.

    void extr(const Register& rd,
              const Register& rn,
              const Register& rm,
              unsigned lsb)


<a id="integer-h">

### HINT ###

System hint (named type).

    void hint(SystemHint code)


### HINT ###

System hint (numbered type).

    void hint(int imm7)


### HLT ###

Halting debug-mode breakpoint.

    void hlt(int code)


<a id="integer-i">

### IC ###

System instruction cache operation.

    void ic(InstructionCacheOp op, const Register& rt)


### ISB ###

Instruction synchronization barrier.

    void isb()


<a id="integer-l">

### LDADD ###

Atomic add on word or doubleword in memory _(Armv8.1)_

    void ldadd(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDA ###

Atomic add on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldadda(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDAB ###

Atomic add on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void ldaddab(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDAH ###

Atomic add on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldaddah(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDAL ###

Atomic add on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldaddal(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDALB ###

Atomic add on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldaddalb(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDALH ###

Atomic add on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldaddalh(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDB ###

Atomic add on byte in memory _(Armv8.1)_

    void ldaddb(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDH ###

Atomic add on halfword in memory _(Armv8.1)_

    void ldaddh(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDL ###

Atomic add on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void ldaddl(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDLB ###

Atomic add on byte in memory, with Store-release semantics _(Armv8.1)_

    void ldaddlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDADDLH ###

Atomic add on halfword in memory, with Store-release semantics _(Armv8.1)_

    void ldaddlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDAPR ###

Load-Acquire RCpc Register word or doubleword _(Armv8.3)_

    void ldapr(const Register& rt, const MemOperand& src)


### LDAPRB ###

Load-Acquire RCpc Register byte _(Armv8.3)_

    void ldaprb(const Register& rt, const MemOperand& src)


### LDAPRH ###

Load-Acquire RCpc Register halfword _(Armv8.3)_

    void ldaprh(const Register& rt, const MemOperand& src)


### LDAPUR ###

Load-acquire RCpc Register word or double-word (with unscaled offset) _(Armv8.4)_.

    void ldapur(const Register& rt, const MemOperand& src)


### LDAPURB ###

Load-acquire RCpc Register byte (with unscaled offset) _(Armv8.4)_.

    void ldapurb(const Register& rt, const MemOperand& src)


### LDAPURH ###

Load-acquire RCpc Register half-word (with unscaled offset) _(Armv8.4)_.

    void ldapurh(const Register& rt, const MemOperand& src)


### LDAPURSB ###

Load-acquire RCpc Register signed byte (with unscaled offset) _(Armv8.4)_.

    void ldapursb(const Register& rt, const MemOperand& src)


### LDAPURSH ###

Load-acquire RCpc Register signed half-word (with unscaled offset) _(Armv8.4)_.

    void ldapursh(const Register& rt, const MemOperand& src)


### LDAPURSW ###

Load-acquire RCpc Register signed word (with unscaled offset) _(Armv8.4)_.

    void ldapursw(const Register& xt, const MemOperand& src)


### LDAR ###

Load-acquire register.

    void ldar(const Register& rt, const MemOperand& src)


### LDARB ###

Load-acquire byte.

    void ldarb(const Register& rt, const MemOperand& src)


### LDARH ###

Load-acquire half-word.

    void ldarh(const Register& rt, const MemOperand& src)


### LDAXP ###

Load-acquire exclusive register pair.

    void ldaxp(const Register& rt, const Register& rt2, const MemOperand& src)


### LDAXR ###

Load-acquire exclusive register.

    void ldaxr(const Register& rt, const MemOperand& src)


### LDAXRB ###

Load-acquire exclusive byte.

    void ldaxrb(const Register& rt, const MemOperand& src)


### LDAXRH ###

Load-acquire exclusive half-word.

    void ldaxrh(const Register& rt, const MemOperand& src)


### LDCLR ###

Atomic bit clear on word or doubleword in memory _(Armv8.1)_

    void ldclr(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRA ###

Atomic bit clear on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldclra(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRAB ###

Atomic bit clear on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void ldclrab(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRAH ###

Atomic bit clear on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldclrah(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRAL ###

Atomic bit clear on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldclral(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRALB ###

Atomic bit clear on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldclralb(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRALH ###

Atomic bit clear on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldclralh(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRB ###

Atomic bit clear on byte in memory _(Armv8.1)_

    void ldclrb(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRH ###

Atomic bit clear on halfword in memory _(Armv8.1)_

    void ldclrh(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRL ###

Atomic bit clear on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void ldclrl(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRLB ###

Atomic bit clear on byte in memory, with Store-release semantics _(Armv8.1)_

    void ldclrlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDCLRLH ###

Atomic bit clear on halfword in memory, with Store-release semantics _(Armv8.1)_

    void ldclrlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDEOR ###

Atomic exclusive OR on word or doubleword in memory _(Armv8.1)_

    void ldeor(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORA ###

Atomic exclusive OR on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldeora(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORAB ###

Atomic exclusive OR on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void ldeorab(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORAH ###

Atomic exclusive OR on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldeorah(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORAL ###

Atomic exclusive OR on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldeoral(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORALB ###

Atomic exclusive OR on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldeoralb(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORALH ###

Atomic exclusive OR on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldeoralh(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORB ###

Atomic exclusive OR on byte in memory _(Armv8.1)_

    void ldeorb(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORH ###

Atomic exclusive OR on halfword in memory _(Armv8.1)_

    void ldeorh(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORL ###

Atomic exclusive OR on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void ldeorl(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORLB ###

Atomic exclusive OR on byte in memory, with Store-release semantics _(Armv8.1)_

    void ldeorlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDEORLH ###

Atomic exclusive OR on halfword in memory, with Store-release semantics _(Armv8.1)_

    void ldeorlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDLAR ###

Load LORelease register _(Armv8.1)_.

    void ldlar(const Register& rt, const MemOperand& src)


### LDLARB ###

Load LORelease byte _(Armv8.1)_.

    void ldlarb(const Register& rt, const MemOperand& src)


### LDLARH ###

Load LORelease half-word _(Armv8.1)_.

    void ldlarh(const Register& rt, const MemOperand& src)


### LDNP ###

Load integer or FP register pair, non-temporal.

    void ldnp(const CPURegister& rt,
              const CPURegister& rt2,
              const MemOperand& src)


### LDP ###

Load integer or FP register pair.

    void ldp(const CPURegister& rt,
             const CPURegister& rt2,
             const MemOperand& src)


### LDPSW ###

Load word pair with sign extension.

    void ldpsw(const Register& xt, const Register& xt2, const MemOperand& src)


### LDR ###

Load integer or FP register from literal pool.

    void ldr(const CPURegister& rt, RawLiteral* literal)


### LDR ###

Load integer or FP register from pc + imm19 << 2.

    void ldr(const CPURegister& rt, int64_t imm19)


### LDR ###

Load integer or FP register.

    void ldr(const CPURegister& rt,
             const MemOperand& src,
             LoadStoreScalingOption option = PreferScaledOffset)


### LDRAA ###

Load double-word with pointer authentication, using data key A and a modifier of zero _(Armv8.3)_.

    void ldraa(const Register& xt, const MemOperand& src)


### LDRAB ###

Load double-word with pointer authentication, using data key B and a modifier of zero _(Armv8.3)_.

    void ldrab(const Register& xt, const MemOperand& src)


### LDRB ###

Load byte.

    void ldrb(const Register& rt,
              const MemOperand& src,
              LoadStoreScalingOption option = PreferScaledOffset)


### LDRH ###

Load half-word.

    void ldrh(const Register& rt,
              const MemOperand& src,
              LoadStoreScalingOption option = PreferScaledOffset)


### LDRSB ###

Load byte with sign extension.

    void ldrsb(const Register& rt,
               const MemOperand& src,
               LoadStoreScalingOption option = PreferScaledOffset)


### LDRSH ###

Load half-word with sign extension.

    void ldrsh(const Register& rt,
               const MemOperand& src,
               LoadStoreScalingOption option = PreferScaledOffset)


### LDRSW ###

Load word with sign extension from literal pool.

    void ldrsw(const Register& xt, RawLiteral* literal)


### LDRSW ###

Load word with sign extension from pc + imm19 << 2.

    void ldrsw(const Register& xt, int64_t imm19)


### LDRSW ###

Load word with sign extension.

    void ldrsw(const Register& xt,
               const MemOperand& src,
               LoadStoreScalingOption option = PreferScaledOffset)


### LDSET ###

Atomic bit set on word or doubleword in memory _(Armv8.1)_

    void ldset(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETA ###

Atomic bit set on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldseta(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETAB ###

Atomic bit set on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsetab(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETAH ###

Atomic bit set on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsetah(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETAL ###

Atomic bit set on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsetal(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETALB ###

Atomic bit set on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsetalb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETALH ###

Atomic bit set on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsetalh(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETB ###

Atomic bit set on byte in memory _(Armv8.1)_

    void ldsetb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETH ###

Atomic bit set on halfword in memory _(Armv8.1)_

    void ldseth(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETL ###

Atomic bit set on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void ldsetl(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETLB ###

Atomic bit set on byte in memory, with Store-release semantics _(Armv8.1)_

    void ldsetlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSETLH ###

Atomic bit set on halfword in memory, with Store-release semantics _(Armv8.1)_

    void ldsetlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAX ###

Atomic signed maximum on word or doubleword in memory _(Armv8.1)_

    void ldsmax(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXA ###

Atomic signed maximum on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsmaxa(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXAB ###

Atomic signed maximum on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsmaxab(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXAH ###

Atomic signed maximum on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsmaxah(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXAL ###

Atomic signed maximum on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsmaxal(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXALB ###

Atomic signed maximum on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsmaxalb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXALH ###

Atomic signed maximum on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsmaxalh(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXB ###

Atomic signed maximum on byte in memory _(Armv8.1)_

    void ldsmaxb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXH ###

Atomic signed maximum on halfword in memory _(Armv8.1)_

    void ldsmaxh(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXL ###

Atomic signed maximum on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void ldsmaxl(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXLB ###

Atomic signed maximum on byte in memory, with Store-release semantics _(Armv8.1)_

    void ldsmaxlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMAXLH ###

Atomic signed maximum on halfword in memory, with Store-release semantics _(Armv8.1)_

    void ldsmaxlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMIN ###

Atomic signed minimum on word or doubleword in memory _(Armv8.1)_

    void ldsmin(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINA ###

Atomic signed minimum on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsmina(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINAB ###

Atomic signed minimum on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsminab(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINAH ###

Atomic signed minimum on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldsminah(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINAL ###

Atomic signed minimum on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsminal(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINALB ###

Atomic signed minimum on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsminalb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINALH ###

Atomic signed minimum on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldsminalh(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINB ###

Atomic signed minimum on byte in memory _(Armv8.1)_

    void ldsminb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINH ###

Atomic signed minimum on halfword in memory _(Armv8.1)_

    void ldsminh(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINL ###

Atomic signed minimum on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void ldsminl(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINLB ###

Atomic signed minimum on byte in memory, with Store-release semantics _(Armv8.1)_

    void ldsminlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDSMINLH ###

Atomic signed minimum on halfword in memory, with Store-release semantics _(Armv8.1)_

    void ldsminlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAX ###

Atomic unsigned maximum on word or doubleword in memory _(Armv8.1)_

    void ldumax(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXA ###

Atomic unsigned maximum on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldumaxa(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXAB ###

Atomic unsigned maximum on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void ldumaxab(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXAH ###

Atomic unsigned maximum on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldumaxah(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXAL ###

Atomic unsigned maximum on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldumaxal(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXALB ###

Atomic unsigned maximum on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldumaxalb(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXALH ###

Atomic unsigned maximum on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void ldumaxalh(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXB ###

Atomic unsigned maximum on byte in memory _(Armv8.1)_

    void ldumaxb(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXH ###

Atomic unsigned maximum on halfword in memory _(Armv8.1)_

    void ldumaxh(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXL ###

Atomic unsigned maximum on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void ldumaxl(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXLB ###

Atomic unsigned maximum on byte in memory, with Store-release semantics _(Armv8.1)_

    void ldumaxlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMAXLH ###

Atomic unsigned maximum on halfword in memory, with Store-release semantics _(Armv8.1)_

    void ldumaxlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMIN ###

Atomic unsigned minimum on word or doubleword in memory _(Armv8.1)_

    void ldumin(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINA ###

Atomic unsigned minimum on word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void ldumina(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINAB ###

Atomic unsigned minimum on byte in memory, with Load-acquire semantics _(Armv8.1)_

    void lduminab(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINAH ###

Atomic unsigned minimum on halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void lduminah(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINAL ###

Atomic unsigned minimum on word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void lduminal(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINALB ###

Atomic unsigned minimum on byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void lduminalb(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINALH ###

Atomic unsigned minimum on halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void lduminalh(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINB ###

Atomic unsigned minimum on byte in memory _(Armv8.1)_

    void lduminb(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINH ###

Atomic unsigned minimum on halfword in memory _(Armv8.1)_

    void lduminh(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINL ###

Atomic unsigned minimum on word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void lduminl(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINLB ###

Atomic unsigned minimum on byte in memory, with Store-release semantics _(Armv8.1)_

    void lduminlb(const Register& rs, const Register& rt, const MemOperand& src)


### LDUMINLH ###

Atomic unsigned minimum on halfword in memory, with Store-release semantics _(Armv8.1)_

    void lduminlh(const Register& rs, const Register& rt, const MemOperand& src)


### LDUR ###

Load integer or FP register (with unscaled offset).

    void ldur(const CPURegister& rt,
              const MemOperand& src,
              LoadStoreScalingOption option = PreferUnscaledOffset)


### LDURB ###

Load byte (with unscaled offset).

    void ldurb(const Register& rt,
               const MemOperand& src,
               LoadStoreScalingOption option = PreferUnscaledOffset)


### LDURH ###

Load half-word (with unscaled offset).

    void ldurh(const Register& rt,
               const MemOperand& src,
               LoadStoreScalingOption option = PreferUnscaledOffset)


### LDURSB ###

Load byte with sign extension (and unscaled offset).

    void ldursb(const Register& rt,
                const MemOperand& src,
                LoadStoreScalingOption option = PreferUnscaledOffset)


### LDURSH ###

Load half-word with sign extension (and unscaled offset).

    void ldursh(const Register& rt,
                const MemOperand& src,
                LoadStoreScalingOption option = PreferUnscaledOffset)


### LDURSW ###

Load word with sign extension.

    void ldursw(const Register& xt,
                const MemOperand& src,
                LoadStoreScalingOption option = PreferUnscaledOffset)


### LDXP ###

Load exclusive register pair.

    void ldxp(const Register& rt, const Register& rt2, const MemOperand& src)


### LDXR ###

Load exclusive register.

    void ldxr(const Register& rt, const MemOperand& src)


### LDXRB ###

Load exclusive byte.

    void ldxrb(const Register& rt, const MemOperand& src)


### LDXRH ###

Load exclusive half-word.

    void ldxrh(const Register& rt, const MemOperand& src)


### LSL ###

Logical shift left.

    void lsl(const Register& rd, const Register& rn, unsigned shift)


### LSLV ###

Logical shift left by variable.

    void lslv(const Register& rd, const Register& rn, const Register& rm)


### LSR ###

Logical shift right.

    void lsr(const Register& rd, const Register& rn, unsigned shift)


### LSRV ###

Logical shift right by variable.

    void lsrv(const Register& rd, const Register& rn, const Register& rm)


<a id="integer-m">

### MADD ###

Multiply and accumulate.

    void madd(const Register& rd,
              const Register& rn,
              const Register& rm,
              const Register& ra)


### MNEG ###

Negated multiply.

    void mneg(const Register& rd, const Register& rn, const Register& rm)


### MOV ###

Move immediate, aliases for movz, movn, orr.

    void mov(const Register& rd, uint64_t imm)


### MOV ###

Move register to register.

    void mov(const Register& rd, const Register& rn)


### MOVK ###

Move immediate and keep.

    void movk(const Register& rd, uint64_t imm, int shift = -1)


### MOVN ###

Move inverted immediate.

    void movn(const Register& rd, uint64_t imm, int shift = -1)


### MOVZ ###

Move immediate.

    void movz(const Register& rd, uint64_t imm, int shift = -1)


### MRS ###

Move to register from system register.

    void mrs(const Register& xt, SystemRegister sysreg)


### MSR ###

Move from register to system register.

    void msr(SystemRegister sysreg, const Register& xt)


### MSUB ###

Multiply and subtract.

    void msub(const Register& rd,
              const Register& rn,
              const Register& rm,
              const Register& ra)


### MUL ###

Multiply.

    void mul(const Register& rd, const Register& rn, const Register& rm)


### MVN ###

Move inverted operand to register.

    void mvn(const Register& rd, const Operand& operand)


<a id="integer-n">

### NEG ###

Negate.

    void neg(const Register& rd, const Operand& operand)


### NEGS ###

Negate and update status flags.

    void negs(const Register& rd, const Operand& operand)


### NGC ###

Negate with carry bit.

    void ngc(const Register& rd, const Operand& operand)


### NGCS ###

Negate with carry bit and update status flags.

    void ngcs(const Register& rd, const Operand& operand)


### NOP ###

No-op.

    void nop()


<a id="integer-o">

### ORN ###

Bitwise nor (A | ~B).

    void orn(const Register& rd, const Register& rn, const Operand& operand)


### ORR ###

Bitwise or (A | B).

    void orr(const Register& rd, const Register& rn, const Operand& operand)


<a id="integer-p">

### PACDA ###

Pointer Authentication Code for Data address, using key A _(Armv8.3)_.

    void pacda(const Register& xd, const Register& xn)


### PACDB ###

Pointer Authentication Code for Data address, using key B _(Armv8.3)_.

    void pacdb(const Register& xd, const Register& xn)


### PACDZA ###

Pointer Authentication Code for Data address, using key A and a modifier of zero _(Armv8.3)_.

    void pacdza(const Register& xd)


### PACDZB ###

Pointer Authentication Code for Data address, using key B and a modifier of zero _(Armv8.3)_.

    void pacdzb(const Register& xd)


### PACGA ###

Pointer Authentication Code, using Generic key _(Armv8.3)_.

    void pacga(const Register& xd, const Register& xn, const Register& xm)


### PACIA ###

Pointer Authentication Code for Instruction address, using key A _(Armv8.3)_.

    void pacia(const Register& xd, const Register& rn)


### PACIA1716 ###

Pointer Authentication Code for Instruction address, using key A, with address in x17 and modifier in x16 _(Armv8.3)_.

    void pacia1716()


### PACIASP ###

Pointer Authentication Code for Instruction address, using key A, with address in LR and modifier in SP _(Armv8.3)_.

    void paciasp()


### PACIAZ ###

Pointer Authentication Code for Instruction address, using key A, with address in LR and a modifier of zero _(Armv8.3)_.

    void paciaz()


### PACIB ###

Pointer Authentication Code for Instruction address, using key B _(Armv8.3)_.

    void pacib(const Register& xd, const Register& xn)


### PACIB1716 ###

Pointer Authentication Code for Instruction address, using key B, with address in x17 and modifier in x16 _(Armv8.3)_.

    void pacib1716()


### PACIBSP ###

Pointer Authentication Code for Instruction address, using key B, with address in LR and modifier in SP _(Armv8.3)_.

    void pacibsp()


### PACIBZ ###

Pointer Authentication Code for Instruction address, using key B, with address in LR and a modifier of zero _(Armv8.3)_.

    void pacibz()


### PACIZA ###

Pointer Authentication Code for Instruction address, using key A and a modifier of zero _(Armv8.3)_.

    void paciza(const Register& xd)


### PACIZB ###

Pointer Authentication Code for Instruction address, using key B and a modifier of zero _(Armv8.3)_.

    void pacizb(const Register& xd)


### PRFM ###

Prefetch from pc + imm19 << 2 (allowing unallocated hints).

    void prfm(int op, int64_t imm19)


### PRFM ###

Prefetch from pc + imm19 << 2.

    void prfm(PrefetchOperation op, int64_t imm19)


### PRFM ###

Prefetch memory (allowing unallocated hints).

    void prfm(int op,
              const MemOperand& addr,
              LoadStoreScalingOption option = PreferScaledOffset)


### PRFM ###

Prefetch memory in the literal pool (allowing unallocated hints).

    void prfm(int op, RawLiteral* literal)


### PRFM ###

Prefetch memory in the literal pool.

    void prfm(PrefetchOperation op, RawLiteral* literal)


### PRFM ###

Prefetch memory.

    void prfm(PrefetchOperation op,
              const MemOperand& addr,
              LoadStoreScalingOption option = PreferScaledOffset)


### PRFUM ###

Prefetch memory (with unscaled offset).

    void prfum(PrefetchOperation op,
               const MemOperand& addr,
               LoadStoreScalingOption option = PreferUnscaledOffset)


### PRFUM ###

Prefetch memory (with unscaled offset, allowing unallocated hints).

    void prfum(int op,
               const MemOperand& addr,
               LoadStoreScalingOption option = PreferUnscaledOffset)


<a id="integer-r">

### RBIT ###

Bit reverse.

    void rbit(const Register& rd, const Register& rn)


### RET ###

Branch to register with return hint.

    void ret(const Register& xn = lr)


### RETAA ###

Return from subroutine, with pointer authentication. Using key A _(Armv8.3)_.

    void retaa()


### RETAB ###

Return from subroutine, with pointer authentication. Using key B _(Armv8.3)_.

    void retab()


### REV ###

Reverse bytes.

    void rev(const Register& rd, const Register& rn)


### REV16 ###

Reverse bytes in 16-bit half words.

    void rev16(const Register& rd, const Register& rn)


### REV32 ###

Reverse bytes in 32-bit words.

    void rev32(const Register& xd, const Register& xn)


### REV64 ###

Reverse bytes in 64-bit general purpose register, an alias for rev _(Armv8.2)_.

    void rev64(const Register& xd, const Register& xn)


### RMIF ###

Rotate register right and insert into NZCV flags under the control of a mask _(Armv8.4)_.

    void rmif(const Register& xn, unsigned rotation, StatusFlags flags)


### ROR ###

Rotate right.

    void ror(const Register& rd, const Register& rs, unsigned shift)


### RORV ###

Rotate right by variable.

    void rorv(const Register& rd, const Register& rn, const Register& rm)


<a id="integer-s">

### SBC ###

Subtract with carry bit.

    void sbc(const Register& rd, const Register& rn, const Operand& operand)


### SBCS ###

Subtract with carry bit and update status flags.

    void sbcs(const Register& rd, const Register& rn, const Operand& operand)


### SBFIZ ###

Signed bitfield insert with zero at right.

    void sbfiz(const Register& rd,
               const Register& rn,
               unsigned lsb,
               unsigned width)


### SBFM ###

Signed bitfield move.

    void sbfm(const Register& rd,
              const Register& rn,
              unsigned immr,
              unsigned imms)


### SBFX ###

Signed bitfield extract.

    void sbfx(const Register& rd,
              const Register& rn,
              unsigned lsb,
              unsigned width)


### SDIV ###

Signed integer divide.

    void sdiv(const Register& rd, const Register& rn, const Register& rm)


### SETF16 ###

Set NZCV flags from register, treated as an 16-bit value _(Armv8.4)_.

    void setf16(const Register& rn)


### SETF8 ###

Set NZCV flags from register, treated as an 8-bit value _(Armv8.4)_.

    void setf8(const Register& rn)


### SMADDL ###

Signed long multiply and accumulate: 32 x 32 + 64 -> 64-bit.

    void smaddl(const Register& xd,
                const Register& wn,
                const Register& wm,
                const Register& xa)


### SMSUBL ###

Signed long multiply and subtract: 64 - (32 x 32) -> 64-bit.

    void smsubl(const Register& xd,
                const Register& wn,
                const Register& wm,
                const Register& xa)


### SMULH ###

Signed multiply high: 64 x 64 -> 64-bit <127:64>.

    void smulh(const Register& xd, const Register& xn, const Register& xm)


### SMULL ###

Signed long multiply: 32 x 32 -> 64-bit.

    void smull(const Register& xd, const Register& wn, const Register& wm)


### STADD ###

Atomic add on word or doubleword in memory, without return. _(Armv8.1)_

    void stadd(const Register& rs, const MemOperand& src)


### STADDB ###

Atomic add on byte in memory, without return. _(Armv8.1)_

    void staddb(const Register& rs, const MemOperand& src)


### STADDH ###

Atomic add on halfword in memory, without return. _(Armv8.1)_

    void staddh(const Register& rs, const MemOperand& src)


### STADDL ###

Atomic add on word or doubleword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void staddl(const Register& rs, const MemOperand& src)


### STADDLB ###

Atomic add on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void staddlb(const Register& rs, const MemOperand& src)


### STADDLH ###

Atomic add on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void staddlh(const Register& rs, const MemOperand& src)


### STCLR ###

Atomic bit clear on word or doubleword in memory, without return. _(Armv8.1)_

    void stclr(const Register& rs, const MemOperand& src)


### STCLRB ###

Atomic bit clear on byte in memory, without return. _(Armv8.1)_

    void stclrb(const Register& rs, const MemOperand& src)


### STCLRH ###

Atomic bit clear on halfword in memory, without return. _(Armv8.1)_

    void stclrh(const Register& rs, const MemOperand& src)


### STCLRL ###

Atomic bit clear on word or doubleword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stclrl(const Register& rs, const MemOperand& src)


### STCLRLB ###

Atomic bit clear on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stclrlb(const Register& rs, const MemOperand& src)


### STCLRLH ###

Atomic bit clear on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stclrlh(const Register& rs, const MemOperand& src)


### STEOR ###

Atomic exclusive OR on word or doubleword in memory, without return. _(Armv8.1)_

    void steor(const Register& rs, const MemOperand& src)


### STEORB ###

Atomic exclusive OR on byte in memory, without return. _(Armv8.1)_

    void steorb(const Register& rs, const MemOperand& src)


### STEORH ###

Atomic exclusive OR on halfword in memory, without return. _(Armv8.1)_

    void steorh(const Register& rs, const MemOperand& src)


### STEORL ###

Atomic exclusive OR on word or doubleword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void steorl(const Register& rs, const MemOperand& src)


### STEORLB ###

Atomic exclusive OR on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void steorlb(const Register& rs, const MemOperand& src)


### STEORLH ###

Atomic exclusive OR on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void steorlh(const Register& rs, const MemOperand& src)


### STLLR ###

Store LORelease register _(Armv8.1)_.

    void stllr(const Register& rt, const MemOperand& dst)


### STLLRB ###

Store LORelease byte _(Armv8.1)_.

    void stllrb(const Register& rt, const MemOperand& dst)


### STLLRH ###

Store LORelease half-word _(Armv8.1)_.

    void stllrh(const Register& rt, const MemOperand& dst)


### STLR ###

Store-release register.

    void stlr(const Register& rt, const MemOperand& dst)


### STLRB ###

Store-release byte.

    void stlrb(const Register& rt, const MemOperand& dst)


### STLRH ###

Store-release half-word.

    void stlrh(const Register& rt, const MemOperand& dst)


### STLUR ###

Store-release word or double-word (with unscaled offset) _(Armv8.4)_.

    void stlur(const Register& rt, const MemOperand& dst)


### STLURB ###

Store-release byte (with unscaled offset) _(Armv8.4)_.

    void stlurb(const Register& rt, const MemOperand& dst)


### STLURH ###

Store-release half-word (with unscaled offset) _(Armv8.4)_.

    void stlurh(const Register& rt, const MemOperand& dst)


### STLXP ###

Store-release exclusive register pair.

    void stlxp(const Register& rs,
               const Register& rt,
               const Register& rt2,
               const MemOperand& dst)


### STLXR ###

Store-release exclusive register.

    void stlxr(const Register& rs, const Register& rt, const MemOperand& dst)


### STLXRB ###

Store-release exclusive byte.

    void stlxrb(const Register& rs, const Register& rt, const MemOperand& dst)


### STLXRH ###

Store-release exclusive half-word.

    void stlxrh(const Register& rs, const Register& rt, const MemOperand& dst)


### STNP ###

Store integer or FP register pair, non-temporal.

    void stnp(const CPURegister& rt,
              const CPURegister& rt2,
              const MemOperand& dst)


### STP ###

Store integer or FP register pair.

    void stp(const CPURegister& rt,
             const CPURegister& rt2,
             const MemOperand& dst)


### STR ###

Store integer or FP register.

    void str(const CPURegister& rt,
             const MemOperand& dst,
             LoadStoreScalingOption option = PreferScaledOffset)


### STRB ###

Store byte.

    void strb(const Register& rt,
              const MemOperand& dst,
              LoadStoreScalingOption option = PreferScaledOffset)


### STRH ###

Store half-word.

    void strh(const Register& rt,
              const MemOperand& dst,
              LoadStoreScalingOption option = PreferScaledOffset)


### STSET ###

Atomic bit set on word or doubleword in memory, without return. _(Armv8.1)_

    void stset(const Register& rs, const MemOperand& src)


### STSETB ###

Atomic bit set on byte in memory, without return. _(Armv8.1)_

    void stsetb(const Register& rs, const MemOperand& src)


### STSETH ###

Atomic bit set on halfword in memory, without return. _(Armv8.1)_

    void stseth(const Register& rs, const MemOperand& src)


### STSETL ###

Atomic bit set on word or doubleword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsetl(const Register& rs, const MemOperand& src)


### STSETLB ###

Atomic bit set on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsetlb(const Register& rs, const MemOperand& src)


### STSETLH ###

Atomic bit set on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsetlh(const Register& rs, const MemOperand& src)


### STSMAX ###

Atomic signed maximum on word or doubleword in memory, without return. _(Armv8.1)_

    void stsmax(const Register& rs, const MemOperand& src)


### STSMAXB ###

Atomic signed maximum on byte in memory, without return. _(Armv8.1)_

    void stsmaxb(const Register& rs, const MemOperand& src)


### STSMAXH ###

Atomic signed maximum on halfword in memory, without return. _(Armv8.1)_

    void stsmaxh(const Register& rs, const MemOperand& src)


### STSMAXL ###

Atomic signed maximum on word or doubleword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsmaxl(const Register& rs, const MemOperand& src)


### STSMAXLB ###

Atomic signed maximum on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsmaxlb(const Register& rs, const MemOperand& src)


### STSMAXLH ###

Atomic signed maximum on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsmaxlh(const Register& rs, const MemOperand& src)


### STSMIN ###

Atomic signed minimum on word or doubleword in memory, without return. _(Armv8.1)_

    void stsmin(const Register& rs, const MemOperand& src)


### STSMINB ###

Atomic signed minimum on byte in memory, without return. _(Armv8.1)_

    void stsminb(const Register& rs, const MemOperand& src)


### STSMINH ###

Atomic signed minimum on halfword in memory, without return. _(Armv8.1)_

    void stsminh(const Register& rs, const MemOperand& src)


### STSMINL ###

Atomic signed minimum on word or doubleword in memory, with Store-release semantics and without return. semantics _(Armv8.1)_

    void stsminl(const Register& rs, const MemOperand& src)


### STSMINLB ###

Atomic signed minimum on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsminlb(const Register& rs, const MemOperand& src)


### STSMINLH ###

Atomic signed minimum on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stsminlh(const Register& rs, const MemOperand& src)


### STUMAX ###

Atomic unsigned maximum on word or doubleword in memory, without return. _(Armv8.1)_

    void stumax(const Register& rs, const MemOperand& src)


### STUMAXB ###

Atomic unsigned maximum on byte in memory, without return. _(Armv8.1)_

    void stumaxb(const Register& rs, const MemOperand& src)


### STUMAXH ###

Atomic unsigned maximum on halfword in memory, without return. _(Armv8.1)_

    void stumaxh(const Register& rs, const MemOperand& src)


### STUMAXL ###

Atomic unsigned maximum on word or doubleword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stumaxl(const Register& rs, const MemOperand& src)


### STUMAXLB ###

Atomic unsigned maximum on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stumaxlb(const Register& rs, const MemOperand& src)


### STUMAXLH ###

Atomic unsigned maximum on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stumaxlh(const Register& rs, const MemOperand& src)


### STUMIN ###

Atomic unsigned minimum on word or doubleword in memory, without return. _(Armv8.1)_

    void stumin(const Register& rs, const MemOperand& src)


### STUMINB ###

Atomic unsigned minimum on byte in memory, without return. _(Armv8.1)_

    void stuminb(const Register& rs, const MemOperand& src)


### STUMINH ###

Atomic unsigned minimum on halfword in memory, without return. _(Armv8.1)_

    void stuminh(const Register& rs, const MemOperand& src)


### STUMINL ###

Atomic unsigned minimum on word or doubleword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stuminl(const Register& rs, const MemOperand& src)


### STUMINLB ###

Atomic unsigned minimum on byte in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stuminlb(const Register& rs, const MemOperand& src)


### STUMINLH ###

Atomic unsigned minimum on halfword in memory, with Store-release semantics and without return. _(Armv8.1)_

    void stuminlh(const Register& rs, const MemOperand& src)


### STUR ###

Store integer or FP register (with unscaled offset).

    void stur(const CPURegister& rt,
              const MemOperand& src,
              LoadStoreScalingOption option = PreferUnscaledOffset)


### STURB ###

Store byte (with unscaled offset).

    void sturb(const Register& rt,
               const MemOperand& dst,
               LoadStoreScalingOption option = PreferUnscaledOffset)


### STURH ###

Store half-word (with unscaled offset).

    void sturh(const Register& rt,
               const MemOperand& dst,
               LoadStoreScalingOption option = PreferUnscaledOffset)


### STXP ###

Store exclusive register pair.

    void stxp(const Register& rs,
              const Register& rt,
              const Register& rt2,
              const MemOperand& dst)


### STXR ###

Store exclusive register.

    void stxr(const Register& rs, const Register& rt, const MemOperand& dst)


### STXRB ###

Store exclusive byte.

    void stxrb(const Register& rs, const Register& rt, const MemOperand& dst)


### STXRH ###

Store exclusive half-word.

    void stxrh(const Register& rs, const Register& rt, const MemOperand& dst)


### SUB ###

Subtract.

    void sub(const Register& rd, const Register& rn, const Operand& operand)


### SUBS ###

Subtract and update status flags.

    void subs(const Register& rd, const Register& rn, const Operand& operand)


### SVC ###

Generate exception targeting EL1.

    void svc(int code)


### SWP ###

Swap word or doubleword in memory _(Armv8.1)_

    void swp(const Register& rs, const Register& rt, const MemOperand& src)


### SWPA ###

Swap word or doubleword in memory, with Load-acquire semantics _(Armv8.1)_

    void swpa(const Register& rs, const Register& rt, const MemOperand& src)


### SWPAB ###

Swap byte in memory, with Load-acquire semantics _(Armv8.1)_

    void swpab(const Register& rs, const Register& rt, const MemOperand& src)


### SWPAH ###

Swap halfword in memory, with Load-acquire semantics _(Armv8.1)_

    void swpah(const Register& rs, const Register& rt, const MemOperand& src)


### SWPAL ###

Swap word or doubleword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void swpal(const Register& rs, const Register& rt, const MemOperand& src)


### SWPALB ###

Swap byte in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void swpalb(const Register& rs, const Register& rt, const MemOperand& src)


### SWPALH ###

Swap halfword in memory, with Load-acquire and Store-release semantics _(Armv8.1)_

    void swpalh(const Register& rs, const Register& rt, const MemOperand& src)


### SWPB ###

Swap byte in memory _(Armv8.1)_

    void swpb(const Register& rs, const Register& rt, const MemOperand& src)


### SWPH ###

Swap halfword in memory _(Armv8.1)_

    void swph(const Register& rs, const Register& rt, const MemOperand& src)


### SWPL ###

Swap word or doubleword in memory, with Store-release semantics _(Armv8.1)_

    void swpl(const Register& rs, const Register& rt, const MemOperand& src)


### SWPLB ###

Swap byte in memory, with Store-release semantics _(Armv8.1)_

    void swplb(const Register& rs, const Register& rt, const MemOperand& src)


### SWPLH ###

Swap halfword in memory, with Store-release semantics _(Armv8.1)_

    void swplh(const Register& rs, const Register& rt, const MemOperand& src)


### SXTB ###

Signed extend byte.

    void sxtb(const Register& rd, const Register& rn)


### SXTH ###

Signed extend halfword.

    void sxth(const Register& rd, const Register& rn)


### SXTW ###

Signed extend word.

    void sxtw(const Register& rd, const Register& rn)


### SYS ###

System instruction with pre-encoded op (op1:crn:crm:op2).

    void sys(int op, const Register& xt = xzr)


### SYS ###

System instruction.

    void sys(int op1, int crn, int crm, int op2, const Register& xt = xzr)


<a id="integer-t">

### TBNZ ###

Test bit and branch to PC offset if not zero.

    void tbnz(const Register& rt, unsigned bit_pos, int64_t imm14)


### TBNZ ###

Test bit and branch to label if not zero.

    void tbnz(const Register& rt, unsigned bit_pos, Label* label)


### TBZ ###

Test bit and branch to PC offset if zero.

    void tbz(const Register& rt, unsigned bit_pos, int64_t imm14)


### TBZ ###

Test bit and branch to label if zero.

    void tbz(const Register& rt, unsigned bit_pos, Label* label)


### TST ###

Bit test and set flags.

    void tst(const Register& rn, const Operand& operand)


<a id="integer-u">

### UBFIZ ###

Unsigned bitfield insert with zero at right.

    void ubfiz(const Register& rd,
               const Register& rn,
               unsigned lsb,
               unsigned width)


### UBFM ###

Unsigned bitfield move.

    void ubfm(const Register& rd,
              const Register& rn,
              unsigned immr,
              unsigned imms)


### UBFX ###

Unsigned bitfield extract.

    void ubfx(const Register& rd,
              const Register& rn,
              unsigned lsb,
              unsigned width)


### UDF ###

Generate undefined instruction exception.

    void udf(int code)


### UDIV ###

Unsigned integer divide.

    void udiv(const Register& rd, const Register& rn, const Register& rm)


### UMADDL ###

Unsigned long multiply and accumulate: 32 x 32 + 64 -> 64-bit.

    void umaddl(const Register& xd,
                const Register& wn,
                const Register& wm,
                const Register& xa)


### UMSUBL ###

Unsigned long multiply and subtract: 64 - (32 x 32) -> 64-bit.

    void umsubl(const Register& xd,
                const Register& wn,
                const Register& wm,
                const Register& xa)


### UMULH ###

Unsigned multiply high: 64 x 64 -> 64-bit <127:64>.

    void umulh(const Register& xd, const Register& xn, const Register& xm)


### UMULL ###

Unsigned long multiply: 32 x 32 -> 64-bit.

    void umull(const Register& xd, const Register& wn, const Register& wm)


### UXTB ###

Unsigned extend byte.

    void uxtb(const Register& rd, const Register& rn)


### UXTH ###

Unsigned extend halfword.

    void uxth(const Register& rd, const Register& rn)


### UXTW ###

Unsigned extend word.

    void uxtw(const Register& rd, const Register& rn)


<a id="integer-x">

### XAFLAG ###

Convert floating-point condition flags from alternative format to Arm format _(Armv8.5)_.

    void xaflag()


### XPACD ###

Strip Pointer Authentication Code of Data address _(Armv8.3)_.

    void xpacd(const Register& xd)


### XPACI ###

Strip Pointer Authentication Code of Instruction address _(Armv8.3)_.

    void xpaci(const Register& xd)


### XPACLRI ###

Strip Pointer Authentication Code of Instruction address in LR _(Armv8.3)_.

    void xpaclri()



AArch64 floating point and NEON instructions
--------------------------------------------

<a id="float-a">

### ABS ###

Absolute value.

    void abs(const VRegister& vd, const VRegister& vn)


### ADD ###

Add.

    void add(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### ADDHN ###

Add narrow returning high half.

    void addhn(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### ADDHN2 ###

Add narrow returning high half (second part).

    void addhn2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### ADDP ###

Add pair of elements scalar.

    void addp(const VRegister& vd, const VRegister& vn)


### ADDP ###

Add pairwise.

    void addp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### ADDV ###

Add across vector.

    void addv(const VRegister& vd, const VRegister& vn)


### AND ###

Bitwise and.

    void and_(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-b">

### BIC ###

Bit clear immediate.

    void bic(const VRegister& vd, const int imm8, const int left_shift = 0)


### BIC ###

Bit clear.

    void bic(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### BIF ###

Bitwise insert if false.

    void bif(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### BIT ###

Bitwise insert if true.

    void bit(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### BSL ###

Bitwise select.

    void bsl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-c">

### CLS ###

Count leading sign bits.

    void cls(const VRegister& vd, const VRegister& vn)


### CLZ ###

Count leading zero bits (vector).

    void clz(const VRegister& vd, const VRegister& vn)


### CMEQ ###

Compare bitwise to zero.

    void cmeq(const VRegister& vd, const VRegister& vn, int value)


### CMEQ ###

Compare equal.

    void cmeq(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### CMGE ###

Compare signed greater than or equal to zero.

    void cmge(const VRegister& vd, const VRegister& vn, int value)


### CMGE ###

Compare signed greater than or equal.

    void cmge(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### CMGT ###

Compare signed greater than zero.

    void cmgt(const VRegister& vd, const VRegister& vn, int value)


### CMGT ###

Compare signed greater than.

    void cmgt(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### CMHI ###

Compare unsigned higher.

    void cmhi(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### CMHS ###

Compare unsigned higher or same.

    void cmhs(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### CMLE ###

Compare signed less than or equal to zero.

    void cmle(const VRegister& vd, const VRegister& vn, int value)


### CMLT ###

Compare signed less than zero.

    void cmlt(const VRegister& vd, const VRegister& vn, int value)


### CMTST ###

Compare bitwise test bits nonzero.

    void cmtst(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### CNT ###

Population count per byte.

    void cnt(const VRegister& vd, const VRegister& vn)


<a id="float-d">

### DUP ###

Duplicate general-purpose register to vector.

    void dup(const VRegister& vd, const Register& rn)


### DUP ###

Duplicate vector element to vector or scalar.

    void dup(const VRegister& vd, const VRegister& vn, int vn_index)


<a id="float-e">

### EOR ###

Bitwise eor.

    void eor(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### EXT ###

Extract vector from pair of vectors.

    void ext(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vm,
             int index)


<a id="float-f">

### FABD ###

FP absolute difference.

    void fabd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FABS ###

FP absolute.

    void fabs(const VRegister& vd, const VRegister& vn)


### FACGE ###

FP absolute greater than or equal.

    void facge(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FACGT ###

FP absolute greater than.

    void facgt(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FADD ###

FP add.

    void fadd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FADDP ###

FP pairwise add scalar.

    void faddp(const VRegister& vd, const VRegister& vn)


### FADDP ###

FP pairwise add vector.

    void faddp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FCADD ###

FP complex add _(Armv8.3)_.

    void fcadd(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int rot)


### FCCMP ###

FP conditional compare.

    void fccmp(const VRegister& vn,
               const VRegister& vm,
               StatusFlags nzcv,
               Condition cond)


### FCCMPE ###

FP conditional signaling compare.

    void fccmpe(const VRegister& vn,
                const VRegister& vm,
                StatusFlags nzcv,
                Condition cond)


### FCMEQ ###

FP compare equal to zero.

    void fcmeq(const VRegister& vd, const VRegister& vn, double imm)


### FCMEQ ###

FP compare equal.

    void fcmeq(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FCMGE ###

FP greater than or equal to zero.

    void fcmge(const VRegister& vd, const VRegister& vn, double imm)


### FCMGE ###

FP greater than or equal.

    void fcmge(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FCMGT ###

FP greater than zero.

    void fcmgt(const VRegister& vd, const VRegister& vn, double imm)


### FCMGT ###

FP greater than.

    void fcmgt(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FCMLA ###

FP complex multiply accumulate (by element) _(Armv8.3)_.

    void fcmla(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index,
               int rot)


### FCMLA ###

FP complex multiply accumulate _(Armv8.3)_.

    void fcmla(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int rot)


### FCMLE ###

FP less than or equal to zero.

    void fcmle(const VRegister& vd, const VRegister& vn, double imm)


### FCMLT ###

FP less than to zero.

    void fcmlt(const VRegister& vd, const VRegister& vn, double imm)


### FCMP ###

FP compare immediate.

    void fcmp(const VRegister& vn, double value)


### FCMP ###

FP compare registers.

    void fcmp(const VRegister& vn, const VRegister& vm)


### FCMPE ###

FP signaling compare immediate.

    void fcmpe(const VRegister& vn, double value)


### FCMPE ###

FP signaling compare registers.

    void fcmpe(const VRegister& vn, const VRegister& vm)


### FCSEL ###

FP conditional select.

    void fcsel(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               Condition cond)


### FCVT ###

FP convert between precisions.

    void fcvt(const VRegister& vd, const VRegister& vn)


### FCVTAS ###

FP convert to signed integer, nearest with ties to away.

    void fcvtas(const Register& rd, const VRegister& vn)


### FCVTAS ###

FP convert to signed integer, nearest with ties to away.

    void fcvtas(const VRegister& vd, const VRegister& vn)


### FCVTAU ###

FP convert to unsigned integer, nearest with ties to away.

    void fcvtau(const Register& rd, const VRegister& vn)


### FCVTAU ###

FP convert to unsigned integer, nearest with ties to away.

    void fcvtau(const VRegister& vd, const VRegister& vn)


### FCVTL ###

FP convert to higher precision.

    void fcvtl(const VRegister& vd, const VRegister& vn)


### FCVTL2 ###

FP convert to higher precision (second part).

    void fcvtl2(const VRegister& vd, const VRegister& vn)


### FCVTMS ###

FP convert to signed integer, round towards -infinity.

    void fcvtms(const Register& rd, const VRegister& vn)


### FCVTMS ###

FP convert to signed integer, round towards -infinity.

    void fcvtms(const VRegister& vd, const VRegister& vn)


### FCVTMU ###

FP convert to unsigned integer, round towards -infinity.

    void fcvtmu(const Register& rd, const VRegister& vn)


### FCVTMU ###

FP convert to unsigned integer, round towards -infinity.

    void fcvtmu(const VRegister& vd, const VRegister& vn)


### FCVTN ###

FP convert to lower precision.

    void fcvtn(const VRegister& vd, const VRegister& vn)


### FCVTN2 ###

FP convert to lower prevision (second part).

    void fcvtn2(const VRegister& vd, const VRegister& vn)


### FCVTNS ###

FP convert to signed integer, nearest with ties to even.

    void fcvtns(const Register& rd, const VRegister& vn)


### FCVTNS ###

FP convert to signed integer, nearest with ties to even.

    void fcvtns(const VRegister& rd, const VRegister& vn)


### FCVTNU ###

FP convert to unsigned integer, nearest with ties to even.

    void fcvtnu(const Register& rd, const VRegister& vn)


### FCVTNU ###

FP convert to unsigned integer, nearest with ties to even.

    void fcvtnu(const VRegister& rd, const VRegister& vn)


### FCVTPS ###

FP convert to signed integer, round towards +infinity.

    void fcvtps(const Register& rd, const VRegister& vn)


### FCVTPS ###

FP convert to signed integer, round towards +infinity.

    void fcvtps(const VRegister& vd, const VRegister& vn)


### FCVTPU ###

FP convert to unsigned integer, round towards +infinity.

    void fcvtpu(const Register& rd, const VRegister& vn)


### FCVTPU ###

FP convert to unsigned integer, round towards +infinity.

    void fcvtpu(const VRegister& vd, const VRegister& vn)


### FCVTXN ###

FP convert to lower precision, rounding to odd.

    void fcvtxn(const VRegister& vd, const VRegister& vn)


### FCVTXN2 ###

FP convert to lower precision, rounding to odd (second part).

    void fcvtxn2(const VRegister& vd, const VRegister& vn)


### FCVTZS ###

FP convert to signed integer or fixed-point, round towards zero.

    void fcvtzs(const Register& rd, const VRegister& vn, int fbits = 0)


### FCVTZS ###

FP convert to signed integer or fixed-point, round towards zero.

    void fcvtzs(const VRegister& vd, const VRegister& vn, int fbits = 0)


### FCVTZU ###

FP convert to unsigned integer or fixed-point, round towards zero.

    void fcvtzu(const Register& rd, const VRegister& vn, int fbits = 0)


### FCVTZU ###

FP convert to unsigned integer or fixed-point, round towards zero.

    void fcvtzu(const VRegister& vd, const VRegister& vn, int fbits = 0)


### FDIV ###

FP divide.

    void fdiv(const VRegister& vd, const VRegister& fn, const VRegister& vm)


### FJCVTZS ###

FP JavaScript convert to signed integer, rounding toward zero _(Armv8.3)_.

    void fjcvtzs(const Register& rd, const VRegister& vn)


### FMADD ###

FP fused multiply-add.

    void fmadd(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               const VRegister& va)


### FMAX ###

FP maximum.

    void fmax(const VRegister& vd, const VRegister& fn, const VRegister& vm)


### FMAXNM ###

FP maximum number.

    void fmaxnm(const VRegister& vd, const VRegister& fn, const VRegister& vm)


### FMAXNMP ###

FP pairwise maximum number scalar.

    void fmaxnmp(const VRegister& vd, const VRegister& vn)


### FMAXNMP ###

FP pairwise maximum number vector.

    void fmaxnmp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMAXNMV ###

FP maximum number across vector.

    void fmaxnmv(const VRegister& vd, const VRegister& vn)


### FMAXP ###

FP pairwise maximum scalar.

    void fmaxp(const VRegister& vd, const VRegister& vn)


### FMAXP ###

FP pairwise maximum vector.

    void fmaxp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMAXV ###

FP maximum across vector.

    void fmaxv(const VRegister& vd, const VRegister& vn)


### FMIN ###

FP minimum.

    void fmin(const VRegister& vd, const VRegister& fn, const VRegister& vm)


### FMINNM ###

FP minimum number.

    void fminnm(const VRegister& vd, const VRegister& fn, const VRegister& vm)


### FMINNMP ###

FP pairwise minimum number scalar.

    void fminnmp(const VRegister& vd, const VRegister& vn)


### FMINNMP ###

FP pairwise minimum number vector.

    void fminnmp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMINNMV ###

FP minimum number across vector.

    void fminnmv(const VRegister& vd, const VRegister& vn)


### FMINP ###

FP pairwise minimum scalar.

    void fminp(const VRegister& vd, const VRegister& vn)


### FMINP ###

FP pairwise minimum vector.

    void fminp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMINV ###

FP minimum across vector.

    void fminv(const VRegister& vd, const VRegister& vn)


### FMLA ###

FP fused multiply-add to accumulator by element.

    void fmla(const VRegister& vd,
              const VRegister& vn,
              const VRegister& vm,
              int vm_index)


### FMLA ###

FP vector multiply accumulate.

    void fmla(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMLAL ###

FP fused multiply-add long to accumulator by element.

    void fmlal(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### FMLAL ###

FP fused multiply-add long to accumulator.

    void fmlal(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMLAL2 ###

FP fused multiply-add long to accumulator (second part).

    void fmlal2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMLAL2 ###

FP fused multiply-add long to accumulator by element (second part).

    void fmlal2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### FMLS ###

FP fused multiply-sub from accumulator by element.

    void fmls(const VRegister& vd,
              const VRegister& vn,
              const VRegister& vm,
              int vm_index)


### FMLS ###

FP vector multiply subtract.

    void fmls(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMLSL ###

FP fused multiply-subtract long to accumulator by element.

    void fmlsl(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### FMLSL ###

FP fused multiply-subtract long to accumulator.

    void fmlsl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMLSL2 ###

FP fused multiply-subtract long to accumulator (second part).

    void fmlsl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMLSL2 ###

FP fused multiply-subtract long to accumulator by element (second part).

    void fmlsl2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### FMOV ###

Move 64-bit register to top half of 128-bit FP register.

    void fmov(const VRegister& vd, int index, const Register& rn)


### FMOV ###

Move FP register to FP register.

    void fmov(const VRegister& vd, const VRegister& fn)


### FMOV ###

Move FP register to register.

    void fmov(const Register& rd, const VRegister& fn)


### FMOV ###

Move double precision immediate to FP register.

    void fmov(const VRegister& vd, double imm)


### FMOV ###

Move half precision immediate to FP register _(Armv8.2)_.

    void fmov(const VRegister& vd, Float16 imm)


### FMOV ###

Move register to FP register.

    void fmov(const VRegister& vd, const Register& rn)


### FMOV ###

Move single precision immediate to FP register.

    void fmov(const VRegister& vd, float imm)


### FMOV ###

Move top half of 128-bit FP register to 64-bit register.

    void fmov(const Register& rd, const VRegister& vn, int index)


### FMSUB ###

FP fused multiply-subtract.

    void fmsub(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               const VRegister& va)


### FMUL ###

FP multiply by element.

    void fmul(const VRegister& vd,
              const VRegister& vn,
              const VRegister& vm,
              int vm_index)


### FMUL ###

FP multiply.

    void fmul(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FMULX ###

FP multiply extended by element.

    void fmulx(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### FMULX ###

FP vector multiply extended.

    void fmulx(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FNEG ###

FP negate.

    void fneg(const VRegister& vd, const VRegister& vn)


### FNMADD ###

FP fused multiply-add and negate.

    void fnmadd(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                const VRegister& va)


### FNMSUB ###

FP fused multiply-subtract and negate.

    void fnmsub(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                const VRegister& va)


### FNMUL ###

FP multiply-negate scalar.

    void fnmul(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FRECPE ###

FP reciprocal estimate.

    void frecpe(const VRegister& vd, const VRegister& vn)


### FRECPS ###

FP reciprocal step.

    void frecps(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FRECPX ###

FP reciprocal exponent scalar.

    void frecpx(const VRegister& vd, const VRegister& vn)


### FRINT32X ###

FP round to 32-bit integer, exact, implicit rounding _(Armv8.5)_.

    void frint32x(const VRegister& vd, const VRegister& vn)


### FRINT32Z ###

FP round to 32-bit integer, towards zero _(Armv8.5)_.

    void frint32z(const VRegister& vd, const VRegister& vn)


### FRINT64X ###

FP round to 64-bit integer, exact, implicit rounding _(Armv8.5)_.

    void frint64x(const VRegister& vd, const VRegister& vn)


### FRINT64Z ###

FP round to 64-bit integer, towards zero _(Armv8.5)_.

    void frint64z(const VRegister& vd, const VRegister& vn)


### FRINTA ###

FP round to integer, nearest with ties to away.

    void frinta(const VRegister& vd, const VRegister& vn)


### FRINTI ###

FP round to integer, implicit rounding.

    void frinti(const VRegister& vd, const VRegister& vn)


### FRINTM ###

FP round to integer, toward minus infinity.

    void frintm(const VRegister& vd, const VRegister& vn)


### FRINTN ###

FP round to integer, nearest with ties to even.

    void frintn(const VRegister& vd, const VRegister& vn)


### FRINTP ###

FP round to integer, toward plus infinity.

    void frintp(const VRegister& vd, const VRegister& vn)


### FRINTX ###

FP round to integer, exact, implicit rounding.

    void frintx(const VRegister& vd, const VRegister& vn)


### FRINTZ ###

FP round to integer, towards zero.

    void frintz(const VRegister& vd, const VRegister& vn)


### FRSQRTE ###

FP reciprocal square root estimate.

    void frsqrte(const VRegister& vd, const VRegister& vn)


### FRSQRTS ###

FP reciprocal square root step.

    void frsqrts(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### FSQRT ###

FP square root.

    void fsqrt(const VRegister& vd, const VRegister& vn)


### FSUB ###

FP subtract.

    void fsub(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-i">

### INS ###

Insert vector element from another vector element.

    void ins(const VRegister& vd,
             int vd_index,
             const VRegister& vn,
             int vn_index)


### INS ###

Insert vector element from general-purpose register.

    void ins(const VRegister& vd, int vd_index, const Register& rn)


<a id="float-l">

### LD1 ###

One-element single structure load to one lane.

    void ld1(const VRegister& vt, int lane, const MemOperand& src)


### LD1 ###

One-element structure load to four registers.

    void ld1(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const VRegister& vt4,
             const MemOperand& src)


### LD1 ###

One-element structure load to one register.

    void ld1(const VRegister& vt, const MemOperand& src)


### LD1 ###

One-element structure load to three registers.

    void ld1(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const MemOperand& src)


### LD1 ###

One-element structure load to two registers.

    void ld1(const VRegister& vt, const VRegister& vt2, const MemOperand& src)


### LD1R ###

One-element single structure load to all lanes.

    void ld1r(const VRegister& vt, const MemOperand& src)


### LD2 ###

Two-element single structure load to one lane.

    void ld2(const VRegister& vt,
             const VRegister& vt2,
             int lane,
             const MemOperand& src)


### LD2 ###

Two-element structure load.

    void ld2(const VRegister& vt, const VRegister& vt2, const MemOperand& src)


### LD2R ###

Two-element single structure load to all lanes.

    void ld2r(const VRegister& vt, const VRegister& vt2, const MemOperand& src)


### LD3 ###

Three-element single structure load to one lane.

    void ld3(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             int lane,
             const MemOperand& src)


### LD3 ###

Three-element structure load.

    void ld3(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const MemOperand& src)


### LD3R ###

Three-element single structure load to all lanes.

    void ld3r(const VRegister& vt,
              const VRegister& vt2,
              const VRegister& vt3,
              const MemOperand& src)


### LD4 ###

Four-element single structure load to one lane.

    void ld4(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const VRegister& vt4,
             int lane,
             const MemOperand& src)


### LD4 ###

Four-element structure load.

    void ld4(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const VRegister& vt4,
             const MemOperand& src)


### LD4R ###

Four-element single structure load to all lanes.

    void ld4r(const VRegister& vt,
              const VRegister& vt2,
              const VRegister& vt3,
              const VRegister& vt4,
              const MemOperand& src)


<a id="float-m">

### MLA ###

Multiply-add by scalar element.

    void mla(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vm,
             int vm_index)


### MLA ###

Multiply-add to accumulator.

    void mla(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### MLS ###

Multiply-subtract by scalar element.

    void mls(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vm,
             int vm_index)


### MLS ###

Multiply-subtract to accumulator.

    void mls(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### MOV ###

Move general-purpose register to a vector element.

    void mov(const VRegister& vd, int vd_index, const Register& rn)


### MOV ###

Move register to register.

    void mov(const VRegister& vd, const VRegister& vn)


### MOV ###

Move vector element to another vector element.

    void mov(const VRegister& vd,
             int vd_index,
             const VRegister& vn,
             int vn_index)


### MOV ###

Move vector element to general-purpose register.

    void mov(const Register& rd, const VRegister& vn, int vn_index)


### MOV ###

Move vector element to scalar.

    void mov(const VRegister& vd, const VRegister& vn, int vn_index)


### MOVI ###

Vector move immediate.

    void movi(const VRegister& vd,
              const uint64_t imm,
              Shift shift = LSL,
              const int shift_amount = 0)


### MUL ###

Multiply by scalar element.

    void mul(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vm,
             int vm_index)


### MUL ###

Multiply.

    void mul(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### MVN ###

Bitwise not.

    void mvn(const VRegister& vd, const VRegister& vn)


### MVNI ###

Vector move inverted immediate.

    void mvni(const VRegister& vd,
              const int imm8,
              Shift shift = LSL,
              const int shift_amount = 0)


<a id="float-n">

### NEG ###

Negate.

    void neg(const VRegister& vd, const VRegister& vn)


### NOT ###

Bitwise not.

    void not_(const VRegister& vd, const VRegister& vn)


<a id="float-o">

### ORN ###

Bitwise orn.

    void orn(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### ORR ###

Bitwise or immediate.

    void orr(const VRegister& vd, const int imm8, const int left_shift = 0)


### ORR ###

Bitwise or.

    void orr(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-p">

### PMUL ###

Polynomial multiply.

    void pmul(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### PMULL ###

Polynomial multiply long.

    void pmull(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### PMULL2 ###

Polynomial multiply long (second part).

    void pmull2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-r">

### RADDHN ###

Rounding add narrow returning high half.

    void raddhn(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### RADDHN2 ###

Rounding add narrow returning high half (second part).

    void raddhn2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### RBIT ###

Reverse bit order.

    void rbit(const VRegister& vd, const VRegister& vn)


### REV16 ###

Reverse elements in 16-bit halfwords.

    void rev16(const VRegister& vd, const VRegister& vn)


### REV32 ###

Reverse elements in 32-bit words.

    void rev32(const VRegister& vd, const VRegister& vn)


### REV64 ###

Reverse elements in 64-bit doublewords.

    void rev64(const VRegister& vd, const VRegister& vn)


### RSHRN ###

Rounding shift right narrow by immediate.

    void rshrn(const VRegister& vd, const VRegister& vn, int shift)


### RSHRN2 ###

Rounding shift right narrow by immediate (second part).

    void rshrn2(const VRegister& vd, const VRegister& vn, int shift)


### RSUBHN ###

Rounding subtract narrow returning high half.

    void rsubhn(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### RSUBHN2 ###

Rounding subtract narrow returning high half (second part).

    void rsubhn2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-s">

### SABA ###

Signed absolute difference and accumulate.

    void saba(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SABAL ###

Signed absolute difference and accumulate long.

    void sabal(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SABAL2 ###

Signed absolute difference and accumulate long (second part).

    void sabal2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SABD ###

Signed absolute difference.

    void sabd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SABDL ###

Signed absolute difference long.

    void sabdl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SABDL2 ###

Signed absolute difference long (second part).

    void sabdl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SADALP ###

Signed pairwise long add and accumulate.

    void sadalp(const VRegister& vd, const VRegister& vn)


### SADDL ###

Signed add long.

    void saddl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SADDL2 ###

Signed add long (second part).

    void saddl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SADDLP ###

Signed pairwise long add.

    void saddlp(const VRegister& vd, const VRegister& vn)


### SADDLV ###

Signed add long across vector.

    void saddlv(const VRegister& vd, const VRegister& vn)


### SADDW ###

Signed add wide.

    void saddw(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SADDW2 ###

Signed add wide (second part).

    void saddw2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SCVTF ###

Convert signed integer or fixed point to FP.

    void scvtf(const VRegister& fd, const Register& rn, int fbits = 0)


### SCVTF ###

Convert signed integer or fixed-point to FP.

    void scvtf(const VRegister& fd, const VRegister& vn, int fbits = 0)


### SDOT ###

Signed dot product _(Armv8.2)_.

    void sdot(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SDOT ###

Signed dot product by element _(Armv8.2)_.

    void sdot(const VRegister& vd,
              const VRegister& vn,
              const VRegister& vm,
              int vm_index)


### SHADD ###

Signed halving add.

    void shadd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SHL ###

Shift left by immediate.

    void shl(const VRegister& vd, const VRegister& vn, int shift)


### SHLL ###

Shift left long by element size.

    void shll(const VRegister& vd, const VRegister& vn, int shift)


### SHLL2 ###

Shift left long by element size (second part).

    void shll2(const VRegister& vd, const VRegister& vn, int shift)


### SHRN ###

Shift right narrow by immediate.

    void shrn(const VRegister& vd, const VRegister& vn, int shift)


### SHRN2 ###

Shift right narrow by immediate (second part).

    void shrn2(const VRegister& vd, const VRegister& vn, int shift)


### SHSUB ###

Signed halving sub.

    void shsub(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SLI ###

Shift left by immediate and insert.

    void sli(const VRegister& vd, const VRegister& vn, int shift)


### SMAX ###

Signed maximum.

    void smax(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMAXP ###

Signed pairwise maximum.

    void smaxp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMAXV ###

Signed maximum across vector.

    void smaxv(const VRegister& vd, const VRegister& vn)


### SMIN ###

Signed minimum.

    void smin(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMINP ###

Signed minimum pairwise.

    void sminp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMINV ###

Signed minimum across vector.

    void sminv(const VRegister& vd, const VRegister& vn)


### SMLAL ###

Signed long multiply-add by scalar element.

    void smlal(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### SMLAL ###

Signed long multiply-add.

    void smlal(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMLAL2 ###

Signed long multiply-add (second part).

    void smlal2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMLAL2 ###

Signed long multiply-add by scalar element (second part).

    void smlal2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### SMLSL ###

Signed long multiply-sub by scalar element.

    void smlsl(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### SMLSL ###

Signed long multiply-sub.

    void smlsl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMLSL2 ###

Signed long multiply-sub (second part).

    void smlsl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMLSL2 ###

Signed long multiply-sub by scalar element (second part).

    void smlsl2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### SMMLA ###

Signed 8-bit integer matrix multiply-accumulate (vector).

    void smmla(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMOV ###

Signed move vector element to general-purpose register.

    void smov(const Register& rd, const VRegister& vn, int vn_index)


### SMULL ###

Signed long multiply by scalar element.

    void smull(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### SMULL ###

Signed long multiply.

    void smull(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMULL2 ###

Signed long multiply (second part).

    void smull2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SMULL2 ###

Signed long multiply by scalar element (second part).

    void smull2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### SQABS ###

Signed saturating absolute value.

    void sqabs(const VRegister& vd, const VRegister& vn)


### SQADD ###

Signed saturating add.

    void sqadd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQDMLAL ###

Signed saturating doubling long multiply-add by element.

    void sqdmlal(const VRegister& vd,
                 const VRegister& vn,
                 const VRegister& vm,
                 int vm_index)


### SQDMLAL ###

Signed saturating doubling long multiply-add.

    void sqdmlal(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQDMLAL2 ###

Signed saturating doubling long multiply-add (second part).

    void sqdmlal2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQDMLAL2 ###

Signed saturating doubling long multiply-add by element (second part).

    void sqdmlal2(const VRegister& vd,
                  const VRegister& vn,
                  const VRegister& vm,
                  int vm_index)


### SQDMLSL ###

Signed saturating doubling long multiply-sub by element.

    void sqdmlsl(const VRegister& vd,
                 const VRegister& vn,
                 const VRegister& vm,
                 int vm_index)


### SQDMLSL ###

Signed saturating doubling long multiply-subtract.

    void sqdmlsl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQDMLSL2 ###

Signed saturating doubling long multiply-sub by element (second part).

    void sqdmlsl2(const VRegister& vd,
                  const VRegister& vn,
                  const VRegister& vm,
                  int vm_index)


### SQDMLSL2 ###

Signed saturating doubling long multiply-subtract (second part).

    void sqdmlsl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQDMULH ###

Signed saturating doubling multiply element returning high half.

    void sqdmulh(const VRegister& vd,
                 const VRegister& vn,
                 const VRegister& vm,
                 int vm_index)


### SQDMULH ###

Signed saturating doubling multiply returning high half.

    void sqdmulh(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQDMULL ###

Signed saturating double long multiply by element.

    void sqdmull(const VRegister& vd,
                 const VRegister& vn,
                 const VRegister& vm,
                 int vm_index)


### SQDMULL ###

Signed saturating doubling long multiply.

    void sqdmull(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQDMULL2 ###

Signed saturating double long multiply by element (second part).

    void sqdmull2(const VRegister& vd,
                  const VRegister& vn,
                  const VRegister& vm,
                  int vm_index)


### SQDMULL2 ###

Signed saturating doubling long multiply (second part).

    void sqdmull2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQNEG ###

Signed saturating negate.

    void sqneg(const VRegister& vd, const VRegister& vn)


### SQRDMLAH ###

Signed saturating rounding doubling multiply accumulate element returning high half _(Armv8.1)_.

    void sqrdmlah(const VRegister& vd,
                  const VRegister& vn,
                  const VRegister& vm,
                  int vm_index)


### SQRDMLAH ###

Signed saturating rounding doubling multiply accumulate returning high half _(Armv8.1)_.

    void sqrdmlah(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQRDMLSH ###

Signed saturating rounding doubling multiply subtract element returning high half _(Armv8.1)_.

    void sqrdmlsh(const VRegister& vd,
                  const VRegister& vn,
                  const VRegister& vm,
                  int vm_index)


### SQRDMLSH ###

Signed saturating rounding doubling multiply subtract returning high half _(Armv8.1)_.

    void sqrdmlsh(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQRDMULH ###

Signed saturating rounding doubling multiply element returning high half.

    void sqrdmulh(const VRegister& vd,
                  const VRegister& vn,
                  const VRegister& vm,
                  int vm_index)


### SQRDMULH ###

Signed saturating rounding doubling multiply returning high half.

    void sqrdmulh(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQRSHL ###

Signed saturating rounding shift left by register.

    void sqrshl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQRSHRN ###

Signed saturating rounded shift right narrow by immediate.

    void sqrshrn(const VRegister& vd, const VRegister& vn, int shift)


### SQRSHRN2 ###

Signed saturating rounded shift right narrow by immediate (second part).

    void sqrshrn2(const VRegister& vd, const VRegister& vn, int shift)


### SQRSHRUN ###

Signed sat rounded shift right unsigned narrow by immediate.

    void sqrshrun(const VRegister& vd, const VRegister& vn, int shift)


### SQRSHRUN2 ###

Signed sat rounded shift right unsigned narrow by immediate (second part).

    void sqrshrun2(const VRegister& vd, const VRegister& vn, int shift)


### SQSHL ###

Signed saturating shift left by immediate.

    void sqshl(const VRegister& vd, const VRegister& vn, int shift)


### SQSHL ###

Signed saturating shift left by register.

    void sqshl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQSHLU ###

Signed saturating shift left unsigned by immediate.

    void sqshlu(const VRegister& vd, const VRegister& vn, int shift)


### SQSHRN ###

Signed saturating shift right narrow by immediate.

    void sqshrn(const VRegister& vd, const VRegister& vn, int shift)


### SQSHRN2 ###

Signed saturating shift right narrow by immediate (second part).

    void sqshrn2(const VRegister& vd, const VRegister& vn, int shift)


### SQSHRUN ###

Signed saturating shift right unsigned narrow by immediate.

    void sqshrun(const VRegister& vd, const VRegister& vn, int shift)


### SQSHRUN2 ###

Signed saturating shift right unsigned narrow by immediate (second part).

    void sqshrun2(const VRegister& vd, const VRegister& vn, int shift)


### SQSUB ###

Signed saturating subtract.

    void sqsub(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SQXTN ###

Signed saturating extract narrow.

    void sqxtn(const VRegister& vd, const VRegister& vn)


### SQXTN2 ###

Signed saturating extract narrow (second part).

    void sqxtn2(const VRegister& vd, const VRegister& vn)


### SQXTUN ###

Signed saturating extract unsigned narrow.

    void sqxtun(const VRegister& vd, const VRegister& vn)


### SQXTUN2 ###

Signed saturating extract unsigned narrow (second part).

    void sqxtun2(const VRegister& vd, const VRegister& vn)


### SRHADD ###

Signed rounding halving add.

    void srhadd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SRI ###

Shift right by immediate and insert.

    void sri(const VRegister& vd, const VRegister& vn, int shift)


### SRSHL ###

Signed rounding shift left by register.

    void srshl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SRSHR ###

Signed rounding shift right by immediate.

    void srshr(const VRegister& vd, const VRegister& vn, int shift)


### SRSRA ###

Signed rounding shift right by immediate and accumulate.

    void srsra(const VRegister& vd, const VRegister& vn, int shift)


### SSHL ###

Signed shift left by register.

    void sshl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SSHLL ###

Signed shift left long by immediate.

    void sshll(const VRegister& vd, const VRegister& vn, int shift)


### SSHLL2 ###

Signed shift left long by immediate (second part).

    void sshll2(const VRegister& vd, const VRegister& vn, int shift)


### SSHR ###

Signed shift right by immediate.

    void sshr(const VRegister& vd, const VRegister& vn, int shift)


### SSRA ###

Signed shift right by immediate and accumulate.

    void ssra(const VRegister& vd, const VRegister& vn, int shift)


### SSUBL ###

Signed subtract long.

    void ssubl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SSUBL2 ###

Signed subtract long (second part).

    void ssubl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SSUBW ###

Signed integer subtract wide.

    void ssubw(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SSUBW2 ###

Signed integer subtract wide (second part).

    void ssubw2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### ST1 ###

One-element single structure store from one lane.

    void st1(const VRegister& vt, int lane, const MemOperand& src)


### ST1 ###

One-element structure store from four registers.

    void st1(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const VRegister& vt4,
             const MemOperand& src)


### ST1 ###

One-element structure store from one register.

    void st1(const VRegister& vt, const MemOperand& src)


### ST1 ###

One-element structure store from three registers.

    void st1(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const MemOperand& src)


### ST1 ###

One-element structure store from two registers.

    void st1(const VRegister& vt, const VRegister& vt2, const MemOperand& src)


### ST2 ###

Two-element single structure store from two lanes.

    void st2(const VRegister& vt,
             const VRegister& vt2,
             int lane,
             const MemOperand& src)


### ST2 ###

Two-element structure store from two registers.

    void st2(const VRegister& vt, const VRegister& vt2, const MemOperand& src)


### ST3 ###

Three-element single structure store from three lanes.

    void st3(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             int lane,
             const MemOperand& src)


### ST3 ###

Three-element structure store from three registers.

    void st3(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const MemOperand& src)


### ST4 ###

Four-element single structure store from four lanes.

    void st4(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const VRegister& vt4,
             int lane,
             const MemOperand& src)


### ST4 ###

Four-element structure store from four registers.

    void st4(const VRegister& vt,
             const VRegister& vt2,
             const VRegister& vt3,
             const VRegister& vt4,
             const MemOperand& src)


### SUB ###

Subtract.

    void sub(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SUBHN ###

Subtract narrow returning high half.

    void subhn(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SUBHN2 ###

Subtract narrow returning high half (second part).

    void subhn2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### SUDOT ###

Dot product with signed and unsigned integers (vector, by element).

    void sudot(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### SUQADD ###

Signed saturating accumulate of unsigned value.

    void suqadd(const VRegister& vd, const VRegister& vn)


### SXTL ###

Signed extend long.

    void sxtl(const VRegister& vd, const VRegister& vn)


### SXTL2 ###

Signed extend long (second part).

    void sxtl2(const VRegister& vd, const VRegister& vn)


<a id="float-t">

### TBL ###

Table lookup from four registers.

    void tbl(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vn2,
             const VRegister& vn3,
             const VRegister& vn4,
             const VRegister& vm)


### TBL ###

Table lookup from one register.

    void tbl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### TBL ###

Table lookup from three registers.

    void tbl(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vn2,
             const VRegister& vn3,
             const VRegister& vm)


### TBL ###

Table lookup from two registers.

    void tbl(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vn2,
             const VRegister& vm)


### TBX ###

Table lookup extension from four registers.

    void tbx(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vn2,
             const VRegister& vn3,
             const VRegister& vn4,
             const VRegister& vm)


### TBX ###

Table lookup extension from one register.

    void tbx(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### TBX ###

Table lookup extension from three registers.

    void tbx(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vn2,
             const VRegister& vn3,
             const VRegister& vm)


### TBX ###

Table lookup extension from two registers.

    void tbx(const VRegister& vd,
             const VRegister& vn,
             const VRegister& vn2,
             const VRegister& vm)


### TRN1 ###

Transpose vectors (primary).

    void trn1(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### TRN2 ###

Transpose vectors (secondary).

    void trn2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-u">

### UABA ###

Unsigned absolute difference and accumulate.

    void uaba(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UABAL ###

Unsigned absolute difference and accumulate long.

    void uabal(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UABAL2 ###

Unsigned absolute difference and accumulate long (second part).

    void uabal2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UABD ###

Unsigned absolute difference.

    void uabd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UABDL ###

Unsigned absolute difference long.

    void uabdl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UABDL2 ###

Unsigned absolute difference long (second part).

    void uabdl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UADALP ###

Unsigned pairwise long add and accumulate.

    void uadalp(const VRegister& vd, const VRegister& vn)


### UADDL ###

Unsigned add long.

    void uaddl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UADDL2 ###

Unsigned add long (second part).

    void uaddl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UADDLP ###

Unsigned pairwise long add.

    void uaddlp(const VRegister& vd, const VRegister& vn)


### UADDLV ###

Unsigned add long across vector.

    void uaddlv(const VRegister& vd, const VRegister& vn)


### UADDW ###

Unsigned add wide.

    void uaddw(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UADDW2 ###

Unsigned add wide (second part).

    void uaddw2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UCVTF ###

Convert unsigned integer or fixed point to FP.

    void ucvtf(const VRegister& fd, const Register& rn, int fbits = 0)


### UCVTF ###

Convert unsigned integer or fixed-point to FP.

    void ucvtf(const VRegister& fd, const VRegister& vn, int fbits = 0)


### UDOT ###

Unsigned dot product _(Armv8.2)_.

    void udot(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UDOT ###

Unsigned dot product by element _(Armv8.2)_.

    void udot(const VRegister& vd,
              const VRegister& vn,
              const VRegister& vm,
              int vm_index)


### UHADD ###

Unsigned halving add.

    void uhadd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UHSUB ###

Unsigned halving sub.

    void uhsub(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMAX ###

Unsigned maximum.

    void umax(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMAXP ###

Unsigned pairwise maximum.

    void umaxp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMAXV ###

Unsigned maximum across vector.

    void umaxv(const VRegister& vd, const VRegister& vn)


### UMIN ###

Unsigned minimum.

    void umin(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMINP ###

Unsigned pairwise minimum.

    void uminp(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMINV ###

Unsigned minimum across vector.

    void uminv(const VRegister& vd, const VRegister& vn)


### UMLAL ###

Unsigned long multiply-add by scalar element.

    void umlal(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### UMLAL ###

Unsigned long multiply-add.

    void umlal(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMLAL2 ###

Unsigned long multiply-add (second part).

    void umlal2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMLAL2 ###

Unsigned long multiply-add by scalar element (second part).

    void umlal2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### UMLSL ###

Unsigned long multiply-sub by scalar element.

    void umlsl(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### UMLSL ###

Unsigned long multiply-sub.

    void umlsl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMLSL2 ###

Unsigned long multiply-sub (second part).

    void umlsl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMLSL2 ###

Unsigned long multiply-sub by scalar element (second part).

    void umlsl2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### UMMLA ###

Unsigned 8-bit integer matrix multiply-accumulate (vector).

    void ummla(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMOV ###

Unsigned move vector element to general-purpose register.

    void umov(const Register& rd, const VRegister& vn, int vn_index)


### UMULL ###

Unsigned long multiply by scalar element.

    void umull(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### UMULL ###

Unsigned long multiply long.

    void umull(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMULL2 ###

Unsigned long multiply (second part).

    void umull2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UMULL2 ###

Unsigned long multiply by scalar element (second part).

    void umull2(const VRegister& vd,
                const VRegister& vn,
                const VRegister& vm,
                int vm_index)


### UQADD ###

Unsigned saturating add.

    void uqadd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UQRSHL ###

Unsigned saturating rounding shift left by register.

    void uqrshl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UQRSHRN ###

Unsigned saturating rounding shift right narrow by immediate.

    void uqrshrn(const VRegister& vd, const VRegister& vn, int shift)


### UQRSHRN2 ###

Unsigned saturating rounding shift right narrow by immediate (second part).

    void uqrshrn2(const VRegister& vd, const VRegister& vn, int shift)


### UQSHL ###

Unsigned saturating shift left by immediate.

    void uqshl(const VRegister& vd, const VRegister& vn, int shift)


### UQSHL ###

Unsigned saturating shift left by register.

    void uqshl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UQSHRN ###

Unsigned saturating shift right narrow by immediate.

    void uqshrn(const VRegister& vd, const VRegister& vn, int shift)


### UQSHRN2 ###

Unsigned saturating shift right narrow by immediate (second part).

    void uqshrn2(const VRegister& vd, const VRegister& vn, int shift)


### UQSUB ###

Unsigned saturating subtract.

    void uqsub(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UQXTN ###

Unsigned saturating extract narrow.

    void uqxtn(const VRegister& vd, const VRegister& vn)


### UQXTN2 ###

Unsigned saturating extract narrow (second part).

    void uqxtn2(const VRegister& vd, const VRegister& vn)


### URECPE ###

Unsigned reciprocal estimate.

    void urecpe(const VRegister& vd, const VRegister& vn)


### URHADD ###

Unsigned rounding halving add.

    void urhadd(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### URSHL ###

Unsigned rounding shift left by register.

    void urshl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### URSHR ###

Unsigned rounding shift right by immediate.

    void urshr(const VRegister& vd, const VRegister& vn, int shift)


### URSQRTE ###

Unsigned reciprocal square root estimate.

    void ursqrte(const VRegister& vd, const VRegister& vn)


### URSRA ###

Unsigned rounding shift right by immediate and accumulate.

    void ursra(const VRegister& vd, const VRegister& vn, int shift)


### USDOT ###

Dot Product with unsigned and signed integers (vector).

    void usdot(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### USDOT ###

Dot product with unsigned and signed integers (vector, by element).

    void usdot(const VRegister& vd,
               const VRegister& vn,
               const VRegister& vm,
               int vm_index)


### USHL ###

Unsigned shift left by register.

    void ushl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### USHLL ###

Unsigned shift left long by immediate.

    void ushll(const VRegister& vd, const VRegister& vn, int shift)


### USHLL2 ###

Unsigned shift left long by immediate (second part).

    void ushll2(const VRegister& vd, const VRegister& vn, int shift)


### USHR ###

Unsigned shift right by immediate.

    void ushr(const VRegister& vd, const VRegister& vn, int shift)


### USMMLA ###

Unsigned and signed 8-bit integer matrix multiply-accumulate (vector).

    void usmmla(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### USQADD ###

Unsigned saturating accumulate of signed value.

    void usqadd(const VRegister& vd, const VRegister& vn)


### USRA ###

Unsigned shift right by immediate and accumulate.

    void usra(const VRegister& vd, const VRegister& vn, int shift)


### USUBL ###

Unsigned subtract long.

    void usubl(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### USUBL2 ###

Unsigned subtract long (second part).

    void usubl2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### USUBW ###

Unsigned subtract wide.

    void usubw(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### USUBW2 ###

Unsigned subtract wide (second part).

    void usubw2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UXTL ###

Unsigned extend long.

    void uxtl(const VRegister& vd, const VRegister& vn)


### UXTL2 ###

Unsigned extend long (second part).

    void uxtl2(const VRegister& vd, const VRegister& vn)


### UZP1 ###

Unzip vectors (primary).

    void uzp1(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### UZP2 ###

Unzip vectors (secondary).

    void uzp2(const VRegister& vd, const VRegister& vn, const VRegister& vm)


<a id="float-x">

### XTN ###

Extract narrow.

    void xtn(const VRegister& vd, const VRegister& vn)


### XTN2 ###

Extract narrow (second part).

    void xtn2(const VRegister& vd, const VRegister& vn)


<a id="float-z">

### ZIP1 ###

Zip vectors (primary).

    void zip1(const VRegister& vd, const VRegister& vn, const VRegister& vm)


### ZIP2 ###

Zip vectors (secondary).

    void zip2(const VRegister& vd, const VRegister& vn, const VRegister& vm)



AArch64 Scalable Vector Extension (SVE) instructions
----------------------------------------------------

<a id="sve-a">

### ABS ###

Absolute value (predicated).

    void abs(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### ADCLB ###

Add with carry long (bottom).

    void adclb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### ADCLT ###

Add with carry long (top).

    void adclt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### ADD ###

Add immediate (unpredicated).

    void add(const ZRegister& zd, const ZRegister& zn, int imm8, int shift = -1)


### ADD ###

Add vectors (predicated).

    void add(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### ADD ###

Add vectors (unpredicated).

    void add(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### ADDHNB ###

Add narrow high part (bottom).

    void addhnb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### ADDHNT ###

Add narrow high part (top).

    void addhnt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### ADDP ###

Add pairwise.

    void addp(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### ADDPL ###

Add multiple of predicate register size to scalar register.

    void addpl(const Register& xd, const Register& xn, int imm6)


### ADDVL ###

Add multiple of vector register size to scalar register.

    void addvl(const Register& xd, const Register& xn, int imm6)


### ADR ###

Compute vector address.

    void adr(const ZRegister& zd, const SVEMemOperand& addr)


### AND ###

Bitwise AND predicates.

    void and_(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### AND ###

Bitwise AND vectors (predicated).

    void and_(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### AND ###

Bitwise AND vectors (unpredicated).

    void and_(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### AND ###

Bitwise AND with immediate (unpredicated).

    void and_(const ZRegister& zd, const ZRegister& zn, uint64_t imm)


### ANDS ###

Bitwise AND predicates.

    void ands(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### ANDV ###

Bitwise AND reduction to scalar.

    void andv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### ASR ###

Arithmetic shift right by 64-bit wide elements (predicated).

    void asr(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### ASR ###

Arithmetic shift right by 64-bit wide elements (unpredicated).

    void asr(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### ASR ###

Arithmetic shift right by immediate (predicated).

    void asr(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             int shift)


### ASR ###

Arithmetic shift right by immediate (unpredicated).

    void asr(const ZRegister& zd, const ZRegister& zn, int shift)


### ASRD ###

Arithmetic shift right for divide by immediate (predicated).

    void asrd(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              int shift)


### ASRR ###

Reversed arithmetic shift right by vector (predicated).

    void asrr(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


<a id="sve-b">

### BCAX ###

Bitwise clear and exclusive OR.

    void bcax(const ZRegister& zd,
              const ZRegister& zn,
              const ZRegister& zm,
              const ZRegister& zk)


### BDEP ###

Scatter lower bits into positions selected by bitmask.

    void bdep(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### BEXT ###

Gather lower bits from positions selected by bitmask.

    void bext(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### BGRP ###

Group bits to right or left as selected by bitmask.

    void bgrp(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### BIC ###

Bitwise clear bits using immediate (unpredicated).

    void bic(const ZRegister& zd, const ZRegister& zn, uint64_t imm)


### BIC ###

Bitwise clear predicates.

    void bic(const PRegisterWithLaneSize& pd,
             const PRegisterZ& pg,
             const PRegisterWithLaneSize& pn,
             const PRegisterWithLaneSize& pm)


### BIC ###

Bitwise clear vectors (predicated).

    void bic(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### BIC ###

Bitwise clear vectors (unpredicated).

    void bic(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### BICS ###

Bitwise clear predicates.

    void bics(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### BRKA ###

Break after first true condition.

    void brka(const PRegisterWithLaneSize& pd,
              const PRegister& pg,
              const PRegisterWithLaneSize& pn)


### BRKAS ###

Break after first true condition.

    void brkas(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const PRegisterWithLaneSize& pn)


### BRKB ###

Break before first true condition.

    void brkb(const PRegisterWithLaneSize& pd,
              const PRegister& pg,
              const PRegisterWithLaneSize& pn)


### BRKBS ###

Break before first true condition.

    void brkbs(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const PRegisterWithLaneSize& pn)


### BRKN ###

Propagate break to next partition.

    void brkn(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### BRKNS ###

Propagate break to next partition.

    void brkns(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const PRegisterWithLaneSize& pn,
               const PRegisterWithLaneSize& pm)


### BRKPA ###

Break after first true condition, propagating from previous partition.

    void brkpa(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const PRegisterWithLaneSize& pn,
               const PRegisterWithLaneSize& pm)


### BRKPAS ###

Break after first true condition, propagating from previous partition.

    void brkpas(const PRegisterWithLaneSize& pd,
                const PRegisterZ& pg,
                const PRegisterWithLaneSize& pn,
                const PRegisterWithLaneSize& pm)


### BRKPB ###

Break before first true condition, propagating from previous partition.

    void brkpb(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const PRegisterWithLaneSize& pn,
               const PRegisterWithLaneSize& pm)


### BRKPBS ###

Break before first true condition, propagating from previous partition.

    void brkpbs(const PRegisterWithLaneSize& pd,
                const PRegisterZ& pg,
                const PRegisterWithLaneSize& pn,
                const PRegisterWithLaneSize& pm)


### BSL ###

Bitwise select.

    void bsl(const ZRegister& zd,
             const ZRegister& zn,
             const ZRegister& zm,
             const ZRegister& zk)


### BSL1N ###

Bitwise select with first input inverted.

    void bsl1n(const ZRegister& zd,
               const ZRegister& zn,
               const ZRegister& zm,
               const ZRegister& zk)


### BSL2N ###

Bitwise select with second input inverted.

    void bsl2n(const ZRegister& zd,
               const ZRegister& zn,
               const ZRegister& zm,
               const ZRegister& zk)


<a id="sve-c">

### CADD ###

Complex integer add with rotate.

    void cadd(const ZRegister& zd,
              const ZRegister& zn,
              const ZRegister& zm,
              int rot)


### CDOT ###

Complex integer dot product (indexed).

    void cdot(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int index,
              int rot)


### CDOT ###

Complex integer dot product.

    void cdot(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int rot)


### CLASTA ###

Conditionally extract element after last to SIMD&FP scalar register.

    void clasta(const VRegister& vd,
                const PRegister& pg,
                const VRegister& vn,
                const ZRegister& zm)


### CLASTA ###

Conditionally extract element after last to general-purpose register.

    void clasta(const Register& rd,
                const PRegister& pg,
                const Register& rn,
                const ZRegister& zm)


### CLASTA ###

Conditionally extract element after last to vector register.

    void clasta(const ZRegister& zd,
                const PRegister& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### CLASTB ###

Conditionally extract last element to SIMD&FP scalar register.

    void clastb(const VRegister& vd,
                const PRegister& pg,
                const VRegister& vn,
                const ZRegister& zm)


### CLASTB ###

Conditionally extract last element to general-purpose register.

    void clastb(const Register& rd,
                const PRegister& pg,
                const Register& rn,
                const ZRegister& zm)


### CLASTB ###

Conditionally extract last element to vector register.

    void clastb(const ZRegister& zd,
                const PRegister& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### CLS ###

Count leading sign bits (predicated).

    void cls(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### CLZ ###

Count leading zero bits (predicated).

    void clz(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### CMLA ###

Complex integer multiply-add with rotate (indexed).

    void cmla(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int index,
              int rot)


### CMLA ###

Complex integer multiply-add with rotate.

    void cmla(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int rot)


### CMP ###



    void cmp(Condition cond,
             const PRegisterWithLaneSize& pd,
             const PRegisterZ& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### CMPEQ ###

Compare vector to 64-bit wide elements.

    void cmpeq(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPEQ ###

Compare vector to immediate.

    void cmpeq(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               int imm5)


### CMPGE ###

Compare vector to 64-bit wide elements.

    void cmpge(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPGE ###

Compare vector to immediate.

    void cmpge(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               int imm5)


### CMPGT ###

Compare vector to 64-bit wide elements.

    void cmpgt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPGT ###

Compare vector to immediate.

    void cmpgt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               int imm5)


### CMPHI ###

Compare vector to 64-bit wide elements.

    void cmphi(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPHI ###

Compare vector to immediate.

    void cmphi(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               unsigned imm7)


### CMPHS ###

Compare vector to 64-bit wide elements.

    void cmphs(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPHS ###

Compare vector to immediate.

    void cmphs(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               unsigned imm7)


### CMPLE ###

Compare vector to 64-bit wide elements.

    void cmple(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPLE ###

Compare vector to immediate.

    void cmple(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               int imm5)


### CMPLO ###

Compare vector to 64-bit wide elements.

    void cmplo(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPLO ###

Compare vector to immediate.

    void cmplo(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               unsigned imm7)


### CMPLS ###

Compare vector to 64-bit wide elements.

    void cmpls(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPLS ###

Compare vector to immediate.

    void cmpls(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               unsigned imm7)


### CMPLT ###

Compare vector to 64-bit wide elements.

    void cmplt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPLT ###

Compare vector to immediate.

    void cmplt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               int imm5)


### CMPNE ###

Compare vector to 64-bit wide elements.

    void cmpne(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### CMPNE ###

Compare vector to immediate.

    void cmpne(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               int imm5)


### CNOT ###

Logically invert boolean condition in vector (predicated).

    void cnot(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### CNT ###

Count non-zero bits (predicated).

    void cnt(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### CNTB ###

Set scalar to multiple of predicate constraint element count.

    void cntb(const Register& rd, int pattern = SVE_ALL, int multiplier = 1)


### CNTD ###

Set scalar to multiple of predicate constraint element count.

    void cntd(const Register& rd, int pattern = SVE_ALL, int multiplier = 1)


### CNTH ###

Set scalar to multiple of predicate constraint element count.

    void cnth(const Register& rd, int pattern = SVE_ALL, int multiplier = 1)


### CNTP ###

Set scalar to active predicate element count.

    void cntp(const Register& xd,
              const PRegister& pg,
              const PRegisterWithLaneSize& pn)


### CNTW ###

Set scalar to multiple of predicate constraint element count.

    void cntw(const Register& rd, int pattern = SVE_ALL, int multiplier = 1)


### COMPACT ###

Shuffle active elements of vector to the right and fill with zero.

    void compact(const ZRegister& zd, const PRegister& pg, const ZRegister& zn)


### CPY ###

Copy SIMD&FP scalar register to vector elements (predicated).

    void cpy(const ZRegister& zd, const PRegisterM& pg, const VRegister& vn)


### CPY ###

Copy general-purpose register to vector elements (predicated).

    void cpy(const ZRegister& zd, const PRegisterM& pg, const Register& rn)


### CPY ###

Copy signed integer immediate to vector elements (predicated).

    void cpy(const ZRegister& zd, const PRegister& pg, int imm8, int shift = -1)


### CTERMEQ ###

Compare and terminate loop.

    void ctermeq(const Register& rn, const Register& rm)


### CTERMNE ###

Compare and terminate loop.

    void ctermne(const Register& rn, const Register& rm)


<a id="sve-d">

### DECB ###

Decrement scalar by multiple of predicate constraint element count.

    void decb(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### DECD ###

Decrement scalar by multiple of predicate constraint element count.

    void decd(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### DECD ###

Decrement vector by multiple of predicate constraint element count.

    void decd(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### DECH ###

Decrement scalar by multiple of predicate constraint element count.

    void dech(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### DECH ###

Decrement vector by multiple of predicate constraint element count.

    void dech(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### DECP ###

Decrement scalar by active predicate element count.

    void decp(const Register& rdn, const PRegisterWithLaneSize& pg)


### DECP ###

Decrement vector by active predicate element count.

    void decp(const ZRegister& zdn, const PRegister& pg)


### DECW ###

Decrement scalar by multiple of predicate constraint element count.

    void decw(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### DECW ###

Decrement vector by multiple of predicate constraint element count.

    void decw(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### DUP ###

Broadcast general-purpose register to vector elements (unpredicated).

    void dup(const ZRegister& zd, const Register& xn)


### DUP ###

Broadcast indexed element to vector (unpredicated).

    void dup(const ZRegister& zd, const ZRegister& zn, unsigned index)


### DUP ###

Broadcast signed immediate to vector elements (unpredicated).

    void dup(const ZRegister& zd, int imm8, int shift = -1)


### DUPM ###

Broadcast logical bitmask immediate to vector (unpredicated).

    void dupm(const ZRegister& zd, uint64_t imm)


<a id="sve-e">

### EON ###

Bitwise exclusive OR with inverted immediate (unpredicated).

    void eon(const ZRegister& zd, const ZRegister& zn, uint64_t imm)


### EOR ###

Bitwise exclusive OR predicates.

    void eor(const PRegisterWithLaneSize& pd,
             const PRegisterZ& pg,
             const PRegisterWithLaneSize& pn,
             const PRegisterWithLaneSize& pm)


### EOR ###

Bitwise exclusive OR vectors (predicated).

    void eor(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### EOR ###

Bitwise exclusive OR vectors (unpredicated).

    void eor(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### EOR ###

Bitwise exclusive OR with immediate (unpredicated).

    void eor(const ZRegister& zd, const ZRegister& zn, uint64_t imm)


### EOR3 ###

Bitwise exclusive OR of three vectors.

    void eor3(const ZRegister& zd,
              const ZRegister& zn,
              const ZRegister& zm,
              const ZRegister& zk)


### EORBT ###

Interleaving exclusive OR (bottom, top).

    void eorbt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### EORS ###

Bitwise exclusive OR predicates.

    void eors(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### EORTB ###

Interleaving exclusive OR (top, bottom).

    void eortb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### EORV ###

Bitwise XOR reduction to scalar.

    void eorv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### EXT ###

Extract vector from pair of vectors.

    void ext(const ZRegister& zd,
             const ZRegister& zn,
             const ZRegister& zm,
             unsigned offset)


<a id="sve-f">

### FABD ###

Floating-point absolute difference (predicated).

    void fabd(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FABS ###

Floating-point absolute value (predicated).

    void fabs(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FACGE ###

Floating-point absolute compare vectors.

    void facge(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FACGT ###

Floating-point absolute compare vectors.

    void facgt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FADD ###

Floating-point add immediate (predicated).

    void fadd(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              double imm)


### FADD ###

Floating-point add vector (predicated).

    void fadd(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FADD ###

Floating-point add vector (unpredicated).

    void fadd(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### FADDA ###

Floating-point add strictly-ordered reduction, accumulating in scalar.

    void fadda(const VRegister& vd,
               const PRegister& pg,
               const VRegister& vn,
               const ZRegister& zm)


### FADDP ###

Floating-point add pairwise.

    void faddp(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FADDV ###

Floating-point add recursive reduction to scalar.

    void faddv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### FCADD ###

Floating-point complex add with rotate (predicated).

    void fcadd(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm,
               int rot)


### FCMEQ ###

Floating-point compare vector with zero.

    void fcmeq(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               double zero)


### FCMEQ ###

Floating-point compare vectors.

    void fcmeq(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FCMGE ###

Floating-point compare vector with zero.

    void fcmge(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               double zero)


### FCMGE ###

Floating-point compare vectors.

    void fcmge(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FCMGT ###

Floating-point compare vector with zero.

    void fcmgt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               double zero)


### FCMGT ###

Floating-point compare vectors.

    void fcmgt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FCMLA ###

Floating-point complex multiply-add by indexed values with rotate.

    void fcmla(const ZRegister& zda,
               const ZRegister& zn,
               const ZRegister& zm,
               int index,
               int rot)


### FCMLA ###

Floating-point complex multiply-add with rotate (predicated).

    void fcmla(const ZRegister& zda,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm,
               int rot)


### FCMLE ###

Floating-point compare vector with zero.

    void fcmle(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               double zero)


### FCMLT ###

Floating-point compare vector with zero.

    void fcmlt(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               double zero)


### FCMNE ###

Floating-point compare vector with zero.

    void fcmne(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               double zero)


### FCMNE ###

Floating-point compare vectors.

    void fcmne(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FCMUO ###

Floating-point compare vectors.

    void fcmuo(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FCPY ###

Copy floating-point immediate to vector elements (predicated).

    void fcpy(const ZRegister& zd, const PRegisterM& pg, double imm)


### FCPY ###

Copy half-precision floating-point immediate to vector elements (predicated).

    void fcpy(const ZRegister& zd, const PRegisterM& pg, Float16 imm)


### FCVT ###

Floating-point convert precision (predicated).

    void fcvt(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FCVTLT ###

Floating-point up convert long (top, predicated).

    void fcvtlt(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FCVTNT ###

Floating-point down convert and narrow (top, predicated).

    void fcvtnt(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FCVTX ###

Floating-point down convert, rounding to odd (predicated).

    void fcvtx(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FCVTXNT ###

Floating-point down convert, rounding to odd (top, predicated).

    void fcvtxnt(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FCVTZS ###

Floating-point convert to signed integer, rounding toward zero (predicated).

    void fcvtzs(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FCVTZU ###

Floating-point convert to unsigned integer, rounding toward zero (predicated).

    void fcvtzu(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FDIV ###

Floating-point divide by vector (predicated).

    void fdiv(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FDIVR ###

Floating-point reversed divide by vector (predicated).

    void fdivr(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FDUP ###

Broadcast floating-point immediate to vector elements.

    void fdup(const ZRegister& zd, double imm)


### FDUP ###

Broadcast half-precision floating-point immediate to vector elements.

    void fdup(const ZRegister& zd, Float16 imm)


### FEXPA ###

Floating-point exponential accelerator.

    void fexpa(const ZRegister& zd, const ZRegister& zn)


### FLOGB ###

Floating-point base 2 logarithm as integer.

    void flogb(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FMAD ###

Floating-point fused multiply-add vectors (predicated), writing multiplicand [Zdn = Za + Zdn * Zm].

    void fmad(const ZRegister& zdn,
              const PRegisterM& pg,
              const ZRegister& zm,
              const ZRegister& za)


### FMAX ###

Floating-point maximum (predicated).

    void fmax(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FMAX ###

Floating-point maximum with immediate (predicated).

    void fmax(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              double imm)


### FMAXNM ###

Floating-point maximum number (predicated).

    void fmaxnm(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### FMAXNM ###

Floating-point maximum number with immediate (predicated).

    void fmaxnm(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                double imm)


### FMAXNMP ###

Floating-point maximum number pairwise.

    void fmaxnmp(const ZRegister& zd,
                 const PRegisterM& pg,
                 const ZRegister& zn,
                 const ZRegister& zm)


### FMAXNMV ###

Floating-point maximum number recursive reduction to scalar.

    void fmaxnmv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### FMAXP ###

Floating-point maximum pairwise.

    void fmaxp(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FMAXV ###

Floating-point maximum recursive reduction to scalar.

    void fmaxv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### FMIN ###

Floating-point minimum (predicated).

    void fmin(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FMIN ###

Floating-point minimum with immediate (predicated).

    void fmin(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              double imm)


### FMINNM ###

Floating-point minimum number (predicated).

    void fminnm(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### FMINNM ###

Floating-point minimum number with immediate (predicated).

    void fminnm(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                double imm)


### FMINNMP ###

Floating-point minimum number pairwise.

    void fminnmp(const ZRegister& zd,
                 const PRegisterM& pg,
                 const ZRegister& zn,
                 const ZRegister& zm)


### FMINNMV ###

Floating-point minimum number recursive reduction to scalar.

    void fminnmv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### FMINP ###

Floating-point minimum pairwise.

    void fminp(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FMINV ###

Floating-point minimum recursive reduction to scalar.

    void fminv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### FMLA ###

Floating-point fused multiply-add by indexed elements (Zda = Zda + Zn * Zm[indexed]).

    void fmla(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int index)


### FMLA ###

Floating-point fused multiply-add vectors (predicated), writing addend [Zda = Zda + Zn * Zm].

    void fmla(const ZRegister& zda,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FMLALB ###

Half-precision floating-point multiply-add long to single-precision (bottom).

    void fmlalb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### FMLALB ###

Half-precision floating-point multiply-add long to single-precision (bottom, indexed).

    void fmlalb(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### FMLALT ###

Half-precision floating-point multiply-add long to single-precision (top).

    void fmlalt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### FMLALT ###

Half-precision floating-point multiply-add long to single-precision (top, indexed).

    void fmlalt(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### FMLS ###

Floating-point fused multiply-subtract by indexed elements (Zda = Zda + -Zn * Zm[indexed]).

    void fmls(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int index)


### FMLS ###

Floating-point fused multiply-subtract vectors (predicated), writing addend [Zda = Zda + -Zn * Zm].

    void fmls(const ZRegister& zda,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FMLSLB ###

Half-precision floating-point multiply-subtract long from single-precision (bottom).

    void fmlslb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### FMLSLB ###

Half-precision floating-point multiply-subtract long from single-precision (bottom, indexed).

    void fmlslb(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### FMLSLT ###

Half-precision floating-point multiply-subtract long from single-precision (top).

    void fmlslt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### FMLSLT ###

Half-precision floating-point multiply-subtract long from single-precision (top, indexed).

    void fmlslt(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### FMMLA ###

Floating-point matrix multiply-accumulate.

    void fmmla(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### FMOV ###

Move 8-bit floating-point immediate to vector elements (predicated).

    void fmov(const ZRegister& zd, const PRegisterM& pg, double imm)


### FMOV ###

Move 8-bit floating-point immediate to vector elements (unpredicated).

    void fmov(const ZRegister& zd, double imm)


### FMSB ###

Floating-point fused multiply-subtract vectors (predicated), writing multiplicand [Zdn = Za + -Zdn * Zm].

    void fmsb(const ZRegister& zdn,
              const PRegisterM& pg,
              const ZRegister& zm,
              const ZRegister& za)


### FMUL ###

Floating-point multiply by immediate (predicated).

    void fmul(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              double imm)


### FMUL ###

Floating-point multiply by indexed elements.

    void fmul(const ZRegister& zd,
              const ZRegister& zn,
              const ZRegister& zm,
              unsigned index)


### FMUL ###

Floating-point multiply vectors (predicated).

    void fmul(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FMUL ###

Floating-point multiply vectors (unpredicated).

    void fmul(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### FMULX ###

Floating-point multiply-extended vectors (predicated).

    void fmulx(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FNEG ###

Floating-point negate (predicated).

    void fneg(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FNMAD ###

Floating-point negated fused multiply-add vectors (predicated), writing multiplicand [Zdn = -Za + -Zdn * Zm].

    void fnmad(const ZRegister& zdn,
               const PRegisterM& pg,
               const ZRegister& zm,
               const ZRegister& za)


### FNMLA ###

Floating-point negated fused multiply-add vectors (predicated), writing addend [Zda = -Zda + -Zn * Zm].

    void fnmla(const ZRegister& zda,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FNMLS ###

Floating-point negated fused multiply-subtract vectors (predicated), writing addend [Zda = -Zda + Zn * Zm].

    void fnmls(const ZRegister& zda,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FNMSB ###

Floating-point negated fused multiply-subtract vectors (predicated), writing multiplicand [Zdn = -Za + Zdn * Zm].

    void fnmsb(const ZRegister& zdn,
               const PRegisterM& pg,
               const ZRegister& zm,
               const ZRegister& za)


### FRECPE ###

Floating-point reciprocal estimate (unpredicated).

    void frecpe(const ZRegister& zd, const ZRegister& zn)


### FRECPS ###

Floating-point reciprocal step (unpredicated).

    void frecps(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### FRECPX ###

Floating-point reciprocal exponent (predicated).

    void frecpx(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRINTA ###

Floating-point round to integral value (predicated).

    void frinta(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRINTI ###

Floating-point round to integral value (predicated).

    void frinti(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRINTM ###

Floating-point round to integral value (predicated).

    void frintm(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRINTN ###

Floating-point round to integral value (predicated).

    void frintn(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRINTP ###

Floating-point round to integral value (predicated).

    void frintp(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRINTX ###

Floating-point round to integral value (predicated).

    void frintx(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRINTZ ###

Floating-point round to integral value (predicated).

    void frintz(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FRSQRTE ###

Floating-point reciprocal square root estimate (unpredicated).

    void frsqrte(const ZRegister& zd, const ZRegister& zn)


### FRSQRTS ###

Floating-point reciprocal square root step (unpredicated).

    void frsqrts(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### FSCALE ###

Floating-point adjust exponent by vector (predicated).

    void fscale(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### FSQRT ###

Floating-point square root (predicated).

    void fsqrt(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### FSUB ###

Floating-point subtract immediate (predicated).

    void fsub(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              double imm)


### FSUB ###

Floating-point subtract vectors (predicated).

    void fsub(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### FSUB ###

Floating-point subtract vectors (unpredicated).

    void fsub(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### FSUBR ###

Floating-point reversed subtract from immediate (predicated).

    void fsubr(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               double imm)


### FSUBR ###

Floating-point reversed subtract vectors (predicated).

    void fsubr(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### FTMAD ###

Floating-point trigonometric multiply-add coefficient.

    void ftmad(const ZRegister& zd,
               const ZRegister& zn,
               const ZRegister& zm,
               int imm3)


### FTSMUL ###

Floating-point trigonometric starting value.

    void ftsmul(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### FTSSEL ###

Floating-point trigonometric select coefficient.

    void ftssel(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


<a id="sve-h">

### HISTCNT ###

Count matching elements in vector.

    void histcnt(const ZRegister& zd,
                 const PRegisterZ& pg,
                 const ZRegister& zn,
                 const ZRegister& zm)


### HISTSEG ###

Count matching elements in vector segments.

    void histseg(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


<a id="sve-i">

### INCB ###

Increment scalar by multiple of predicate constraint element count.

    void incb(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### INCD ###

Increment scalar by multiple of predicate constraint element count.

    void incd(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### INCD ###

Increment vector by multiple of predicate constraint element count.

    void incd(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### INCH ###

Increment scalar by multiple of predicate constraint element count.

    void inch(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### INCH ###

Increment vector by multiple of predicate constraint element count.

    void inch(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### INCP ###

Increment scalar by active predicate element count.

    void incp(const Register& rdn, const PRegisterWithLaneSize& pg)


### INCP ###

Increment vector by active predicate element count.

    void incp(const ZRegister& zdn, const PRegister& pg)


### INCW ###

Increment scalar by multiple of predicate constraint element count.

    void incw(const Register& xdn, int pattern = SVE_ALL, int multiplier = 1)


### INCW ###

Increment vector by multiple of predicate constraint element count.

    void incw(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### INDEX ###

Create index starting from and incremented by general-purpose register.

    void index(const ZRegister& zd, const Register& rn, const Register& rm)


### INDEX ###

Create index starting from and incremented by immediate.

    void index(const ZRegister& zd, int start, int step)


### INDEX ###

Create index starting from general-purpose register and incremented by immediate.

    void index(const ZRegister& zd, const Register& rn, int imm5)


### INDEX ###

Create index starting from immediate and incremented by general-purpose register.

    void index(const ZRegister& zd, int imm5, const Register& rm)


### INSR ###

Insert SIMD&FP scalar register in shifted vector.

    void insr(const ZRegister& zdn, const VRegister& vm)


### INSR ###

Insert general-purpose register in shifted vector.

    void insr(const ZRegister& zdn, const Register& rm)


<a id="sve-l">

### LASTA ###

Extract element after last to SIMD&FP scalar register.

    void lasta(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### LASTA ###

Extract element after last to general-purpose register.

    void lasta(const Register& rd, const PRegister& pg, const ZRegister& zn)


### LASTB ###

Extract last element to SIMD&FP scalar register.

    void lastb(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### LASTB ###

Extract last element to general-purpose register.

    void lastb(const Register& rd, const PRegister& pg, const ZRegister& zn)


### LD1B ###

Contiguous/gather load bytes to vector.

    void ld1b(const ZRegister& zt,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD1D ###

Contiguous/gather load doublewords to vector.

    void ld1d(const ZRegister& zt,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD1H ###

Contiguous/gather load halfwords to vector.

    void ld1h(const ZRegister& zt,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD1RB ###

Load and broadcast unsigned byte to vector.

    void ld1rb(const ZRegister& zt,
               const PRegisterZ& pg,
               const SVEMemOperand& addr)


### LD1RD ###

Load and broadcast doubleword to vector.

    void ld1rd(const ZRegister& zt,
               const PRegisterZ& pg,
               const SVEMemOperand& addr)


### LD1RH ###

Load and broadcast unsigned halfword to vector.

    void ld1rh(const ZRegister& zt,
               const PRegisterZ& pg,
               const SVEMemOperand& addr)


### LD1ROB ###

Contiguous load and replicate thirty-two bytes.

    void ld1rob(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1ROD ###

Contiguous load and replicate four doublewords.

    void ld1rod(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1ROH ###

Contiguous load and replicate sixteen halfwords.

    void ld1roh(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1ROW ###

Contiguous load and replicate eight words.

    void ld1row(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RQB ###

Contiguous load and replicate sixteen bytes.

    void ld1rqb(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RQD ###

Contiguous load and replicate two doublewords.

    void ld1rqd(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RQH ###

Contiguous load and replicate eight halfwords.

    void ld1rqh(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RQW ###

Contiguous load and replicate four words.

    void ld1rqw(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RSB ###

Load and broadcast signed byte to vector.

    void ld1rsb(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RSH ###

Load and broadcast signed halfword to vector.

    void ld1rsh(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RSW ###

Load and broadcast signed word to vector.

    void ld1rsw(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LD1RW ###

Load and broadcast unsigned word to vector.

    void ld1rw(const ZRegister& zt,
               const PRegisterZ& pg,
               const SVEMemOperand& addr)


### LD1SB ###

Contiguous/gather load signed bytes to vector.

    void ld1sb(const ZRegister& zt,
               const PRegisterZ& pg,
               const SVEMemOperand& addr)


### LD1SH ###

Contiguous/gather load signed halfwords to vector.

    void ld1sh(const ZRegister& zt,
               const PRegisterZ& pg,
               const SVEMemOperand& addr)


### LD1SW ###

Contiguous/gather load signed words to vector.

    void ld1sw(const ZRegister& zt,
               const PRegisterZ& pg,
               const SVEMemOperand& addr)


### LD1W ###

Contiguous/gather load words to vector.

    void ld1w(const ZRegister& zt,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD2B ###

Contiguous load two-byte structures to two vectors.

    void ld2b(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD2D ###

Contiguous load two-doubleword structures to two vectors.

    void ld2d(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD2H ###

Contiguous load two-halfword structures to two vectors.

    void ld2h(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD2W ###

Contiguous load two-word structures to two vectors.

    void ld2w(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD3B ###

Contiguous load three-byte structures to three vectors.

    void ld3b(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD3D ###

Contiguous load three-doubleword structures to three vectors.

    void ld3d(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD3H ###

Contiguous load three-halfword structures to three vectors.

    void ld3h(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD3W ###

Contiguous load three-word structures to three vectors.

    void ld3w(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD4B ###

Contiguous load four-byte structures to four vectors.

    void ld4b(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD4D ###

Contiguous load four-doubleword structures to four vectors.

    void ld4d(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD4H ###

Contiguous load four-halfword structures to four vectors.

    void ld4h(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LD4W ###

Contiguous load four-word structures to four vectors.

    void ld4w(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegisterZ& pg,
              const SVEMemOperand& addr)


### LDFF1B ###

Contiguous load first-fault unsigned bytes to vector.

    void ldff1b(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDFF1B ###

Gather load first-fault unsigned bytes to vector (immediate index).

    void ldff1b(const ZRegister& zt,
                const PRegisterZ& pg,
                const ZRegister& zn,
                int imm5)


### LDFF1B ###

Gather load first-fault unsigned bytes to vector.

    void ldff1b(const ZRegister& zt,
                const PRegisterZ& pg,
                const Register& xn,
                const ZRegister& zm)


### LDFF1D ###

Contiguous load first-fault doublewords to vector.

    void ldff1d(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDFF1D ###

Gather load first-fault doublewords to vector (immediate index).

    void ldff1d(const ZRegister& zt,
                const PRegisterZ& pg,
                const ZRegister& zn,
                int imm5)


### LDFF1D ###

Gather load first-fault doublewords to vector (vector index).

    void ldff1d(const ZRegister& zt,
                const PRegisterZ& pg,
                const Register& xn,
                const ZRegister& zm)


### LDFF1H ###

Contiguous load first-fault unsigned halfwords to vector.

    void ldff1h(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDFF1H ###

Gather load first-fault unsigned halfwords to vector (immediate index).

    void ldff1h(const ZRegister& zt,
                const PRegisterZ& pg,
                const ZRegister& zn,
                int imm5)


### LDFF1H ###

Gather load first-fault unsigned halfwords to vector (vector index).

    void ldff1h(const ZRegister& zt,
                const PRegisterZ& pg,
                const Register& xn,
                const ZRegister& zm)


### LDFF1SB ###

Contiguous load first-fault signed bytes to vector.

    void ldff1sb(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDFF1SB ###

Gather load first-fault signed bytes to vector (immediate index).

    void ldff1sb(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const ZRegister& zn,
                 int imm5)


### LDFF1SB ###

Gather load first-fault signed bytes to vector (vector index).

    void ldff1sb(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const Register& xn,
                 const ZRegister& zm)


### LDFF1SH ###

Contiguous load first-fault signed halfwords to vector.

    void ldff1sh(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDFF1SH ###

Gather load first-fault signed halfwords to vector (immediate index).

    void ldff1sh(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const ZRegister& zn,
                 int imm5)


### LDFF1SH ###

Gather load first-fault signed halfwords to vector (vector index).

    void ldff1sh(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const Register& xn,
                 const ZRegister& zm)


### LDFF1SW ###

Contiguous load first-fault signed words to vector.

    void ldff1sw(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDFF1SW ###

Gather load first-fault signed words to vector (immediate index).

    void ldff1sw(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const ZRegister& zn,
                 int imm5)


### LDFF1SW ###

Gather load first-fault signed words to vector (vector index).

    void ldff1sw(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const Register& xn,
                 const ZRegister& zm)


### LDFF1W ###

Contiguous load first-fault unsigned words to vector.

    void ldff1w(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDFF1W ###

Gather load first-fault unsigned words to vector (immediate index).

    void ldff1w(const ZRegister& zt,
                const PRegisterZ& pg,
                const ZRegister& zn,
                int imm5)


### LDFF1W ###

Gather load first-fault unsigned words to vector (vector index).

    void ldff1w(const ZRegister& zt,
                const PRegisterZ& pg,
                const Register& xn,
                const ZRegister& zm)


### LDNF1B ###

Contiguous load non-fault unsigned bytes to vector (immediate index).

    void ldnf1b(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDNF1D ###

Contiguous load non-fault doublewords to vector (immediate index).

    void ldnf1d(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDNF1H ###

Contiguous load non-fault unsigned halfwords to vector (immediate index).

    void ldnf1h(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDNF1SB ###

Contiguous load non-fault signed bytes to vector (immediate index).

    void ldnf1sb(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDNF1SH ###

Contiguous load non-fault signed halfwords to vector (immediate index).

    void ldnf1sh(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDNF1SW ###

Contiguous load non-fault signed words to vector (immediate index).

    void ldnf1sw(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDNF1W ###

Contiguous load non-fault unsigned words to vector (immediate index).

    void ldnf1w(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDNT1B ###

Contiguous load non-temporal bytes to vector.

    void ldnt1b(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDNT1D ###

Contiguous load non-temporal doublewords to vector.

    void ldnt1d(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDNT1H ###

Contiguous load non-temporal halfwords to vector.

    void ldnt1h(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDNT1SB ###

Gather load non-temporal signed bytes.

    void ldnt1sb(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDNT1SH ###

Gather load non-temporal signed halfwords.

    void ldnt1sh(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDNT1SW ###

Gather load non-temporal signed words.

    void ldnt1sw(const ZRegister& zt,
                 const PRegisterZ& pg,
                 const SVEMemOperand& addr)


### LDNT1W ###

Contiguous load non-temporal words to vector.

    void ldnt1w(const ZRegister& zt,
                const PRegisterZ& pg,
                const SVEMemOperand& addr)


### LDR ###

Load SVE predicate/vector register.

    void ldr(const CPURegister& rt, const SVEMemOperand& addr)


### LSL ###

Logical shift left by 64-bit wide elements (predicated).

    void lsl(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### LSL ###

Logical shift left by 64-bit wide elements (unpredicated).

    void lsl(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### LSL ###

Logical shift left by immediate (predicated).

    void lsl(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             int shift)


### LSL ###

Logical shift left by immediate (unpredicated).

    void lsl(const ZRegister& zd, const ZRegister& zn, int shift)


### LSLR ###

Reversed logical shift left by vector (predicated).

    void lslr(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### LSR ###

Logical shift right by 64-bit wide elements (predicated).

    void lsr(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### LSR ###

Logical shift right by 64-bit wide elements (unpredicated).

    void lsr(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### LSR ###

Logical shift right by immediate (predicated).

    void lsr(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             int shift)


### LSR ###

Logical shift right by immediate (unpredicated).

    void lsr(const ZRegister& zd, const ZRegister& zn, int shift)


### LSRR ###

Reversed logical shift right by vector (predicated).

    void lsrr(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


<a id="sve-m">

### MAD ###

Multiply-add vectors (predicated), writing multiplicand [Zdn = Za + Zdn * Zm].

    void mad(const ZRegister& zdn,
             const PRegisterM& pg,
             const ZRegister& zm,
             const ZRegister& za)


### MATCH ###

Detect any matching elements, setting the condition flags.

    void match(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### MLA ###

Multiply-add to accumulator (indexed).

    void mla(const ZRegister& zda,
             const ZRegister& zn,
             const ZRegister& zm,
             int index)


### MLA ###

Multiply-add vectors (predicated), writing addend [Zda = Zda + Zn * Zm].

    void mla(const ZRegister& zda,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### MLS ###

Multiply-subtract from accumulator (indexed).

    void mls(const ZRegister& zda,
             const ZRegister& zn,
             const ZRegister& zm,
             int index)


### MLS ###

Multiply-subtract vectors (predicated), writing addend [Zda = Zda - Zn * Zm].

    void mls(const ZRegister& zda,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### MOV ###

Move SIMD&FP scalar register to vector elements (predicated)

    void mov(const ZRegister& zd, const PRegisterM& pg, const VRegister& vn)


### MOV ###

Move SIMD&FP scalar register to vector elements (unpredicated)

    void mov(const ZRegister& zd, const VRegister& vn)


### MOV ###

Move general-purpose register to vector elements (predicated)

    void mov(const ZRegister& zd, const PRegisterM& pg, const Register& rn)


### MOV ###

Move general-purpose register to vector elements (unpredicated)

    void mov(const ZRegister& zd, const Register& xn)


### MOV ###

Move indexed element to vector elements (unpredicated)

    void mov(const ZRegister& zd, const ZRegister& zn, unsigned index)


### MOV ###

Move logical bitmask immediate to vector (unpredicated).

    void mov(const ZRegister& zd, uint64_t imm)


### MOV ###

Move predicates (merging)

    void mov(const PRegisterWithLaneSize& pd,
             const PRegisterM& pg,
             const PRegisterWithLaneSize& pn)


### MOV ###

Move predicates (unpredicated)

    void mov(const PRegister& pd, const PRegister& pn)


### MOV ###

Move predicates (zeroing)

    void mov(const PRegisterWithLaneSize& pd,
             const PRegisterZ& pg,
             const PRegisterWithLaneSize& pn)


### MOV ###

Move signed immediate to vector elements (unpredicated).

    void mov(const ZRegister& zd, int imm8, int shift)


### MOV ###

Move signed integer immediate to vector elements (predicated)

    void mov(const ZRegister& zd, const PRegister& pg, int imm8, int shift = -1)


### MOV ###

Move vector elements (predicated)

    void mov(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### MOV ###

Move vector register (unpredicated)

    void mov(const ZRegister& zd, const ZRegister& zn)


### MOVPRFX ###

Move prefix (predicated).

    void movprfx(const ZRegister& zd, const PRegister& pg, const ZRegister& zn)


### MOVPRFX ###

Move prefix (unpredicated).

    void movprfx(const ZRegister& zd, const ZRegister& zn)


### MOVS ###

Move predicate (unpredicated), setting the condition flags

    void movs(const PRegister& pd, const PRegister& pn)


### MOVS ###

Move predicates (zeroing), setting the condition flags

    void movs(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn)


### MSB ###

Multiply-subtract vectors (predicated), writing multiplicand [Zdn = Za - Zdn * Zm].

    void msb(const ZRegister& zdn,
             const PRegisterM& pg,
             const ZRegister& zm,
             const ZRegister& za)


### MUL ###

Multiply (indexed).

    void mul(const ZRegister& zd,
             const ZRegister& zn,
             const ZRegister& zm,
             int index)


### MUL ###

Multiply by immediate (unpredicated).

    void mul(const ZRegister& zd, const ZRegister& zn, int imm8)


### MUL ###

Multiply vectors (predicated).

    void mul(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### MUL ###

Multiply vectors (unpredicated).

    void mul(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


<a id="sve-n">

### NAND ###

Bitwise NAND predicates.

    void nand(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### NANDS ###

Bitwise NAND predicates.

    void nands(const PRegisterWithLaneSize& pd,
               const PRegisterZ& pg,
               const PRegisterWithLaneSize& pn,
               const PRegisterWithLaneSize& pm)


### NBSL ###

Bitwise inverted select.

    void nbsl(const ZRegister& zd,
              const ZRegister& zn,
              const ZRegister& zm,
              const ZRegister& zk)


### NEG ###

Negate (predicated).

    void neg(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### NMATCH ###

Detect no matching elements, setting the condition flags.

    void nmatch(const PRegisterWithLaneSize& pd,
                const PRegisterZ& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### NOR ###

Bitwise NOR predicates.

    void nor(const PRegisterWithLaneSize& pd,
             const PRegisterZ& pg,
             const PRegisterWithLaneSize& pn,
             const PRegisterWithLaneSize& pm)


### NORS ###

Bitwise NOR predicates.

    void nors(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### NOT ###

Bitwise invert predicate.

    void not_(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn)


### NOT ###

Bitwise invert vector (predicated).

    void not_(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### NOTS ###

Bitwise invert predicate, setting the condition flags.

    void nots(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn)


<a id="sve-o">

### ORN ###

Bitwise OR inverted predicate.

    void orn(const PRegisterWithLaneSize& pd,
             const PRegisterZ& pg,
             const PRegisterWithLaneSize& pn,
             const PRegisterWithLaneSize& pm)


### ORN ###

Bitwise OR with inverted immediate (unpredicated).

    void orn(const ZRegister& zd, const ZRegister& zn, uint64_t imm)


### ORNS ###

Bitwise OR inverted predicate.

    void orns(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### ORR ###

Bitwise OR predicate.

    void orr(const PRegisterWithLaneSize& pd,
             const PRegisterZ& pg,
             const PRegisterWithLaneSize& pn,
             const PRegisterWithLaneSize& pm)


### ORR ###

Bitwise OR vectors (predicated).

    void orr(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### ORR ###

Bitwise OR vectors (unpredicated).

    void orr(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### ORR ###

Bitwise OR with immediate (unpredicated).

    void orr(const ZRegister& zd, const ZRegister& zn, uint64_t imm)


### ORRS ###

Bitwise OR predicate.

    void orrs(const PRegisterWithLaneSize& pd,
              const PRegisterZ& pg,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### ORV ###

Bitwise OR reduction to scalar.

    void orv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


<a id="sve-p">

### PFALSE ###

Set all predicate elements to false.

    void pfalse(const PRegisterWithLaneSize& pd)


### PFIRST ###

Set the first active predicate element to true.

    void pfirst(const PRegisterWithLaneSize& pd,
                const PRegister& pg,
                const PRegisterWithLaneSize& pn)


### PMUL ###

Polynomial multiply vectors (unpredicated).

    void pmul(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### PMULLB ###

Polynomial multiply long (bottom).

    void pmullb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### PMULLT ###

Polynomial multiply long (top).

    void pmullt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### PNEXT ###

Find next active predicate.

    void pnext(const PRegisterWithLaneSize& pd,
               const PRegister& pg,
               const PRegisterWithLaneSize& pn)


### PRFB ###

Prefetch bytes.

    void prfb(PrefetchOperation prfop,
              const PRegister& pg,
              const SVEMemOperand& addr)


### PRFD ###

Prefetch doublewords.

    void prfd(PrefetchOperation prfop,
              const PRegister& pg,
              const SVEMemOperand& addr)


### PRFH ###

Prefetch halfwords.

    void prfh(PrefetchOperation prfop,
              const PRegister& pg,
              const SVEMemOperand& addr)


### PRFW ###

Prefetch words.

    void prfw(PrefetchOperation prfop,
              const PRegister& pg,
              const SVEMemOperand& addr)


### PTEST ###

Set condition flags for predicate.

    void ptest(const PRegister& pg, const PRegisterWithLaneSize& pn)


### PTRUE ###

Initialise predicate from named constraint.

    void ptrue(const PRegisterWithLaneSize& pd, int pattern = SVE_ALL)


### PTRUES ###

Initialise predicate from named constraint.

    void ptrues(const PRegisterWithLaneSize& pd, int pattern = SVE_ALL)


### PUNPKHI ###

Unpack and widen half of predicate.

    void punpkhi(const PRegisterWithLaneSize& pd,
                 const PRegisterWithLaneSize& pn)


### PUNPKLO ###

Unpack and widen half of predicate.

    void punpklo(const PRegisterWithLaneSize& pd,
                 const PRegisterWithLaneSize& pn)


<a id="sve-r">

### RADDHNB ###

Rounding add narrow high part (bottom).

    void raddhnb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### RADDHNT ###

Rounding add narrow high part (top).

    void raddhnt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### RBIT ###

Reverse bits (predicated).

    void rbit(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### RDFFR ###

Read the first-fault register.

    void rdffr(const PRegisterWithLaneSize& pd)


### RDFFR ###

Return predicate of succesfully loaded elements.

    void rdffr(const PRegisterWithLaneSize& pd, const PRegisterZ& pg)


### RDFFRS ###

Return predicate of succesfully loaded elements.

    void rdffrs(const PRegisterWithLaneSize& pd, const PRegisterZ& pg)


### RDVL ###

Read multiple of vector register size to scalar register.

    void rdvl(const Register& xd, int imm6)


### REV ###

Reverse all elements in a predicate.

    void rev(const PRegisterWithLaneSize& pd, const PRegisterWithLaneSize& pn)


### REV ###

Reverse all elements in a vector (unpredicated).

    void rev(const ZRegister& zd, const ZRegister& zn)


### REVB ###

Reverse bytes / halfwords / words within elements (predicated).

    void revb(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### REVH ###

Reverse bytes / halfwords / words within elements (predicated).

    void revh(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### REVW ###

Reverse bytes / halfwords / words within elements (predicated).

    void revw(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### RSHRNB ###

Rounding shift right narrow by immediate (bottom).

    void rshrnb(const ZRegister& zd, const ZRegister& zn, int shift)


### RSHRNT ###

Rounding shift right narrow by immediate (top).

    void rshrnt(const ZRegister& zd, const ZRegister& zn, int shift)


### RSUBHNB ###

Rounding subtract narrow high part (bottom).

    void rsubhnb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### RSUBHNT ###

Rounding subtract narrow high part (top).

    void rsubhnt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


<a id="sve-s">

### SABA ###

Signed absolute difference and accumulate.

    void saba(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SABALB ###

Signed absolute difference and accumulate long (bottom).

    void sabalb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SABALT ###

Signed absolute difference and accumulate long (top).

    void sabalt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SABD ###

Signed absolute difference (predicated).

    void sabd(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### SABDLB ###

Signed absolute difference long (bottom).

    void sabdlb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SABDLT ###

Signed absolute difference long (top).

    void sabdlt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SADALP ###

Signed add and accumulate long pairwise.

    void sadalp(const ZRegister& zda, const PRegisterM& pg, const ZRegister& zn)


### SADDLB ###

Signed add long (bottom).

    void saddlb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SADDLBT ###

Signed add long (bottom + top).

    void saddlbt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SADDLT ###

Signed add long (top).

    void saddlt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SADDV ###

Signed add reduction to scalar.

    void saddv(const VRegister& dd, const PRegister& pg, const ZRegister& zn)


### SADDWB ###

Signed add wide (bottom).

    void saddwb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SADDWT ###

Signed add wide (top).

    void saddwt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SBCLB ###

Subtract with carry long (bottom).

    void sbclb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SBCLT ###

Subtract with carry long (top).

    void sbclt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SCVTF ###

Signed integer convert to floating-point (predicated).

    void scvtf(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### SDIV ###

Signed divide (predicated).

    void sdiv(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### SDIVR ###

Signed reversed divide (predicated).

    void sdivr(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SDOT ###

Signed dot product by indexed quadtuplet.

    void sdot(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int index)


### SDOT ###

Signed dot product.

    void sdot(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SEL ###

Conditionally select elements from two predicates.

    void sel(const PRegisterWithLaneSize& pd,
             const PRegister& pg,
             const PRegisterWithLaneSize& pn,
             const PRegisterWithLaneSize& pm)


### SEL ###

Conditionally select elements from two vectors.

    void sel(const ZRegister& zd,
             const PRegister& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### SETFFR ###

Initialise the first-fault register to all true.

    void setffr()


### SHADD ###

Signed halving addition.

    void shadd(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SHRNB ###

Shift right narrow by immediate (bottom).

    void shrnb(const ZRegister& zd, const ZRegister& zn, int shift)


### SHRNT ###

Shift right narrow by immediate (top).

    void shrnt(const ZRegister& zd, const ZRegister& zn, int shift)


### SHSUB ###

Signed halving subtract.

    void shsub(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SHSUBR ###

Signed halving subtract reversed vectors.

    void shsubr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SLI ###

Shift left and insert (immediate).

    void sli(const ZRegister& zd, const ZRegister& zn, int shift)


### SMAX ###

Signed maximum vectors (predicated).

    void smax(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### SMAX ###

Signed maximum with immediate (unpredicated).

    void smax(const ZRegister& zd, const ZRegister& zn, int imm8)


### SMAXP ###

Signed maximum pairwise.

    void smaxp(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SMAXV ###

Signed maximum reduction to scalar.

    void smaxv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### SMIN ###

Signed minimum vectors (predicated).

    void smin(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### SMIN ###

Signed minimum with immediate (unpredicated).

    void smin(const ZRegister& zd, const ZRegister& zn, int imm8)


### SMINP ###

Signed minimum pairwise.

    void sminp(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SMINV ###

Signed minimum reduction to scalar.

    void sminv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### SMLALB ###

Signed multiply-add long to accumulator (bottom).

    void smlalb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SMLALB ###

Signed multiply-add long to accumulator (bottom, indexed).

    void smlalb(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### SMLALT ###

Signed multiply-add long to accumulator (top).

    void smlalt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SMLALT ###

Signed multiply-add long to accumulator (top, indexed).

    void smlalt(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### SMLSLB ###

Signed multiply-subtract long from accumulator (bottom).

    void smlslb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SMLSLB ###

Signed multiply-subtract long from accumulator (bottom, indexed).

    void smlslb(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### SMLSLT ###

Signed multiply-subtract long from accumulator (top).

    void smlslt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SMLSLT ###

Signed multiply-subtract long from accumulator (top, indexed).

    void smlslt(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### SMMLA ###

Signed integer matrix multiply-accumulate.

    void smmla(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SMULH ###

Signed multiply returning high half (predicated).

    void smulh(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SMULH ###

Signed multiply returning high half (unpredicated).

    void smulh(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SMULLB ###

Signed multiply long (bottom).

    void smullb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SMULLB ###

Signed multiply long (bottom, indexed).

    void smullb(const ZRegister& zd,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### SMULLT ###

Signed multiply long (top).

    void smullt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SMULLT ###

Signed multiply long (top, indexed).

    void smullt(const ZRegister& zd,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### SPLICE ###

Splice two vectors under predicate control.

    void splice(const ZRegister& zd,
                const PRegister& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SQABS ###

Signed saturating absolute value.

    void sqabs(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### SQADD ###

Signed saturating add immediate (unpredicated).

    void sqadd(const ZRegister& zd,
               const ZRegister& zn,
               int imm8,
               int shift = -1)


### SQADD ###

Signed saturating add vectors (unpredicated).

    void sqadd(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SQADD ###

Signed saturating addition (predicated).

    void sqadd(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SQCADD ###

Saturating complex integer add with rotate.

    void sqcadd(const ZRegister& zd,
                const ZRegister& zn,
                const ZRegister& zm,
                int rot)


### SQDECB ###

Signed saturating decrement scalar by multiple of 8-bit predicate constraint element count.

    void sqdecb(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQDECB ###

Signed saturating decrement scalar by multiple of 8-bit predicate constraint element count.

    void sqdecb(const Register& xd,
                const Register& wn,
                int pattern,
                int multiplier)


### SQDECD ###

Signed saturating decrement scalar by multiple of 64-bit predicate constraint element count.

    void sqdecd(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQDECD ###

Signed saturating decrement scalar by multiple of 64-bit predicate constraint element count.

    void sqdecd(const Register& xd,
                const Register& wn,
                int pattern = SVE_ALL,
                int multiplier = 1)


### SQDECD ###

Signed saturating decrement vector by multiple of 64-bit predicate constraint element count.

    void sqdecd(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### SQDECH ###

Signed saturating decrement scalar by multiple of 16-bit predicate constraint element count.

    void sqdech(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQDECH ###

Signed saturating decrement scalar by multiple of 16-bit predicate constraint element count.

    void sqdech(const Register& xd,
                const Register& wn,
                int pattern = SVE_ALL,
                int multiplier = 1)


### SQDECH ###

Signed saturating decrement vector by multiple of 16-bit predicate constraint element count.

    void sqdech(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### SQDECP ###

Signed saturating decrement scalar by active predicate element count.

    void sqdecp(const Register& xd,
                const PRegisterWithLaneSize& pg,
                const Register& wn)


### SQDECP ###

Signed saturating decrement scalar by active predicate element count.

    void sqdecp(const Register& xdn, const PRegisterWithLaneSize& pg)


### SQDECP ###

Signed saturating decrement vector by active predicate element count.

    void sqdecp(const ZRegister& zdn, const PRegister& pg)


### SQDECW ###

Signed saturating decrement scalar by multiple of 32-bit predicate constraint element count.

    void sqdecw(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQDECW ###

Signed saturating decrement scalar by multiple of 32-bit predicate constraint element count.

    void sqdecw(const Register& xd,
                const Register& wn,
                int pattern = SVE_ALL,
                int multiplier = 1)


### SQDECW ###

Signed saturating decrement vector by multiple of 32-bit predicate constraint element count.

    void sqdecw(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### SQDMLALB ###

Signed saturating doubling multiply-add long to accumulator (bottom).

    void sqdmlalb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SQDMLALB ###

Signed saturating doubling multiply-add long to accumulator (bottom, indexed).

    void sqdmlalb(const ZRegister& zda,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQDMLALBT ###

Signed saturating doubling multiply-add long to accumulator (bottom x top).

    void sqdmlalbt(const ZRegister& zda,
                   const ZRegister& zn,
                   const ZRegister& zm)


### SQDMLALT ###

Signed saturating doubling multiply-add long to accumulator (top).

    void sqdmlalt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SQDMLALT ###

Signed saturating doubling multiply-add long to accumulator (top, indexed).

    void sqdmlalt(const ZRegister& zda,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQDMLSLB ###

Signed saturating doubling multiply-subtract long from accumulator (bottom).

    void sqdmlslb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SQDMLSLB ###

Signed saturating doubling multiply-subtract long from accumulator (bottom, indexed).

    void sqdmlslb(const ZRegister& zda,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQDMLSLBT ###

Signed saturating doubling multiply-subtract long from accumulator (bottom x top).

    void sqdmlslbt(const ZRegister& zda,
                   const ZRegister& zn,
                   const ZRegister& zm)


### SQDMLSLT ###

Signed saturating doubling multiply-subtract long from accumulator (top).

    void sqdmlslt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SQDMLSLT ###

Signed saturating doubling multiply-subtract long from accumulator (top, indexed).

    void sqdmlslt(const ZRegister& zda,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQDMULH ###

Signed saturating doubling multiply high (indexed).

    void sqdmulh(const ZRegister& zd,
                 const ZRegister& zn,
                 const ZRegister& zm,
                 int index)


### SQDMULH ###

Signed saturating doubling multiply high (unpredicated).

    void sqdmulh(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SQDMULLB ###

Signed saturating doubling multiply long (bottom).

    void sqdmullb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SQDMULLB ###

Signed saturating doubling multiply long (bottom, indexed).

    void sqdmullb(const ZRegister& zd,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQDMULLT ###

Signed saturating doubling multiply long (top).

    void sqdmullt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SQDMULLT ###

Signed saturating doubling multiply long (top, indexed).

    void sqdmullt(const ZRegister& zd,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQINCB ###

Signed saturating increment scalar by multiple of 8-bit predicate constraint element count.

    void sqincb(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQINCB ###

Signed saturating increment scalar by multiple of 8-bit predicate constraint element count.

    void sqincb(const Register& xd,
                const Register& wn,
                int pattern = SVE_ALL,
                int multiplier = 1)


### SQINCD ###

Signed saturating increment scalar by multiple of 64-bit predicate constraint element count.

    void sqincd(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQINCD ###

Signed saturating increment scalar by multiple of 64-bit predicate constraint element count.

    void sqincd(const Register& xd,
                const Register& wn,
                int pattern,
                int multiplier)


### SQINCD ###

Signed saturating increment vector by multiple of 64-bit predicate constraint element count.

    void sqincd(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### SQINCH ###

Signed saturating increment scalar by multiple of 16-bit predicate constraint element count.

    void sqinch(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQINCH ###

Signed saturating increment scalar by multiple of 16-bit predicate constraint element count.

    void sqinch(const Register& xd,
                const Register& wn,
                int pattern = SVE_ALL,
                int multiplier = 1)


### SQINCH ###

Signed saturating increment vector by multiple of 16-bit predicate constraint element count.

    void sqinch(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### SQINCP ###

Signed saturating increment scalar by active predicate element count.

    void sqincp(const Register& xd,
                const PRegisterWithLaneSize& pg,
                const Register& wn)


### SQINCP ###

Signed saturating increment scalar by active predicate element count.

    void sqincp(const Register& xdn, const PRegisterWithLaneSize& pg)


### SQINCP ###

Signed saturating increment vector by active predicate element count.

    void sqincp(const ZRegister& zdn, const PRegister& pg)


### SQINCW ###

Signed saturating increment scalar by multiple of 32-bit predicate constraint element count.

    void sqincw(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### SQINCW ###

Signed saturating increment scalar by multiple of 32-bit predicate constraint element count.

    void sqincw(const Register& xd,
                const Register& wn,
                int pattern = SVE_ALL,
                int multiplier = 1)


### SQINCW ###

Signed saturating increment vector by multiple of 32-bit predicate constraint element count.

    void sqincw(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### SQNEG ###

Signed saturating negate.

    void sqneg(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### SQRDCMLAH ###

Saturating rounding doubling complex integer multiply-add high with rotate (indexed).

    void sqrdcmlah(const ZRegister& zda,
                   const ZRegister& zn,
                   const ZRegister& zm,
                   int index,
                   int rot)


### SQRDCMLAH ###

Saturating rounding doubling complex integer multiply-add high with rotate.

    void sqrdcmlah(const ZRegister& zda,
                   const ZRegister& zn,
                   const ZRegister& zm,
                   int rot)


### SQRDMLAH ###

Signed saturating rounding doubling multiply-add high to accumulator (indexed).

    void sqrdmlah(const ZRegister& zda,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQRDMLAH ###

Signed saturating rounding doubling multiply-add high to accumulator (unpredicated).

    void sqrdmlah(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SQRDMLSH ###

Signed saturating rounding doubling multiply-subtract high from accumulator (indexed).

    void sqrdmlsh(const ZRegister& zda,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQRDMLSH ###

Signed saturating rounding doubling multiply-subtract high from accumulator (unpredicated).

    void sqrdmlsh(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### SQRDMULH ###

Signed saturating rounding doubling multiply high (indexed).

    void sqrdmulh(const ZRegister& zd,
                  const ZRegister& zn,
                  const ZRegister& zm,
                  int index)


### SQRDMULH ###

Signed saturating rounding doubling multiply high (unpredicated).

    void sqrdmulh(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SQRSHL ###

Signed saturating rounding shift left by vector (predicated).

    void sqrshl(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SQRSHLR ###

Signed saturating rounding shift left reversed vectors (predicated).

    void sqrshlr(const ZRegister& zd,
                 const PRegisterM& pg,
                 const ZRegister& zn,
                 const ZRegister& zm)


### SQRSHRNB ###

Signed saturating rounding shift right narrow by immediate (bottom).

    void sqrshrnb(const ZRegister& zd, const ZRegister& zn, int shift)


### SQRSHRNT ###

Signed saturating rounding shift right narrow by immediate (top).

    void sqrshrnt(const ZRegister& zd, const ZRegister& zn, int shift)


### SQRSHRUNB ###

Signed saturating rounding shift right unsigned narrow by immediate (bottom).

    void sqrshrunb(const ZRegister& zd, const ZRegister& zn, int shift)


### SQRSHRUNT ###

Signed saturating rounding shift right unsigned narrow by immediate (top).

    void sqrshrunt(const ZRegister& zd, const ZRegister& zn, int shift)


### SQSHL ###

Signed saturating shift left by immediate.

    void sqshl(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               int shift)


### SQSHL ###

Signed saturating shift left by vector (predicated).

    void sqshl(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SQSHLR ###

Signed saturating shift left reversed vectors (predicated).

    void sqshlr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SQSHLU ###

Signed saturating shift left unsigned by immediate.

    void sqshlu(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                int shift)


### SQSHRNB ###

Signed saturating shift right narrow by immediate (bottom).

    void sqshrnb(const ZRegister& zd, const ZRegister& zn, int shift)


### SQSHRNT ###

Signed saturating shift right narrow by immediate (top).

    void sqshrnt(const ZRegister& zd, const ZRegister& zn, int shift)


### SQSHRUNB ###

Signed saturating shift right unsigned narrow by immediate (bottom).

    void sqshrunb(const ZRegister& zd, const ZRegister& zn, int shift)


### SQSHRUNT ###

Signed saturating shift right unsigned narrow by immediate (top).

    void sqshrunt(const ZRegister& zd, const ZRegister& zn, int shift)


### SQSUB ###

Signed saturating subtract immediate (unpredicated).

    void sqsub(const ZRegister& zd,
               const ZRegister& zn,
               int imm8,
               int shift = -1)


### SQSUB ###

Signed saturating subtract vectors (unpredicated).

    void sqsub(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SQSUB ###

Signed saturating subtraction (predicated).

    void sqsub(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SQSUBR ###

Signed saturating subtraction reversed vectors (predicated).

    void sqsubr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SQXTNB ###

Signed saturating extract narrow (bottom).

    void sqxtnb(const ZRegister& zd, const ZRegister& zn)


### SQXTNT ###

Signed saturating extract narrow (top).

    void sqxtnt(const ZRegister& zd, const ZRegister& zn)


### SQXTUNB ###

Signed saturating unsigned extract narrow (bottom).

    void sqxtunb(const ZRegister& zd, const ZRegister& zn)


### SQXTUNT ###

Signed saturating unsigned extract narrow (top).

    void sqxtunt(const ZRegister& zd, const ZRegister& zn)


### SRHADD ###

Signed rounding halving addition.

    void srhadd(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SRI ###

Shift right and insert (immediate).

    void sri(const ZRegister& zd, const ZRegister& zn, int shift)


### SRSHL ###

Signed rounding shift left by vector (predicated).

    void srshl(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### SRSHLR ###

Signed rounding shift left reversed vectors (predicated).

    void srshlr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SRSHR ###

Signed rounding shift right by immediate.

    void srshr(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               int shift)


### SRSRA ###

Signed rounding shift right and accumulate (immediate).

    void srsra(const ZRegister& zda, const ZRegister& zn, int shift)


### SSHLLB ###

Signed shift left long by immediate (bottom).

    void sshllb(const ZRegister& zd, const ZRegister& zn, int shift)


### SSHLLT ###

Signed shift left long by immediate (top).

    void sshllt(const ZRegister& zd, const ZRegister& zn, int shift)


### SSRA ###

Signed shift right and accumulate (immediate).

    void ssra(const ZRegister& zda, const ZRegister& zn, int shift)


### SSUBLB ###

Signed subtract long (bottom).

    void ssublb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SSUBLBT ###

Signed subtract long (bottom - top).

    void ssublbt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SSUBLT ###

Signed subtract long (top).

    void ssublt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SSUBLTB ###

Signed subtract long (top - bottom).

    void ssubltb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SSUBWB ###

Signed subtract wide (bottom).

    void ssubwb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SSUBWT ###

Signed subtract wide (top).

    void ssubwt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### ST1B ###

Contiguous/scatter store bytes from vector.

    void st1b(const ZRegister& zt,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST1D ###

Contiguous/scatter store doublewords from vector.

    void st1d(const ZRegister& zt,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST1H ###

Contiguous/scatter store halfwords from vector.

    void st1h(const ZRegister& zt,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST1W ###

Contiguous/scatter store words from vector.

    void st1w(const ZRegister& zt,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST2B ###

Contiguous store two-byte structures from two vectors.

    void st2b(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST2D ###

Contiguous store two-doubleword structures from two vectors,

    void st2d(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST2H ###

Contiguous store two-halfword structures from two vectors.

    void st2h(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST2W ###

Contiguous store two-word structures from two vectors.

    void st2w(const ZRegister& zt1,
              const ZRegister& zt2,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST3B ###

Contiguous store three-byte structures from three vectors.

    void st3b(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST3D ###

Contiguous store three-doubleword structures from three vectors.

    void st3d(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST3H ###

Contiguous store three-halfword structures from three vectors.

    void st3h(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST3W ###

Contiguous store three-word structures from three vectors.

    void st3w(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST4B ###

Contiguous store four-byte structures from four vectors.

    void st4b(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST4D ###

Contiguous store four-doubleword structures from four vectors.

    void st4d(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST4H ###

Contiguous store four-halfword structures from four vectors.

    void st4h(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegister& pg,
              const SVEMemOperand& addr)


### ST4W ###

Contiguous store four-word structures from four vectors.

    void st4w(const ZRegister& zt1,
              const ZRegister& zt2,
              const ZRegister& zt3,
              const ZRegister& zt4,
              const PRegister& pg,
              const SVEMemOperand& addr)


### STNT1B ###

Contiguous store non-temporal bytes from vector.

    void stnt1b(const ZRegister& zt,
                const PRegister& pg,
                const SVEMemOperand& addr)


### STNT1D ###

Contiguous store non-temporal doublewords from vector.

    void stnt1d(const ZRegister& zt,
                const PRegister& pg,
                const SVEMemOperand& addr)


### STNT1H ###

Contiguous store non-temporal halfwords from vector.

    void stnt1h(const ZRegister& zt,
                const PRegister& pg,
                const SVEMemOperand& addr)


### STNT1W ###

Contiguous store non-temporal words from vector.

    void stnt1w(const ZRegister& zt,
                const PRegister& pg,
                const SVEMemOperand& addr)


### STR ###

Store SVE predicate/vector register.

    void str(const CPURegister& rt, const SVEMemOperand& addr)


### SUB ###

Subtract immediate (unpredicated).

    void sub(const ZRegister& zd, const ZRegister& zn, int imm8, int shift = -1)


### SUB ###

Subtract vectors (predicated).

    void sub(const ZRegister& zd,
             const PRegisterM& pg,
             const ZRegister& zn,
             const ZRegister& zm)


### SUB ###

Subtract vectors (unpredicated).

    void sub(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SUBHNB ###

Subtract narrow high part (bottom).

    void subhnb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SUBHNT ###

Subtract narrow high part (top).

    void subhnt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### SUBR ###

Reversed subtract from immediate (unpredicated).

    void subr(const ZRegister& zd, const ZRegister& zn, int imm8, int shift = -1)


### SUBR ###

Reversed subtract vectors (predicated).

    void subr(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### SUDOT ###

Signed by unsigned integer indexed dot product.

    void sudot(const ZRegister& zda,
               const ZRegister& zn,
               const ZRegister& zm,
               int index)


### SUNPKHI ###

Signed unpack and extend half of vector.

    void sunpkhi(const ZRegister& zd, const ZRegister& zn)


### SUNPKLO ###

Signed unpack and extend half of vector.

    void sunpklo(const ZRegister& zd, const ZRegister& zn)


### SUQADD ###

Signed saturating addition of unsigned value.

    void suqadd(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### SXTB ###

Signed byte extend (predicated).

    void sxtb(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### SXTH ###

Signed halfword extend (predicated).

    void sxth(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### SXTW ###

Signed word extend (predicated).

    void sxtw(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


<a id="sve-t">

### TBL ###

Programmable table lookup in one or two vector table (zeroing).

    void tbl(const ZRegister& zd,
             const ZRegister& zn1,
             const ZRegister& zn2,
             const ZRegister& zm)


### TBL ###

Programmable table lookup/permute using vector of indices into a vector.

    void tbl(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### TBX ###

Programmable table lookup in single vector table (merging).

    void tbx(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### TRN1 ###

Interleave even or odd elements from two predicates.

    void trn1(const PRegisterWithLaneSize& pd,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### TRN1 ###

Interleave even or odd elements from two vectors.

    void trn1(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### TRN2 ###

Interleave even or odd elements from two predicates.

    void trn2(const PRegisterWithLaneSize& pd,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### TRN2 ###

Interleave even or odd elements from two vectors.

    void trn2(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


<a id="sve-u">

### UABA ###

Unsigned absolute difference and accumulate.

    void uaba(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UABALB ###

Unsigned absolute difference and accumulate long (bottom).

    void uabalb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UABALT ###

Unsigned absolute difference and accumulate long (top).

    void uabalt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UABD ###

Unsigned absolute difference (predicated).

    void uabd(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### UABDLB ###

Unsigned absolute difference long (bottom).

    void uabdlb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UABDLT ###

Unsigned absolute difference long (top).

    void uabdlt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UADALP ###

Unsigned add and accumulate long pairwise.

    void uadalp(const ZRegister& zda, const PRegisterM& pg, const ZRegister& zn)


### UADDLB ###

Unsigned add long (bottom).

    void uaddlb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UADDLT ###

Unsigned add long (top).

    void uaddlt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UADDV ###

Unsigned add reduction to scalar.

    void uaddv(const VRegister& dd, const PRegister& pg, const ZRegister& zn)


### UADDWB ###

Unsigned add wide (bottom).

    void uaddwb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UADDWT ###

Unsigned add wide (top).

    void uaddwt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UCVTF ###

Unsigned integer convert to floating-point (predicated).

    void ucvtf(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### UDIV ###

Unsigned divide (predicated).

    void udiv(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### UDIVR ###

Unsigned reversed divide (predicated).

    void udivr(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UDOT ###

Unsigned dot product by indexed quadtuplet.

    void udot(const ZRegister& zda,
              const ZRegister& zn,
              const ZRegister& zm,
              int index)


### UDOT ###

Unsigned dot product.

    void udot(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UHADD ###

Unsigned halving addition.

    void uhadd(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UHSUB ###

Unsigned halving subtract.

    void uhsub(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UHSUBR ###

Unsigned halving subtract reversed vectors.

    void uhsubr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### UMAX ###

Unsigned maximum vectors (predicated).

    void umax(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### UMAX ###

Unsigned maximum with immediate (unpredicated).

    void umax(const ZRegister& zd, const ZRegister& zn, int imm8)


### UMAXP ###

Unsigned maximum pairwise.

    void umaxp(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UMAXV ###

Unsigned maximum reduction to scalar.

    void umaxv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### UMIN ###

Unsigned minimum vectors (predicated).

    void umin(const ZRegister& zd,
              const PRegisterM& pg,
              const ZRegister& zn,
              const ZRegister& zm)


### UMIN ###

Unsigned minimum with immediate (unpredicated).

    void umin(const ZRegister& zd, const ZRegister& zn, int imm8)


### UMINP ###

Unsigned minimum pairwise.

    void uminp(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UMINV ###

Unsigned minimum reduction to scalar.

    void uminv(const VRegister& vd, const PRegister& pg, const ZRegister& zn)


### UMLALB ###

Unsigned multiply-add long to accumulator (bottom).

    void umlalb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UMLALB ###

Unsigned multiply-add long to accumulator (bottom, indexed).

    void umlalb(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### UMLALT ###

Unsigned multiply-add long to accumulator (top).

    void umlalt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UMLALT ###

Unsigned multiply-add long to accumulator (top, indexed).

    void umlalt(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### UMLSLB ###

Unsigned multiply-subtract long from accumulator (bottom).

    void umlslb(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UMLSLB ###

Unsigned multiply-subtract long from accumulator (bottom, indexed).

    void umlslb(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### UMLSLT ###

Unsigned multiply-subtract long from accumulator (top).

    void umlslt(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UMLSLT ###

Unsigned multiply-subtract long from accumulator (top, indexed).

    void umlslt(const ZRegister& zda,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### UMMLA ###

Unsigned integer matrix multiply-accumulate.

    void ummla(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### UMULH ###

Unsigned multiply returning high half (predicated).

    void umulh(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UMULH ###

Unsigned multiply returning high half (unpredicated).

    void umulh(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UMULLB ###

Unsigned multiply long (bottom).

    void umullb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UMULLB ###

Unsigned multiply long (bottom, indexed).

    void umullb(const ZRegister& zd,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### UMULLT ###

Unsigned multiply long (top).

    void umullt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UMULLT ###

Unsigned multiply long (top, indexed).

    void umullt(const ZRegister& zd,
                const ZRegister& zn,
                const ZRegister& zm,
                int index)


### UQADD ###

Unsigned saturating add immediate (unpredicated).

    void uqadd(const ZRegister& zd,
               const ZRegister& zn,
               int imm8,
               int shift = -1)


### UQADD ###

Unsigned saturating add vectors (unpredicated).

    void uqadd(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UQADD ###

Unsigned saturating addition (predicated).

    void uqadd(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UQDECB ###

Unsigned saturating decrement scalar by multiple of 8-bit predicate constraint element count.

    void uqdecb(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQDECD ###

Unsigned saturating decrement scalar by multiple of 64-bit predicate constraint element count.

    void uqdecd(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQDECD ###

Unsigned saturating decrement vector by multiple of 64-bit predicate constraint element count.

    void uqdecd(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### UQDECH ###

Unsigned saturating decrement scalar by multiple of 16-bit predicate constraint element count.

    void uqdech(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQDECH ###

Unsigned saturating decrement vector by multiple of 16-bit predicate constraint element count.

    void uqdech(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### UQDECP ###

Unsigned saturating decrement scalar by active predicate element count.

    void uqdecp(const Register& rdn, const PRegisterWithLaneSize& pg)


### UQDECP ###

Unsigned saturating decrement vector by active predicate element count.

    void uqdecp(const ZRegister& zdn, const PRegister& pg)


### UQDECW ###

Unsigned saturating decrement scalar by multiple of 32-bit predicate constraint element count.

    void uqdecw(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQDECW ###

Unsigned saturating decrement vector by multiple of 32-bit predicate constraint element count.

    void uqdecw(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### UQINCB ###

Unsigned saturating increment scalar by multiple of 8-bit predicate constraint element count.

    void uqincb(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQINCD ###

Unsigned saturating increment scalar by multiple of 64-bit predicate constraint element count.

    void uqincd(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQINCD ###

Unsigned saturating increment vector by multiple of 64-bit predicate constraint element count.

    void uqincd(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### UQINCH ###

Unsigned saturating increment scalar by multiple of 16-bit predicate constraint element count.

    void uqinch(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQINCH ###

Unsigned saturating increment vector by multiple of 16-bit predicate constraint element count.

    void uqinch(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### UQINCP ###

Unsigned saturating increment scalar by active predicate element count.

    void uqincp(const Register& rdn, const PRegisterWithLaneSize& pg)


### UQINCP ###

Unsigned saturating increment vector by active predicate element count.

    void uqincp(const ZRegister& zdn, const PRegister& pg)


### UQINCW ###

Unsigned saturating increment scalar by multiple of 32-bit predicate constraint element count.

    void uqincw(const Register& rdn, int pattern = SVE_ALL, int multiplier = 1)


### UQINCW ###

Unsigned saturating increment vector by multiple of 32-bit predicate constraint element count.

    void uqincw(const ZRegister& zdn, int pattern = SVE_ALL, int multiplier = 1)


### UQRSHL ###

Unsigned saturating rounding shift left by vector (predicated).

    void uqrshl(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### UQRSHLR ###

Unsigned saturating rounding shift left reversed vectors (predicated).

    void uqrshlr(const ZRegister& zd,
                 const PRegisterM& pg,
                 const ZRegister& zn,
                 const ZRegister& zm)


### UQRSHRNB ###

Unsigned saturating rounding shift right narrow by immediate (bottom).

    void uqrshrnb(const ZRegister& zd, const ZRegister& zn, int shift)


### UQRSHRNT ###

Unsigned saturating rounding shift right narrow by immediate (top).

    void uqrshrnt(const ZRegister& zd, const ZRegister& zn, int shift)


### UQSHL ###

Unsigned saturating shift left by immediate.

    void uqshl(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               int shift)


### UQSHL ###

Unsigned saturating shift left by vector (predicated).

    void uqshl(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UQSHLR ###

Unsigned saturating shift left reversed vectors (predicated).

    void uqshlr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### UQSHRNB ###

Unsigned saturating shift right narrow by immediate (bottom).

    void uqshrnb(const ZRegister& zd, const ZRegister& zn, int shift)


### UQSHRNT ###

Unsigned saturating shift right narrow by immediate (top).

    void uqshrnt(const ZRegister& zd, const ZRegister& zn, int shift)


### UQSUB ###

Unsigned saturating subtract immediate (unpredicated).

    void uqsub(const ZRegister& zd,
               const ZRegister& zn,
               int imm8,
               int shift = -1)


### UQSUB ###

Unsigned saturating subtract vectors (unpredicated).

    void uqsub(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UQSUB ###

Unsigned saturating subtraction (predicated).

    void uqsub(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### UQSUBR ###

Unsigned saturating subtraction reversed vectors (predicated).

    void uqsubr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### UQXTNB ###

Unsigned saturating extract narrow (bottom).

    void uqxtnb(const ZRegister& zd, const ZRegister& zn)


### UQXTNT ###

Unsigned saturating extract narrow (top).

    void uqxtnt(const ZRegister& zd, const ZRegister& zn)


### URECPE ###

Unsigned reciprocal estimate (predicated).

    void urecpe(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### URHADD ###

Unsigned rounding halving addition.

    void urhadd(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### URSHL ###

Unsigned rounding shift left by vector (predicated).

    void urshl(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               const ZRegister& zm)


### URSHLR ###

Unsigned rounding shift left reversed vectors (predicated).

    void urshlr(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### URSHR ###

Unsigned rounding shift right by immediate.

    void urshr(const ZRegister& zd,
               const PRegisterM& pg,
               const ZRegister& zn,
               int shift)


### URSQRTE ###

Unsigned reciprocal square root estimate (predicated).

    void ursqrte(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### URSRA ###

Unsigned rounding shift right and accumulate (immediate).

    void ursra(const ZRegister& zda, const ZRegister& zn, int shift)


### USDOT ###

Unsigned by signed integer dot product.

    void usdot(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### USDOT ###

Unsigned by signed integer indexed dot product.

    void usdot(const ZRegister& zda,
               const ZRegister& zn,
               const ZRegister& zm,
               int index)


### USHLLB ###

Unsigned shift left long by immediate (bottom).

    void ushllb(const ZRegister& zd, const ZRegister& zn, int shift)


### USHLLT ###

Unsigned shift left long by immediate (top).

    void ushllt(const ZRegister& zd, const ZRegister& zn, int shift)


### USMMLA ###

Unsigned by signed integer matrix multiply-accumulate.

    void usmmla(const ZRegister& zda, const ZRegister& zn, const ZRegister& zm)


### USQADD ###

Unsigned saturating addition of signed value.

    void usqadd(const ZRegister& zd,
                const PRegisterM& pg,
                const ZRegister& zn,
                const ZRegister& zm)


### USRA ###

Unsigned shift right and accumulate (immediate).

    void usra(const ZRegister& zda, const ZRegister& zn, int shift)


### USUBLB ###

Unsigned subtract long (bottom).

    void usublb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### USUBLT ###

Unsigned subtract long (top).

    void usublt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### USUBWB ###

Unsigned subtract wide (bottom).

    void usubwb(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### USUBWT ###

Unsigned subtract wide (top).

    void usubwt(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UUNPKHI ###

Unsigned unpack and extend half of vector.

    void uunpkhi(const ZRegister& zd, const ZRegister& zn)


### UUNPKLO ###

Unsigned unpack and extend half of vector.

    void uunpklo(const ZRegister& zd, const ZRegister& zn)


### UXTB ###

Unsigned byte extend (predicated).

    void uxtb(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### UXTH ###

Unsigned halfword extend (predicated).

    void uxth(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### UXTW ###

Unsigned word extend (predicated).

    void uxtw(const ZRegister& zd, const PRegisterM& pg, const ZRegister& zn)


### UZP1 ###

Concatenate even or odd elements from two predicates.

    void uzp1(const PRegisterWithLaneSize& pd,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### UZP1 ###

Concatenate even or odd elements from two vectors.

    void uzp1(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### UZP2 ###

Concatenate even or odd elements from two predicates.

    void uzp2(const PRegisterWithLaneSize& pd,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### UZP2 ###

Concatenate even or odd elements from two vectors.

    void uzp2(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


<a id="sve-w">

### WHILEGE ###

While decrementing signed scalar greater than or equal to scalar.

    void whilege(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILEGT ###

While decrementing signed scalar greater than scalar.

    void whilegt(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILEHI ###

While decrementing unsigned scalar higher than scalar.

    void whilehi(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILEHS ###

While decrementing unsigned scalar higher or same as scalar.

    void whilehs(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILELE ###

While incrementing signed scalar less than or equal to scalar.

    void whilele(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILELO ###

While incrementing unsigned scalar lower than scalar.

    void whilelo(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILELS ###

While incrementing unsigned scalar lower or same as scalar.

    void whilels(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILELT ###

While incrementing signed scalar less than scalar.

    void whilelt(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILERW ###

While free of read-after-write conflicts.

    void whilerw(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WHILEWR ###

While free of write-after-read/write conflicts.

    void whilewr(const PRegisterWithLaneSize& pd,
                 const Register& rn,
                 const Register& rm)


### WRFFR ###

Write the first-fault register.

    void wrffr(const PRegisterWithLaneSize& pn)


<a id="sve-x">

### XAR ###

Bitwise exclusive OR and rotate right by immediate.

    void xar(const ZRegister& zd,
             const ZRegister& zn,
             const ZRegister& zm,
             int shift)


<a id="sve-z">

### ZIP1 ###

Interleave elements from two half predicates.

    void zip1(const PRegisterWithLaneSize& pd,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### ZIP1 ###

Interleave elements from two half vectors.

    void zip1(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)


### ZIP2 ###

Interleave elements from two half predicates.

    void zip2(const PRegisterWithLaneSize& pd,
              const PRegisterWithLaneSize& pn,
              const PRegisterWithLaneSize& pm)


### ZIP2 ###

Interleave elements from two half vectors.

    void zip2(const ZRegister& zd, const ZRegister& zn, const ZRegister& zm)



Additional or pseudo instructions
---------------------------------

<a id="pseudo-b">

### BIND ###

Bind a label to the current PC.

    void bind(Label* label)


<a id="pseudo-d">

### DC ###

Emit data in the instruction stream.

    void dc(T data)


### DC32 ###

Emit 32 bits of data into the instruction stream.

    void dc32(uint32_t data)


### DC64 ###

Emit 64 bits of data into the instruction stream.

    void dc64(uint64_t data)


### DCI ###

Emit raw instructions into the instruction stream.

    void dci(Instr raw_inst)


<a id="pseudo-p">

### PLACE ###

Place a literal at the current PC.

    void place(RawLiteral* literal)



