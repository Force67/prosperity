/*
 * PS4Delta : PS4 emulation and research project
 *
 * NID -> handler table for HLE libfmod. NIDs captured from the game's import
 * list (DELTA_NID_TRACE=libfmod). All point at the generic FMOD_OK stub for
 * now; out-param-bearing calls (System_Create etc.) get real handlers as the
 * boot call sequence is mapped (DELTA_HLE_TRACE).
 */

#include "libfmod.h"

static const runtime::funcInfo functions[] = {
    {0x05d13cf8fca1a227, (void *)&fmodStub},
    {0x28ef2d6fc0e400cc, (void *)&fmodStub},
    {0x343310e59f107564, (void *)&fmodStub},
    {0x39c08767c25556a6, (void *)&fmodStub},
    {0x3aee0b1e89022b69, (void *)&fmodStub},
    {0x4a315d0939538a3c, (void *)&fmodStub},
    {0x4cbcf0af8532dfb8, (void *)&fmodStub},
    {0x53c0136ff78bb1f0, (void *)&fmodStub},
    {0x57a4b75d3f681334, (void *)&fmodStub},
    {0x5a972cd7ea851955, (void *)&fmodStub},
    {0x65f532d873b502fb, (void *)&fmodStub},
    {0x675ad8028508849e, (void *)&fmodStub},
    {0x7f67baf875528482, (void *)&fmodStub},
    {0x8224fdada3de03d6, (void *)&fmodStub},
    {0x8a13554f8763c6d3, (void *)&fmodStub},
    {0x8cbf8d8f8ccf7362, (void *)&fmodStub},
    {0x95c5bd457e7529e3, (void *)&fmodStub},
    {0x9bd795564d82e92a, (void *)&fmodStub},
    {0x9c915a1a6789beb8, (void *)&fmodStub},
    {0xa3685c2ba17c221e, (void *)&fmodStub},
    {0xa58980eeb87d8de6, (void *)&fmodStub},
    {0xa94a2565ed8d55ac, (void *)&fmodStub},
    {0xb1ec989091aa156b, (void *)&fmodStub},
    {0xb67f23029b98a20c, (void *)&fmodStub},
    {0xba95702b34f688fd, (void *)&fmodStub},
    {0xc7206cb9346da045, (void *)&fmodStub},
    {0xca29a14303752cc2, (void *)&fmodStub},
    {0xcba528ee30d7be10, (void *)&fmodStub},
    {0xd2401275feaede74, (void *)&fmodStub},
    {0xe44bf5fb31d56d76, (void *)&fmodStub},
    {0xeb7d118fc9a66b51, (void *)&fmodStub},
    {0xf0004274322a5c9b, (void *)&fmodStub},
    {0xf205582285fef486, (void *)&fmodStub},
    {0xf54483efd9cfab5c, (void *)&fmodStub},
    {0xf8684fb0b8472cbb, (void *)&fmodStub},
    {0xf8807c2f288abc81, (void *)&fmodStub},
    {0xfd96b528c3564714, (void *)&fmodStub},
};

MODULE_INIT(libfmod);

// Linker anchor (see vprx.cpp): without an external reference the whole _api.cpp
// object is dropped from the static lib and the module never self-registers.
extern "C" int vprx_anchor_libfmod = 1;
