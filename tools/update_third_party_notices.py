#!/usr/bin/env python3

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LICENSES = (
    ('LibAtrac9', 'vendor/libatrac9/LICENSE'),
    ('zlib', 'vendor/zlib/LICENSE'),
    ('Capstone', 'vendor/capstone/LICENSES/LICENSE.TXT'),
    ('Capstone LLVM components', 'vendor/capstone/LICENSES/LICENSE_LLVM.TXT'),
    ('Capstone BSD components', 'vendor/capstone/LICENSES/LICENSE_BSD_3_CLAUSE.txt'),
    ('Xbyak', 'vendor/xbyak/COPYRIGHT'),
    ('Equilibrium', 'vendor/equilibrium/LICENSE.md'),
    ('ICU components', 'vendor/equilibrium/base/external/icu/LICENSE'),
    ('xoshiro256**', 'vendor/equilibrium/base/external/xoshiro256ss/LICENSE'),
    ('Mbed TLS', 'vendor/mbedtls/LICENSE'),
    ('UnRAR', 'vendor/unrar/license.txt'),
    ('SPIRV-Cross', 'vendor/SPIRV-Cross/LICENSE'),
    ('Khronos headers', 'vendor/SPIRV-Cross/LICENSES/LicenseRef-KhronosFreeUse.txt'),
    ('Dear ImGui', 'vendor/equilibrium/external/tracy/imgui/LICENSE.txt'),
    ('Tracy profiler components', 'vendor/equilibrium/external/tracy/LICENSE'),
    ('Tracy client', 'vendor/tracy/LICENSE'),
    ('libbacktrace', 'vendor/tracy/public/libbacktrace/LICENSE'),
    ('GoogleTest (tests)', 'vendor/googletest/LICENSE'),
    ('FEX', 'vendor/fex/LICENSE'),
    ('FEXCore', 'vendor/fex/FEXCore/LICENSE'),
    ('fmt (FEX)', 'vendor/fex/External/fmt/LICENSE'),
    ('xxHash (FEX)', 'vendor/fex/External/xxhash/LICENSE'),
    ('robin-map (FEX)', 'vendor/fex/External/robin-map/LICENSE'),
    ('tiny-json (FEX)', 'vendor/fex/External/tiny-json/LICENSE'),
    ('range-v3 (FEX)', 'vendor/fex/External/range-v3/LICENSE.txt'),
    ('Cephes (FEX)', 'vendor/fex/External/cephes/LICENSE'),
    ('cpp-optparse (FEX)', 'vendor/fex/Source/Common/cpp-optparse/LICENSE'),
    ('VIXL (FEX)', 'vendor/fex/External/vixl/LICENCE'),
)

HEADERS = (
    ('SPIRV-Cross copyright', 'vendor/SPIRV-Cross/spirv_cross.cpp', 'first'),
    ('Khronos header copyright', 'vendor/SPIRV-Cross/spirv.hpp', 'lines'),
    ('Mbed TLS copyright', 'vendor/mbedtls/library/aes.c', 'first'),
    ('stb_image', 'vendor/equilibrium/external/tracy/profiler/src/stb_image.h', 'last'),
    ('LZ4 (Tracy)', 'vendor/tracy/public/common/tracy_lz4.hpp', 'first'),
    ('SoftFloat (FEX)', 'vendor/fex/External/SoftFloat-3e/include/SoftFloat-3e/softfloat.h', 'first'),
)


def main():
    sections = [
        'Third-party notices\n\n'
        'Prosperity\'s license is in LICENSE. Bundled components retain their own\n'
        'licenses and copyright notices, reproduced below. Inclusion depends on\n'
        'the platform and build options; test dependencies are also listed.\n\n'
        'Libraries supplied separately by the operating system or package manager\n'
        '(including SDL, Vulkan, OpenGL/EGL, FFmpeg, SPIRV-Tools, shaderc, vkd3d,\n'
        'and DXC) retain the notices supplied with their packages. If distributing\n'
        'those libraries with Prosperity, include their version-specific notices\n'
        'and any required source alongside this file.\n\n'
        'Regenerate after dependency updates: python3 tools/update_third_party_notices.py\n'
    ]
    for name, source in LICENSES:
        text = (ROOT / source).read_text()
        sections.append(f'\n{"=" * 72}\n{name}\nSource: {source}\n{"=" * 72}\n\n{text}\n')
    for name, source, position in HEADERS:
        text = (ROOT / source).read_text()
        if position == 'lines':
            notice = text.split('\n\n', 1)[0]
        else:
            start = text.rfind('/*') if position == 'last' else text.index('/*')
            end = text.index('*/', start) + 2
            notice = text[start:end]
        sections.append(f'\n{"=" * 72}\n{name}\nSource: {source}\n{"=" * 72}\n\n{notice}\n')
    (ROOT / 'THIRD_PARTY_NOTICES.txt').write_text(''.join(sections))


if __name__ == '__main__':
    main()
