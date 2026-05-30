#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <utl/file.h>

namespace crypto {
#if 0

	// convert SELF to elf and return new image in out
	bool convert_self(utl::File&, std::vector<uint8_t> &&out);
	bool convert_self(utl::File&, const std::wstring& dest);
#endif
}