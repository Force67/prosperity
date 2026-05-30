#pragma once

/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base/strings/xstring.h>

namespace utl {
base::StringW make_abs_path(const base::StringW &relative);
base::String make_abs_path(const base::String &relative);
}
