#pragma once

#if defined(_WIN32)
#include <d3d12.h>
#include <dxgi1_6.h>
#else
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
// clang-format off: vkd3d_windows.h must come first.
#include <vkd3d_windows.h>
#include <vkd3d_d3d12.h>
#include <vkd3d_utils.h>
// clang-format on
#endif

#undef min
#undef max
