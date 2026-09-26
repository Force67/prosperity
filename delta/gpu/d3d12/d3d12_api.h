#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The platform's D3D12 headers. Include after every standard library header:
 * the Windows compatibility layer defines min and max as macros.
 *
 * On Linux these are vkd3d's headers, which declare the same interfaces:
 *   - vkd3d_windows.h supplies the COM shims and must come first, or the
 *     generated header reaches for <windows.h>.
 *   - WIDL_EXPLICIT_AGGREGATE_RETURNS: a method returning a struct by value
 *     (GetCPUDescriptorHandleForHeapStart) is called through a vtable slot
 *     that takes a hidden out-pointer; without this the two disagree.
 *   - __uuidof does not compile under -std=c++20 here, so interfaces are
 *     queried with the IID_ID3D12Foo constants. Exactly one translation unit
 *     defines INITGUID before including this header (dxguid on Windows).
 */

#if defined(_WIN32)
#include <d3d12.h>
#include <dxgi1_6.h>
#else
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <vkd3d_windows.h>
#include <vkd3d_d3d12.h>
#include <vkd3d_utils.h>
#endif

#undef min
#undef max
