/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#pragma once

// Tracy instrumentation. Every macro compiles away without DELTA_TRACY; with
// it they cost a flag check until a viewer or tracy-capture connects.
//
//   DELTA_ZONE("gpu.draw");            scope named by a literal
//   DELTA_ZONE_FN();                   scope named after the function
//   DELTA_ZONE_DYN(name);              scope named at run time (costs more)
//   DELTA_ZONE_TEXTF("n=%u", n);       annotate the innermost DELTA_ZONE
//   DELTA_ZONE_VALUE(u64);
//   DELTA_FRAME_MARK();                one guest frame presented
//   DELTA_PLOT("name", value);         a counter track
//   DELTA_THREAD_NAME("name");         label the calling thread
//   DELTA_MESSAGE(text, length);       a timeline message

#if defined(DELTA_TRACY)

#include <cstring>

#include "tracy/Tracy.hpp"

#define DELTA_ZONE(name) ZoneScopedN(name)
#define DELTA_ZONE_FN() ZoneScoped
#define DELTA_ZONE_DYN(name) ZoneTransientN(___tracy_scoped_zone, name, true)
#define DELTA_ZONE_TEXTF(...) ZoneTextF(__VA_ARGS__)
#define DELTA_ZONE_VALUE(value) ZoneValue(value)
#define DELTA_FRAME_MARK() FrameMark
#define DELTA_PLOT(name, value) TracyPlot(name, value)
#define DELTA_THREAD_NAME(name) tracy::SetThreadName(name)
#define DELTA_MESSAGE(text, length) TracyMessage(text, length)

#else

#define DELTA_ZONE(name)
#define DELTA_ZONE_FN()
#define DELTA_ZONE_DYN(name)
#define DELTA_ZONE_TEXTF(...)
#define DELTA_ZONE_VALUE(value)
#define DELTA_FRAME_MARK()
#define DELTA_PLOT(name, value)
#define DELTA_THREAD_NAME(name)
#define DELTA_MESSAGE(text, length)

#endif
