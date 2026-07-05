#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Forwarding header. The PS4 PM4/GCN command processor moved to gpu/ps4/; this
 * keeps the stable public include path ("gpu/cmd_processor.h") working for
 * external callers (the Gnm/VideoOut HLE, gc_dev/dce_dev) so their source is
 * untouched. The public gpu:: submit API is declared in ps4/cmd_processor.h.
 */

#include "ps4/cmd_processor.h"
