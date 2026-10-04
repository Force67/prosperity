#pragma once

#include "bit_reader.h"
#include "error_codes.h"
#include "structures.h"

At9Status ReadScaleFactors(Channel* channel, BitReaderCxt* br);
