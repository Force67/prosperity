#pragma once

#include "unpack.h"

At9Status CreateGradient(Block* block);
void CalculateMask(Channel* channel);
void CalculatePrecisions(Channel* channel);
void GenerateGradientCurves();
