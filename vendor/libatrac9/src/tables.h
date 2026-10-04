#pragma once

#include "structures.h"

extern const ChannelConfig ChannelConfigs[6];
extern const unsigned char MaxHuffPrecision[2];
extern const unsigned char MinBandCount[2];
extern const unsigned char MaxExtensionBand[2];
extern const unsigned char SamplingRateIndexToFrameSamplesPower[16];
extern const unsigned char MaxBandCount[16];
extern const unsigned char BandToQuantUnitCount[19];
extern const unsigned char QuantUnitToCoeffCount[30];
extern const int QuantUnitToCoeffIndex[31];
extern const unsigned char QuantUnitToCodebookIndex[30];
extern const int SampleRates[16];
extern const unsigned char ScaleFactorWeights[8][32];
extern const double SpectrumScale[32];
extern const double QuantizerInverseStepSize[16];
extern const double QuantizerStepSize[16];
extern const double QuantizerFineStepSize[16];

extern double MdctWindow[3][256];
extern double ImdctWindow[3][256];
extern double SinTables[9][256];
extern double CosTables[9][256];
extern int ShuffleTables[9][256];
