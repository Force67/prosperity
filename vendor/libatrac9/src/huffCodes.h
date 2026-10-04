#pragma once

#include "bit_reader.h"

typedef struct
{
	const unsigned char* Bits;
	const unsigned short* Codes;
	unsigned char* Lookup;
	const int Length;
	const int ValueCount;
	const int ValueCountPower;
	const int ValueBits;
	const int ValueMax;
	const int MaxBitSize;
} HuffmanCodebook;

int ReadHuffmanValue(const HuffmanCodebook* huff, BitReaderCxt* br, int isSigned);
void DecodeHuffmanValues(int* spectrum, int index, int bandCount, const HuffmanCodebook* huff, const int* values);
void InitHuffmanCodebook(const HuffmanCodebook* codebook);

extern HuffmanCodebook HuffmanScaleFactorsUnsigned[7];
extern HuffmanCodebook HuffmanScaleFactorsSigned[6];
extern HuffmanCodebook HuffmanSpectrum[2][8][4];
