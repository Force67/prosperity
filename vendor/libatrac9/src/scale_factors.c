#include "scale_factors.h"
#include "huffCodes.h"
#include "tables.h"
#include "utility.h"
#include <string.h>

static void ReadClcOffset(Channel* channel, BitReaderCxt* br);
static void ReadVlcDeltaOffset(Channel* channel, BitReaderCxt* br);
static void ReadVlcDistanceToBaseline(Channel* channel, BitReaderCxt* br, const int* baseline, const int baselineLength);
static void ReadVlcDeltaOffsetWithBaseline(Channel* channel, BitReaderCxt* br, const int* baseline, const int baselineLength);

At9Status ReadScaleFactors(Channel * channel, BitReaderCxt * br)
{
	memset(channel->ScaleFactors, 0, sizeof(channel->ScaleFactors));

	channel->ScaleFactorCodingMode = ReadInt(br, 2);
	if (channel->ChannelIndex == 0)
	{
		switch (channel->ScaleFactorCodingMode)
		{
		case 0:
			ReadVlcDeltaOffset(channel, br);
			break;
		case 1:
			ReadClcOffset(channel, br);
			break;
		case 2:
			if (channel->Block->FirstInSuperframe) return ERR_UNPACK_SCALE_FACTOR_MODE_INVALID;
			ReadVlcDistanceToBaseline(channel, br, channel->ScaleFactorsPrev, channel->Block->QuantizationUnitsPrev);
			break;
		case 3:
			if (channel->Block->FirstInSuperframe) return ERR_UNPACK_SCALE_FACTOR_MODE_INVALID;
			ReadVlcDeltaOffsetWithBaseline(channel, br, channel->ScaleFactorsPrev, channel->Block->QuantizationUnitsPrev);
			break;
		}
	}
	else
	{
		switch (channel->ScaleFactorCodingMode)
		{
		case 0:
			ReadVlcDeltaOffset(channel, br);
			break;
		case 1:
			ReadVlcDistanceToBaseline(channel, br, channel->Block->Channels[0].ScaleFactors, channel->Block->ExtensionUnit);
			break;
		case 2:
			ReadVlcDeltaOffsetWithBaseline(channel, br, channel->Block->Channels[0].ScaleFactors, channel->Block->ExtensionUnit);
			break;
		case 3:
			if (channel->Block->FirstInSuperframe) return ERR_UNPACK_SCALE_FACTOR_MODE_INVALID;
			ReadVlcDistanceToBaseline(channel, br, channel->ScaleFactorsPrev, channel->Block->QuantizationUnitsPrev);
			break;
		}
	}

	for (int i = 0; i < channel->Block->ExtensionUnit; i++)
	{
		if (channel->ScaleFactors[i] < 0 || channel->ScaleFactors[i] > 31)
		{
			return ERR_UNPACK_SCALE_FACTOR_OOB;
		}
	}

	memcpy(channel->ScaleFactorsPrev, channel->ScaleFactors, sizeof(channel->ScaleFactors));

	return ERR_SUCCESS;
}

static void ReadClcOffset(Channel* channel, BitReaderCxt* br)
{
	const int maxBits = 5;
	int* sf = channel->ScaleFactors;
	const int bitLength = ReadInt(br, 2) + 2;
	const int baseValue = bitLength < maxBits ? ReadInt(br, maxBits) : 0;

	for (int i = 0; i < channel->Block->ExtensionUnit; i++)
	{
		sf[i] = ReadInt(br, bitLength) + baseValue;
	}
}

static void ReadVlcDeltaOffset(Channel* channel, BitReaderCxt* br)
{
	const int weightIndex = ReadInt(br, 3);
	const unsigned char* weights = ScaleFactorWeights[weightIndex];

	int* sf = channel->ScaleFactors;
	const int baseValue = ReadInt(br, 5);
	const int bitLength = ReadInt(br, 2) + 3;
	const HuffmanCodebook* codebook = &HuffmanScaleFactorsUnsigned[bitLength];

	sf[0] = ReadInt(br, bitLength);

	for (int i = 1; i < channel->Block->ExtensionUnit; i++)
	{
		const int delta = ReadHuffmanValue(codebook, br, 0);
		sf[i] = (sf[i - 1] + delta) & (codebook->ValueMax - 1);
	}

	for (int i = 0; i < channel->Block->ExtensionUnit; i++)
	{
		sf[i] += baseValue - weights[i];
	}
}

static void ReadVlcDistanceToBaseline(Channel* channel, BitReaderCxt* br, const int* baseline, const int baselineLength)
{
	int* sf = channel->ScaleFactors;
	const int bitLength = ReadInt(br, 2) + 2;
	const HuffmanCodebook* codebook = &HuffmanScaleFactorsSigned[bitLength];
	const int unitCount = Min(channel->Block->ExtensionUnit, baselineLength);

	for (int i = 0; i < unitCount; i++)
	{
		const int distance = ReadHuffmanValue(codebook, br, TRUE);
		sf[i] = (baseline[i] + distance) & 31;
	}

	for (int i = unitCount; i < channel->Block->ExtensionUnit; i++)
	{
		sf[i] = ReadInt(br, 5);
	}
}

static void ReadVlcDeltaOffsetWithBaseline(Channel* channel, BitReaderCxt* br, const int* baseline, const int baselineLength)
{
	int* sf = channel->ScaleFactors;
	const int baseValue = ReadOffsetBinary(br, 5);
	const int bitLength = ReadInt(br, 2) + 1;
	const HuffmanCodebook* codebook = &HuffmanScaleFactorsUnsigned[bitLength];
	const int unitCount = Min(channel->Block->ExtensionUnit, baselineLength);

	sf[0] = ReadInt(br, bitLength);

	for (int i = 1; i < unitCount; i++)
	{
		const int delta = ReadHuffmanValue(codebook, br, FALSE);
		sf[i] = (sf[i - 1] + delta) & (codebook->ValueMax - 1);
	}

	for (int i = 0; i < unitCount; i++)
	{
		sf[i] += baseValue + baseline[i];
	}

	for (int i = unitCount; i < channel->Block->ExtensionUnit; i++)
	{
		sf[i] = ReadInt(br, 5);
	}
}
