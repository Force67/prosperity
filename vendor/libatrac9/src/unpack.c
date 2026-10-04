#include "unpack.h"
#include "band_extension.h"
#include "bit_allocation.h"
#include "huffCodes.h"
#include "scale_factors.h"
#include "tables.h"
#include "utility.h"
#include <string.h>

static At9Status UnpackBlock(Block* block, BitReaderCxt* br);
static At9Status ReadBlockHeader(Block* block, BitReaderCxt* br);
static At9Status UnpackStandardBlock(Block* block, BitReaderCxt* br);
static At9Status ReadBandParams(Block* block, BitReaderCxt* br);
static At9Status ReadGradientParams(Block* block, BitReaderCxt* br);
static At9Status ReadStereoParams(Block* block, BitReaderCxt* br);
static At9Status ReadExtensionParams(Block* block, BitReaderCxt* br);
static void UpdateCodedUnits(Channel* channel);
static void CalculateSpectrumCodebookIndex(Channel* channel);

static At9Status ReadSpectra(Channel* channel, BitReaderCxt* br);
static At9Status ReadSpectraFine(Channel* channel, BitReaderCxt* br);

static At9Status UnpackLfeBlock(Block* block, BitReaderCxt* br);
static void DecodeLfeScaleFactors(Channel* channel, BitReaderCxt* br);
static void CalculateLfePrecision(Channel* channel);
static void ReadLfeSpectra(Channel* channel, BitReaderCxt* br);

At9Status UnpackFrame(Frame* frame, BitReaderCxt* br)
{
	const int blockCount = frame->Config->ChannelConfig.BlockCount;

	for (int i = 0; i < blockCount; i++)
	{
		ERROR_CHECK(UnpackBlock(&frame->Blocks[i], br));

		if (frame->Blocks[i].FirstInSuperframe && frame->IndexInSuperframe)
		{
			return ERR_UNPACK_SUPERFRAME_FLAG_INVALID;
		}
	}

	frame->IndexInSuperframe++;

	if (frame->IndexInSuperframe == frame->Config->FramesPerSuperframe)
	{
		frame->IndexInSuperframe = 0;
	}

	return ERR_SUCCESS;
}

static At9Status UnpackBlock(Block* block, BitReaderCxt* br)
{
	ERROR_CHECK(ReadBlockHeader(block, br));

	if (block->BlockType == LFE)
	{
		ERROR_CHECK(UnpackLfeBlock(block, br));
	}
	else
	{
		ERROR_CHECK(UnpackStandardBlock(block, br));
	}

	AlignPosition(br, 8);
	return ERR_SUCCESS;
}

static At9Status ReadBlockHeader(Block* block, BitReaderCxt* br)
{
	block->FirstInSuperframe = !ReadInt(br, 1);
	block->ReuseBandParams = ReadInt(br, 1);

	if (block->FirstInSuperframe && block->ReuseBandParams && block->BlockType != LFE)
	{
		return ERR_UNPACK_REUSE_BAND_PARAMS_INVALID;
	}

	return ERR_SUCCESS;
}

static At9Status UnpackStandardBlock(Block* block, BitReaderCxt* br)
{
	if (!block->ReuseBandParams)
	{
		ERROR_CHECK(ReadBandParams(block, br));
	}

	ERROR_CHECK(ReadGradientParams(block, br));
	ERROR_CHECK(CreateGradient(block));
	ERROR_CHECK(ReadStereoParams(block, br));
	ERROR_CHECK(ReadExtensionParams(block, br));

	for (int i = 0; i < block->ChannelCount; i++)
	{
		Channel* channel = &block->Channels[i];
		UpdateCodedUnits(channel);

		ERROR_CHECK(ReadScaleFactors(channel, br));
		CalculateMask(channel);
		CalculatePrecisions(channel);
		CalculateSpectrumCodebookIndex(channel);

		ERROR_CHECK(ReadSpectra(channel, br));
		ERROR_CHECK(ReadSpectraFine(channel, br));
	}

	block->QuantizationUnitsPrev = block->BandExtensionEnabled ? block->ExtensionUnit : block->QuantizationUnitCount;
	return ERR_SUCCESS;
}

static At9Status ReadBandParams(Block* block, BitReaderCxt* br)
{
	const int minBandCount = MinBandCount[block->Config->HighSampleRate];
	const int maxExtensionBand = MaxExtensionBand[block->Config->HighSampleRate];
	block->BandCount = ReadInt(br, 4) + minBandCount;
	block->QuantizationUnitCount = BandToQuantUnitCount[block->BandCount];

	if (block->BandCount > MaxBandCount[block->Config->SampleRateIndex])
	{
		return ERR_UNPACK_BAND_PARAMS_INVALID;
	}

	if (block->BlockType == Stereo)
	{
		block->StereoBand = ReadInt(br, 4);
		block->StereoBand += minBandCount;
		block->StereoQuantizationUnit = BandToQuantUnitCount[block->StereoBand];
	}
	else
	{
		block->StereoBand = block->BandCount;
	}

	if (block->StereoBand > block->BandCount)
	{
		return ERR_UNPACK_BAND_PARAMS_INVALID;
	}

	block->BandExtensionEnabled = ReadInt(br, 1);
	if (block->BandExtensionEnabled)
	{
		block->ExtensionBand = ReadInt(br, 4);
		block->ExtensionBand += minBandCount;

		if (block->ExtensionBand < block->BandCount || block->ExtensionBand > maxExtensionBand)
		{
			return ERR_UNPACK_BAND_PARAMS_INVALID;
		}

		block->ExtensionUnit = BandToQuantUnitCount[block->ExtensionBand];
	}
	else
	{
		block->ExtensionBand = block->BandCount;
		block->ExtensionUnit = block->QuantizationUnitCount;
	}

	return ERR_SUCCESS;
}

static At9Status ReadGradientParams(Block* block, BitReaderCxt* br)
{
	block->GradientMode = ReadInt(br, 2);
	if (block->GradientMode > 0)
	{
		block->GradientEndUnit = 31;
		block->GradientEndValue = 31;
		block->GradientStartUnit = ReadInt(br, 5);
		block->GradientStartValue = ReadInt(br, 5);
	}
	else
	{
		block->GradientStartUnit = ReadInt(br, 6);
		block->GradientEndUnit = ReadInt(br, 6) + 1;
		block->GradientStartValue = ReadInt(br, 5);
		block->GradientEndValue = ReadInt(br, 5);
	}
	block->GradientBoundary = ReadInt(br, 4);

	if (block->GradientBoundary > block->QuantizationUnitCount)
	{
		return ERR_UNPACK_GRAD_BOUNDARY_INVALID;
	}
	if (block->GradientStartUnit < 0 || block->GradientStartUnit >= 48)
	{
		return ERR_UNPACK_GRAD_START_UNIT_OOB;
	}
	if (block->GradientEndUnit < 0 || block->GradientEndUnit >= 48)
	{
		return ERR_UNPACK_GRAD_END_UNIT_OOB;
	}
	if (block->GradientStartUnit > block->GradientEndUnit)
	{
		return ERR_UNPACK_GRAD_END_UNIT_INVALID;
	}
	if (block->GradientStartValue < 0 || block->GradientStartValue >= 32)
	{
		return ERR_UNPACK_GRAD_START_VALUE_OOB;
	}
	if (block->GradientEndValue < 0 || block->GradientEndValue >= 32)
	{
		return ERR_UNPACK_GRAD_END_VALUE_OOB;
	}

	return ERR_SUCCESS;
}

static At9Status ReadStereoParams(Block* block, BitReaderCxt* br)
{
	if (block->BlockType != Stereo) return ERR_SUCCESS;

	block->PrimaryChannelIndex = ReadInt(br, 1);
	block->HasJointStereoSigns = ReadInt(br, 1);
	if (block->HasJointStereoSigns)
	{
		for (int i = block->StereoQuantizationUnit; i < block->QuantizationUnitCount; i++)
		{
			block->JointStereoSigns[i] = ReadInt(br, 1);
		}
	}
	else
	{
		memset(block->JointStereoSigns, 0, sizeof(block->JointStereoSigns));
	}

	return ERR_SUCCESS;
}

static void BexReadHeader(Channel* channel, BitReaderCxt* br, int bexBand)
{
	const int bexMode = ReadInt(br, 2);
	channel->BexMode = bexBand > 2 ? bexMode : 4;
	channel->BexValueCount = BexEncodedValueCounts[channel->BexMode][bexBand];
}

static void BexReadData(Channel* channel, BitReaderCxt* br, int bexBand)
{
	for (int i = 0; i < channel->BexValueCount; i++)
	{
		const int dataLength = BexDataLengths[channel->BexMode][bexBand][i];
		channel->BexValues[i] = ReadInt(br, dataLength);
	}
}

static At9Status ReadExtensionParams(Block* block, BitReaderCxt* br)
{
	int bexBand = 0;
	if (block->BandExtensionEnabled)
	{
		bexBand = BexGroupInfo[block->QuantizationUnitCount - 13].BandCount;
		if (block->BlockType == Stereo)
		{
			BexReadHeader(&block->Channels[1], br, bexBand);
		}
		else
		{
			br->Position += 1;
		}
	}
	block->HasExtensionData = ReadInt(br, 1);

	if (!block->HasExtensionData) return ERR_SUCCESS;
	if (!block->BandExtensionEnabled)
	{
		block->BexMode = ReadInt(br, 2);
		block->BexDataLength = ReadInt(br, 5);
		br->Position += block->BexDataLength;
		return ERR_SUCCESS;
	}

	BexReadHeader(&block->Channels[0], br, bexBand);

	block->BexDataLength = ReadInt(br, 5);
	if (block->BexDataLength == 0) return ERR_SUCCESS;
	const int bexDataEnd = br->Position + block->BexDataLength;

	BexReadData(&block->Channels[0], br, bexBand);

	if (block->BlockType == Stereo)
	{
		BexReadData(&block->Channels[1], br, bexBand);
	}

	// Make sure we didn't read too many bits
	if (br->Position > bexDataEnd)
	{
		return ERR_UNPACK_EXTENSION_DATA_INVALID;
	}

	return ERR_SUCCESS;
}

static void UpdateCodedUnits(Channel* channel)
{
	if (channel->Block->PrimaryChannelIndex == channel->ChannelIndex)
	{
		channel->CodedQuantUnits = channel->Block->QuantizationUnitCount;
	}
	else
	{
		channel->CodedQuantUnits = channel->Block->StereoQuantizationUnit;
	}
}

static void CalculateSpectrumCodebookIndex(Channel* channel)
{
	memset(channel->CodebookSet, 0, sizeof(channel->CodebookSet));
	const int quantUnits = channel->CodedQuantUnits;
	int* sf = channel->ScaleFactors;

	if (quantUnits <= 1) return;
	if (channel->Config->HighSampleRate) return;

	// Temporarily setting this value allows for simpler code by
	// making the last value a non-special case.
	const int originalScaleTmp = sf[quantUnits];
	sf[quantUnits] = sf[quantUnits - 1];

	int avg = 0;
	if (quantUnits > 12)
	{
		for (int i = 0; i < 12; i++)
		{
			avg += sf[i];
		}
		avg = (avg + 6) / 12;
	}

	for (int i = 8; i < quantUnits; i++)
	{
		const int prevSf = sf[i - 1];
		const int nextSf = sf[i + 1];
		const int minSf = Min(prevSf, nextSf);
		if (sf[i] - minSf >= 3 || sf[i] - prevSf + sf[i] - nextSf >= 3)
		{
			channel->CodebookSet[i] = 1;
		}
	}

	for (int i = 12; i < quantUnits; i++)
	{
		if (channel->CodebookSet[i] == 0)
		{
			const int minSf = Min(sf[i - 1], sf[i + 1]);
			if (sf[i] - minSf >= 2 && sf[i] >= avg - (QuantUnitToCoeffCount[i] == 16 ? 1 : 0))
			{
				channel->CodebookSet[i] = 1;
			}
		}
	}

	sf[quantUnits] = originalScaleTmp;
}

static At9Status ReadSpectra(Channel* channel, BitReaderCxt* br)
{
	int values[16];
	memset(channel->QuantizedSpectra, 0, sizeof(channel->QuantizedSpectra));
	const int maxHuffPrecision = MaxHuffPrecision[channel->Config->HighSampleRate];

	for (int i = 0; i < channel->CodedQuantUnits; i++)
	{
		const int subbandCount = QuantUnitToCoeffCount[i];
		const int precision = channel->Precisions[i] + 1;
		if (precision <= maxHuffPrecision)
		{
			const HuffmanCodebook* huff = &HuffmanSpectrum[channel->CodebookSet[i]][precision][QuantUnitToCodebookIndex[i]];
			const int groupCount = subbandCount >> huff->ValueCountPower;
			for (int j = 0; j < groupCount; j++)
			{
				values[j] = ReadHuffmanValue(huff, br, FALSE);
			}

			DecodeHuffmanValues(channel->QuantizedSpectra, QuantUnitToCoeffIndex[i], subbandCount, huff, values);
		}
		else
		{
			const int subbandIndex = QuantUnitToCoeffIndex[i];
			for (int j = subbandIndex; j < QuantUnitToCoeffIndex[i + 1]; j++)
			{
				channel->QuantizedSpectra[j] = ReadSignedInt(br, precision);
			}
		}
	}

	return ERR_SUCCESS;
}

static At9Status ReadSpectraFine(Channel* channel, BitReaderCxt* br)
{
	memset(channel->QuantizedSpectraFine, 0, sizeof(channel->QuantizedSpectraFine));

	for (int i = 0; i < channel->CodedQuantUnits; i++)
	{
		if (channel->PrecisionsFine[i] > 0)
		{
			const int overflowBits = channel->PrecisionsFine[i] + 1;
			const int startSubband = QuantUnitToCoeffIndex[i];
			const int endSubband = QuantUnitToCoeffIndex[i + 1];

			for (int j = startSubband; j < endSubband; j++)
			{
				channel->QuantizedSpectraFine[j] = ReadSignedInt(br, overflowBits);
			}
		}
	}
	return ERR_SUCCESS;
}

static At9Status UnpackLfeBlock(Block* block, BitReaderCxt* br)
{
	Channel* channel = &block->Channels[0];
	block->QuantizationUnitCount = 2;

	DecodeLfeScaleFactors(channel, br);
	CalculateLfePrecision(channel);
	channel->CodedQuantUnits = block->QuantizationUnitCount;
	ReadLfeSpectra(channel, br);

	return ERR_SUCCESS;
}

static void DecodeLfeScaleFactors(Channel* channel, BitReaderCxt* br)
{
	memset(channel->ScaleFactors, 0, sizeof(channel->ScaleFactors));
	for (int i = 0; i < channel->Block->QuantizationUnitCount; i++)
	{
		channel->ScaleFactors[i] = ReadInt(br, 5);
	}
}

static void CalculateLfePrecision(Channel* channel)
{
	Block* block = channel->Block;
	const int precision = block->ReuseBandParams ? 8 : 4;
	for (int i = 0; i < block->QuantizationUnitCount; i++)
	{
		channel->Precisions[i] = precision;
		channel->PrecisionsFine[i] = 0;
	}
}

static void ReadLfeSpectra(Channel* channel, BitReaderCxt* br)
{
	memset(channel->QuantizedSpectra, 0, sizeof(channel->QuantizedSpectra));

	for (int i = 0; i < channel->CodedQuantUnits; i++)
	{
		if (channel->Precisions[i] <= 0) continue;

		const int precision = channel->Precisions[i] + 1;
		for (int j = QuantUnitToCoeffIndex[i]; j < QuantUnitToCoeffIndex[i + 1]; j++)
		{
			channel->QuantizedSpectra[j] = ReadSignedInt(br, precision);
		}
	}
}
