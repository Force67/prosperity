#include "quantization.h"
#include "tables.h"
#include <string.h>

static void DequantizeQuantUnit(Channel* channel, int band);
static void ScaleSpectrumChannel(Channel* channel);

void DequantizeSpectra(Block* block)
{
	for (int i = 0; i < block->ChannelCount; i++)
	{
		Channel* channel = &block->Channels[i];
		memset(channel->Spectra, 0, sizeof(channel->Spectra));

		for (int j = 0; j < channel->CodedQuantUnits; j++)
		{
			DequantizeQuantUnit(channel, j);
		}
	}
}

static void DequantizeQuantUnit(Channel* channel, int band)
{
	const int subBandIndex = QuantUnitToCoeffIndex[band];
	const int subBandCount = QuantUnitToCoeffCount[band];
	const double stepSize = QuantizerStepSize[channel->Precisions[band]];
	const double stepSizeFine = QuantizerFineStepSize[channel->PrecisionsFine[band]];

	for (int sb = 0; sb < subBandCount; sb++)
	{
		const double coarse = channel->QuantizedSpectra[subBandIndex + sb] * stepSize;
		const double fine = channel->QuantizedSpectraFine[subBandIndex + sb] * stepSizeFine;
		channel->Spectra[subBandIndex + sb] = coarse + fine;
	}
}

void ScaleSpectrumBlock(Block* block)
 {
	 for (int i = 0; i < block->ChannelCount; i++)
	 {
		 ScaleSpectrumChannel(&block->Channels[i]);
	 }
 }

static void ScaleSpectrumChannel(Channel* channel)
{
	 const int quantUnitCount = channel->Block->QuantizationUnitCount;
	 double* spectra = channel->Spectra;

	 for (int i = 0; i < quantUnitCount; i++)
	 {
		 for (int sb = QuantUnitToCoeffIndex[i]; sb < QuantUnitToCoeffIndex[i + 1]; sb++)
		 {
			 spectra[sb] *= SpectrumScale[channel->ScaleFactors[i]];
		 }
	 }
 }