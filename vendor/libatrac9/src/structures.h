#pragma once

#define CONFIG_DATA_SIZE 4
#define MAX_CHANNEL_COUNT 8
#define MAX_BLOCK_COUNT 5
#define MAX_BLOCK_CHANNEL_COUNT 2
#define MAX_FRAME_SAMPLES 256
#define MAX_BEX_VALUES 4

#define MAX_QUANT_UNITS 30

typedef struct Frame Frame;
typedef struct Block Block;

typedef enum BlockType {
	Mono = 0,
	Stereo = 1,
	LFE = 2
} BlockType;

typedef struct {
	unsigned char BlockCount;
	unsigned char ChannelCount;
	unsigned char Types[MAX_BLOCK_COUNT];
} ChannelConfig;

typedef struct {
	unsigned char ConfigData[CONFIG_DATA_SIZE];
	int SampleRateIndex;
	int ChannelConfigIndex;
	int FrameBytes;
	int SuperframeIndex;

	ChannelConfig ChannelConfig;
	int ChannelCount;
	int SampleRate;
	int HighSampleRate;
	int FramesPerSuperframe;
	int FrameSamplesPower;
	int FrameSamples;
	int SuperframeBytes;
	int SuperframeSamples;
} ConfigData;

typedef struct {
	int Initialized;
	unsigned short StateA;
	unsigned short StateB;
	unsigned short StateC;
	unsigned short StateD;
} RngCxt;

typedef struct {
	int Bits;
	int Size;
	double Scale;
	double ImdctPrevious[MAX_FRAME_SAMPLES];
	double* Window;
	double* SinTable;
	double* CosTable;
} Mdct;

typedef struct {
	Frame* Frame;
	Block* Block;
	ConfigData* Config;
	int ChannelIndex;

	Mdct Mdct;

	double Pcm[MAX_FRAME_SAMPLES];
	double Spectra[MAX_FRAME_SAMPLES];

	int CodedQuantUnits;
	int ScaleFactorCodingMode;

	int ScaleFactors[31];
	int ScaleFactorsPrev[31];

	int Precisions[MAX_QUANT_UNITS];
	int PrecisionsFine[MAX_QUANT_UNITS];
	int PrecisionMask[MAX_QUANT_UNITS];

	int CodebookSet[MAX_QUANT_UNITS];

	int QuantizedSpectra[MAX_FRAME_SAMPLES];
	int QuantizedSpectraFine[MAX_FRAME_SAMPLES];

	int BexMode;
	int BexValueCount;
	int BexValues[MAX_BEX_VALUES];

	RngCxt Rng;
} Channel;

struct Block {
	Frame* Frame;
	ConfigData* Config;
	enum BlockType BlockType;
	int BlockIndex;
	Channel Channels[MAX_BLOCK_CHANNEL_COUNT];
	int ChannelCount;
	int FirstInSuperframe;
	int ReuseBandParams;

	int BandCount;
	int StereoBand;
	int ExtensionBand;
	int QuantizationUnitCount;
	int StereoQuantizationUnit;
	int ExtensionUnit;
	int QuantizationUnitsPrev;

	int Gradient[31];
	int GradientMode;
	int GradientStartUnit;
	int GradientStartValue;
	int GradientEndUnit;
	int GradientEndValue;
	int GradientBoundary;

	int PrimaryChannelIndex;
	int HasJointStereoSigns;
	int JointStereoSigns[MAX_QUANT_UNITS];

	int BandExtensionEnabled;
	int HasExtensionData;
	int BexDataLength;
	int BexMode;
};

struct Frame {
	int IndexInSuperframe;
	ConfigData* Config;
	Channel* Channels[MAX_CHANNEL_COUNT];
	Block Blocks[MAX_BLOCK_COUNT];
};

typedef struct {
	int Initialized;
	int Wlength;
	ConfigData Config;
	Frame Frame;
} Atrac9Handle;

typedef struct {
	char GroupBUnit;
	char GroupCUnit;
	char BandCount;
} BexGroup;

typedef struct {
	int Channels;
	int ChannelConfigIndex;
	int SamplingRate;
	int SuperframeSize;
	int FramesInSuperframe;
	int FrameSamples;
	int Wlength;
	unsigned char ConfigData[CONFIG_DATA_SIZE];
} CodecInfo;
