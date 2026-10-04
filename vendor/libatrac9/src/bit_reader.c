#include "bit_reader.h"
#include "utility.h"

static int PeekIntFallback(BitReaderCxt* br, int bitCount);

void InitBitReaderCxt(BitReaderCxt* br, const void * buffer)
{
	br->Buffer = buffer;
	br->Position = 0;
}

int ReadInt(BitReaderCxt* br, const int bits)
{
	const int value = PeekInt(br, bits);
	br->Position += bits;
	return value;
}

int ReadSignedInt(BitReaderCxt* br, const int bits)
{
	const int value = PeekInt(br, bits);
	br->Position += bits;
	return SignExtend32(value, bits);
}

int ReadOffsetBinary(BitReaderCxt* br, const int bits)
{
	const int offset = 1 << (bits - 1);
	const int value = PeekInt(br, bits) - offset;
	br->Position += bits;
	return value;
}

int PeekInt(BitReaderCxt* br, const int bits)
{
	const int byteIndex = br->Position / 8;
	const int bitIndex = br->Position % 8;
	const unsigned char* buffer = br->Buffer;

	if (bits <= 9)
	{
		int value = buffer[byteIndex] << 8 | buffer[byteIndex + 1];
		value &= 0xFFFF >> bitIndex;
		value >>= 16 - bits - bitIndex;
		return value;
	}

	if (bits <= 17)
	{
		int value = buffer[byteIndex] << 16 | buffer[byteIndex + 1] << 8 | buffer[byteIndex + 2];
		value &= 0xFFFFFF >> bitIndex;
		value >>= 24 - bits - bitIndex;
		return value;
	}

	if (bits <= 25)
	{
		int value = buffer[byteIndex] << 24
			| buffer[byteIndex + 1] << 16
			| buffer[byteIndex + 2] << 8
			| buffer[byteIndex + 3];

		value &= (int)(0xFFFFFFFF >> bitIndex);
		value >>= 32 - bits - bitIndex;
		return value;
	}
	return PeekIntFallback(br, bits);
}

void AlignPosition(BitReaderCxt* br, const unsigned int multiple)
{
	const int position = br->Position;
	if (position % multiple == 0)
	{
		return;
	}

	br->Position = position + multiple - position % multiple;
}

static int PeekIntFallback(BitReaderCxt* br, int bitCount)
{
	int value = 0;
	int byteIndex = br->Position / 8;
	int bitIndex = br->Position % 8;
	const unsigned char* buffer = br->Buffer;

	while (bitCount > 0)
	{
		if (bitIndex >= 8)
		{
			bitIndex = 0;
			byteIndex++;
		}

		int bitsToRead = bitCount;
		if (bitsToRead > 8 - bitIndex)
		{
			bitsToRead = 8 - bitIndex;
		}

		const int mask = 0xFF >> bitIndex;
		const int currentByte = (mask & buffer[byteIndex]) >> (8 - bitIndex - bitsToRead);

		value = (value << bitsToRead) | currentByte;
		bitIndex += bitsToRead;
		bitCount -= bitsToRead;
	}
	return value;
}