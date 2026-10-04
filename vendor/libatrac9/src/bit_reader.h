#pragma once

typedef struct {
	const unsigned char * Buffer;
	int Position;
} BitReaderCxt;

// Make MSVC compiler happy. Leave const in for value parameters

void InitBitReaderCxt(BitReaderCxt* br, const void * buffer);
int PeekInt(BitReaderCxt* br, const int bits);
int ReadInt(BitReaderCxt* br, const int bits);
int ReadSignedInt(BitReaderCxt* br, const int bits);
int ReadOffsetBinary(BitReaderCxt* br, const int bits);
void AlignPosition(BitReaderCxt* br, const unsigned int multiple);
