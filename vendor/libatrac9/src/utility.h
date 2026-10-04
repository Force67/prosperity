#pragma once

#define FALSE 0
#define TRUE 1

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int Max(int a, int b);
int Min(int a, int b);
unsigned int BitReverse32(unsigned int value, int bitCount);
int SignExtend32(int value, int bits);
short Clamp16(int value);
int Round(double x);
