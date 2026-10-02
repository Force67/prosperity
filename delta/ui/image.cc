#if defined(__linux__) && !defined(__ANDROID__)
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif
