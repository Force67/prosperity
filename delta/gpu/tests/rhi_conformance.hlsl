// Shaders for the RHI conformance test. One source, compiled two ways: to
// SPIR-V by glslc (with TARGET_VULKAN defined) at build time, and to DXBC by
// D3DCompile at run time. Both backends then run the same program, which is
// what makes a pixel difference between them a backend bug rather than a
// shader difference.
//
// Binding convention (rhi/types.h): a bind group is an HLSL register space,
// and a binding index is the register slot within it, unique across register
// classes. Push constants live at b0 in space 8.

#ifdef TARGET_VULKAN
#define PUSH_CONSTANTS [[vk::push_constant]]
#define PUSH_REGISTER
#else
#define PUSH_CONSTANTS
#define PUSH_REGISTER : register(b0, space8)
#endif

PUSH_CONSTANTS cbuffer Push PUSH_REGISTER {
  float4 tint;
};

// Group 0: a texture and its sampler.
Texture2D color_tex : register(t0, space0);
SamplerState color_samp : register(s1, space0);

// Group 1: a dynamic constant buffer, offset per draw.
cbuffer Xform : register(b0, space1) {
  float4 offset;
};

// Group 2: a storage buffer for the compute scenario.
RWByteAddressBuffer out_buf : register(u0, space2);

struct SolidOut {
  float4 pos : SV_Position;
  float4 color : COLOR0;
};

SolidOut vs_solid(float3 pos : ATTRIB0, float4 color : ATTRIB1) {
  SolidOut o;
  o.pos = float4(pos, 1.0);
  o.color = color;
  return o;
}

float4 ps_solid(SolidOut i) : SV_Target {
  return i.color;
}

struct MrtOut {
  float4 first : SV_Target0;
  float4 second : SV_Target1;
};

MrtOut ps_mrt(SolidOut i) {
  MrtOut o;
  o.first = i.color;
  o.second = float4(i.color.a, i.color.b, i.color.g, i.color.r);
  return o;
}

struct TexOut {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
};

TexOut vs_tex(float2 pos : ATTRIB0, float2 uv : ATTRIB1) {
  TexOut o;
  o.pos = float4(pos + offset.xy, 0.0, 1.0);
  o.uv = uv;
  return o;
}

float4 ps_tex(TexOut i) : SV_Target {
  return color_tex.Sample(color_samp, i.uv) * tint;
}

[numthreads(8, 1, 1)]
void cs_fill(uint3 id : SV_DispatchThreadID) {
  out_buf.Store(id.x * 4, id.x * asuint(tint.x) + 1);
}
