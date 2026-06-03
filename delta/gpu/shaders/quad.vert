#version 450
layout(location = 0) in vec2 inPos;            // screen-space position (VB0)
layout(push_constant) uniform PC { mat4 mvp; } pc;
layout(location = 0) out vec2 vUv;
void main() {
    vec4 p = pc.mvp * vec4(inPos, 0.0, 1.0);
    p.z = 0.0;   // 2D: avoid GL-style z=-1 getting clipped by Vulkan's [0,1] depth
    gl_Position = p;
    vUv = p.xy * 0.5 + 0.5;
}
