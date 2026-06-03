#version 450
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUv;
layout(push_constant) uniform PC { mat4 mvp; } pc;
layout(location = 0) out vec2 vUv;
void main() {
    vec4 p = pc.mvp * vec4(inPos, 0.0, 1.0);
    p.z = 0.0;
    gl_Position = p;
    vUv = inUv;
}
