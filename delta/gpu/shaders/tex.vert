#version 450
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUv;
layout(push_constant) uniform PC { mat4 mvp; uint clipUV; } pc;
layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUv;
void main() {
    vec4 p = pc.mvp * vec4(inPos, 0.0, 1.0); p.z = 0.0;
    gl_Position = p;
    vColor = inColor;
    // Render-to-texture composite: sample the source RT at the screen position
    // (a 1:1 fullscreen blit) rather than the per-vertex uv attribute.
    vUv = (pc.clipUV != 0u) ? (p.xy * 0.5 + 0.5) : inUv;
}
