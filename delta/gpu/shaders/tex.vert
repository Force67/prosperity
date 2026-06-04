#version 450
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUv;
layout(push_constant) uniform PC { mat4 mvp; uint clipUV; } pc;
layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUv;
layout(location = 2) flat out uint vClip;
void main() {
    vec4 p = pc.mvp * vec4(inPos, 0.0, 1.0); p.z = 0.0;
    gl_Position = p;
    vClip = pc.clipUV;
    // Render-to-texture composite (clipUV set): a neutral fullscreen blit of the
    // source RT: sample at the screen position and don't modulate by the
    // per-vertex attributes (whose format/offset varies per draw and is unknown
    // for the composite). Regular sprites use their vertex uv + color.
    if (pc.clipUV != 0u) {
        vColor = vec4(1.0);
        vUv = p.xy * 0.5 + 0.5;
    } else {
        vColor = inColor;
        vUv = inUv;
    }
}
