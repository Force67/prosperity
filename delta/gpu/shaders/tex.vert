#version 450
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUv;
layout(push_constant) uniform PC { mat4 mvp; uint clipUV; } pc;
layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUv;
void main() {
    vec4 p = pc.mvp * vec4(inPos, 0.0, 1.0); p.z = 0.0;
    if (pc.clipUV != 0u) {
        gl_Position = p;
        vColor = vec4(1.0);
        vUv = vec2(p.x * 0.5 + 0.5, p.y * 0.5 + 0.5);
    } else {
        gl_Position = vec4(-p.x, p.y, p.z, p.w);
        vColor = inColor;
        vUv = inUv;
    }
}
