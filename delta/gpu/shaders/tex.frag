#version 450
layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUv;
layout(location = 2) flat in uint vClip;
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 c = texture(tex, vUv) * vColor;
    // Fullscreen render-to-texture composite: copy the source RT opaquely. Its
    // alpha is a scratch value left by sprite blending, not coverage, so keeping
    // it would make the SRC_ALPHA blend fade the whole scene to the black clear.
    // Force alpha 1 so the composite lands the full scene RGB onto the scanout.
    if (vClip != 0u) c.a = 1.0;
    outColor = c;
}
