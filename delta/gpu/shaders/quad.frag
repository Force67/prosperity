#version 450
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;
void main() {
    // Visible gradient: proves the geometry+transform reached the framebuffer.
    outColor = vec4(vUv, 0.6, 1.0);
}
