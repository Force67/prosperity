#version 450
layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUv;
layout(location = 2) flat in uint vClip;
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 c = texture(tex, vUv) * vColor;
    // clipUV==1 (fullscreen scene composite): copy the source RT opaquely. Its alpha
    // is a scratch value left by sprite blending, not coverage, so keeping it would
    // make the SRC_ALPHA blend fade the whole scene to the black clear.
    if (vClip == 1u) c.a = 1.0;
    // clipUV==2 (room-layer composite): the room is composed of stacked layers (a
    // floor layer, then a wall layer with an empty interior), each REPLACE-blended on
    // hardware. Their stored alpha is unreliable (the floor texels are alpha 0), so
    // derive coverage from colour: a coloured texel is opaque, a black texel is empty.
    // With SRC_ALPHA blend the wall layer's black interior then keeps the floor below.
    else if (vClip == 2u) c.a = max(max(c.r, c.g), c.b) > 0.0039 ? 1.0 : 0.0;
    outColor = c;
}
