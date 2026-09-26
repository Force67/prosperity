#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out0;
layout(set = 0, binding = 0) uniform usampler2D tex;
layout(push_constant) uniform Push { vec4 coord; } pc;
void main() { out0 = vec4(textureLod(tex, pc.coord.xy, 0.0)) / 255.0; }
