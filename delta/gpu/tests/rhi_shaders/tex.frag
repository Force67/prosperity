#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out0;
layout(set = 0, binding = 0) uniform sampler2D tex;
void main() { out0 = texture(tex, v_uv); }
