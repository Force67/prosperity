#version 450
layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 v_color;
layout(push_constant) uniform Push { vec4 offset; } pc;
layout(set = 1, binding = 0) uniform Tint { vec4 tint; } u;
void main() {
  v_color = color * u.tint;
  gl_Position = vec4(pos + pc.offset.xy, pc.offset.z, 1.0);
}
