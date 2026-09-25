#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 offset;
layout(location = 0) out vec4 v_color;
void main() {
  v_color = color;
  gl_Position = vec4(pos.xy + offset, pos.z, 1.0);
  gl_PointSize = 1.0;
}
