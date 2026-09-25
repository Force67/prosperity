#version 450
// Attribute locations past what some GL drivers have, with gaps.
layout(location = 20) in vec3 pos;
layout(location = 27) in vec4 color;
layout(location = 0) out vec4 v_color;
void main() {
  v_color = color;
  gl_Position = vec4(pos, 1.0);
}
