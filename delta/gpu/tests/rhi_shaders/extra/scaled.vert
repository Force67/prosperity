#version 450
// Colour from an 8-bit scaled attribute: 0..255 as floats.
layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 v_color;
void main() {
  gl_Position = vec4(pos, 0.0, 1.0);
  v_color = color / 255.0;
}
