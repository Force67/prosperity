#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 out0;
layout(location = 1) out vec4 out1;
void main() {
  out0 = v_color;
  out1 = vec4(1.0) - v_color;
}
