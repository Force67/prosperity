#version 450
// Writes three varyings; varyings.frag reads only the second.
layout(location = 0) out vec4 v0;
layout(location = 1) out vec4 v1;
layout(location = 2) out vec2 v2;
void main() {
  vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
  v0 = vec4(1.0, 0.0, 0.0, 1.0);
  v1 = vec4(0.0, 1.0, 0.0, 1.0);
  v2 = vec2(0.5);
}
