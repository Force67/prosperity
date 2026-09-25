#version 450
layout(location = 0) out vec2 v_uv;
void main() {
  vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  v_uv = p;
  // uv (0,0) lands at the top-left of the target under a y-up viewport.
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
