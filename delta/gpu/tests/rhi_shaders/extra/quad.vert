#version 450
// A rectangle (triangle strip) in clip space at depth z, in a flat colour.
layout(push_constant) uniform Push {
  vec4 rect;  // x0, y0, x1, y1
  vec4 color;
  vec4 z;
} pc;
layout(location = 0) out vec4 v_color;
void main() {
  vec2 c = vec2(gl_VertexIndex & 1, (gl_VertexIndex >> 1) & 1);
  gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, c), pc.z.x, 1.0);
  v_color = pc.color;
}
