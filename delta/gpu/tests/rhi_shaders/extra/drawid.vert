#version 450
// One point per vertex at column `slot` of row 0 (under a y-up viewport),
// coloured with the vertex and instance index.
layout(location = 0) in float vertex_slot;
layout(location = 1) in float instance_slot;
layout(location = 0) out vec4 v_color;
void main() {
  float slot = vertex_slot + instance_slot;
  gl_Position = vec4(-1.0 + (slot + 0.5) / 32.0, 1.0 - 1.0 / 64.0, 0.0, 1.0);
  gl_PointSize = 1.0;
  v_color = vec4(float(gl_VertexIndex), float(gl_InstanceIndex), 0.0, 255.0) /
            255.0;
}
