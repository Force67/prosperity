#version 450
// One point in, the right half of the target out.
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
layout(location = 0) in vec4 v_color[];
layout(location = 0) out vec4 g_color;
void main() {
  for (int i = 0; i < 4; i++) {
    g_color = v_color[0];
    gl_Position = vec4(float(i & 1), float(i >> 1) * 2.0 - 1.0, 0.0, 1.0);
    EmitVertex();
  }
  EndPrimitive();
}
