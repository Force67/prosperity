#version 450
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) out vec4 out0;
void main() { out0 = vec4(gl_BaryCoordEXT, 1.0); }
