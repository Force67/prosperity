#version 450

layout(push_constant) uniform Background {
  vec2 resolution;
  float time;
  float style;
  float pulse;
  float opacity;
} background;
layout(location = 0) out vec4 color;

const vec3 blue = vec3(0.10, 0.38, 0.95);
const vec3 ice = vec3(0.65, 0.86, 1.0);
const vec2 apex = vec2(0.0, -0.72);
const vec2 left = vec2(-0.66, 0.48);
const vec2 right = vec2(0.66, 0.48);

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
mat2 rotate(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat2(c, -s, s, c);
}
float segment(vec2 p, vec2 a, vec2 b) {
  vec2 ab = b - a;
  return length(p - a - ab * clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0));
}
float logo(vec2 p) {
  float outer = min(segment(p, apex, left),
                    min(segment(p, left, right), segment(p, right, apex)));
  float inner = min(segment(p, vec2(0.16, -0.06), vec2(0.39, 0.35)),
                    segment(p, vec2(0.39, 0.35), vec2(-0.38, 0.35)));
  return min(outer - 0.042, inner - 0.020);
}
float triangle(vec2 p) {
  p.x = abs(p.x);
  return max(p.y - 0.6, p.x * 0.866025 - p.y * 0.5 - 0.3);
}
vec3 atmosphere(vec2 p) {
  float glow = exp(-dot(p - vec2(0.8, 0.10), p - vec2(0.8, 0.10)) * 0.85);
  return vec3(0.016, 0.022, 0.036) + vec3(0.010, 0.025, 0.060) * glow;
}
float selectionWave(vec2 p) {
  float radius = background.pulse * 0.8;
  return exp(-pow(length(p - vec2(0.75, 0.0)) - radius, 2.0) * 30.0) *
         exp(-background.pulse * 1.8);
}

vec3 current(vec2 p, float t) {
  vec3 result = atmosphere(p);
  for (int layer = 0; layer < 4; ++layer) {
    float z = float(layer);
    float scale = 3.5 - z * 0.72;
    vec2 flow = rotate(-0.17 + 0.055 * sin(t * 0.07)) * p;
    flow += vec2(t * (0.024 + z * 0.014), t * 0.018);
    flow.y += 0.075 * sin(flow.x * 2.0 + t * 0.19 + z);
    vec2 grid = flow * scale + z * 9.4;
    vec2 cell = floor(grid);
    float id = hash(cell + z);
    vec2 q = fract(grid) - 0.5;
    q -= vec2(sin(t * 0.18 + id * 6.28), cos(t * 0.13 + id * 6.28)) * 0.045;
    q = rotate((id - 0.5) * 0.55 + sin(t * 0.08 + id) * 0.08) * q;
    float size = mix(0.22, 0.43, id);
    float d = abs(logo(q / size)) * size;
    float line = exp(-d * 420.0) * 0.65;
    float bloom = exp(-d * 45.0) * 0.07;
    float wave = pow(0.5 + 0.5 * sin(cell.x * 0.4 + cell.y * 0.7 - t * 0.65), 7.0);
    float light = (0.12 + 0.8 * wave) * (0.45 + z * 0.16);
    result += mix(blue, ice, id * 0.6 + wave * 0.4) * (line + bloom) * light;
  }
  float d = abs(logo((p - vec2(0.87, 0.17)) / 1.05));
  result += blue * exp(-d * 12.0) * 0.035;
  result += ice * selectionWave(p) * 0.075;
  return result;
}

vec3 glass(vec2 p, float t) {
  vec3 result = atmosphere(p);
  vec2 q = p - vec2(0.78, 0.10);
  float yaw = 0.40 * sin(t * 0.22);
  q = rotate(0.04 * sin(t * 0.17)) * q;
  q.x /= cos(yaw);
  q.y += q.x * 0.06 * sin(t * 0.22);
  q /= 0.95;
  float d = logo(q);
  float aa = max(fwidth(d), 0.001);
  float mask = 1.0 - smoothstep(-aa, aa, d);
  float extrusion = 1.0 - smoothstep(-aa, aa, logo(q - vec2(0.035, 0.025)));
  result += vec3(0.025, 0.09, 0.22) * extrusion;
  vec2 gradient = vec2(logo(q + vec2(0.001, 0)) - logo(q - vec2(0.001, 0)),
                        logo(q + vec2(0, 0.001)) - logo(q - vec2(0, 0.001))) / 0.002;
  float bevel = smoothstep(-0.043, 0.008, d);
  vec3 normal = normalize(vec3(gradient * bevel * 1.4, 0.7));
  vec3 light = normalize(vec3(sin(t * 0.31), -0.8, 1.0));
  float specular = pow(max(dot(normal, normalize(light + vec3(0, 0, 1))), 0.0), 26.0);
  float rim = pow(1.0 - abs(normal.z), 1.5);
  float strip = pow(0.5 + 0.5 * sin(q.y * 4.0 + normal.x * 3.0 - t * 0.55), 20.0);
  vec3 material = atmosphere(p + normal.xy * 0.22) * 1.5 + blue * 0.14;
  material += ice * (0.7 * specular + rim * 0.55 + strip * 0.45);
  material += vec3(0.9, 0.97, 1.0) * exp(-abs(d) * 200.0) * 0.7;
  result = mix(result, material, mask * 0.92);
  result += blue * exp(-abs(d) * 15.0) * 0.12;
  float caustic = pow(abs(sin(p.x * 6.0 + t * 0.2) * sin(p.y * 8.0 - t * 0.3)), 12.0);
  result += ice * caustic * exp(-dot(p - vec2(0.8, 0.65), p - vec2(0.8, 0.65)) * 5.0) * 0.12;
  for (int i = 0; i < 3; ++i) {
    float phase = t * 0.16 + float(i) * 2.094;
    vec2 orbit = vec2(cos(phase) * 0.86, sin(phase) * 0.57);
    vec2 fragment = rotate(phase) * (p - vec2(0.78, 0.1) - orbit) / 0.075;
    float edge = abs(triangle(fragment));
    result += mix(blue, ice, sin(phase) * 0.5 + 0.5) * exp(-edge * 6.0) * 0.28;
  }
  result += ice * selectionWave(p) * 0.06;
  return result;
}

vec2 logoPoint(float index) {
  float part = floor(index * 5.0);
  float along = fract(index * 5.0);
  if (part < 1.0) return mix(apex, left, along);
  if (part < 2.0) return mix(left, right, along);
  if (part < 3.0) return mix(right, apex, along);
  if (part < 4.0) return mix(vec2(0.16, -0.06), vec2(0.39, 0.35), along);
  return mix(vec2(0.39, 0.35), vec2(-0.38, 0.35), along);
}
vec3 signal(vec2 p, float t) {
  vec3 result = atmosphere(p);
  float formed = smoothstep(-0.25, 0.65, cos(t * 0.48));
  vec2 center = vec2(0.8, 0.1);
  float ghost = abs(logo((p - center) / 0.92));
  result += blue * exp(-ghost * 80.0) * formed * 0.11;
  for (int i = 0; i < 180; ++i) {
    float id = float(i);
    float random = hash(vec2(id, 5.0));
    vec2 target = logoPoint((id + 0.5) / 180.0) * 0.92;
    vec2 stream = vec2(sin(id * 0.058 + t * 0.30) * 0.95 + cos(id * 0.19 + t * 0.21) * 0.25,
                        cos(id * 0.044 + t * 0.27) * 0.64 + sin(id * 0.21 - t * 0.18) * 0.18);
    vec2 position = center + mix(stream, target, formed);
    position += (1.0 - formed) * vec2(sin(t + id), cos(t * 0.7 + id)) * 0.024;
    vec2 delta = p - position;
    float distance2 = dot(delta, delta);
    if (distance2 > 0.014) continue;
    vec3 tint = mix(blue, ice, random);
    float size = mix(0.006, 0.017, random);
    float d = triangle(rotate(t * (1.0 - formed) + id) * delta / size);
    float particle = 1.0 - smoothstep(-0.1, 0.15, d);
    float glow = exp(-distance2 * 1400.0) * 0.08;
    float flicker = 0.65 + 0.35 * sin(t * 1.1 + id * 0.77);
    result += tint * (particle * 0.85 + glow) * flicker;
  }
  result += ice * selectionWave(p) * 0.045;
  return result;
}

vec3 anniversary(vec2 p, float t) {
  const vec3 gold = vec3(0.95, 0.64, 0.27);
  const vec3 champagne = vec3(1.0, 0.91, 0.70);
  vec2 center = vec2(0.78, 0.10);
  vec2 q = rotate(0.035 * sin(t * 0.17)) * (p - center) / 0.88;
  float radius = length(q);
  float angle = atan(q.y, q.x);
  vec3 result = vec3(0.018, 0.019, 0.025);
  result += gold * exp(-radius * radius * 1.3) * 0.023;
  float halo = abs(radius - 0.76);
  result += gold * exp(-halo * 18.0) * 0.08;
  float distance = abs(logo(q));
  float illumination = 0.3 + 0.7 * pow(0.5 + 0.5 * cos(angle - t * 0.32), 10.0);
  result += champagne * exp(-distance * 240.0) * illumination * 0.95;
  result += gold * exp(-distance * 25.0) * 0.15;
  for (int i = 0; i < 3; ++i) {
    float layer = float(i);
    vec2 ring = rotate(-0.40 + layer * 0.40 + sin(t * 0.09) * 0.14) * (p - center);
    float inclination = 0.32 + layer * 0.05;
    ring.y /= inclination;
    float orbit = abs(length(ring) - (0.82 + layer * 0.095)) * inclination;
    float sweep = pow(0.5 + 0.5 * cos(atan(ring.y, ring.x) - t * 0.22 - layer), 5.0);
    result += mix(gold, champagne, sweep) *
              (exp(-orbit * 600.0) * 0.28 + exp(-orbit * 45.0) * 0.025) *
              (0.25 + 0.75 * sweep);
  }
  vec2 dust = p * 28.0 + vec2(t * 0.10, -t * 0.12);
  vec2 cell = floor(dust);
  float id = hash(cell);
  vec2 offset = vec2(hash(cell + 2.3), hash(cell + 7.1));
  vec2 local = fract(dust) - offset;
  float speck = exp(-dot(local, local) * 90.0) * step(0.93, id);
  float shimmer = pow(0.5 + 0.5 * sin(t * 0.55 + id * 6.28), 3.0);
  result += champagne * speck * shimmer * 0.22;
  result += gold * selectionWave(p) * 0.08;
  return result;
}

void main() {
  vec2 p = (gl_FragCoord.xy - background.resolution * 0.5) / background.resolution.y * 2.0;
  float t = background.time;
  vec3 result = background.style < 1.5 ? current(p, t)
                : background.style < 2.5 ? glass(p, t)
                : background.style < 3.5 ? signal(p, t) : anniversary(p, t);
  // The left side stays quiet beneath the title and Play button.
  float readable = mix(0.24, 1.0, smoothstep(-1.1, 0.45, p.x));
  float vignette = 1.0 - 0.3 * smoothstep(0.6, 2.2, length(p));
  result *= readable * vignette;
  result = vec3(1.0) - exp(-result * 1.3);
  color = vec4(result, background.opacity);
}
