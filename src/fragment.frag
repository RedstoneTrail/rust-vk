#version 460

layout(location = 0) out vec4 f_color;
// layout(location = 1) in vec3 vertex_position;
// layout(location = 2) in vec3 normal;
// layout(location = 3) in vec3 origin;
// layout(location = 4) in vec3 relative_position;
// layout(location = 5) in vec2 uv;

layout(location = 0) in vec2 tex_coords;
layout(set = 0, binding = 0) uniform sampler2D tex;

float vec3dotproduct(vec3 v1, vec3 v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

void main() {
        // float red = 1.0;

        // red *= abs(vec3dotproduct(normalize(normal), vec3(0.0, 0.0, -1.0)));
        // red *= pow(length(relative_position), 2);

        // f_color = vec4(vec3(1.0, vec2(0.0)) * (abs(vec3dotproduct(normalize(normal), vec3(0.0, 0.0, -1.0)))) * (length(vertex_position)), 1.0);
        // f_color = vec4(red, vec2(0.0), 1.0);
        // f_color = vec4(uv, 0.0, 1.0);

        // f_color.xyz = fragColor;

        // f_color.xyz += fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));

        f_color = texture(tex, tex_coords);
}
