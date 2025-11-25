#version 460

// layout(location = 1) in vec3 position;
layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 0) buffer Data {
	uint data[];
} data;
layout(push_constant) uniform PushConstants {
	float rotation;
	vec2 resolution;
} push_constants;

void main () {
	// if (gl_FragCoord.x < gl_FragCoord.y) {
	// 	f_color = vec4(vec2(0.0), vec2(1.0));
	// } else {
	// 	f_color = vec4(gl_FragCoord.x / 2560, gl_FragCoord.y / 1600, 0.0, 1.0);
	// }
	f_color = vec4(1.0, vec2(0.0), 1.0);
}
