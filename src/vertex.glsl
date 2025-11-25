#version 460

layout(location = 0) in vec2 position;
layout(push_constant) uniform PushConstants {
	float rotation;
	vec2 resolution;
} push_constants;

void main() {
	float pi = radians(180);

	float rotation = push_constants.rotation * 2 * pi;

	mat2 rotation_matrix = mat2(cos(rotation), sin(rotation), - sin(rotation), cos(rotation));

	gl_Position = vec4(rotation_matrix * position, 0.0, 1.0);
}
