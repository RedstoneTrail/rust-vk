#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
// layout(location = 3) in vec2 inTexCoord;

layout(push_constant) uniform PushConstants {
        vec2 rotation;
        vec2 resolution;
        vec3 origin;
        float side_length;
} push_constants;

// layout(set = 0, location = 1) out vec3 vertex_position;
// layout(set = 0, location = 2) out vec3 normal;
// layout(set = 0, location = 3) out vec3 origin;
// layout(set = 0, location = 4) out vec3 relative_position;
// layout(set = 0, location = 5) out vec2 uvout;

layout(location = 0) out vec2 tex_coords;

void main() {
        float pi = radians(180);

        // convert from turns to radians for trig functions
        vec2 radian_rotation = push_constants.rotation.yx * 2 * pi;

        // rotation on y axis followed by rotation on y-axis
        mat3 rotation_matrix = mat3(
                        cos(radian_rotation.y), 0.0, sin(radian_rotation.y),
                        0.0, 1.0, 0.0,
                        -sin(radian_rotation.y), 0.0, cos(radian_rotation.y)
                ) * mat3(
                                1.0, 0.0, 0.0,
                                0.0, cos(radian_rotation.x), -sin(radian_rotation.x),
                                0.0, sin(radian_rotation.x), cos(radian_rotation.x)
                        );

        vec4 out_position = vec4(vec3(rotation_matrix * position), 1.0);

        // relative_position = position;
        // normal = rotation_matrix * vec3(vec2(0.0), -1.0);
        // origin = push_constants.origin;

        out_position.z *= 0.01;
        out_position.xyz += push_constants.origin;
        gl_Position.xyz = out_position.xyz;
        gl_Position.xy *= push_constants.side_length / push_constants.resolution.xy;
        // vertex_position = out_position.xyz;
        // uvout = uv;

        tex_coords = uv;
}
