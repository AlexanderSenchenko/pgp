#version 430 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;

uniform mat4 MVP;

out vec4 vs_color;

void main()
{
	gl_Position = MVP * vec4(pos,1);
	vs_color=vec4(color,1.0);
}