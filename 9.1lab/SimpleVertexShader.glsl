#version 430 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 color;

uniform mat4 scaleMatrix;
uniform mat4 rotateMatrixY;
uniform mat4 rotateMatrixX;
uniform mat4 translateMatrix;

out vec4 vs_color;

void main()
{
	gl_Position = translateMatrix*rotateMatrixX*rotateMatrixY*scaleMatrix*vec4(pos,1);
	vs_color=vec4(color,1.0);
}