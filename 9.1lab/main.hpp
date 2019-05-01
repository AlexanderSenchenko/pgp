#ifndef MAIN_HPP
#define MAIN_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// #include <glm/gtx/transform.hpp>
// using namespace glm;

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "shader.hpp"

GLFWwindow *window;
GLuint bufferID;
GLuint colorBufferID;
GLuint progHandle;
GLuint vertexArrayID;

const unsigned int window_width = 1024;
const unsigned int window_height = 768;

const int num_of_verticies = 3;

int initGL();
int initBuffer();
int initColorBuffer();
int translateTriangles();
int scaleTriangles();

int testMoveTrianglesCursor();
int testRotateMatrix();

#endif