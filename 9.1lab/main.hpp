#ifndef MAIN_HPP
#define MAIN_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

glm::mat4 scaleMatrix;
glm::mat4 rotateMatrixY;
glm::mat4 rotateMatrixX;
glm::mat4 translateMatrix;

GLfloat valueX = 0.0f;
GLfloat valueY = 0.0f;

GLfloat scale = 1.0f;

float angleY = 0.0f;
float angleX = 0.0f;

int initGL();
int initBuffer();
int initColorBuffer();

int translateTrianglesKey();
int translateTrianglesCursor();
int scaleTrianglesKey();
int rotateTrianglesKey();

int randColorBuffer();

#endif