#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
using namespace glm;

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

int initGL();
int initBuffer();
int initColorBuffer();

GLuint LoadShaders(const char *vertex_file_path,
									const char *fragment_file_path);

GLFWwindow *window;
GLuint bufferID;
GLuint colorBufferID;
GLuint progHandle;
GLuint vertexArrayID;

const unsigned int window_width = 512;
const unsigned int window_height = 512;

const int num_of_verticies = 3;

int main()
{
	if (initGL() == -1)
		return -1;

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	// glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);

	progHandle = LoadShaders("SimpleVertexShader.vertexshader",
								"SimpleFragmentShader.fragmentshader");

	initBuffer();
	initColorBuffer();

	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		#if 1
		glUseProgram(progHandle);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, bufferID);
		glVertexAttribPointer(
			0,
			3,
			GL_FLOAT,
			GL_FALSE,
			0,
			(void*) 0
		);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
		glVertexAttribPointer(
			1,
			3,
			GL_FLOAT,
			GL_FALSE,
			0,
			(void*) 0
		);

		glDrawArrays(GL_TRIANGLES, 0, 3);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		#endif

		glfwSwapBuffers(window);
		glfwPollEvents();
	} while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
								glfwWindowShouldClose(window) == 0);

	glDeleteBuffers(1, &bufferID);
	glDeleteBuffers(1, &colorBufferID);
	glDeleteVertexArrays(1, &vertexArrayID);
	glDeleteProgram(progHandle);

	glfwTerminate();

	return 0;
}

int initBuffer()
{
	glGenBuffers(1, &bufferID);

	glBindBuffer(GL_ARRAY_BUFFER, bufferID);

	#if 0
		static const GLfloat vertex_buffer_data[] = {
				-0.9f, -0.9f, -0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
				0.9f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f,
				};

		glBufferData(GL_ARRAY_BUFFER, 6*num_of_verticies*sizeof(float),
									vertex_buffer_data, GL_STATIC_DRAW );
	#endif

	#if 1
		static const GLfloat vertex_buffer_data[] = { 
				-0.9f, -0.9f, -0.0f,
				0.0f, 0.0f, 0.0f,
				0.9f, -0.5f, 0.0f,
				};

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data),
									vertex_buffer_data, GL_STATIC_DRAW);
	#endif
	
	return 0;
}

int initColorBuffer()
{
	glGenBuffers(1, &colorBufferID);

	glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);

	static const GLfloat vertex_buffer_data[] = { 
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f,
			};

	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data),
								vertex_buffer_data, GL_STATIC_DRAW);

	return 0;
}

int initGL()
{
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	window = glfwCreateWindow(window_width, window_height,
									"Tamplate window", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to opne GLFW window\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	return 0;
}

GLuint LoadShaders(const char *vertex_file_path,
										const char *fragment_file_path)
{
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if (VertexShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	} else {
		printf("Impossible to open %s. Are you in the right directory ?\
			Don't forget to read the FAQ !\n", vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if (FragmentShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << FragmentShaderStream.rdbuf();
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL,
									&VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL,
										&FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL,
												&ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}

	
	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);
	
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}
