#include "main.hpp"

int main()
{
	if (initGL() == -1)
		return -1;

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);

	progHandle = LoadShaders("SimpleVertexShader.glsl",
								"SimpleFragmentShader.glsl");

	GLuint MatrixID = glGetUniformLocation(progHandle, "MVP");

	glm::mat4 translateMatrix;
	glm::mat4 MVP;

	GLfloat valueX = 0.0f;
	GLfloat valueY = 0.0f;

	// initBuffer();
	glGenBuffers(1, &bufferID);

	glm::vec3 vertex_buffer_data[] = {
		glm::vec3(-0.9f, -0.9f, -0.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.9f, -0.5f, 0.0f)
	};

	initColorBuffer();

	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		translateMatrix = glm::translate(glm::mat4(1.0f),
					glm::vec3(valueX, valueY, 0.0f));

		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			valueX -= 0.01f;
			translateMatrix = glm::translate(glm::mat4(1.0f),
					glm::vec3(valueX, valueY, 0.0f));
		}
		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			valueX += 0.01f;
			translateMatrix = glm::translate(glm::mat4(1.0f),
					glm::vec3(valueX, valueY, 0.0f));
		}
		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
			valueY += 0.01f;
			translateMatrix = glm::translate(glm::mat4(1.0f),
					glm::vec3(valueX, valueY, 0.0f));
		}
		if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			valueY -= 0.01f;
			translateMatrix = glm::translate(glm::mat4(1.0f),
					glm::vec3(valueX, valueY, 0.0f));
		}

		MVP = translateMatrix;

		glBindBuffer(GL_ARRAY_BUFFER, bufferID);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data),
								vertex_buffer_data, GL_STATIC_DRAW);

		glUseProgram(progHandle);

		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

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
				-0.9f, -0.9f, -0.0f,	// left down
				0.0f, 0.0f, 0.0f,
				0.9f, -0.5f, 0.0f		// right down
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

	static const GLfloat color_buffer_data[] = { 
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f
			};

	glBufferData(GL_ARRAY_BUFFER, sizeof(color_buffer_data),
								color_buffer_data, GL_STATIC_DRAW);

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
