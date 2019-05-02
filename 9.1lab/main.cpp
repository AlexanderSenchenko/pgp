#include "main.hpp"

int main()
{
	if (initGL() == -1)
		return -1;

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	// glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_TRUE);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);

	progHandle = LoadShaders("SimpleVertexShader.glsl",
								"SimpleFragmentShader.glsl");

	GLuint scaleMatrixID = glGetUniformLocation(progHandle,
													"scaleMatrix");
	GLuint rotateMatrixIDY = glGetUniformLocation(progHandle,
													"rotateMatrixY");
	GLuint rotateMatrixIDX = glGetUniformLocation(progHandle,
													"rotateMatrixX");
	GLuint translateMatrixID = glGetUniformLocation(progHandle,
												"translateMatrix");

	scaleMatrix = glm::scale(glm::mat4(1.0f),
								glm::vec3(scale, scale, scale));
	rotateMatrixY = glm::rotate(glm::mat4(1.0f),
								0.0f, glm::vec3(0.0f, 1.0f, 0.0f));
	rotateMatrixX = glm::rotate(glm::mat4(1.0f),
								0.0f, glm::vec3(1.0f, 0.0f, 0.0f));
	translateMatrix = glm::translate(glm::mat4(1.0f),
								glm::vec3(valueX, valueY, 0.0f));
	
	initBuffer();
	initColorBuffer();

	// glfwGetWindowSize(window, &width, &height);
	glfwSetCursorPos(window, window_width / 2, window_height / 2);

	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		scaleTrianglesKey();
		rotateTrianglesKey();
		translateTrianglesCursor();
		// translateTrianglesKey();

		glUseProgram(progHandle);

		glUniformMatrix4fv(scaleMatrixID, 1, GL_FALSE,
												&scaleMatrix[0][0]);
		glUniformMatrix4fv(rotateMatrixIDY, 1, GL_FALSE,
												&rotateMatrixY[0][0]);
		glUniformMatrix4fv(rotateMatrixIDX, 1, GL_FALSE,
												&rotateMatrixX[0][0]);
		glUniformMatrix4fv(translateMatrixID, 1, GL_FALSE,
												&translateMatrix[0][0]);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, bufferID);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);

		glDrawArrays(GL_TRIANGLES, 0, 3 * 4);

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

int rotateTrianglesKey()
{
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		angleY -= 0.01f;
		rotateMatrixY = glm::rotate(glm::mat4(1.0f),
								angleY, glm::vec3(0.0f, 1.0f, 0.0f));
	}
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		angleY += 0.01f;
		rotateMatrixY = glm::rotate(glm::mat4(1.0f),
								angleY, glm::vec3(0.0f, 1.0f, 0.0f));
	}
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		angleX += 0.01f;
		rotateMatrixX = glm::rotate(glm::mat4(1.0f),
								angleX, glm::vec3(1.0f, 0.0f, 0.0f));
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		angleX -= 0.01f;
		rotateMatrixX = glm::rotate(glm::mat4(1.0f),
								angleX, glm::vec3(1.0f, 0.0f, 0.0f));
	}

	return 0;
}

int translateTrianglesCursor()
{

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	valueY = ((window_height / 2) - ypos) / 100;
	valueX = (xpos - (window_width / 2)) / 100;

	translateMatrix = glm::translate(glm::mat4(1.0f),
								glm::vec3(valueX, valueY, 0.0f));
	#if 0
	printf("%f %f\n", xpos, ypos);
	printf("valueY = %f valueX = %f\n", valueY, valueX);
	#endif
	return 0;
}

int scaleTrianglesKey()
{
	if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS) {
		scale += 0.01f;
		scaleMatrix = glm::scale(glm::mat4(1.0f),
			glm::vec3(scale, scale, scale));
	}
	if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS) {
		scale -= 0.01f;
		scaleMatrix = glm::scale(glm::mat4(1.0f),
			glm::vec3(scale, scale, scale));
	}
	return 0;
}

int translateTrianglesKey()
{
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

	return 0;
}

int initBuffer()
{
	glGenBuffers(1, &bufferID);

	glBindBuffer(GL_ARRAY_BUFFER, bufferID);

	#if 0
		static const GLfloat vertex_buffer_data[] = { 
			-0.9f, -0.9f, -0.0f,	// left down
			0.0f, 0.0f, 0.0f,
			0.9f, -0.5f, 0.0f		// right down
		};
	#endif

	#if 1
		glm::vec3 vertex_buffer_data[] = {
			glm::vec3(-0.9f, -0.9f, -0.0f),
			glm::vec3(0.0f, 0.2f, 0.0f),
			glm::vec3(0.9f, -0.5f, 0.0f),
			glm::vec3(-0.9f, -0.9f, -0.0f),
			glm::vec3(0.0f, 0.2f, 0.0f),
			glm::vec3(0.0f, 0.0f, 0.5f),
			glm::vec3(-0.9f, -0.9f, -0.0f),
			glm::vec3(0.9f, -0.5f, 0.0f),
			glm::vec3(0.0f, 0.0f, 0.5f),
			glm::vec3(0.0f, 0.2f, 0.0f),
			glm::vec3(0.9f, -0.5f, 0.0f),
			glm::vec3(0.0f, 0.0f, 0.5f)
		};
	#endif

	#if 0
		static GLfloat vertex_buffer_data[] = {
			-1.0f,-1.0f,-1.0f, // Треугольник 1 : начало
			-1.0f,-1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f, // Треугольник 1 : конец
			1.0f, 1.0f,-1.0f, // Треугольник 2 : начало
			-1.0f,-1.0f,-1.0f,
			-1.0f, 1.0f,-1.0f, // Треугольник 2 : конец
			1.0f,-1.0f, 1.0f,
			-1.0f,-1.0f,-1.0f,
			1.0f,-1.0f,-1.0f,
			1.0f, 1.0f,-1.0f,
			1.0f,-1.0f,-1.0f,
			-1.0f,-1.0f,-1.0f,
			-1.0f,-1.0f,-1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f,-1.0f,
			1.0f,-1.0f, 1.0f,
			-1.0f,-1.0f, 1.0f,
			-1.0f,-1.0f,-1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f,-1.0f, 1.0f,
			1.0f,-1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f,-1.0f,-1.0f,
			1.0f, 1.0f,-1.0f,
			1.0f,-1.0f,-1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f,-1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f,-1.0f,
			-1.0f, 1.0f,-1.0f,
			1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f,-1.0f,
			-1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f,
			1.0f,-1.0f, 1.0f
		};
	#endif

	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data),
								vertex_buffer_data, GL_STATIC_DRAW);
	
	return 0;
}

int initColorBuffer()
{
	glGenBuffers(1, &colorBufferID);

	glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);

	#if 0
		static const GLfloat color_buffer_data[] = { 
			1.0f, 0.0f, 0.0f,	// read
			0.0f, 1.0f, 0.0f,	// green
			0.0f, 0.0f, 1.0f 	// blue
		};
	#endif

	#if 1
		static const GLfloat color_buffer_data[] = { 
			0.583f,  0.771f,  0.014f,
			0.609f,  0.115f,  0.436f,
			0.327f,  0.483f,  0.844f,
			0.822f,  0.569f,  0.201f,
			0.435f,  0.602f,  0.223f,
			0.310f,  0.747f,  0.185f,
			0.597f,  0.770f,  0.761f,
			0.559f,  0.436f,  0.730f,
			0.359f,  0.583f,  0.152f,
			0.483f,  0.596f,  0.789f,
			0.559f,  0.861f,  0.639f,
			0.195f,  0.548f,  0.859f
		};
	#endif

	#if 0
		static const GLfloat color_buffer_data[] = {
			0.583f,  0.771f,  0.014f,
			0.609f,  0.115f,  0.436f,
			0.327f,  0.483f,  0.844f,
			0.822f,  0.569f,  0.201f,
			0.435f,  0.602f,  0.223f,
			0.310f,  0.747f,  0.185f,
			0.597f,  0.770f,  0.761f,
			0.559f,  0.436f,  0.730f,
			0.359f,  0.583f,  0.152f,
			0.483f,  0.596f,  0.789f,
			0.559f,  0.861f,  0.639f,
			0.195f,  0.548f,  0.859f,
			0.014f,  0.184f,  0.576f,
			0.771f,  0.328f,  0.970f,
			0.406f,  0.615f,  0.116f,
			0.676f,  0.977f,  0.133f,
			0.971f,  0.572f,  0.833f,
			0.140f,  0.616f,  0.489f,
			0.997f,  0.513f,  0.064f,
			0.945f,  0.719f,  0.592f,
			0.543f,  0.021f,  0.978f,
			0.279f,  0.317f,  0.505f,
			0.167f,  0.620f,  0.077f,
			0.347f,  0.857f,  0.137f,
			0.055f,  0.953f,  0.042f,
			0.714f,  0.505f,  0.345f,
			0.783f,  0.290f,  0.734f,
			0.722f,  0.645f,  0.174f,
			0.302f,  0.455f,  0.848f,
			0.225f,  0.587f,  0.040f,
			0.517f,  0.713f,  0.338f,
			0.053f,  0.959f,  0.120f,
			0.393f,  0.621f,  0.362f,
			0.673f,  0.211f,  0.457f,
			0.820f,  0.883f,  0.371f,
			0.982f,  0.099f,  0.879f
		};
	#endif

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
