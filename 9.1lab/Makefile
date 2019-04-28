GLEW := -I/usr/include/libdrm -L/usr/lib64 -lGLEW -lGLU -lGL
GLFW := -I/usr/local/include -L/usr/local/lib -lglfw3
GLM := -I/usr/local/include
GLUT := -I/usr/local/include -L/usr/local/lib -lglut
HELP := -lm -lXrandr -lXi -lX11 -lXxf86vm -lpthread -ldl

all:
	g++ -Wall main.cpp -o main $(GLEW) $(GLFW) $(GLUT) $(HELP)