#include <stdio.h>
#include <stdlib.h>

int main()
{
	int rows = 8;
	int columns = 8;
	int size = rows * columns;
	float *matrix1, *matrix2;

	matrix1 = (float*) calloc(size, sizeof(float));
	matrix2 = (float*) calloc(size, sizeof(float));

	for (int i = 0; i < rows / 2; i++) {
		for (int j = 0; j < columns; j++) {
			matrix1[i * columns + j] = i * columns + j;
		}
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%0.1f\t", matrix1[i * columns + j]);
		}
		printf("\n");
	}
	printf("\n");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			matrix2[j * columns + i] = matrix1[i * columns + j];
		}
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%0.1f\t", matrix2[i * columns + j]);
		}
		printf("\n");
	}
	printf("\n");

	free(matrix1);
	free(matrix2);

	return 0;
}
