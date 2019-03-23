#include <cuda.h>
#include <stdio.h>

void printMatrix(float *matrix, int rows, int columns)
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++)
			printf("%g\t", matrix[i * columns + j]);
		printf("\n");
	}
	printf("\n");
}

void Output(float* a, int N)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			fprintf(stdout, "%g\t", a[j + i * N]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
}

__global__ void initMatrix_1D(float *matrix)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	matrix[i] = i;
}

__global__ void initMatrix(float *matrix)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int I = gridDim.x * blockDim.x;

	matrix[i + j * I] = (float) (i + j * I);
}

__global__ void transp(float *matrix, float *matrix_t, int N)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	matrix_t[y * N + x] = matrix[x * N + y];
}

int main(int argc, char *argv[])
{	
	int N = (argc > 1) ? atoi(argv[1]) : 4;
	int size_matrix = N * N;
	int block_x = (argc > 2) ? atoi(argv[2]) : 1;
	int block_y = block_x;

	float *dmatrix1, *hmatrix1;
	float *dmatrix2, *hmatrix2;

	cudaMalloc((void**) &dmatrix1, size_matrix * sizeof(float));
	cudaMalloc((void**) &dmatrix2, size_matrix * sizeof(float));
	hmatrix1 = (float*) calloc(size_matrix, sizeof(float));
	hmatrix2 = (float*) calloc(size_matrix, sizeof(float));

	dim3 dimGrid = dim3(N / block_x, N / block_y, 1);
	dim3 dimBlock = dim3(block_x, block_y, 1);

	printf("Size matrix(%dx%d): %d\n", N, N, N * N);
	printf("gridDim.x = %d gridDim.y = %d\n", dimGrid.x, dimGrid.y);
	printf("blockDim.x = %d blockDim.y = %d\n", dimBlock.x, dimBlock.y);

	initMatrix<<<dimGrid, dimBlock>>>(dmatrix1);
	cudaDeviceSynchronize();

	cudaMemcpy(hmatrix1, dmatrix1, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	Output(hmatrix1, N);

	transp<<<dimGrid, dimBlock>>>(dmatrix1, dmatrix2, N);
	cudaDeviceSynchronize();

	cudaMemcpy(hmatrix2, dmatrix2, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	Output(hmatrix2, N);

	#if 0
		float *test_matrix = (float*) calloc(size_matrix, sizeof(float));
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				test_matrix[j * N + i] = hmatrix1[i * N + j];
			}
		}
		
		Output(test_matrix, N);

		free(test_matrix);
	#endif

	cudaFree(dmatrix1);
	cudaFree(dmatrix2);
	free(hmatrix1);
	free(hmatrix2);

	return 0;
}
