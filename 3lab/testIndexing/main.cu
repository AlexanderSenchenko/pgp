#include <cuda.h>
#include <stdio.h>

void printMatrix(float *matrix, int rows, int columns)
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++)
			printf("%g ", matrix[i * rows + j]);
		printf("\n");
	}
	printf("\n");
}

#define CUDA_CHECK_RETURN(value)\
{\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
 		fprintf(stderr, "Error %s at line %d in file %s\n",\
 		cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
 		exit(1);\
	}\
}

__global__ void initMatrix_2D_I(float *matrix, float value)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int I = gridDim.x * blockDim.x;

	matrix[j * I + i] = value;
}

__global__ void initMatrix_2D_J(float *matrix, float value)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int J = gridDim.y * blockDim.y;

	matrix[j + i * J] = value;
}

int main(int argc, char *argv[])
{
	int rows = (argc > 1) ? atoi(argv[1]) : 32;
	int columns = (argc > 2) ? atoi(argv[2]) : 32;
	int size_matrix = rows * columns;

	int block_x = (argc > 3) ? atoi(argv[3]) : 4;
	int thread_x = (argc > 4) ? atoi(argv[4]) : 8;

	float time1, time2, time3, time4;

	cudaEvent_t stop, start;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// float *dmatrix1, *hmatrix1;
	// float *dmatrix2, *hmatrix2;
	float *dmatrix;
	// float *hmatrix1, *hmatrix2, *hmatrix3, *hmatrix4;

	// cudaMalloc((void**) &dmatrix1, size_matrix * sizeof(float));
	// cudaMalloc((void**) &dmatrix2, size_matrix * sizeof(float));	
	cudaMalloc((void**) &dmatrix, size_matrix * sizeof(float));

	printf("Size matrix (%d * %d): %d\n", rows, columns, size_matrix);
	printf("Threads: %d\n\n", block_x * block_x * thread_x * thread_x);

	#if 1	
	// Matrix 1 (block_x * block_y, thread_x) 
	float *hmatrix1 = (float*) calloc(size_matrix, sizeof(float));
	
	cudaEventRecord(start, 0);

	initMatrix_2D_I<<<dim3(block_x, block_x), dim3(thread_x * thread_x)>>>(dmatrix, 1.0);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();

	cudaEventElapsedTime(&time1, start, stop);

	cudaMemcpy(hmatrix1, dmatrix, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	if (argc > 5 && atoi(argv[5]) == 1)
		printMatrix(hmatrix1, rows, columns);
	
	free(hmatrix1);
	#endif

	#if 1
	// Matrix 2 (block_x, thread_x * thread_y)
	float *hmatrix2 = (float*) calloc(size_matrix, sizeof(float));

	cudaEventRecord(start, 0);

	initMatrix_2D_I<<<dim3(block_x * block_x), dim3(thread_x, thread_x)>>>(dmatrix, 2.0);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();

	cudaEventElapsedTime(&time2, start, stop);

	cudaMemcpy(hmatrix2, dmatrix, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	if (argc > 5 && atoi(argv[5]) == 1)
		printMatrix(hmatrix2, rows, columns);
	
	free(hmatrix2);
	#endif

	#if 1
	// Matrix 3 (block_x * block_y, thread_x)
	float *hmatrix3 = (float*) calloc(size_matrix, sizeof(float));
	
	cudaEventRecord(start, 0);

	initMatrix_2D_J<<<dim3(block_x, block_x), dim3(thread_x * thread_x)>>>(dmatrix, 3.0);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();

	cudaEventElapsedTime(&time3, start, stop);

	cudaMemcpy(hmatrix3, dmatrix, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);
	
	if (argc > 5 && atoi(argv[5]) == 1)
		printMatrix(hmatrix3, rows, columns);

	printf("Time(%dx%d, %d)_J: %.8f\n", block_x, block_x, thread_x * thread_x, time3);

	free(hmatrix3);
	#endif

	#if 1
	// Matrix 4 (block_x, thread_x * thread_y)
	float *hmatrix4 = (float*) calloc(size_matrix, sizeof(float));	

	cudaEventRecord(start, 0);

	initMatrix_2D_J<<<dim3(block_x * block_x), dim3(thread_x, thread_x)>>>(dmatrix, 4.0);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaDeviceSynchronize();

	cudaEventElapsedTime(&time4, start, stop);
	
	cudaMemcpy(hmatrix4, dmatrix, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	if (argc > 5 && atoi(argv[5]) == 1)
		printMatrix(hmatrix4, rows, columns);

	printf("Time(%d, %dx%d)_J: %.8f\n\n", block_x * block_x, thread_x, thread_x, time4);
	
	free(hmatrix4);
	#endif

	printf("Time(%dx%d, %d)_I: %.8f\n", block_x, block_x, thread_x * thread_x, time1);
	printf("Time(%d, %dx%d)_I: %.8f\n", block_x * block_x, thread_x, thread_x, time2);
	printf("Time(%dx%d, %d)_J: %.8f\n", block_x, block_x, thread_x * thread_x, time3);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//cudaFree(dmatrix1);
	//cudaFree(dmatrix2);
	cudaFree(dmatrix);
	// free(hmatrix1);
	// free(hmatrix2);
	// free(hmatrix3);
	// free(hmatrix4);

	return 0;
}
