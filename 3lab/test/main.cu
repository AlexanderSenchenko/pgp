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

__global__ void initMatrix_1D(float *matrix)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	matrix[i] = i;
}

__global__ void initMatrix_2D_I(float *matrix)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int I = gridDim.x * blockDim.x;
	// int J = gridDim.y * blockDim.y;

	// matrix[j * I + i] = j * I + i;

	// matrix[i + j * I] = i;
	// matrix[i + j * I] = j;
	// matrix[i + j * I] = I;
	// matrix[i + j * I] = J;
	matrix[i + j * I] = threadIdx.x;
	// matrix[i + j * I] = threadIdx.y;
	// matrix[i + j * I] = gridDim.x;
	// matrix[i + j * I] = gridDim.y;
	// matrix[i + j * I] = blockDim.x;
	// matrix[i + j * I] = blockDim.y;
	// matrix[i + j * I] = blockIdx.x;
	// matrix[i + j * I] = blockIdx.y;
}

__global__ void initMatrix_2D_J(float *matrix)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	// int I = gridDim.x * blockDim.x;
	int J = gridDim.y * blockDim.y;

	matrix[j + i * J] = j + i * J;
}

__global__ void transp(float *matrix1, float *matrix2, float *check)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int I = gridDim.x * blockDim.x;	

	matrix2[i * I + j] = matrix1[j * I + i];
	check[i * I + j] = j * I + i;
}

int main(int argc, char *argv[])
{
	int blocks = (argc > 1) ? atoi(argv[1]) : 8;
	int threads = (argc > 2) ? atoi(argv[2]) : 4;
	int rows = 8;
	int columns = 8;
	int size_matrix = rows * columns;

	float *dmatrix1, *hmatrix1;
	float *dmatrix2, *hmatrix2;
	float *dcheck, *hcheck;
	float *dmatrix3, *hmatrix3;

	cudaMalloc((void**) &dmatrix1, size_matrix * sizeof(float));
	cudaMalloc((void**) &dmatrix2, size_matrix * sizeof(float));
	cudaMalloc((void**) &dcheck, size_matrix * sizeof(float));
	cudaMalloc((void**) &dmatrix3, size_matrix * sizeof(float));
	
	hmatrix1 = (float*) calloc(size_matrix, sizeof(float));
	hmatrix2 = (float*) calloc(size_matrix, sizeof(float));
	hcheck = (float*) calloc(size_matrix, sizeof(float));
	hmatrix3 = (float*) calloc(size_matrix, sizeof(float));

	initMatrix_2D_I<<<dim3(2, 2), dim3(8)>>>(dmatrix3);
	cudaDeviceSynchronize();
	cudaMemcpy(hmatrix3, dmatrix3, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	printMatrix(hmatrix3, rows, columns);

	// initMatrix_1D<<<dim3(blocks), dim3(threads)>>>(dmatrix1);
	initMatrix_2D_I<<<dim3(blocks), dim3(2, 2)>>>(dmatrix1);
	cudaDeviceSynchronize();
	cudaMemcpy(hmatrix1, dmatrix1, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	printMatrix(hmatrix1, rows, columns);

#if 0
	initMatrix_2D_J<<<dim3(blocks), dim3(2, 2)>>>(dmatrix2);
	cudaDeviceSynchronize();
	cudaMemcpy(hmatrix2, dmatrix2, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);

	printMatrix(hmatrix2, rows, columns);
#endif

	transp<<<dim3(blocks), dim3(2, 2)>>>(dmatrix1, dmatrix2, dcheck);	
	cudaDeviceSynchronize();

	cudaMemcpy(hmatrix2, dmatrix2, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(hmatrix2, rows, columns);

	cudaMemcpy(hcheck, dcheck, size_matrix * sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(hcheck, rows, columns);

	cudaFree(dmatrix1);
	cudaFree(dmatrix2);
	cudaFree(dcheck);
	free(hmatrix1);
	free(hmatrix2);
	free(hcheck);

	return 0;
}
