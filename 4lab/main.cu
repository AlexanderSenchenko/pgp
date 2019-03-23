#include <stdio.h>
#include <cuda.h>

void Output(float* a, int N)
{
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			fprintf(stdout, "%g\t",a[j + i * N]);
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n\n\n");
}

__global__ void gInitializeStorage(float* storage_d)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int N = blockDim.x * gridDim.x;

	storage_d[i + j * N] = (float) (i + j * N);
}

__global__ void gTranspose0(float* storage_d, float* storage_d_t)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int N = blockDim.x * gridDim.x;

	storage_d_t[j + i * N] = storage_d[i + j * N];
}

__global__ void gTranspose11(float* storage_d, float* storage_d_t)
{
	extern __shared__ float buffer[];
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int N = blockDim.x * gridDim.x;

	buffer[threadIdx.y + threadIdx.x * blockDim.y] = storage_d[i + j * N];
	__syncthreads();

	i = threadIdx.x + blockIdx.y * blockDim.x;
	j = threadIdx.y + blockIdx.x * blockDim.y;

	storage_d_t[i + j * N] = buffer[threadIdx.x + threadIdx.y * blockDim.x];
}

#define SH_DIM 32
__global__ void gTranspose12(float* storage_d, float* storage_d_t)
{
	__shared__ float buffer_s[SH_DIM][SH_DIM];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int N = blockDim.x * gridDim.x;

	buffer_s[threadIdx.y][threadIdx.x] = storage_d[i + j * N];
	__syncthreads();

	i = threadIdx.x + blockIdx.y * blockDim.x;
	j = threadIdx.y + blockIdx.x * blockDim.y;
	storage_d_t[i + j * N] = buffer_s[threadIdx.x][threadIdx.y];
}

__global__ void gTranspose2(float* storage_d, float* storage_d_t)
{
	__shared__ float buffer[SH_DIM][SH_DIM + 1];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int N = blockDim.x * gridDim.x;

	buffer[threadIdx.y][threadIdx.x] = storage_d[i + j * N];
	__syncthreads();

	i = threadIdx.x + blockIdx.y * blockDim.x;
	j = threadIdx.y + blockIdx.x * blockDim.y;
	storage_d_t[i + j * N] = buffer[threadIdx.x][threadIdx.y];
}

int main(int argc, char const *argv[])
{
	if (argc < 3) {
		fprintf(stderr, "USAGE: matrix <dimension of matrix> <dimension_of_threads>\n");
		return -1;
	}

	int N = (argc > 1) ? atoi(argv[1]) : 256;
	int blocks_x = (argc > 2) ? atoi(argv[2]) : 32;
	int blocks_y = blocks_x;
	int dim_of_threads = blocks_x;

	dim3 dimGrid = dim3(N / blocks_x, N / blocks_y, 1);
	dim3 dimBlock = dim3(blocks_x, blocks_y, 1);

	printf("Size matrix(%dx%d): %d\n", N, N, N * N);
	printf("gridDim.x = %d gridDim.y = %d\n", dimGrid.x, dimGrid.y);
	printf("blockDim.x = %d blockDim.y = %d\n", dimBlock.x, dimBlock.y);

	if (N % dim_of_threads) {
		fprintf(stderr, "change dimensions\n");
		return -1;
	}

	int dim_of_blocks = N / dim_of_threads;
	const int max_size = 1 << 8;

	if(dim_of_blocks > max_size) {
		fprintf(stderr, "too many blocks\n");
		return -1;
	}

	float *storage_d, *storage_d_t, *storage_h;

	cudaMalloc((void**) &storage_d, N * N * sizeof(float));
	cudaMalloc((void**) &storage_d_t, N * N * sizeof(float));
	storage_h = (float*) calloc(N * N, sizeof(float));

	gInitializeStorage<<<dimGrid,dimBlock>>>(storage_d);
	cudaDeviceSynchronize();

	memset(storage_h, 0.0, N * N * sizeof(float));
	cudaMemcpy(storage_h, storage_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	if (argc > 3)
		Output(storage_h, N);

	#if 1
	// Transpose 0
	gTranspose0<<<dimGrid, dimBlock>>>(storage_d, storage_d_t);
	cudaDeviceSynchronize();

	memset(storage_h, 0.0, N * N * sizeof(float));
	cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	if (argc > 3)
		Output(storage_h, N);
	#endif

	#if 1
	// Transpose 11
	gTranspose11<<<dimGrid, dimBlock,
					dim_of_threads * dim_of_threads * sizeof(float)>>>(storage_d, storage_d_t);
	cudaDeviceSynchronize();

	memset(storage_h, 0.0, N * N * sizeof(float));
	cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	if (argc > 3)
		Output(storage_h, N);
	#endif

	#if 1
	// Transpose 12
	gTranspose12<<<dimGrid, dimBlock>>>(storage_d, storage_d_t);
	cudaDeviceSynchronize();

	memset(storage_h, 0.0 , N * N * sizeof(float));
	cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	if (argc > 3)
		Output(storage_h, N);
	#endif

	#if 1
	// Transpose 2
	gTranspose2<<<dimGrid, dimBlock>>>(storage_d, storage_d_t);
	cudaDeviceSynchronize();

	memset(storage_h, 0.0, N * N * sizeof(float));
	cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	if (argc > 3)
		Output(storage_h, N);
	#endif

	cudaFree(storage_d);
	cudaFree(storage_d_t);
	free(storage_h);

	return 0;
}
