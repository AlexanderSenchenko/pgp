#include <cuda.h>
#include <stdio.h>

__global__ void gTest(float* a) {
	a[threadIdx.x + blockDim.x * blockIdx.x]
		= (float) (threadIdx.x + blockDim.x * blockIdx.x);
}

__global__ void gSGEVV(float* a, float* b, float* c) {
	c[threadIdx.x + blockDim.x * blockIdx.x] =
		a[threadIdx.x + blockDim.x * blockIdx.x] +
		b[threadIdx.x + blockDim.x * blockIdx.x];

}

__global__ void gInitArray(float* a, float x) {
	a[threadIdx.x + blockDim.x * blockIdx.x] = x;
}

int main(int argc, char *argv[]) {
	float *a, *d;

	int num_of_blocks = (argc > 1) ? atoi(argv[1]) : 32;
	int threads_per_block = (argc > 2) ? atoi(argv[2]) : 512;

	int N = num_of_blocks * threads_per_block;
	int size_array = N;

	cudaMalloc((void**) &a, size_array * sizeof(float));
	d = (float*) calloc(size_array, sizeof(float));

	gInitArray<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(a, 1.0);

	cudaDeviceSynchronize();

	cudaMemcpy(d, a, size_array * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Blocks: %d\nThreads: %d\n", num_of_blocks, threads_per_block);
	printf("Size array: %d\n", size_array);

	free(d);
	cudaFree(a);
	
	return 0;
}
