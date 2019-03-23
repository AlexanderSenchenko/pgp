#include <cuda.h>
#include <stdio.h>

__global__ void initVector(float* vector, float value)
{
	vector[threadIdx.x + blockDim.x * blockIdx.x] = value;
}

int main(int argc, char *argv[])
{
	int blocks = 1024;
	int threads = 1;
	int size_vector = blocks * threads;
	float time;
	float value = 1.0;

	float *dvector, *hvector;

	cudaMalloc((void**) &dvector, size_vector * sizeof(float));
	hvector = (float*) calloc(size_vector, sizeof(float));

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (; threads <= 1024; blocks /= 2, threads *= 2, value++) {
		cudaEventRecord(start, 0);

		initVector<<<dim3(blocks), dim3(threads)>>>(dvector, value);	
	
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaDeviceSynchronize();

		cudaEventElapsedTime(&time, start, stop);

		cudaMemcpy(hvector, dvector, size_vector * sizeof(float), cudaMemcpyDeviceToHost);

		printf("Time(%d, %d):\t%.8f\n", blocks, threads, time);

		#if 0
		for (int i = 0; i < size_vector; i++) {
			printf("%g ", hvector[i]);
		}
		printf("\n");
		#endif

		time = 0.0;
		initVector<<<dim3(blocks), dim3(threads)>>>(dvector, 0.0);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dvector);
	free(hvector);

	return 0;
}

