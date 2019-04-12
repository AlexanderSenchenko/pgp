#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < N) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}

__global__ void gSumVector(int *a, int *b, int *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N)
		c[idx] = a[idx] * b[idx];
}

__global__ void gScalarMultVect(int *a, int *b, int *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N)
		c[idx] = a[idx] * b[idx];
}

int main()
{
	cudaDeviceProp prop;
	int whichDevice;

	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
   
	if (!prop.deviceOverlap) {
		printf("Device does not support overlapping\n");
		return 0;
	}

	float time1, time2, time3, time4;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *host_a, *host_b, *host_c;
	cudaHostAlloc((void**) &host_a, FULL_DATA_SIZE * sizeof(int),
	cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_b, FULL_DATA_SIZE * sizeof(int),
	cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_c, FULL_DATA_SIZE * sizeof(int),
	cudaHostAllocDefault);

	int *dev_a, *dev_b, *dev_c;
	cudaMalloc((void**) &dev_a, N * sizeof(int));
	cudaMalloc((void**) &dev_b, N * sizeof(int));
	cudaMalloc((void**) &dev_c, N * sizeof(int));

	int *dev_a0, *dev_a1, *dev_b0, *dev_b1, *dev_c0, *dev_c1;
	cudaMalloc((void**) &dev_a0, N * sizeof(int));
	cudaMalloc((void**) &dev_a1, N * sizeof(int));
	cudaMalloc((void**) &dev_b0, N * sizeof(int));
	cudaMalloc((void**) &dev_b1, N * sizeof(int));
	cudaMalloc((void**) &dev_c0, N * sizeof(int));
	cudaMalloc((void**) &dev_c1, N * sizeof(int));

	cudaStream_t stream, stream0, stream1;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	cudaEventRecord(start, 0);
	for (int i = 0; i < FULL_DATA_SIZE; i += N) {
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);

		kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}

	cudaStreamSynchronize(stream);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time1, start, stop);

	cudaEventRecord(start, 0);
	for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
		cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);

		kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);

		cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);

		cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

		kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

		cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time2, start, stop);

	cudaEventRecord(start,0);
	for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
		cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

		cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
		kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

		cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time3, start, stop);

	int *host_a1, *host_b1, *host_c1;
	host_a1 = (int*) calloc(FULL_DATA_SIZE, sizeof(int));
	host_b1 = (int*) calloc(FULL_DATA_SIZE, sizeof(int));
	host_c1 = (int*) calloc(FULL_DATA_SIZE, sizeof(int));

	cudaEventRecord(start, 0);
	for (int i = 0; i < FULL_DATA_SIZE; i += N) {
		cudaMemcpy(dev_a, host_a1 + i, N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, host_b1 + i, N * sizeof(int), cudaMemcpyHostToDevice);

		kernel<<<N / 256, 256>>>(dev_a, dev_b, dev_c);

		cudaMemcpy(host_c1 + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time4, start, stop);

	printf("%.4f ms\n", time1);
	printf("%.4f ms\n", time2);
	printf("%.4f ms\n", time3);
	printf("%.4f ms\n", time4);

	free(host_a1);
	free(host_b1);
	free(host_c1);

	cudaStreamDestroy(stream);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaFree(dev_a0);
	cudaFree(dev_a1);
	cudaFree(dev_b0);
	cudaFree(dev_b1);
	cudaFree(dev_c0);
	cudaFree(dev_c1);

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
