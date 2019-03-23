#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

double wtime()
{
 	struct timeval tv;
 	gettimeofday(&tv, NULL);
 	return tv.tv_sec + tv.tv_usec * 1E-6;
}

__global__ void gTest(float* a)
{
	a[threadIdx.x + blockDim.x * blockIdx.x] 
		= (float)(threadIdx.x + blockDim.x * blockIdx.x);
}

void firstStart()
{
	float *da, *ha;
	int num_of_blocks = 2, threads_per_block = 32;
	int N = num_of_blocks * threads_per_block;

	ha = (float*) calloc(N, sizeof(float));
	cudaMalloc((void**) &da, N * sizeof(float));

	gTest<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(da);
	cudaDeviceSynchronize();
	cudaMemcpy(ha, da, N * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++)
		printf("%g\n", ha[i]);
	
	free(ha);
	cudaFree(da);
}

__global__ void gSGEVV(float* a, float* b, float* c)
{
	c[threadIdx.x + blockDim.x * blockIdx.x] =
		a[threadIdx.x + blockDim.x * blockIdx.x] +
		b[threadIdx.x + blockDim.x * blockIdx.x];
}

__global__ void gInitArray(float* a, float x)
{
	a[threadIdx.x + blockDim.x * blockIdx.x] = x;
}

__global__ void gSGEVV_iter(float* a, float* b, float* c, int n, int N)
{
	for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += N)
		c[i] = a[i] + b[i];
}

__global__ void gInitArray_iter(float* a, float x, int n, int N)
{
	for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += N)
		a[i] = x;
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

int main(int argc, char *argv[])
{
	float *a, *b, *c, *d;

	int num_of_blocks = (argc > 1) ? atoi(argv[1]) : 1024;
	// int threads_per_block = (argc > 2) ? atoi(argv[2]) : 32;
	int threads_per_block = 1;

	int N = num_of_blocks * threads_per_block;
	int size_array;

	double time;

	float elapsedTime;

	cudaEvent_t stop, start;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	#if 0
		size_array = N;

		cudaMalloc((void**) &a, size_array * sizeof(float));
		cudaMalloc((void**) &b, size_array * sizeof(float));
		cudaMalloc((void**) &c, size_array * sizeof(float));
		d = (float*) calloc(size_array, sizeof(float));

		for (int i = num_of_blocks; threads_per_block <= 128; i /= 2, threads_per_block *= 2) {
			gInitArray<<<dim3(i), dim3(threads_per_block)>>>(a, 1.0);
			gInitArray<<<dim3(i), dim3(threads_per_block)>>>(b, 2.0);

			time = wtime();
			gSGEVV<<<dim3(i), dim3(threads_per_block)>>>(a, b, c);
			cudaDeviceSynchronize();
			time = wtime() - time;

			cudaMemcpy(d, c, size_array * sizeof(float), cudaMemcpyDeviceToHost);

			if (argc > 2 && atoi(argv[2]) == 1) { 
				for (int j = 0; j < N; j++)
					printf("%g ", d[j]);
				printf("\n");
			}

			printf("Blocks: %d,\tThreads: %d,\t", i, threads_per_block);
			printf("Time: %.8f sec.\n", time);
			
			gInitArray<<<dim3(i), dim3(threads_per_block)>>>(c, 0.0);
		}

		free(d);
		cudaFree(a);
		cudaFree(b);
		cudaFree(c);
	#endif

	#if 1
		int rank = (argc > 3) ? atoi(argv[3]) : 10;

		// size_array = 1 << rank;
		size_array = 1 << 10;

		CUDA_CHECK_RETURN(cudaMalloc((void**) &a, size_array * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &b, size_array * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &c, size_array * sizeof(float)));
		d = (float*) calloc(size_array, sizeof(float));

		printf("Size vector: %d(10^%d)\n", size_array, rank);

		// int blocks = num_of_blocks;
		// int threads = threads_per_block;

		int blocks = 1;
		int threads = 1025;

		// for (; threads <= 128; blocks /= 2, threads *= 2) {
			gInitArray_iter<<<dim3(blocks), dim3(threads)>>>(a, 1.0, size_array, N);
			gInitArray_iter<<<dim3(blocks), dim3(threads)>>>(b, 2.0, size_array, N);

			time = wtime();
			cudaEventRecord(start, 0);
						
			gSGEVV_iter<<<dim3(blocks), dim3(threads)>>>(a, b, c, size_array, N);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaGetLastError());
			
			time = wtime() - time;
			cudaEventRecord(stop, 0);

			cudaEventSynchronize(stop);

			cudaEventElapsedTime(&elapsedTime, start, stop);

			CUDA_CHECK_RETURN(cudaMemcpy(d, c, size_array * sizeof(float), cudaMemcpyDeviceToHost));

			if (argc > 2 && atoi(argv[2]) == 1) { 
				for (int j = 0; j < size_array; j++)
					printf("%g ", d[j]);
				printf("\n");
			}

			printf("Blocks: %d\tThreads: %d\t", blocks, threads);
			// printf("Time: %.8f sec.\t", time);
			printf("Time(2): %.8f sec.\n", elapsedTime);
			
			gInitArray_iter<<<dim3(blocks), dim3(threads)>>>(c, 0.0, size_array, N);
		// }

		free(d);
		cudaFree(a);
		cudaFree(b);
		cudaFree(c);
	#endif

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	#if 0
		FILE *out_1024_1 = fopen("1024_1.txt", "w");
		FILE *out_512_2 = fopen("512_2.txt", "w");
		FILE *out_128_8 = fopen("128_8.txt", "w");
		FILE *out_32_32 = fopen("32_32.txt", "w");
		FILE *out_8_128 = fopen("8_128.txt", "w");

		for (int rank = 10; rank <= 23; rank++) {
			size_array = 1 << rank;

			cudaMalloc((void**) &a, size_array * sizeof(float));
			cudaMalloc((void**) &b, size_array * sizeof(float));
			cudaMalloc((void**) &c, size_array * sizeof(float));
			d = (float*) calloc(size_array, sizeof(float));

			printf("Size vector: %d(10^%d)\n", size_array, rank);

			int blocks = num_of_blocks;
			int threads = threads_per_block;

			for (; threads <= 128; blocks /= 2, threads *= 2) {
				gInitArray_iter<<<dim3(blocks), dim3(threads)>>>(a, 1.0, size_array, N);
				gInitArray_iter<<<dim3(blocks), dim3(threads)>>>(b, 2.0, size_array, N);

				time = wtime();
				gSGEVV_iter<<<dim3(blocks), dim3(threads)>>>(a, b, c, size_array, N);
				cudaDeviceSynchronize();
				time = wtime() - time;

				cudaMemcpy(d, c, size_array * sizeof(float), cudaMemcpyDeviceToHost);

				if (argc > 2 && atoi(argv[2]) == 1) { 
					for (int j = 0; j < size_array; j++)
						printf("%g ", d[j]);
					printf("\n");
				}

				printf("Blocks: %d\tThreads: %d\t", blocks, threads);
				printf("Time: %.8f sec.\n", time);
			
				switch (threads) {
					case 1:
						fprintf(out_1024_1, "%d %.8f\n", size_array, time);
						break;
					case 2:
						fprintf(out_512_2, "%d %.8f\n", size_array, time);
						break;
					case 8:
						fprintf(out_128_8, "%d %.8f\n", size_array, time);
						break;
					case 32:
						fprintf(out_32_32, "%d %.8f\n", size_array, time);
						break;
					case 128:
						fprintf(out_8_128, "%d %.8f\n", size_array, time);
						break;
					default:
						break;
				}

				gInitArray_iter<<<dim3(blocks), dim3(threads)>>>(c, 0.0, size_array, N);
			}

			free(d);
			cudaFree(a);
			cudaFree(b);
			cudaFree(c);
		}

		fclose(out_1024_1);
		fclose(out_512_2);
		fclose(out_128_8);
		fclose(out_32_32);
		fclose(out_8_128);
	#endif

	#if 0
		time = wtime();
		gSGEVV<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(a, b, c);
		cudaDeviceSynchronize();
		time = wtime() - time;

		cudaMemcpy(d, c, size_array * sizeof(float), cudaMemcpyDeviceToHost);
		
		for (int i = 0; i < N; i++)
			printf("%g\n", d[i]);

		printf("Blocks: %d, Threads: %d, ", num_of_blocks, threads_per_block);
		printf("Time: %.6f sec.\n", time);
	#endif

	#if 0
		firstStart();

		free(d);
		cudaFree(a);
		cudaFree(b);
		cudaFree(c);
	#endif

	return 0;
}

