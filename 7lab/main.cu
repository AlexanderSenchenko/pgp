#include <stdio.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#define STEPS 1
#define LENGTH 1024
#define U 0.1f
#define H 0.1f
#define TAU 0.2f
#define THREADS_PER_BLOCK 256

__global__ void kernel(float *fn_1, float *fn)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i == 0)
		fn[i] = fn_1[i] + 1;
	else
		fn[i] = fn_1[i] + (U * TAU / H) * (fn_1[i - 1] - fn_1[i]);
}

struct stepFunctor
{
	__host__ __device__
	float operator()(thrust::tuple<float&, float&> tuple)
	{
		float valueI_1 = thrust::get<0>(tuple);
		float valueI = thrust::get<1>(tuple); 
		return valueI + (U * TAU / H) * (valueI_1 - valueI);
	}
};

struct stepLeftEdgeFunctor
{
	__host__ __device__ float operator()(float fn_1)
	{ return fn_1 + 1; }
};

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float time;

	float *fn_1;
	float *fn;
	cudaMalloc((void**) &fn_1, LENGTH * sizeof(float));
	cudaMalloc((void**) &fn, LENGTH * sizeof(float));

	// PURE CUDA
	float *vector = new float[LENGTH];
	for (int i = 0; i < LENGTH; i++)
		vector[i] = i;

	cudaMemcpy(fn_1, vector, LENGTH * sizeof(float),
											cudaMemcpyHostToDevice);
	cudaMemcpy(fn,   vector, LENGTH * sizeof(float),
											cudaMemcpyHostToDevice);

#ifdef PRINTS
	float *temp = new float[LENGTH];
	cudaMemcpy(temp, fn_1, LENGTH * sizeof(float),
											cudaMemcpyDeviceToHost);
	for(int i = 0; i < LENGTH; i++)
		printf("%5.2f\t", temp[i]);
	printf("\n");
#endif

	cudaEventRecord(start);
	for (int i = 0; i < STEPS; i++) {
		kernel<<<LENGTH / THREADS_PER_BLOCK,
									THREADS_PER_BLOCK>>>(fn_1, fn);
		cudaDeviceSynchronize();
		cudaMemcpy(fn_1, fn, LENGTH * sizeof(float),
											cudaMemcpyDeviceToDevice);
		
#ifdef PRINTS
		cudaMemcpy(temp, fn, LENGTH * sizeof(float),
											cudaMemcpyDeviceToHost);
		for(int i = 0; i < LENGTH; i++)
			printf("%5.2f\t", temp[i]);
		printf("\n");
#endif
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Pure cuda: %f ms\n", time);

	// THRUST
	thrust::device_vector<float> vectorFn_1(LENGTH), vectorFn(LENGTH);
	thrust::sequence(vectorFn_1.begin(), vectorFn_1.end());	

#ifdef PRINTS
	thrust::host_vector<float> vectorHost(LENGTH);
	vectorHost = vectorFn_1;
	for(int i = 0; i < LENGTH; i++)
		printf("%5.2f\t", vectorHost[i]);
	printf("\n");
#endif
	
	cudaEventRecord(start);
	for (int i = 0; i < STEPS; i++) {
		thrust::transform(
			thrust::make_zip_iterator(
				thrust::make_tuple(vectorFn_1.begin(), 
									vectorFn_1.begin() + 1 )),
			thrust::make_zip_iterator(
				thrust::make_tuple(vectorFn_1.end() - 1,
									vectorFn_1.end() )),
			vectorFn.begin() + 1,
			stepFunctor()
		);
		thrust::transform(vectorFn_1.begin(), vectorFn_1.begin() + 1,
								vectorFn.begin(), stepLeftEdgeFunctor());
		vectorFn_1 = vectorFn;

#ifdef PRINTS
		vectorHost = vectorFn;
		for(int i = 0; i < LENGTH; i++)
			printf("%5.2f\t", vectorHost[i]);
		printf("\n");
#endif
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Thrust: %f ms\n", time);

#ifdef PRINTS
	delete[] temp;
#endif
	delete[] vector;
	cudaFree(fn_1);
	cudaFree(fn);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
