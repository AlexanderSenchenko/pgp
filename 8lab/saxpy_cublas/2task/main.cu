#include <cufft.h>
#include <stdio.h>
#include <malloc.h>

#define NX 64
#define BATCH 1
#define pi 3.141592

float array_data[][4] = {
	1,  2, 142, 999,
	1,  1, 999, 999,
	1,  3, 129, 999,
	1,  4, 112, 999,
	1,  5, 111, 999,
	1,  6,  98, 999,
	1,  7,  93, 999,
	1,  8, 128, 999,
	1,  9,  98, 999,
	1, 10, 106, 999,
	1, 11, 105, 999,
	1, 12, 124, 999,
	1, 13, 178, 999,
	1, 14, 124, 999,
	1, 15, 121, 999,
	1, 16, 138, 999,
	1, 17, 157, 999,
	1, 18, 152, 999,
	1, 19, 146, 999,
	1, 20,  83, 999,
	1, 21, 102, 999,
	1, 22, 122, 999,
	1, 23, 147, 999,
	1, 24,  79, 999,
	1, 25, 116, 999,
	1, 26, 111, 999,
	1, 27, 169, 999,
	1, 28, 200, 999,
	1, 29, 283, 999,
	1, 30, 340, 999,
	1, 31, 130, 999,
	2,  1, 204, 999,
	2,  2, 126, 999,
	2,  3, 118, 999,
	2,  4, 178, 999,
	2,  5, 160, 999,
	2,  6, 197, 999,
	2,  7, 159, 999,
	2,  8, 162, 999,
	2,  9, 136, 999,
	2, 10, 160, 999,
	2, 11, 167, 999,
	2, 12, 201, 999,
	2, 13, 196, 999,
	2, 14, 203, 999,
	2, 15, 267, 999,
	2, 16, 202, 999,
	2, 17,  98, 999,
	2, 18, 244, 999,
	2, 19, 252, 999,
	2, 20, 152, 999,
	2, 21, 243, 999,
	2, 22, 327, 999,
	2, 23, 267, 999,
	2, 24, 275, 999,
	2, 25, 292, 999,
	2, 26, 240, 999,
	2, 27, 230, 999,
	2, 28, 151, 999,
	3,  1, 153, 999,
	3,  2, 164, 999,
	3,  3,  92, 999,
	3,  4,  66, 999,
	3,  5,  74, 999,
	3,  6,  94, 999,
	3,  7,  98, 999,
	3,  8, 144, 999,
	3,  9, 118, 999,
	3, 10, 135, 999,
	3, 11, 188, 999,
	3, 12, 195, 999,
	3, 13, 156, 999,
	3, 14, 239, 999,
	3, 15, 240, 999,
	3, 16, 245, 999,
	3, 17, 189, 999,
	3, 18, 220, 999,
	3, 19, 241, 999,
	3, 20, 212, 999,
	3, 21, 231, 999,
	3, 22, 231, 999,
	3, 23, 224, 999,
	3, 24, 194, 999,
	3, 25, 147, 999,
	3, 26, 202, 999,
	3, 27, 132, 999,
	3, 28, 164, 999,
	3, 29, 198, 999,
	3, 30, 172, 999,
	3, 31, 159, 999,
	4,  1, 118, 999,
	4,  2,  94, 999,
	4,  3, 116, 999,
	4,  4, 167, 999,
	4,  5, 207, 999,
	4,  6, 219, 999,
	4,  7, 207, 999,
	4,  8, 190, 999,
	4,  9, 183, 999,
	4, 10, 220, 999,
	4, 11, 220, 999,
	4, 12, 320, 999,
	4, 13, 250, 999,
	4, 14, 265, 999,
	4, 15, 267, 999,
	4, 16, 201, 999,
	4, 17, 205, 999,
	4, 18, 206, 999,
	4, 19, 265, 999,
	4, 20, 219, 999,
	4, 21,  82, 999,
	4, 22, 100, 999,
	4, 23,  68, 999,
	4, 24,  43, 999,
	4, 25,  54, 999,
	4, 26,  93, 999,
	4, 27,  99, 999,
	4, 28, 115, 999,
	4, 29, 151, 999,
	4, 30, 177, 999,
	5,  1, 123, 999,
	5,  2,  77, 999,
	5,  3,  97, 999,
	5,  4, 146, 999,
	5,  5, 141, 999,
	5,  6, 135, 999,
	5,  7, 110, 999,
	5,  8, 131, 999,
	5,  9, 179, 999,
	5, 10, 163, 999,
	5, 11, 219, 999,
	5, 12, 184, 999,
	5, 13, 176, 999,
	5, 14, 182, 999,
	5, 15, 154, 999,
	5, 16, 106, 999,
	5, 17, 116, 999,
	5, 18, 146, 999,
	5, 19, 176, 999,
	5, 20, 160, 999,
	5, 21, 111, 999,
	5, 22, 999, 999,
	5, 23, 124, 999,
	5, 24, 135, 999,
	5, 25, 167, 999,
	5, 26, 210, 999,
	5, 27, 224, 999,
	5, 28, 999, 999,
	5, 29, 191, 999,
	5, 30, 170, 999,
	5, 31, 196, 999,
	6,  1, 219, 999,
	6,  2, 216, 999,
	6,  3, 190, 999,
	6,  4, 232, 999,
	6,  5, 211, 999,
	6,  6, 199, 999,
	6,  7, 204, 999,
	6,  8, 253, 999,
	6,  9, 246, 999,
	6, 10, 239, 999,
	6, 11, 321, 999,
	6, 12, 330, 999,
	6, 13, 263, 999,
	6, 14, 288, 999,
	6, 15, 220, 999,
	6, 16, 181, 999,
	6, 17, 163, 999,
	6, 18, 148, 999,
	6, 19, 185, 999,
	6, 20, 155, 999,
	6, 21, 182, 999,
	6, 22, 160, 999,
	6, 23, 136, 999,
	6, 24, 136, 999,
	6, 25, 157, 999,
	6, 26, 179, 999,
	6, 27, 165, 999,
	6, 28, 243, 999,
	6, 29, 249, 999,
	6, 30, 247, 999,
	7,  1, 245, 999,
	7,  2, 298, 999,
	7,  3, 342, 999,
	7,  4, 366, 999,
	7,  5, 331, 999,
	7,  6, 247, 999,
	7,  7, 218, 999,
	7,  8, 163, 999,
	7,  9, 218, 999,
	7, 10, 212, 999,
	7, 11, 168, 999,
	7, 12, 199, 999,
	7, 13, 165, 999,
	7, 14, 165, 999,
	7, 15, 118, 999,
	7, 16, 109, 999,
	7, 17, 106, 999,
	7, 18, 113, 999,
	7, 19, 140, 999,
	7, 20, 143, 999,
	7, 21, 173, 999,
	7, 22, 242, 999,
	7, 23, 195, 999,
	7, 24, 225, 999,
	7, 25, 236, 999,
	7, 26, 242, 999,
	7, 27, 216, 999,
	7, 28, 223, 999,
	7, 29, 203, 999,
	7, 30, 191, 999,
	7, 31, 194, 999,
	8,  1, 126, 999,
	8,  2, 159, 999,
	8,  3, 194, 999,
	8,  4, 181, 999,
	8,  5, 131, 999,
	8,  6,  96, 999,
	8,  7, 124, 999,
	8,  8, 138, 999,
	8,  9, 107, 999,
	8, 10,  89, 999,
	8, 11,  91, 999,
	8, 12,  97, 999,
	8, 13, 103, 999,
	8, 14, 138, 999,
	8, 15, 167, 999,
	8, 16, 248, 999,
	8, 17, 281, 999,
	8, 18, 352, 999,
	8, 19, 285, 999,
	8, 20, 289, 999,
	8, 21, 299, 999,
	8, 22, 263, 999,
	8, 23, 267, 999,
	8, 24, 999, 999,
	8, 25, 235, 999,
	8, 26, 181, 999,
	8, 27, 176, 999,
	8, 28, 153, 999,
	8, 29, 156, 999,
	8, 30, 196, 999,
	8, 31, 198, 999,
	9,  1, 167, 999,
	9,  2, 168, 999,
	9,  3, 149, 999,
	9,  4, 147, 999,
	9,  5, 136, 999,
	9,  6, 167, 999,
	9,  7,  91, 999,
	9,  8, 127, 999,
	9,  9, 149, 999,
	9, 10, 114, 999,
	9, 11, 113, 999,
	9, 12, 138, 999,
	9, 13, 163, 999,
	9, 14, 161, 999,
	9, 15, 152, 999,
	9, 16, 154, 999,
	9, 17, 105, 999,
	9, 18, 107, 999,
	9, 19, 171, 999,
	9, 20, 105, 999,
	9, 21,  88, 999,
	9, 22, 156, 999,
	9, 23, 153, 999,
	9, 24, 163, 999,
	9, 25, 125, 999,
	9, 26, 139, 999,
	9, 27, 141, 999,
	9, 28, 180, 999,
	9, 29, 154, 999,
	9, 30, 155, 999,
   10,  1, 179, 999,
   10,  2, 179, 999,
   10,  3, 210, 999,
   10,  4, 212, 999,
   10,  5, 235, 999,
   10,  6, 199, 999,
   10,  7,  85, 999,
   10,  8, 138, 999,
   10,  9, 146, 999,
   10, 10, 200, 999,
   10, 11, 199, 999,
   10, 12, 200, 999,
   10, 13, 168, 999,
   10, 14, 146, 999,
   10, 15, 202, 999,
   10, 16, 169, 999,
   10, 17, 129, 999,
   10, 18, 158, 999,
   10, 19, 114, 999,
   10, 20, 129, 999,
   10, 21, 154, 999,
   10, 22, 113, 999,
   10, 23, 131, 999,
   10, 24,  68, 999,
   10, 25, 130, 999,
   10, 26, 116, 999,
   10, 27, 999, 999,
   10, 28, 999, 999,
   10, 29, 180, 999,
   10, 30, 999, 999,
   10, 31, 266, 999,
   11,  1, 248, 999,
   11,  2, 224, 999,
   11,  3, 158, 999,
   11,  4, 136, 999,
   11,  5, 130, 999,
   11,  6, 164, 999,
   11,  7, 171, 999,
   11,  8, 157, 999,
   11,  9, 143, 999,
   11, 10, 184, 999,
   11, 11, 177, 999,
   11, 12, 126, 999,
   11, 13, 124, 999,
   11, 14, 117, 999,
   11, 15, 118, 999,
   11, 16,  92, 999,
   11, 17, 999, 999,
   11, 18,  87, 999,
   11, 19, 103, 999,
   11, 20, 119, 999,
   11, 21, 120, 999,
   11, 22, 102, 999,
   11, 23, 119, 999,
   11, 24, 999, 999,
   11, 25, 124, 999,
   11, 26,  82, 999,
   11, 27,  87, 999,
   11, 28, 117, 999,
   11, 29, 105, 999,
   11, 30, 105, 999,
   12,  1, 999, 999,
   12,  2,  68, 999,
   12,  3, 130, 999,
   12,  4, 113, 999,
   12,  5, 153, 999,
   12,  6, 167, 999,
   12,  7, 212, 999,
   12,  8, 186, 999,
   12,  9, 234, 999,
   12, 10, 195, 999,
   12, 11, 209, 999,
   12, 12, 185, 999,
   12, 13, 125, 999,
   12, 14, 242, 999,
   12, 15, 999, 999,
   12, 16, 118, 999,
   12, 17,  99, 999,
   12, 18, 164, 999,
   12, 19, 164, 999,
   12, 20, 167, 999,
   12, 21, 177, 999,
   12, 22, 426, 999,
   12, 23, 147, 999,
   12, 24, 157, 999,
   12, 25, 167, 999,
   12, 26, 158, 999,
   12, 27, 999, 999,
   12, 28, 172, 999,
   12, 29, 999, 999,
   12, 30, 182, 999,
   12, 31, 169, 999,
};

__global__ void gInitData(cufftComplex *data, cufftComplex *in_data)
{
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	float x = in_data[i].x*2.0f*pi/(NX);
	data[i].x=cosf(x)-3.0f*sinf(x);
	data[i].y=0.0f;
}

int main()
{
	cufftHandle plan;
	cufftComplex *in_data;
	cufftComplex *data;
	cufftComplex *data_h=(cufftComplex*)calloc(NX,sizeof(cufftComplex));
	cufftComplex *in_data_h=(cufftComplex*)calloc(NX,sizeof(cufftComplex));

	cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
	cudaMalloc((void**)&in_data, sizeof(cufftComplex)*NX*BATCH);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return -1;
	}

	for (int i = 0; i < NX; i++) {
		in_data_h[i].x = array_data[i][2];
	}
	cudaMemcpy(in_data, in_data_h, NX*sizeof(cufftComplex),
	cudaMemcpyHostToDevice);
	
	gInitData<<<1, NX>>>(data, in_data);
	cudaDeviceSynchronize();
	
	if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return -1;
	}
	
	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return -1;
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return -1;
	} 
	
	cudaMemcpy(data_h, data, NX*sizeof(cufftComplex),
	cudaMemcpyDeviceToHost);

	for(int i=0;i<NX;i++)
		printf("%g\t\t%g\n", data_h[i].x, data_h[i].y);
	
	cufftDestroy(plan);
	cudaFree(data);
	free(data_h);

	return 0;
}
