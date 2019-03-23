#include <cuda.h>
#include <stdio.h>

int main()
{
	int count;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&count);

	printf("Count CUDA device = %i\n", count);

	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		
		printf("Device %d\n", i);
		printf("Compute capability            : %d.%d\n", prop.major, prop.minor);
		printf("Name                          : %s\n", prop.name);
		printf("Total Global Memory           : %d byte\n", prop.totalGlobalMem);
		printf("Shared memory per block       : %d byte\n", prop.sharedMemPerBlock);
		printf("Registers per block           : %d\n", prop.regsPerBlock);
		printf("Warp size                     : %d\n", prop.warpSize);
		printf("Max threads per block         : %d\n", prop.maxThreadsPerBlock);
		printf("Total constant memory         : %d byte\n", prop.totalConstMem);
		printf("Clock Rate                    : %d kHz\n", prop.clockRate);
		printf("Texture Alignment             : %u\n", prop.textureAlignment);
		printf("Device Overlap                : %d\n", prop.deviceOverlap);
		printf("Multiprocessor Count          : %d\n", prop.multiProcessorCount);
		printf("Max Threads Dim               : %d %d %d\n", prop.maxThreadsDim[0],
								prop.maxThreadsDim[1],
								prop.maxThreadsDim[2]);
		printf("Max Grid Size                 : %d %d %d\n", prop.maxGridSize[0],
								prop.maxGridSize[1],
								prop.maxGridSize[2]);
		printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
		// printf("%d", prop.maxBlocksPerMultiProcessor);

		printf("\n");
	}

	return 0;
}
