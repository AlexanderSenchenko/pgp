#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <sys/time.h>

void example_saxpy();

double wtime() {
 	struct timeval tv;
 	gettimeofday(&tv, NULL);
 	return tv.tv_sec + tv.tv_usec * 1E-6;
}

__host__ void print_array(float *data1, float *data2,
								int num_elem, const char *prefix)
{
	printf("\n%s", prefix);
	for (int i = 0; i < num_elem; i++)
		printf("\n%2d: %2.4f %2.4f ", i + 1, data1[i], data2[i]);
}

int main(int argc, const char **argv)
{
	double time = wtime();

	const int num_elem = (argc > 1) ? atoi(argv[1]) : 8;
	const size_t size_in_bytes = (num_elem * sizeof(float));
   
	float *A_dev;
	cudaMalloc((void **) &A_dev, size_in_bytes);
	float *B_dev;
	cudaMalloc((void **) &B_dev, size_in_bytes);
	
	float *A_h;
	cudaMallocHost((void **) &A_h, size_in_bytes);
	float *B_h;
	cudaMallocHost((void **) &B_h, size_in_bytes);
   
	memset(A_h, 0, size_in_bytes);
	memset(B_h, 0, size_in_bytes);
	
	// Инициализация библиотеки CUBLAS
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);
	
	for (int i = 0; i < num_elem; i++)
		A_h[i] = (float) i;

	if (argc < 1)
		print_array(A_h, B_h, num_elem, "Before Set");
	
	const int num_rows = num_elem;
	const int num_cols = 1;
	const size_t elem_size = sizeof(float);
	
	// Копирование матрицы с числом строк num_elem и одним столбцом
	// с хоста на устройство
	cublasSetMatrix(num_rows, num_cols, elem_size, A_h, num_rows,
													A_dev, num_rows);
	
	// Очищаем массив на устройстве
	cudaMemset(B_dev, 0, size_in_bytes);
	
	// выполнение SingleAlphaXPlusY
	const int stride = 1;
	float alpha = 2.0F;
	cublasSaxpy(cublas_handle, num_elem, &alpha, A_dev, stride,
														B_dev, stride);
	
	// Копирование матриц с числом строк num_elem и одним столбцом
	// с устройства на хост
	cublasGetMatrix(num_rows, num_cols, elem_size, A_dev, num_rows,
														A_h, num_rows);
	cublasGetMatrix(num_rows, num_cols, elem_size, B_dev, num_rows,
														B_h, num_rows);
	
	// Удостоверяемся, что все асинхронные вызовы выполнены
	// const int default_stream = 0;
	// cudaStreamSynchronize(default_stream);
	cudaStreamSynchronize(0);

	// Print out the arrays
	if (argc < 1) {
		print_array(A_h, B_h, num_elem, "After Set");
		printf("\n");
	}
	
	// Освобождаем ресурсы на устройстве
	cublasDestroy(cublas_handle);
	cudaFree(A_dev);
	cudaFree(B_dev);
	
	// Освобождаем ресурсы на хосте
	cudaFreeHost(A_h);
	cudaFreeHost(A_h);
	cudaFreeHost(B_h);
	
	//сброс устройства, подготовка для выполнения новых программ
	cudaDeviceReset();

	time = wtime() - time;
	printf("Time SAXPY cuBLAS: %.6f s\n", time);

	return 0;
}
