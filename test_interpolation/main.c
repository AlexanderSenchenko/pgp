#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include <sys/time.h>

#define M_PI 3.14159265358979323846
#define COEF 48
#define VERTCOUNT COEF * COEF * 2	// 48 * 48  * 2 = 4608
#define RADIUS 10.0f
#define FGSIZE 20
#define FGSHIFT FGSIZE / 2			// 20 / 2  = 10
#define IMIN(A, B) (A < B ? A : B)	// 32 < 18,99609375 ?
#define THREADSPERBLOCK 256
#define BLOCKSPERGRID\
	IMIN(32, (VERTCOUNT + THREADSPERBLOCK - 1) / THREADSPERBLOCK)
// IMIN(32, (4608 + 256 - 1) / 256) = 
//	IMIN(32, (4863) / 256) = 
//	IMIN(32, 18,99609375)

typedef float(*ptr_f)(float, float, float);

struct Vertex
{
	float x, y, z;
};

float func(float x, float y, float z)
{
	return (0.5 * sqrtf(15.0 / M_PI)) * (0.5 * sqrtf(15.0 / M_PI)) *
			z * z * y * y * sqrtf(1.0f - z * z / RADIUS / RADIUS) / RADIUS /
			RADIUS / RADIUS / RADIUS;
}

float check(struct Vertex *v, ptr_f f)
{
	float sum = 0.0f;
	for (int i = 0; i < VERTCOUNT; ++i)
		sum += f(v[i].x, v[i].y, v[i].z);
	return sum;
}

void calc_f(float *arr_f, int x_size, int y_size, int z_size, ptr_f f)
{
	for (int x = 0; x < x_size; ++x)
		for (int y = 0; y < y_size; ++y)
			for (int z = 0; z < z_size; ++z)
				arr_f[z_size * (x * y_size + y) + z] =
					f(x - FGSHIFT, y - FGSHIFT, z - FGSHIFT);
}

void init_vertex()
{
	struct Vertex *temp_vert = (struct Vertex*) malloc(sizeof(struct Vertex)
								 * VERTCOUNT);
	int i = 0;
	
	for (int iphi = 0; iphi < 2 * COEF; ++iphi) {
		for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i) {
			float phi = iphi * M_PI / COEF;
			float psi = ipsi * M_PI / COEF;
			temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
			temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
			temp_vert[i].z = RADIUS * cosf(psi);
		}
	}

	printf("sumcheck = %f\n", check(temp_vert, &func) * M_PI * M_PI
								/ COEF / COEF);
	free(temp_vert);
}

float linearInterpolationCalc(int x_size, int y_size, int z_size,
							float x, float y, float z,
							int i, int j, int k, float *arr, float div)
{
	return (arr[z_size * ((i - 1) * y_size + (j - 1)) + (k - 1)] / div) *
			(i - x) * (j - y) * (k - z) +
			(arr[z_size * ((i - 1) * y_size + (j - 1)) + k] / div) *
			(i - x) * (j - y) * (z - (k - 1)) +
			(arr[z_size * ((i - 1) * y_size + j) + (k - 1)] / div) *
			(i - x) * (y - (j - 1)) * (k - z) +
			(arr[z_size * ((i - 1) * y_size + j) + k] / div) *
			(i - x) * (y - (j - 1)) * (z - (k - 1)) +
			(arr[z_size * (i * y_size + (j - 1)) + (k - 1)] / div) *
			(x - (i - 1)) * (j - y) * (k - z) +
			(arr[z_size * (i * y_size + (j - 1)) + k] / div) *
			(x - (i - 1)) * (j - y) * (z - (k - 1)) +
			(arr[z_size * (i * y_size + j) + (k - 1)] / div) *
			(x - (i - 1)) * (y - (j - 1)) * (k - z) +
			(arr[z_size * (i * y_size + j) + k] / div) *
			(x - (i - 1)) * (y - (j - 1)) * (z - (k - 1));
}

/*
 * div = (x2 - x1) * (y2 - y1) * (z2 -z1)
 *
 * f(x, y, z) =
 * (arr[x1][y1][z1] / div) * (x2 - x) * (y2 - y) * (z2 - z) +
 * (arr[x1][y1][z2] / div) * (x2 - x) * (y2 - y) * (z - z1) +
 * (arr[x1][y2][z1] / div) * (x2 - x) * (y - y1) * (z2 - z) +
 * (arr[x1][y2][z2] / div) * (x2 - x) * (y - y1) * (z - z1) +
 * (arr[x2][y1][z1] / div) * (x - x1) * (y2 - y) * (z2 - z) +
 * (arr[x2][y1][z2] / div) * (x - x1) * (y2 - y) * (z - z1) +
 * (arr[x2][y2][z1] / div) * (x - x1) * (y - y1) * (z2 - z) +
 * (arr[x2][y2][z2] / div) * (x - x1) * (y - y1) * (z - z1)
 */

float getRand()
{
	return (float)rand() / RAND_MAX;
}

float linearInterpolation(float *arr, int n_iter)
{
	float sum = 0.0f;
	// float sum_t = 0.0f;
	float div = 0.0f;

	float x, y, z;

	srand(time(NULL));

	for (int iter = 0; iter < n_iter; iter++) {
		x = getRand() * FGSIZE;
		y = getRand() * FGSIZE;
		z = getRand() * FGSIZE;

		for (int i = 1; i <= FGSIZE; i++) {
			for (int j = 1; j <= FGSIZE; j++) {
				for (int k = 1; k <= FGSIZE; k++) {
					if ((x < i && x > i - 1) &&
						(y < j && y > j - 1) &&
						(z < k && z > k - 1)) {
						// printf("%d-%d %d-%d %d-%d\n", (i - 1), i, (j - 1), j, (k - 1), k);
						// printf("f=%f\n", arr[FGSIZE * ((i - 1) * FGSIZE + (j - 1)) + (k - 1)]);
						div = (i - (i - 1)) * (j - (j - 1)) * (k - (k - 1));
						sum += linearInterpolationCalc(FGSIZE, FGSIZE, FGSIZE,
												x, y, z, i, j, k, arr, div);
						// printf("%f %f %f\n", x, y, z);
						// printf("f(i)=%f\n", sum_t);

						// printf("%f\n", arr[FGSIZE * (i * FGSIZE + j) + k]);
						// printf("%f\n", sum_t);
						// printf("%f\n", arr[FGSIZE * ((i - 1) * FGSIZE + (j - 1)) + (k - 1)]);
						// sum += sum_t;
						div = 0.0f;
					}
				}
			}
		}
	}

	// for (int i = 0; i < FGSIZE; i++) {
	// 	for (int j = 0; j < FGSIZE; j++) {
	// 		for (int k = 0; k < FGSIZE; k++) {
	// 			sum += arr[FGSIZE * (i * FGSIZE + j) + k];
	// 		}
	// 	}
	// }

	// printf("%f\n", arr[2]);
	// printf("%f\n", arr[FGSIZE * (0 * FGSIZE + 0) + 2]);

	return sum;
}

int main(void)
{
	float *arr = (float*) malloc(sizeof(float) * FGSIZE * FGSIZE * FGSIZE);

	printf("Block per grid: %d\n", BLOCKSPERGRID);
	printf("Thread per block: %d\n", THREADSPERBLOCK);
	
	init_vertex();
	calc_f(arr, FGSIZE, FGSIZE, FGSIZE, &func);

	float sum = linearInterpolation(arr, VERTCOUNT);
	printf("Linear interpolation sum = %f\n", sum * M_PI * M_PI
												/ COEF / COEF);

	free(arr);

	return 0;
}
