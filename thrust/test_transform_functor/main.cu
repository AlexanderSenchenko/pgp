#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <cstdio>
#include <cmath>

struct range_functor
{
	float h;
	range_functor(float _h) : h(_h) {}
	__host__ __device__ float operator()(float x)
	{
		return h * x;
	}
};

struct sin_functor
{
	__device__ float operator()(float x)
	{
		return __sinf(x);
	}
};

int main()
{
	range_functor R(0.02);
	sin_functor Sin;

	fprintf(stderr, "%g\n", R(30.0f));

	thrust::host_vector<float> h1(1 << 8);
	thrust::host_vector<float> h2(1 << 8);
	thrust::device_vector<float> d1(1 << 8);
	thrust::device_vector<float> d2(1 << 8);
	thrust::sequence(thrust::device, d1.begin(), d2.end());
	thrust::transform(d1.begin(), d1.end(), d1.begin(), R);
	thrust::transform(d1.begin(), d1.end(), d2.begin(), Sin);

	h2 = d2;
	h1 = d1;

	for (int i = 0; i < (1 << 8); i++)
		printf("%g\t%g\n", h1[i], h2[i]);

	return 0;
}