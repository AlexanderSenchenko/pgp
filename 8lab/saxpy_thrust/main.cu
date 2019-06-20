#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

double wtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1E-6;
}

struct saxpy_functor
{
	const float a;

	saxpy_functor(float _a) : a(_a) {}
	
	__host__ __device__ float operator() (thrust::tuple<float&, float&> t)
	{
		float x = thrust::get<0>(t);
		float y = thrust::get<1>(t);

		return a * x + y;
	}
};

void saxpy(float a, thrust::device_vector<float> &x,
									thrust::device_vector<float> &y)
{
	saxpy_functor func(a);
	thrust::transform(
		thrust::make_zip_iterator(
			thrust::make_tuple(x.begin(), y.begin() )),
		thrust::make_zip_iterator(
			thrust::make_tuple(x.end(), y.end() )),
		y.begin(),
		func
	);
}

__host__ void print_array(thrust::host_vector<float> &data1,
	thrust::host_vector<float> &data2, int num_elem, const char *prefix)
{
	printf("\n%s", prefix);
	for (int i = 0; i < num_elem; i++)
		printf("\n%2d: %2.4f %2.4f ", i + 1, data1[i], data2[i]);
}

int main(int argc, const char **argv)
{
	double time = wtime();

	int num_elem = (argc > 1) ? std::atoi(argv[1]) : 8;
	const float a = 2.0;

	thrust::host_vector<float> h1(num_elem);
	thrust::host_vector<float> h2(num_elem);

	thrust::sequence(h1.begin(), h1.end());
	thrust::fill(h2.begin(), h2.end(), 0.0);

	if (argc < 1)
		print_array(h1, h2, num_elem, "Before Set");

	thrust::device_vector<float> d1 = h1;
	thrust::device_vector<float> d2 = h2;

	saxpy(a, d1, d2);

	h2 = d2;
	h1 = d1;

	if (argc < 1) {
		print_array(h1, h2, num_elem, "After Set");
		printf("\n");
	}

	time = wtime() - time;
	printf("Time SAXPY thrust: %.6f s\n", time);
	
	return 0;
}
