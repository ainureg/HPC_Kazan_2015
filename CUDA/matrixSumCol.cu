#include <iostream>

void check(const char *file, const int line, cudaError_t err) {
	if (err != cudaSuccess) {
		std::cerr << file << ":" << line
			<< " CUDA call failed with error: "
			<< cudaGetErrorString(err)
			<< std::endl;
		std::terminate();
	}
}

#define CHECK(x) check(__FILE__, __LINE__, (x))

__global__ void sum(const float *a, float *b) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int n = blockDim.x * gridDim.x;
	float sum = 0;
	int idx = col;
	for (int row = 0; row < n; row++) {
		sum += a[idx];
		idx += n;
	}
	b[col] = sum;
}
int main() {
	const int N = 8192;
	float *ha, *hb;
	float *da, *db;

	ha = new float [N * N];
	hb = new float [N];
	CHECK(cudaMalloc(&da, N * N * sizeof(float)));
	CHECK(cudaMalloc(&db, N * sizeof(float)));

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			ha[i * N + j] = sin(i * N + j);

	CHECK(cudaMemcpy(da, ha, N * N * sizeof(float), cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));

	CHECK(cudaEventRecord(start, 0));

	dim3 block(1024);
	dim3 grid(N / block.x);
	sum<<<grid, block>>>(da, db);

	CHECK(cudaEventRecord(stop, 0));

	CHECK(cudaEventSynchronize(stop));

	float timems;
	CHECK(cudaEventElapsedTime(&timems, start, stop));
	std::cout << "Kernel elapsed time: " << timems << " ms" << std::endl;

	CHECK(cudaMemcpy(hb, db, N * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 10; i++)
		std::cout << "b[" << i << "] = " << hb[i] << std::endl; 
	std::cout << "..." << std::endl;
	for (int i = N - 10; i < N; i++)
		std::cout << "b[" << i << "] = " << hb[i] << std::endl; 

	CHECK(cudaFree(da));
	CHECK(cudaFree(db));
	delete[] ha;
	delete[] hb;

	return 0;
}
