#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "add.cuh"

__global__
void add(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}
}

Add::Add() 
{

}

Add::~Add()
{

}

void Add::AddVertByRand()
{
	size_t tmpVertexPointerSize;
	float3* tmpVertexPointer;
	cudaGraphicsMapResources(1, &vertexBuf, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&tmpVertexPointer, &tmpVertexPointerSize, vertexBuf);


}
