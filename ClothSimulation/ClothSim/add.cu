
#include "device_launch_parameters.h"

#include "add.cuh"
#include <curand.h>
#include <curand_kernel.h>

__global__
void add(unsigned int size, float time, float3* vertexBuf) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;
	
	float3 pos = vertexBuf[index];

	curandState_t state;

	curand_init(0, 0, 0, &state);

	int randomVal = (curand(&state) % 3);

	pos.y = sin(time * 0.001f + blockIdx.x * threadIdx.x * 0.5f) * 10;

	vertexBuf[index] = pos;



}

Add::Add() 
{
}

Add::~Add()
{

}

void Add::BindBuffers(HeightMap * map)
{
	cudaGraphicsGLRegisterBuffer(&vertexBuf, map->getVertexBuffer(), cudaGraphicsMapFlagsNone);

	//dim3 block(256, 1, 1);
	////dim3 grid((size + block.x - 1) / block.x, 1, 1);
}

void Add::AddByRand(unsigned int size, float time)
{
	std::size_t tmpVertexPointerSize;
	float3* tmpVertexPointer;
	cudaGraphicsMapResources(1, &vertexBuf, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&tmpVertexPointer, &tmpVertexPointerSize, vertexBuf);

	dim3 block(256, 1, 1);
	dim3 grid((size + block.x - 1) / block.x, 1, 1);

	add << <grid, block >> > (size, time, tmpVertexPointer);
	
	cudaGraphicsUnmapResources(1, &vertexBuf, 0);

}
