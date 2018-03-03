
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


__global__
void Integrate(unsigned int size, float time, float3* vertexBuf, float3 grav, float damping) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;

	float3 pos = vertexBuf[index];
	float3 oldPos = pos;

	//pos.x = pos.x + damping + grav.x*(time*time);
	pos.y = pos.y - damping + grav.y*(time*time);
	//pos.y = pos.y + damping + grav.z*(time*time);

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

void Add::IntergrateTest(unsigned int size, float time, float damping, Vector3 gravity)
{
	std::size_t tmpVertexPointerSize;
	float3* tmpVertexPointer;
	cudaGraphicsMapResources(1, &vertexBuf, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&tmpVertexPointer, &tmpVertexPointerSize, vertexBuf);

	dim3 block(256, 1, 1);
	dim3 grid((size + block.x - 1) / block.x, 1, 1);

	float3 grav;
	grav.x = gravity.x;
	grav.y = gravity.y;
	grav.z = gravity.z;

	Integrate << <grid, block >> > (size, 0.01, tmpVertexPointer, grav, damping);

	cudaGraphicsUnmapResources(1, &vertexBuf, 0);

}
