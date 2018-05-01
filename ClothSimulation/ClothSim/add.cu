
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
void ClothPhys(unsigned int size, float time, float3* vertexBuf, float3 grav, float damping, float3* oldPositions) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;

	/*if (index == 0 || index == 256)
	return;*/

	float3 pos = vertexBuf[index];
	float3 old = oldPositions[index];
	float3 temp = pos;

	float3 vel = make_float3((pos.x - old.x) / time, (pos.y - old.y) / time, (pos.z - old.z) / time);
	float3 force = make_float3(grav.x * 1 + vel.x*damping, grav.x * 1 + vel.x*damping, grav.x * 1 + vel.x*damping);
	
	if (pos.y >= -5000.0f) {
		pos.y = pos.y + (pos.y - old.y) * damping + grav.y * (time*time);
	}
	pos.x = pos.x + (pos.x - old.x) * damping + grav.x * (time*time);
	pos.z = pos.z + (pos.z-old.z) * damping + grav.z * (time*time);
	


	vertexBuf[index] = pos;
	oldPositions[index] = temp;

}

__global__
void Integrate(unsigned int size, float time, float3* vertexBuf, float3 grav, float damping, float3* oldPositions) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= size)
		return;

	//if (index == 65791/* || index == 256*/)
	//	return;

	float3 pos = vertexBuf[index];
	float3 old = oldPositions[index];
	float3 temp = pos;

	float3 vel = make_float3((pos.x - old.x) / time, (pos.y - old.y) / time, (pos.z - old.z) / time);
	float3 force = make_float3(grav.x * 1 + vel.x*damping, grav.x * 1 + vel.x*damping, grav.x * 1 + vel.x*damping);
	float3 acc = make_float3(force.x / 1, force.y / 1, force.z / 1);

	pos.x = pos.x + (pos.x-old.x) * damping  + grav.x * (time*time);
	if (pos.y >= -5000.0f) {
		pos.y = pos.y +  (pos.y-old.y) * damping + grav.y * (time*time);
	}
	
	pos.z = pos.z + (pos.z-old.z) * damping + grav.z * (time*time);
	
	
	////pos = pos  + (pos - old) * damping + grav * (time*time);

	/*if (pos.y < 0.0f) {
		pos.y = 0.0f;
	}*/


	//pos.x = pos.x + (pos.x - old.x) + acc.x * (time*time);
	//if (pos.y >= -5000.0f) {
		//pos.y = pos.y + (pos.y - old.y) + acc.y * (time*time);
	//}

	
	//pos.z = pos.z + (pos.z - old.z) + acc.z * (time*time);

	float3 pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9,
		pos10, pos11, pos12;

	float3 correctionVec = make_float3(0, 0, 0);

	
	if (index < 65792) {
		pos1 = vertexBuf[index + 257];
		float3 temp;
		temp.x = pos1.x - pos.x;
		temp.y = pos1.y - pos.y;
		temp.z = pos1.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}



	if (index >= 257) {
		pos2 = vertexBuf[index - 257];
		float3 temp;
		temp.x = pos2.x - pos.x;
		temp.y = pos2.y - pos.y;
		temp.z = pos2.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}


	if (index % 257 != 256) {
		pos3 = vertexBuf[index + 1];
		float3 temp;
		temp.x = pos3.x - pos.x;
		temp.y = pos3.y - pos.y;
		temp.z = pos3.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}


	if (index % 257 != 0) {
		pos4 = vertexBuf[index - 1];
		float3 temp;
		temp.x = pos4.x - pos.x;
		temp.y = pos4.y - pos.y;
		temp.z = pos4.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	//vertexBuf[index] = pos;


	if (index < 65535) {
		pos5 = vertexBuf[index + 514];
		float3 temp;
		temp.x = pos5.x - pos.x;
		temp.y = pos5.y - pos.y;
		temp.z = pos5.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 32.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	if (index > 514) {
		pos6 = vertexBuf[index - 514];
		float3 temp;
		temp.x = pos6.x - pos.x;
		temp.y = pos6.y - pos.y;
		temp.z = pos6.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 32.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	if (index % 257 != 0 && index % 257 != 1) {
		pos7 = vertexBuf[index - 2];
		float3 temp;
		temp.x = pos7.x - pos.x;
		temp.y = pos7.y - pos.y;
		temp.z = pos7.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 32.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	if (index % 257 != 255 && index % 257 != 256) {
		pos8 = vertexBuf[index + 2];
		float3 temp;
		temp.x = pos8.x - pos.x;
		temp.y = pos8.y - pos.y;
		temp.z = pos8.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 32.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	if (index >= 257 && index % 257 != 0) {
		pos9 = vertexBuf[index - 257 - 1];
		float3 temp;
		temp.x = pos9.x - pos.x;
		temp.y = pos9.y - pos.y;
		temp.z = pos9.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 22.62f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	if (index >= 257 && index % 257 != 256) {
		pos10 = vertexBuf[index - 257 + 1];
		float3 temp;
		temp.x = pos10.x - pos.x;
		temp.y = pos10.y - pos.y;
		temp.z = pos10.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 22.62f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	if (index<65792 && index % 257 != 0) {
		pos11 = vertexBuf[index + 257 - 1];
		float3 temp;
		temp.x = pos11.x - pos.x;
		temp.y = pos11.y - pos.y;
		temp.z = pos11.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 22.62f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	if (index<65792 && index % 257 != 256) {
		pos12 = vertexBuf[index + 257 + 1];
		float3 temp;
		temp.x = pos12.x - pos.x;
		temp.y = pos12.y - pos.y;
		temp.z = pos12.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 22.62f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}

	pos.x = pos.x + (correctionVec.x*0.5);
	pos.y = pos.y + (correctionVec.y*0.5);
	pos.z = pos.z + (correctionVec.z*0.5);


	vertexBuf[index] = pos;
	oldPositions[index] = temp;



}


__global__
void SphereConstraint(float3* vertexBuf, float3 spherePoint, float size, float radius) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;

	float3 pos = vertexBuf[index];

	float3 delta;
	delta.x = pos.x - spherePoint.x;
	delta.y = pos.y - spherePoint.y;
	delta.z = pos.z - spherePoint.z;

	float dist = sqrt((delta.x*delta.x)+(delta.y*delta.y)+(delta.z*delta.z));

	if (dist < radius) {
		pos.x = spherePoint.x + delta.x *(radius / dist);
		pos.y = spherePoint.y + delta.y *(radius / dist);
		pos.z = spherePoint.z + delta.z *(radius / dist);

	}


	vertexBuf[index] = pos;

}

__global__
void DistanceConstraint(float3* vertexBuf, float size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;

	float3 pos = vertexBuf[index];
	float3 pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9,
		pos10, pos11, pos12;

	float3 correctionVec = make_float3(0,0,0);

	if (index < 65792) {
		pos1 = vertexBuf[index+257];
		float3 temp;
		temp.x = pos1.x - pos.x;
		temp.y = pos1.y - pos.y;
		temp.z = pos1.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}



	if (index > 257) {
		pos2 = vertexBuf[index - 257];
		float3 temp;
		temp.x = pos2.x - pos.x;
		temp.y = pos2.y - pos.y;
		temp.z = pos2.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}


	if (index % 257 != 256) {
		pos3 = vertexBuf[index + 1];
		float3 temp;
		temp.x = pos3.x - pos.x;
		temp.y = pos3.y - pos.y;
		temp.z = pos3.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}


	if (index % 257 != 0) {
		pos4 = vertexBuf[index - 1];
		float3 temp;
		temp.x = pos4.x - pos.x;
		temp.y = pos4.y - pos.y;
		temp.z = pos4.z - pos.z;

		float currentLen = sqrt((temp.x*temp.x) + (temp.y*temp.y) + (temp.z*temp.z));
		float diff = (currentLen - 16.0f) / currentLen;
		correctionVec.x = correctionVec.x + (temp.x * 0.2 * diff);
		correctionVec.y = correctionVec.y + (temp.y * 0.2 * diff);
		correctionVec.z = correctionVec.z + (temp.z * 0.2 * diff);
	}
	//vertexBuf[index] = pos;

	


	pos.x = pos.x - (correctionVec.x);
	pos.y = pos.y - (correctionVec.y);
	pos.z = pos.z - (correctionVec.z);

	vertexBuf[index] = pos;
}



__global__
void SpringConstraint(float3* vertexBuf, float size, int vertIndex, float3 massGrav) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;
	
	if (index % vertIndex != 0) {
		return;
	}

	float3 pos = vertexBuf[index];
	


	vertexBuf[index] = pos;
}

Add::Add(unsigned int size)
{
	cudaMalloc((void**)&oldPositions, size * sizeof(float3));
}

Add::~Add()
{
	cudaFree(oldPositions);
}

void Add::BindBuffers(HeightMap * map)
{
	cudaGraphicsGLRegisterBuffer(&vertexBuf, map->getVertexBuffer(), cudaGraphicsMapFlagsNone);
	/*cudaGraphicsGLRegisterBuffer(&oldPos, map->getVertexBuffer(), cudaGraphicsMapFlagsNone);*/

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
	//float3* oldPositions;
	cudaGraphicsMapResources(1, &vertexBuf, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&tmpVertexPointer, &tmpVertexPointerSize, vertexBuf);

	/*cudaGraphicsMapResources(1, &vertexBuf, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&oldPositions, &tmpVertexPointerSize, oldPos);*/

	dim3 block(256, 1, 1);
	dim3 grid((size + block.x - 1) / block.x, 1, 1);

	float3 grav;
	grav.x = gravity.x;
	grav.y = gravity.y;
	grav.z = gravity.z;

	//-2000.0f, 2000.0f, -2000.0f
	float3 spherePoint;
	spherePoint.x = 2000.0f;
	spherePoint.y = -2000.0f;
	spherePoint.z = 2000.0f;

	/*float3 massGrav;
	massGrav.x = c->getMassGrav().x;
	massGrav.y = c->getMassGrav().y;
	massGrav.z = c->getMassGrav().z;*/

	SphereConstraint << <grid, block >> > (tmpVertexPointer, spherePoint, size, 1010);
	Integrate << <grid, block >> > (size, 0.25, tmpVertexPointer, grav, damping, oldPositions);
	//FloorConstraint << <grid, block >> > (tmpVertexPointer, 0.0f, size);
	
	//SpringConstraint <<<grid, block>>>(tmpVertexPointer, )
	//DistanceConstraint << <grid, block >> > (tmpVertexPointer, size);



	cudaGraphicsUnmapResources(1, &vertexBuf, 0);
	//cudaGraphicsUnmapResources(1, &oldPos, 0);

}
