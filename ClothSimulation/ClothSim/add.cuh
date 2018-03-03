#include "..\..\nclgl\HeightMap.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class Add
{
public:
	Add();
	~Add();

	void BindBuffers(HeightMap* map);

	void AddByRand(unsigned int size, float time);
	void IntergrateTest(unsigned int size, float time, float damping, Vector3 gravity);

private:
	cudaGraphicsResource* vertexBuf;
};