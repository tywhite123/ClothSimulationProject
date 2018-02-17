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

private:
	cudaGraphicsResource* vertexBuf;
};