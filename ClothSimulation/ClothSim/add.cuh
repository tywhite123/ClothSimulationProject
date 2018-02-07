#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "..\..\nclgl\HeightMap.h"

class Add
{
public:
	Add();
	~Add();

	void AddVertByRand(HeightMap* map);

private:
	cudaGraphicsResource* vertexBuf;
};