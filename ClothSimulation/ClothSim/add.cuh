#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class Add
{
public:
	Add();
	~Add();

	void AddVertByRand();

private:
	cudaGraphicsResource* vertexBuf;
};