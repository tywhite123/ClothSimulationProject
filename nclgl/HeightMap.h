#pragma once

#include <string>
#include <iostream>
#include <fstream>

#include "Mesh.h"

#define RAW_HEIGHT 257
#define RAW_WIDTH 257

#define HEIGHTMAP_X 16.0f
#define HEIGHTMAP_Z 16.0f
#define HEIGHTMAP_Y 0.0f
#define HEIGHTMAP_TEX_X 1.0f / 16.0f
#define HEIGHTMAP_TEX_Z 1.0f / 16.0f

class HeightMap : public Mesh
{
public:
	HeightMap(std::string name);
	~HeightMap(void) {};

	GLuint getVertexBuffer() { return bufferObject[VERTEX_BUFFER]; }

	float getXSize() { return vertices[66048].x; }
	float getYSize() { return vertices[66048].y; }
	float getZSize() { return vertices[66048].z; }
	
	Vector3 getVerticesAt(int x) { return vertices[x]; }
};

