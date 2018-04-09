#pragma once

#include "../../nclgl/OGLRenderer.h"

#include "add.cuh"
#include "../../nclgl/Camera.h"
#include "../../nclgl/HeightMap.h"
#include "../../nclgl/SceneNode.h"

class Renderer : public OGLRenderer
{
public:
	Renderer(Window &parent);
	virtual ~Renderer(void);

	virtual void RenderScene();
	virtual void UpdateScene(float msec);

protected:
	HeightMap* heightMap;
	Camera* camera;
	Mesh* sphere;
	Add* add;
	SceneNode* root;

	float time = 0;

};