#include "Renderer.h"


Renderer::Renderer(Window &parent) : OGLRenderer(parent)
{
	heightMap = new HeightMap(TEXTUREDIR"terrain.raw");
	camera = new Camera();
	sphere = Mesh::GenerateSphere();

	add = new Add();


	camera->SetPosition(Vector3(2363.0f, 768.0f, 4961.0f));


	currentShader = new Shader(SHADERDIR"TexturedVertex.glsl", SHADERDIR"TexturedFragment.glsl");

	if (!currentShader->LinkProgram()) {
		return;
	}

	heightMap->SetTexture(SOIL_load_OGL_texture(TEXTUREDIR"Barren Reds.JPG", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS));

	if (!heightMap->GetTexture()) {
		return;
	}


	SetTextureRepeating(heightMap->GetTexture(), true);

	sphere->SetTexture(SOIL_load_OGL_texture(TEXTUREDIR"Barren Reds.JPG", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS));

	if (!sphere->GetTexture()) {
		return;
	}

	SetTextureRepeating(sphere->GetTexture(), true);

	/*root = new SceneNode();

	SceneNode* cloth = new SceneNode();
	cloth->SetMesh(heightMap);
	root->AddChild;*/

	projMatrix = Matrix4::Perspective(1.0f, 10000.0f, (float)width / (float)height, 45.0f);

	
	add->BindBuffers(heightMap);
	
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	init = true;
}


Renderer::~Renderer()
{
	delete heightMap;
	delete camera;
	delete sphere;
}

void Renderer::UpdateScene(float msec)
{
	camera->UpdateCamera(msec);
	viewMatrix = camera->BuildViewMatrix();


	time += msec;

	//add->AddByRand(heightMap->GetNumVerts(), time);
	//add->IntergrateTest(heightMap->GetNumVerts(), msec, 0.99, Vector3(0.0, -1.0, 0.0));
}

void Renderer::RenderScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(currentShader->GetProgram());
	UpdateShaderMatrices();

	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "diffuseTex"), 0);

	//heightMap->Draw();
	//sphere->Draw();

	

	glUseProgram(0);
	SwapBuffers();
}