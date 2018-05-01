#include "Renderer.h"


Renderer::Renderer(Window &parent) : OGLRenderer(parent)
{
	heightMap = new HeightMap(TEXTUREDIR"terrain.raw");
	camera = new Camera();
	sphere = Mesh::GenerateSphere();

	add = new Add(heightMap->GetNumVerts());
	camera->SetPosition(Vector3(2363.0f, 768.0f, 4961.0f));
	//cloth = new Cloth(256, 256, heightMap->getXSize(), heightMap->getZSize(), heightMap->getYSize());


	currentShader = new Shader(SHADERDIR"TexturedVertex.glsl", SHADERDIR"TexturedFragment.glsl");

	if (!currentShader->LinkProgram()) {
		return;
	}

	heightMap->SetTexture(SOIL_load_OGL_texture(TEXTUREDIR"cloth.JPG", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS));

	if (!heightMap->GetTexture()) {
		return;
	}


	SetTextureRepeating(heightMap->GetTexture(), true);

	sphere->SetTexture(SOIL_load_OGL_texture(TEXTUREDIR"Barren Reds.JPG", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS));

	if (!sphere->GetTexture()) {
		return;
	}

	SetTextureRepeating(sphere->GetTexture(), true);

	root = new SceneNode();

	SceneNode* cloth = new SceneNode();
	cloth->SetMesh(heightMap);
	cloth->SetTransform(Matrix4::Translation(Vector3(-2000.0f, 2000.0f, -2000.0f)));
	root->AddChild(cloth);

	SceneNode* sphereNode = new SceneNode();
	sphereNode->SetMesh(sphere);
	sphereNode->SetTransform(Matrix4::Translation(Vector3(0.0f, 0.0f, 0.0f)));
	root->AddChild(sphereNode);


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
	root->Update(msec);

	time += msec;

	//add->AddByRand(heightMap->GetNumVerts(), time);
	add->IntergrateTest(heightMap->GetNumVerts(), msec, 0.999, Vector3(0.0, -9.81f, 0.0)/1);
}

void Renderer::DrawNode(SceneNode * n)
{
	if (n->GetMesh())
	{
		Matrix4 transform = n->GetWorldTransform() * Matrix4::Scale(n->GetModelScale());

		glUniformMatrix4fv(glGetUniformLocation(currentShader->GetProgram(), "modelMatrix"), 1, false, (float*)&transform);
		glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "useTexture"), (int)n->GetMesh()->GetTexture());

		n->Draw();
	}

	for (vector<SceneNode*>::const_iterator i = n->GetChildIteratorStart(); i != n->GetChildIteratorEnd(); ++i)
	{
		DrawNode(*i);
	}
}

void Renderer::RenderScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(currentShader->GetProgram());
	UpdateShaderMatrices();

	glUniform1i(glGetUniformLocation(currentShader->GetProgram(), "diffuseTex"), 0);

	/*heightMap->Draw();
	sphere->Draw();*/

	DrawNode(root);

	glUseProgram(0);
	SwapBuffers();
}