#include "Mesh.h"



Mesh::Mesh(void)
{
	for (int i = 0; i < MAX_BUFFER; ++i) {
		bufferObject[i] = 0;
	}
	glGenVertexArrays(1, &arrayObject);

	numVertices = 0;
	vertices = NULL;
	colours = NULL;
	type = GL_TRIANGLES;

	indices = NULL;
	numIndices = 0;

	texture = 0;
	textureCoords = NULL;
}


Mesh::~Mesh(void)
{
	glDeleteVertexArrays(1, &arrayObject);
	glDeleteBuffers(MAX_BUFFER, bufferObject);
	glDeleteTextures(1, &texture);
	delete[] textureCoords;
	delete[] vertices;
	delete[] colours;
	delete[] indices;
}

void Mesh::Draw()
{
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindVertexArray(arrayObject);
	if (bufferObject[INDEX_BUFFER]) {
		glDrawElements(type, numIndices, GL_UNSIGNED_INT, 0);
	}
	else {
		glDrawArrays(type, 0, numVertices);
	}
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

Mesh * Mesh::GenerateTriangle()
{
	Mesh* m = new Mesh();
	m->numVertices = 3;

	m->vertices = new Vector3[m->numVertices];
	m->vertices[0] = Vector3(0.0f, 0.5f, 0.0f);
	m->vertices[1] = Vector3(0.5f, -0.5f, 0.0f);
	m->vertices[2] = Vector3(-0.5f, -0.5f, 0.0f);

	m->textureCoords = new Vector2[m->numVertices];
	m->textureCoords[0] = Vector2(0.5f, 0.0f);
	m->textureCoords[1] = Vector2(1.0f, 1.0f);
	m->textureCoords[2] = Vector2(0.0f, 1.0f);

	m->colours = new Vector4[m->numVertices];
	m->colours[0] = Vector4(1.0f, 0.0f, 0.0f, 1.0f);
	m->colours[1] = Vector4(0.0f, 1.0f, 0.0f, 1.0f);
	m->colours[2] = Vector4(0.0f, 0.0f, 1.0f, 1.0f);

	m->BufferData();

	return m;
}

Mesh* Mesh::GenerateQuad() {
	Mesh* m = new Mesh();
	m->numVertices = 4;
	m->type = GL_TRIANGLE_STRIP;

	m->vertices = new Vector3[m->numVertices];
	m->textureCoords = new Vector2[m->numVertices];
	m->colours = new Vector4[m->numVertices];

	m->vertices = new Vector3[m->numVertices];
	m->vertices[0] = Vector3(-1.0f, -1.0f, 0.0f);
	m->vertices[1] = Vector3(-1.0f, 1.0f, 0.0f);
	m->vertices[2] = Vector3(1.0f, -1.0f, 0.0f);
	m->vertices[3] = Vector3(1.0f, 1.0f, 0.0f);

	m->textureCoords = new Vector2[m->numVertices];
	m->textureCoords[0] = Vector2(0.0f, 1.0f);
	m->textureCoords[1] = Vector2(0.0f, 0.0f);
	m->textureCoords[2] = Vector2(1.0f, 1.0f);
	m->textureCoords[2] = Vector2(1.0f, 0.0f);

	for (int i = 0; i < 4; ++i) {
		m->colours[i] = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
	}

	m->BufferData();
	return m;

}

Mesh * Mesh::GenerateSphere()
{
	const int RADIUS = 1000;
	const Vector3 center = Vector3(0, 0, 0);
	vector<Vector3> vertices = vector<Vector3>();
	//for (float theta = 0.0f; theta < PI; theta += PI / 10) {
	for (float phi = 0.0f; phi < 2.01f * PI; phi += PI / 20) {
		for (float theta = 0.0f; theta < PI+0.125f; theta += PI / 20) {
			float x, y, z;
			x = RADIUS * cos(theta) * sin(phi) + center.x;
			y = RADIUS * sin(theta) * sin(phi) + center.y;
			z = RADIUS * cos(phi) + center.z;
			vertices.push_back(Vector3(x, y, z));
			x = RADIUS * cos(theta) * sin(phi + PI/20) + center.x;
			y = RADIUS * sin(theta) * sin(phi + PI/20) + center.y;
			z = RADIUS * cos(phi + PI/20) + center.z;
			vertices.push_back(Vector3(x, y, z));
		}
	}

	Mesh* m = new Mesh();
	m->numVertices = vertices.size();
	m->type = GL_TRIANGLE_STRIP;



	m->vertices = new Vector3[m->numVertices];
	m->colours = new Vector4[m->numVertices];
	m->textureCoords = new Vector2[m->numVertices];

	for (int i = 0; i < m->numVertices; ++i) {
		m->vertices[i] = vertices[i];
		m->colours[i] = Vector4(1.0f, 1.0f, 1.0f, 1.0f);
		if (i % 3 == 0) {
			m->textureCoords[i] = Vector2(0.0f, 0.0f);
		}
		else if (i % 3 == 1) {
			m->textureCoords[i] = Vector2(0.0f,	1.0f);
		}
		else {
			m->textureCoords[i] = Vector2(1.0f, 0.0f);
			m->textureCoords[i] = Vector2(1.0f, 1.0f);
		}
	}

	m->BufferData();
	return m;
}

void Mesh::BufferData()
{
	glBindVertexArray(arrayObject);
	glGenBuffers(1, &bufferObject[VERTEX_BUFFER]);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObject[VERTEX_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector3), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(VERTEX_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VERTEX_BUFFER);

	if (textureCoords) {
		glGenBuffers(1, &bufferObject[TEXTURE_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER, bufferObject[TEXTURE_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector2), textureCoords, GL_STATIC_DRAW);
		glVertexAttribPointer(TEXTURE_BUFFER, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(TEXTURE_BUFFER);
	}

	if (colours) {
		glGenBuffers(1, &bufferObject[COLOUR_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER, bufferObject[COLOUR_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vector4), colours, GL_STATIC_DRAW);
		glVertexAttribPointer(COLOUR_BUFFER, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(COLOUR_BUFFER);
	}

	if (indices) {
		glGenBuffers(1, &bufferObject[INDEX_BUFFER]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject[INDEX_BUFFER]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(GLuint), indices, GL_STATIC_DRAW);
	}

	glBindVertexArray(0);
}


