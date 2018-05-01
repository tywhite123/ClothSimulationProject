#pragma once

#include <vector>
#include "../../nclgl/Vector3.h"

enum SpringType {
	STRUCTURAL, SHEAR, BEND 
};

struct Spring {
	SpringType springType;
	//float ks, kd;
	int x, z;
};

struct Node {
	int x, z;
	int vertIndex;
	float xPos, yPos, zPos;

	float mass;

	std::vector<Spring> springs;
};

class Cloth
{
public:
	Cloth(int width, int height, float clothX, float clothZ, float clothY);
	~Cloth();

	Vector3 getMassGrav() { return massGrav; }
	float getStrucLen() { return strucLen; }
	Vector3 getShearLen(){ return shearLen; }
	float getBendLen() { return bendLen; }

	Node getNodeAt(int i) { return nodes.at(i); }
	
protected:
	float mass;
	int xNodes;
	int zNodes;

	std::vector<Node> nodes;

	float strucLen;
	Vector3 shearLen;
	float bendLen;

	Vector3 massGrav;
};

