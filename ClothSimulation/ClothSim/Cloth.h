#pragma once
class Cloth
{
public:
	Cloth(int x, int y);
	~Cloth();

protected:
	float mass;
	int xNodes;
	int yNodes;
};

