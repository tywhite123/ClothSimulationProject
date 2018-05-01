#include "Cloth.h"

Cloth::Cloth(int width, int height, float clothX, float clothZ, float clothY) : xNodes(width), zNodes(height), mass(1)
{
	int vert = 0;
	for (int i = 0; i <= width; ++i) {
		for (int j = 0; j <= height; ++j) {
			
			float x = clothX / (1 / xNodes-i);
			float z = clothZ / (1 / zNodes-j);
			Node n;
			n.x = i;
			n.z = j;
			n.vertIndex = vert;
			n.xPos = x;
			n.yPos = clothY;
			n.zPos = z;
			n.mass = mass;

			//Structural
			Spring s;
			s.springType = STRUCTURAL;
			if (i <= width && i > 0) {
				s.x = i - 1;
				s.z = j;
				n.springs.push_back(s);
			}
			if (j <= height && j > 0) {
				s.x = i - 1;
				s.z = j;
				n.springs.push_back(s);
			}
			if (i >= 0 && i < width) {
				s.x = i + 1;
				s.z = j;
				n.springs.push_back(s);
			}
			if (j >= 0 && j < height) {
				s.x = i;
				s.z = j + 1;
				n.springs.push_back(s);
			}
			//Shear
			s.springType = SHEAR;
			if (i >= 0 && j >= 0) {
				s.x = i + 1;
				s.z = j + 1;
				n.springs.push_back(s);
			}
			if (i >= 0 && j <= height) {
				s.x = i + 1;
				s.z = j - 1;
				n.springs.push_back(s);
			}
			if (j >= 0 && i <= width) {
				s.x = i - 1;
				s.z = j + 1;
				n.springs.push_back(s);
			}
			if (i <= width && j <= height) {
				s.x = i - 1;
				s.z = j - 1;
				n.springs.push_back(s);
			}
			//Bend
			s.springType = BEND;
			if (i <= width && i > 1) {
				s.x = i - 2;
				s.z = j;
				n.springs.push_back(s);
			}
			if (j <= height && j > 1) {
				s.x = i - 2;
				s.z = j;
				n.springs.push_back(s);
			}
			if (i >= 0 && i < width-1) {
				s.x = i + 2;
				s.z = j;
				n.springs.push_back(s);
			}
			if (j >= 0 && j < height-1) {
				s.x = i;
				s.z = j + 2;
				n.springs.push_back(s);
			}


			nodes.push_back(n);

			strucLen = nodes.at(1).xPos - nodes.at(0).xPos;
			shearLen = Vector3(nodes.at(1).xPos - nodes.at(0).xPos, nodes.at(1).yPos - nodes.at(0).yPos, nodes.at(1).zPos - nodes.at(0).zPos);
			bendLen = (nodes.at(1).xPos - nodes.at(0).xPos) * 2;

			massGrav = Vector3(0.0f, -9.8f, 0.0f) * mass;
			vert += (66048 / (width*height));

		}
	}
}

Cloth::~Cloth()
{
}
