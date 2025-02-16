#include "geometry.h"

int Geometry::G_Init(int *L, std::string output_dir_)
{
	Lx = L[0];
	Ly = L[1];
	Lz = L[2];
	output_dir = output_dir;
	
	return 0;
}
