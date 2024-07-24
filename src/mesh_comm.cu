#include "mesh.h"
#include "solver.h"

int Mesh::M_Interpolate(int i_dev, int L, int var)
{
	solver->S_Interpolate(i_dev, L, var);

	return 0;
}

int Mesh::M_Average(int i_dev, int L, int var)
{
	solver->S_Average(i_dev, L, var);

	return 0;
}
