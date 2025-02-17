#include "cppspec.h"
#include "geometry.h"
#include "mesh.h"
#include "solver.h"

int ReadInputFile(std::string input_file_directory, std::map<std::string, int> &input_map_int, std::map<std::string, double> &input_map_dbl, std::map<std::string, std::string> &input_map_str);

int main(int argc, char *argv[])
{
	// Read input file and use map to make solver input.
	std::string input_file_directory = "../input/";
	std::map<std::string, int> input_map_int;
	std::map<std::string, double> input_map_dbl;
	std::map<std::string, std::string> input_map_str;
	ReadInputFile(input_file_directory, input_map_int, input_map_dbl, input_map_str);
	
	// Create a new geometry and import from input folder.
	Geometry geometry(input_map_int, input_map_dbl, input_map_str);
	geometry.G_ImportSTL_ASCII("bunny_ascii.stl");
	//geometry.G_ImportBoundariesFromTextFile(0);
	//geometry.G_Convert_IndexListsToCoordList(0);
	geometry.G_Init_Arrays_CoordsList_CPU(0);
	geometry.G_PrintSTL(0);
	
	// Create mesh and LBM solver.
	Mesh mesh(input_map_int, input_map_dbl, input_map_str);
	mesh.geometry = &geometry;
	mesh.M_NewSolver_LBM_BGK(input_map_int, input_map_dbl, input_map_str);
	
	// Solver loop (includes rendering and printing).
	mesh.M_AdvanceLoop();
	
	return 0;
}
