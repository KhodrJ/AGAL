#include "main.h"

int ReadInputFile(std::map<std::string, int> *input_map_int, std::map<std::string, double> *input_map_dbl, std::string *output_dir);

int main(int argc, char *argv[])
{
	// Read input file and use map to make solver input.
	std::map<std::string, int> input_map_int;
	std::map<std::string, double> input_map_dbl;
	std::string output_dir;
	ReadInputFile(&input_map_int, &input_map_dbl, &output_dir);
	
	// Create mesh and LBM solver.
	Mesh mesh = Mesh(input_map_int, input_map_dbl, output_dir);
	
	// Solver loop (includes rendering and printing).
	mesh.M_AdvanceLoop();
	
	return 0;
}
