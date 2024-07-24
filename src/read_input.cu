#include "main.h"

int ParseNxFactor(std::string str_val, int Nx)
{
	// Returns the result of A*Nx where A is the specified integer. Nx must come second (so not Nx*A).
	int pos = str_val.find("*");
	int A = std::stoi(str_val.substr(0, pos));
	
	if (pos != std::string::npos)
		return A*Nx;
	else
		return A;
}

int ReadInputFile(std::map<std::string, int> *input_map_int, std::map<std::string, double> *input_map_dbl, std::string *output_dir)
{
	// Setup default input map.
	std::ifstream input = std::ifstream("../input/input.txt");
	*input_map_int = std::map<std::string, int> // Define map of integer parameters with default values.
	{
		{"MAX_LEVELS",              5},
		{"MAX_LEVELS_INTERIOR",     4},
		{"N_LEVEL_START",           0},
		{"Nx",                      64},
		{"N_RESTART",               1},
		{"PERIODIC_X",              0},
		{"PERIODIC_Y",              0},
		{"PERIODIC_Z",              0},
		{"S_LES",                   0},
		{"P_REFINE",                32},
		{"N_REFINE_START",          -3},
		{"N_REFINE_INC",            1},
		{"N_PROBE",                 0},
		{"N_PROBE_DENSITY",         4},
		{"N_PROBE_FREQUENCY",       32},
		{"N_PROBE_FORCE",           0},
		{"N_PROBE_F_FREQUENCY",     16},
		{"N_PROBE_AVE",             0},
		{"N_PROBE_AVE_FREQUENCY",   1},
		{"N_PROBE_AVE_START",       100*64},
		{"N_PRINT_LEVELS",          2},
		{"P_OUTPUT",                1*64},
		{"N_OUTPUT",                10},
		{"N_OUTPUT_START",          0}
	};
	*input_map_dbl = std::map<std::string, double> // Define map of double parameters with default values.
	{
		{"L_c",                     1.0},
		{"L_fy",                    1.0},
		{"L_fz",                    1.0},
		{"v0",                      5.00e-5},
		{"V_PROBE_TOL",             1e-4},
	};
	*output_dir = "../out/TEST_UPLOAD/";
	
	// Extract Nx first.
	int Nx = -1;
	std::string str_name;
	std::string str_val;
	for (std::string line; std::getline(input, line);) 
	{
		if (line.length() > 0 && line.at(0) != '#') // Ignore comments.
		{
			std::stringstream ss(line);
			ss >> str_name >> str_val;
			
			if (str_name == "Nx")
				Nx = std::stoi(str_val);
		}
	}
	
	// Read rest of input file.
	input.clear();
	input.seekg(0);
	for (std::string line; std::getline(input, line);) 
	{
		if (line.length() > 0 && line.at(0) != '#') // Ignore comments.
		{
			std::stringstream ss(line);
			ss >> str_name >> str_val;
			
			if (str_name == "P_DIR_NAME")
				*output_dir = str_val;
			else
			{
				if ( (*input_map_int).count(str_name) == 1 )
					(*input_map_int)[str_name] = ParseNxFactor(str_val, Nx);
				else if ( (*input_map_dbl).count(str_name) == 1 )
					(*input_map_dbl)[str_name] = std::stod(str_val);
				else
					std::cout << "Error: Invalid input (" << str_name << ") skipped..." << std::endl;
			}
		}
	}
	
	// Check (debug).
// 	for (std::map<std::string, int>::iterator it = (*input_map_int).begin(); it != (*input_map_int).end(); ++it)
// 	{
// 		std::cout << it->first << " " << it->second << std::endl;
// 	}
// 	for (std::map<std::string, double>::iterator it = (*input_map_dbl).begin(); it != (*input_map_dbl).end(); ++it)
// 	{
// 		std::cout << it->first << " " << it->second << std::endl;
// 	}
	
	return 0;
}
