#include "geometry.h"

int Geometry::G_ImportSTL_ASCII(std::string filename)
{
	// Parameters.
	std::ifstream input = std::ifstream(input_dir + "geometry/" + filename);
	std::string line = "";
	std::string word = "";
	
	// Get the first line and ignore it (solid Exported ... ).
	std::getline(input, line);
	
	// Now loop over the various facets and add face vertices to the vectors.
	std::getline(input, line);          // To start the loop (face normal ... ).
	while (word != "endsolid")
	{
		std::getline(input, line); // outer loop (ignore)
		
		std::getline(input, line); // vertex ... (ignore)
		std::getline(input, line); // vertex ... (ignore)
		std::getline(input, line); // vertex ... (ignore)
		
		std::getline(input, line); // endloop (ignore)
		std::getline(input, line); // endfacet (ignore)
		std::getline(input, line); // facet normal ... (of next facet)   OR   endsolid Exported ...
		std::stringstream ss(line);
		ss >> word;
	}
	
	// Close file.
	input.close();

	return 0;
}
