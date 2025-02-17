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
	int counter = 0;
	double v1x = 0.0;
	double v1y = 0.0;
	double v1z = 0.0;
	double v2x = 0.0;
	double v2y = 0.0;
	double v2z = 0.0;
	double v3x = 0.0;
	double v3y = 0.0;
	double v3z = 0.0;
	std::getline(input, line);          // To start the loop (face normal ... ).
	std::stringstream ss(line);
	while (word != "endsolid")
	{
		std::getline(input, line); // outer loop (ignore)
		
		std::getline(input, line); // vertex ... (ignore)
		ss.clear();
		ss.str(line);
		ss >> word >> v1x >> v1y >> v1z;
		std::getline(input, line); // vertex ... (ignore)
		ss.clear();
		ss.str(line);
		ss >> word >> v2x >> v2y >> v2z;
		std::getline(input, line); // vertex ... (ignore)
		ss.clear();
		ss.str(line);
		ss >> word >> v3x >> v3y >> v3z;
		
		v_geom_f_face_1_X.push_back(v1x);
		v_geom_f_face_1_Y.push_back(v1y);
		v_geom_f_face_1_Z.push_back(v1z);
		v_geom_f_face_2_X.push_back(v2x);
		v_geom_f_face_2_Y.push_back(v2y);
		v_geom_f_face_2_Z.push_back(v2z);
		v_geom_f_face_3_X.push_back(v3x);
		v_geom_f_face_3_Y.push_back(v3y);
		v_geom_f_face_3_Z.push_back(v3z);
		
		std::getline(input, line); // endloop (ignore)
		std::getline(input, line); // endfacet (ignore)
		std::getline(input, line); // facet normal ... (of next facet)   OR   endsolid Exported ...
		ss = std::stringstream(line);
		ss >> word;
		counter++;
	}
	
	// Close file.
	init_coords_list = 1;
	input.close();

	return 0;
}
