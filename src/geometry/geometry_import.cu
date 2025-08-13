/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_ImportSTL_ASCII(std::string filename)
{
	// Parameters.
	std::ifstream input = std::ifstream(input_dir + "geometry/" + filename);
	std::string line = "";
	std::string word = "";
	
	// Get the first line and ignore it (solid Exported ... ).
	std::getline(input, line);
	
	// Now loop over the various facets and add face vertices to the vectors.
	int counter = 0;
	ufloat_g_t v1x = 0.0;
	ufloat_g_t v1y = 0.0;
	ufloat_g_t v1z = 0.0;
	ufloat_g_t v2x = 0.0;
	ufloat_g_t v2y = 0.0;
	ufloat_g_t v2z = 0.0;
	ufloat_g_t v3x = 0.0;
	ufloat_g_t v3y = 0.0;
	ufloat_g_t v3z = 0.0;
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
		
		// Check for duplicate vertices (degenerate triangles).
		bool C1 = !(v1x == v2x && v1y == v2y && v1z == v2z);
		bool C2 = !(v1x == v3x && v1y == v3y && v1z == v3z);
		bool C3 = !(v3x == v2x && v3y == v2y && v3z == v2z);
		
		if (C1 && C2 && C3)
		{
			v_geom_f_face_1_X.push_back(v1x);
			v_geom_f_face_1_Y.push_back(v1y);
			v_geom_f_face_1_Z.push_back(v1z);
			v_geom_f_face_2_X.push_back(v2x);
			v_geom_f_face_2_Y.push_back(v2y);
			v_geom_f_face_2_Z.push_back(v2z);
			v_geom_f_face_3_X.push_back(v3x);
			v_geom_f_face_3_Y.push_back(v3y);
			v_geom_f_face_3_Z.push_back(v3z);
			counter++;
		}
		
		std::getline(input, line); // endloop (ignore)
		std::getline(input, line); // endfacet (ignore)
		std::getline(input, line); // facet normal ... (of next facet)   OR   endsolid Exported ...
		ss = std::stringstream(line);
		ss >> word;
	}
	
	// Close file.
	init_coords_list = 1;
	input.close();
	std::cout << "[-] Finished importing " << filename << ", added " << counter << " triangles..." << std::endl;
	
	return 0;
}
