#include "geometry.h"

int Geometry::G_PrintSTL(int i_dev)
{
	std::cout << "[-] Printing STL file..." << std::endl;
	std::ofstream stl = std::ofstream(output_dir + "geometry.stl");
	stl << "solid Geometry" << std::endl;
	
	// Compute facet normals and print.
	for (int i = 0; i < n_faces[i_dev]; i++)
	{
		ufloat_g_t v1x = geom_f_face_X[i_dev][i + 0*n_faces_a[i_dev]];
		ufloat_g_t v1y = geom_f_face_X[i_dev][i + 1*n_faces_a[i_dev]];
		ufloat_g_t v1z = geom_f_face_X[i_dev][i + 2*n_faces_a[i_dev]];
		ufloat_g_t v2x = geom_f_face_X[i_dev][i + 3*n_faces_a[i_dev]];
		ufloat_g_t v2y = geom_f_face_X[i_dev][i + 4*n_faces_a[i_dev]];
		ufloat_g_t v2z = geom_f_face_X[i_dev][i + 5*n_faces_a[i_dev]];
		ufloat_g_t v3x = geom_f_face_X[i_dev][i + 6*n_faces_a[i_dev]];
		ufloat_g_t v3y = geom_f_face_X[i_dev][i + 7*n_faces_a[i_dev]];
		ufloat_g_t v3z = geom_f_face_X[i_dev][i + 8*n_faces_a[i_dev]];
#if (N_DIM==2)
		ufloat_g_t n1 = v2y-v1y;
		ufloat_g_t n2 = -(v2x-v1x);
		ufloat_g_t n3 = 0.0;
#else
		ufloat_g_t dx1 = v2x-v1x;
		ufloat_g_t dy1 = v2y-v1y;
		ufloat_g_t dz1 = v2z-v1z;
		ufloat_g_t dx2 = v3x-v1x;
		ufloat_g_t dy2 = v3y-v1y;
		ufloat_g_t dz2 = v3z-v1z;
		ufloat_g_t n1 = dy1*dz2-dz1*dy2;
		ufloat_g_t n2 = dz1*dx2-dx1*dz2;
		ufloat_g_t n3 = dx1*dy2-dy1*dx2;
#endif
		
		stl << "facet normal " << n1 << " " << n2 << " " << n3 << std::endl;
		stl << "    outer loop" << std::endl;
		stl << "        vertex " << v1x << " " << v1y << " " << v1z << std::endl;
		stl << "        vertex " << v2x << " " << v2y << " " << v2z << std::endl;
		stl << "        vertex " << v3x << " " << v3y << " " << v3z << std::endl;
		stl << "    endloop" << std::endl;
		stl << "endfacet" << std::endl;
	}
	
	stl << "endsolid Geometry" << std::endl;
	stl.close();
	std::cout << "[-] Finished printing STL file..." << std::endl;
	
	return 0;
}

int Geometry::G_PrintOBJ(int i_dev)
{
	std::ofstream obj = std::ofstream(output_dir + "geometry.obj");
	obj << "# Geometry." << std::endl << std::endl;;
	
	// Print vertices.
	obj << "Vertices." << std::endl;
	for (int i = 0; i < n_nodes[i_dev]; i++)
		obj << "v " << geom_f_node_X[i_dev][i + 0*n_nodes[i_dev]] << " " << geom_f_node_X[i_dev][i + 1*n_nodes[i_dev]] << " " << geom_f_node_X[i_dev][i + 2*n_nodes[i_dev]] << std::endl;
	obj << std::endl;
	
	// Print facs.
	obj << "Faces." << std::endl;
	for (int i = 0; i < n_faces[i_dev]; i++)
		obj << "f " << geom_ID_face[i_dev][i + 0*n_faces[i_dev]]+1 << " " << geom_ID_face[i_dev][i + 1*n_faces[i_dev]]+1 << " " << geom_ID_face[i_dev][i + 2*n_faces[i_dev]]+1 << std::endl;
	obj.close();
	std::cout << "[-] Finished printing OBJ file..." << std::endl;
	
	return 0;
}
