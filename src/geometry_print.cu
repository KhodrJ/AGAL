#include "geometry.h"

int Geometry::PrintSTL(int i_dev)
{
	std::ofstream stl = std::ofstream(output_dir + "geometry.stl");
	stl << "solid Geometry" << std::endl;
	
	// Compute facet normals and print.
	for (int i = 0; i < n_faces[i_dev]; i++)
	{
		int p1 = geom_ID_face[i_dev][i + 0*n_faces[i_dev]];
		int p2 = geom_ID_face[i_dev][i + 1*n_faces[i_dev]];
		int p3 = geom_ID_face[i_dev][i + 2*n_faces[i_dev]];
		double v1x = geom_f_node_X[i_dev][p1 + 0*n_nodes[i_dev]];
		double v1y = geom_f_node_X[i_dev][p1 + 1*n_nodes[i_dev]];
		double v1z = geom_f_node_X[i_dev][p1 + 2*n_nodes[i_dev]];
		double v2x = geom_f_node_X[i_dev][p2 + 0*n_nodes[i_dev]];
		double v2y = geom_f_node_X[i_dev][p2 + 1*n_nodes[i_dev]];
		double v2z = geom_f_node_X[i_dev][p2 + 2*n_nodes[i_dev]];
		double v3x = geom_f_node_X[i_dev][p3 + 0*n_nodes[i_dev]];
		double v3y = geom_f_node_X[i_dev][p3 + 1*n_nodes[i_dev]];
		double v3z = geom_f_node_X[i_dev][p3 + 2*n_nodes[i_dev]];
		
// 		int p1 = (*v_geom_ID_face_1)[i];
// 		int p2 = (*v_geom_ID_face_2)[i];
// 		int p3 = (*v_geom_ID_face_3)[i];
// 		double v1x = (*v_geom_f_node_X)[p1];
// 		double v1y = (*v_geom_f_node_Y)[p1];
// 		double v1z = (*v_geom_f_node_Z)[p1];
// 		double v2x = (*v_geom_f_node_X)[p2];
// 		double v2y = (*v_geom_f_node_Y)[p2];
// 		double v2z = (*v_geom_f_node_Z)[p2];
// 		double v3x = (*v_geom_f_node_X)[p3];
// 		double v3y = (*v_geom_f_node_Y)[p3];
// 		double v3z = (*v_geom_f_node_Z)[p3];
#if (N_DIM==2)
		double n1 = v2y-v1y;
		double n2 = -(v2x-v1x);
		double n3 = 0.0;
#else
		double dx1 = v2x-v1x;
		double dy1 = v2y-v1y;
		double dz1 = v2z-v1z;
		double dx2 = v3x-v1x;
		double dy2 = v3y-v1y;
		double dz2 = v3z-v1z;
		double n1 = dy1*dz2-dz1*dy2;
		double n2 = dz1*dx2-dx1*dz2;
		double n3 = dx1*dy2-dy1*dx2;
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
	
	return 0;
}

int Geometry::PrintOBJ(int i_dev)
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
	
// 	// Print vertices.
// 	obj << "Vertices." << std::endl;
// 	for (int i = 0; i < (*v_geom_f_node_X).size(); i++)
// 		obj << "v " << (*v_geom_f_node_X)[i] << " " << (*v_geom_f_node_Y)[i] << " " << (*v_geom_f_node_Z)[i] << std::endl;
// 	obj << std::endl;
// 	
// 	// Print facs.
// 	obj << "Faces." << std::endl;
// 	for (int i = 0; i < (*v_geom_ID_face_1).size(); i++)
// 		obj << "f " << (*v_geom_ID_face_1)[i]+1 << " " << (*v_geom_ID_face_2)[i]+1 << " " << (*v_geom_ID_face_3)[i]+1 << std::endl;
	
	obj.close();

	return 0;
}
