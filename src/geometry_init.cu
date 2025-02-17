#include "geometry.h"

int Geometry::G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str)
{
	Lx                      = params_dbl["L_c"];
	Ly                      = params_dbl["L_fy"]*Lx;
	Lz                      = params_dbl["L_fz"]*Lx;
	input_dir               = params_str["I_DIR_NAME"];
	output_dir              = params_str["P_DIR_NAME"];
	
	return 0;
}

int Geometry::G_Init_Arrays_IndexLists_CPU(int i_dev)
{
	if (init_index_lists == 0)
	{
		G_UpdateCounts(i_dev);
		geom_f_node_X[i_dev] = new double[3*n_nodes_a[i_dev]];
		geom_ID_face[i_dev] = new int[3*n_faces_a[i_dev]];
		geom_ID_face_attr[i_dev] = new double[n_faces_a[i_dev]];
		for (int j = 0; j < n_nodes_a[i_dev]; j++)
		{
			geom_f_node_X[i_dev][j + 0*n_nodes_a[i_dev]] = 0.0;
			geom_f_node_X[i_dev][j + 1*n_nodes_a[i_dev]] = 0.0;
			geom_f_node_X[i_dev][j + 2*n_nodes_a[i_dev]] = 0.0;
			if (j < n_nodes[i_dev])
			{
				geom_f_node_X[i_dev][j + 0*n_nodes_a[i_dev]] = v_geom_f_node_X[j];
				geom_f_node_X[i_dev][j + 1*n_nodes_a[i_dev]] = v_geom_f_node_Y[j];
				geom_f_node_X[i_dev][j + 2*n_nodes_a[i_dev]] = v_geom_f_node_Z[j];
			}
		}
		for (int j = 0; j < n_faces_a[i_dev]; j++)
		{
			geom_ID_face[i_dev][j + 0*n_faces_a[i_dev]] = -1;
			geom_ID_face[i_dev][j + 1*n_faces_a[i_dev]] = -1;
			geom_ID_face[i_dev][j + 2*n_faces_a[i_dev]] = -1;
			if (j < n_faces[i_dev])
			{
				geom_ID_face[i_dev][j + 0*n_faces_a[i_dev]] = v_geom_ID_face_1[j];
				geom_ID_face[i_dev][j + 1*n_faces_a[i_dev]] = v_geom_ID_face_2[j];
				geom_ID_face[i_dev][j + 2*n_faces_a[i_dev]] = v_geom_ID_face_3[j];
			}
		}
		init_index_lists = 1;
	}
	else
		std::cout << "[-] Index lists already built, reset first to rebuild..." << std::endl;
	
	return 0;
}

int Geometry::G_Init_Arrays_CoordsList_CPU(int i_dev)
{
	if (init_coords_list == 0)
	{
		G_UpdateCounts(i_dev);
		geom_f_face_X[i_dev] = new double[9*n_faces_a[i_dev]];
		for (int j = 0; j < n_faces_a[i_dev]; j++)
		{
			geom_f_face_X[i_dev][j + 0*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 1*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 2*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 3*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 4*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 5*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 6*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 7*n_faces_a[i_dev]] = 0.0;
			geom_f_face_X[i_dev][j + 8*n_faces_a[i_dev]] = 0.0;
			if (j < n_faces[i_dev])
			{
				geom_f_face_X[i_dev][j + 0*n_faces_a[i_dev]] = v_geom_f_face_1_X[j];
				geom_f_face_X[i_dev][j + 1*n_faces_a[i_dev]] = v_geom_f_face_1_Y[j];
				geom_f_face_X[i_dev][j + 2*n_faces_a[i_dev]] = v_geom_f_face_1_Z[j];
				geom_f_face_X[i_dev][j + 3*n_faces_a[i_dev]] = v_geom_f_face_2_X[j];
				geom_f_face_X[i_dev][j + 4*n_faces_a[i_dev]] = v_geom_f_face_2_Y[j];
				geom_f_face_X[i_dev][j + 5*n_faces_a[i_dev]] = v_geom_f_face_2_Z[j];
				geom_f_face_X[i_dev][j + 6*n_faces_a[i_dev]] = v_geom_f_face_3_X[j];
				geom_f_face_X[i_dev][j + 7*n_faces_a[i_dev]] = v_geom_f_face_3_Y[j];
				geom_f_face_X[i_dev][j + 8*n_faces_a[i_dev]] = v_geom_f_face_3_Z[j];
			}
		}
		
		init_coords_list = 1;
	}
	else
		std::cout << "[-] Coords list already built, reset first to rebuild..." << std::endl;
	
	return 0;
}

int Geometry::G_UpdateCounts(int i_dev)
{
	n_nodes[i_dev] = v_geom_f_node_X.size();
	n_faces[i_dev] = v_geom_ID_face_1.size();
	n_nodes_a[i_dev] = n_nodes[i_dev] + 256-(n_nodes[i_dev]%256);
	n_faces_a[i_dev] = n_faces[i_dev] + 256-(n_faces[i_dev]%256);
	
	return 0;
}
