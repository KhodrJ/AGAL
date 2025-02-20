#include "geometry.h"

int Geometry::G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str)
{
	Lx                      = params_dbl["L_c"];
	Ly                      = params_dbl["L_fy"]*Lx;
	Lz                      = params_dbl["L_fz"]*Lx;
	input_dir               = params_str["I_DIR_NAME"];
	output_dir              = params_str["P_DIR_NAME"];
	G_NEAR_WALL_DISTANCE    = params_dbl["G_NEAR_WALL_DISTANCE"];
	G_BIN_DENSITY           = params_int["G_BIN_DENSITY"];
	G_BIN_FRAC              = params_int["G_BIN_FRAC"];
	
	return 0;
}

int Geometry::G_Init_Arrays_IndexLists_CPU(int i_dev)
{
	if (v_geom_f_node_X.size() > 0)
	{
		init_index_lists = 1;
		G_UpdateCounts(i_dev);
		geom_f_node_X[i_dev] = new ufloat_g_t[3*n_nodes_a[i_dev]];
		geom_ID_face[i_dev] = new int[3*n_faces_a[i_dev]];
		geom_ID_face_attr[i_dev] = new ufloat_g_t[n_faces_a[i_dev]];
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
	}
	else
		std::cout << "[-] Index lists already built, reset first to rebuild..." << std::endl;
	
	return 0;
}

int Geometry::G_Init_Arrays_CoordsList_CPU(int i_dev)
{
	if (v_geom_f_face_1_X.size() > 0)
	{
		init_coords_list = 1;
		G_UpdateCounts(i_dev);
		
		std::cout << "[-] Initializing the CPU coords list array..." << std::endl;
		geom_f_face_X[i_dev] = new ufloat_g_t[9*n_faces_a[i_dev]];
// 		std::fill_n(geom_f_face_X[i_dev], 9*n_faces_a[i_dev], (ufloat_g_t)0.0);
// 		std::copy(v_geom_f_face_1_X.begin(), v_geom_f_face_1_X.end(), &geom_f_face_X[i_dev][0*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_1_Y.begin(), v_geom_f_face_1_Y.end(), &geom_f_face_X[i_dev][1*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_1_Z.begin(), v_geom_f_face_1_Z.end(), &geom_f_face_X[i_dev][2*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_2_X.begin(), v_geom_f_face_2_X.end(), &geom_f_face_X[i_dev][3*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_2_Y.begin(), v_geom_f_face_2_Y.end(), &geom_f_face_X[i_dev][4*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_2_Z.begin(), v_geom_f_face_2_X.end(), &geom_f_face_X[i_dev][5*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_3_X.begin(), v_geom_f_face_3_X.end(), &geom_f_face_X[i_dev][6*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_3_Y.begin(), v_geom_f_face_3_Y.end(), &geom_f_face_X[i_dev][7*n_faces_a[i_dev]]);
// 		std::copy(v_geom_f_face_3_Z.begin(), v_geom_f_face_3_Z.end(), &geom_f_face_X[i_dev][8*n_faces_a[i_dev]]);
		
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
		
		//gpuErrchk( cudaMalloc((void **)&c_cells_ID_mask[i_dev], n_maxcells*sizeof(int)) );
		//gpuErrchk( cudaMemcpy(c_geom_f_node_X[i_dev], geom_f_node_X[i_dev], 3*n_nodes[i_dev]*sizeof(double), cudaMemcpyHostToDevice) );
		
		std::cout << "[-] Initializing the GPU coords list array..." << std::endl;
		gpuErrchk( cudaMalloc((void **)&c_geom_f_face_X[i_dev], 9*n_faces_a[i_dev]*sizeof(ufloat_g_t)) );
		gpuErrchk( cudaMemcpy(c_geom_f_face_X[i_dev], geom_f_face_X[i_dev], 9*n_faces_a[i_dev]*sizeof(ufloat_g_t), cudaMemcpyHostToDevice) );
		std::cout << "[-] Finished copying the coords list array to the GPU..." << std::endl;
		cudaDeviceSynchronize();
	}
	else
		std::cout << "[-] Coords list already built, reset first to rebuild..." << std::endl;
	
	return 0;
}

int Geometry::G_UpdateCounts(int i_dev)
{
	if (init_index_lists)
	{
		n_nodes[i_dev] = v_geom_f_node_X.size();
		n_faces[i_dev] = v_geom_ID_face_1.size();
		n_nodes_a[i_dev] = n_nodes[i_dev] + 256-(n_nodes[i_dev]%256);
		n_faces_a[i_dev] = n_faces[i_dev] + 256-(n_faces[i_dev]%256);
	}
	if (init_coords_list)
	{
		n_faces[i_dev] = v_geom_f_face_1_X.size();
		n_nodes[i_dev] = 3*n_faces[i_dev];
		n_nodes_a[i_dev] = n_nodes[i_dev] + 256-(n_nodes[i_dev]%256);
		n_faces_a[i_dev] = n_faces[i_dev] + 256-(n_faces[i_dev]%256);
	}
	
	if (!init_index_lists && !init_coords_list)
		std::cout << "[-] Warning: neither set of lists has been loaded. Count is zero..." << std::endl;
	
	return 0;
}
