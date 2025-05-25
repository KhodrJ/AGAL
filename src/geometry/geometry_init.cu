/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str)
{
	// From input.
	Lx                      = params_dbl["L_c"];
	Ly                      = params_dbl["L_fy"]*Lx;
	Lz                      = params_dbl["L_fz"]*Lx;
	Nx                      = params_int["Nx"];
	input_dir               = params_str["I_DIR_NAME"];
	output_dir              = params_str["P_DIR_NAME"];
	MAX_LEVELS_WALL         = params_int["MAX_LEVELS_WALL"];
	G_NEAR_WALL_DISTANCE    = params_dbl["G_NEAR_WALL_DISTANCE"];
	G_FILENAME              = params_str["G_FILENAME"];
	G_LOADTYPE              = params_int["G_LOADTYPE"];
	G_PRINT                 = params_int["G_PRINT"];
	G_BIN_DENSITY           = params_int["G_BIN_DENSITY"];
	G_BIN_FRAC              = params_int["G_BIN_FRAC"];
	
	// Derived.
	dx = Lx/Nx;
	dx = (ufloat_t)(dx / pow(2.0, (ufloat_t)MAX_LEVELS_WALL));
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_Init_Arrays_IndexLists_CPU(int i_dev)
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

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_Init_Arrays_CoordsList_CPU(int i_dev)
{
	if (v_geom_f_face_1_X.size() > 0)
	{
		// Constants/
		constexpr int N_DIM = AP->N_DIM;
		ufloat_g_t odenom = (ufloat_g_t)1.0 / (ufloat_g_t)N_DIM;
		ufloat_g_t eps = (ufloat_g_t)0.0;
		if (std::is_same<ufloat_g_t, float>::value) eps = FLT_EPSILON;
		if (std::is_same<ufloat_g_t, double>::value) eps = DBL_EPSILON;
		eps = eps*(ufloat_g_t)100;
		std::cout << "Using EPS=" << eps << std::endl;
		
		// Update tracker variable and face count.
		init_coords_list = 1;
		G_UpdateCounts(i_dev);
		
		std::cout << "[-] Initializing the CPU coords list array..." << std::endl;
		geom_f_face_X[i_dev] = new ufloat_g_t[9*n_faces_a[i_dev]];
		
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
// 				geom_f_face_X[i_dev][j + 0*n_faces_a[i_dev]] = v_geom_f_face_1_X[j];
// 				geom_f_face_X[i_dev][j + 1*n_faces_a[i_dev]] = v_geom_f_face_1_Y[j];
// 				geom_f_face_X[i_dev][j + 2*n_faces_a[i_dev]] = v_geom_f_face_1_Z[j];
// 				geom_f_face_X[i_dev][j + 3*n_faces_a[i_dev]] = v_geom_f_face_2_X[j];
// 				geom_f_face_X[i_dev][j + 4*n_faces_a[i_dev]] = v_geom_f_face_2_Y[j];
// 				geom_f_face_X[i_dev][j + 5*n_faces_a[i_dev]] = v_geom_f_face_2_Z[j];
// 				geom_f_face_X[i_dev][j + 6*n_faces_a[i_dev]] = v_geom_f_face_3_X[j];
// 				geom_f_face_X[i_dev][j + 7*n_faces_a[i_dev]] = v_geom_f_face_3_Y[j];
// 				geom_f_face_X[i_dev][j + 8*n_faces_a[i_dev]] = v_geom_f_face_3_Z[j];
				
				ufloat_g_t vx1 = v_geom_f_face_1_X[j];
				ufloat_g_t vy1 = v_geom_f_face_1_Y[j];
				ufloat_g_t vz1 = v_geom_f_face_1_Z[j];
				ufloat_g_t vx2 = v_geom_f_face_2_X[j];
				ufloat_g_t vy2 = v_geom_f_face_2_Y[j];
				ufloat_g_t vz2 = v_geom_f_face_2_Z[j];
				ufloat_g_t vx3 = v_geom_f_face_3_X[j];
				ufloat_g_t vy3 = v_geom_f_face_3_Y[j];
				ufloat_g_t vz3 = v_geom_f_face_3_Z[j];
				
				ufloat_g_t vxc = odenom*(vx1 + vx2 + vx3);
				ufloat_g_t vyc = odenom*(vy1 + vy2 + vy3);
				ufloat_g_t vzc = odenom*(vz1 + vz2 + vz3);
				ufloat_g_t vxe;
				ufloat_g_t vye;
				ufloat_g_t vze;
				ufloat_g_t norm;
				
				// Vertex 1.
				vxe = vx1 - vxc;
				vye = vy1 - vyc;
				vze = vz1 - vzc;
				norm = sqrt(vxe*vxe + vye*vye + vze*vze);
				vxe /= norm; vye /= norm; vze /= norm;
// 				vx1 = vx1 + eps*vxe;
// 				vy1 = vy1 + eps*vye;
// 				vz1 = vz1 + eps*vze;
				vx1 = vx1 + eps;
				vy1 = vy1 + eps;
				vz1 = vz1 + eps;
				
				// Vertex 2.
				vxe = vx2 - vxc;
				vye = vy2 - vyc;
				vze = vz2 - vzc;
				norm = sqrt(vxe*vxe + vye*vye + vze*vze);
				vxe /= norm; vye /= norm; vze /= norm;
// 				vx2 = vx2 + eps*vxe;
// 				vy2 = vy2 + eps*vye;
// 				vz2 = vz2 + eps*vze;
				vx2 = vx2 + eps;
				vy2 = vy2 + eps;
				vz2 = vz2 + eps;
				
				// Vertex 3.
				vxe = vx3 - vxc;
				vye = vy3 - vyc;
				vze = vz3 - vzc;
				norm = sqrt(vxe*vxe + vye*vye + vze*vze);
				vxe /= norm; vye /= norm; vze /= norm;
// 				vx3 = vx3 + eps*vxe;
// 				vy3 = vy3 + eps*vye;
// 				vz3 = vz3 + eps*vze;
				vx3 = vx3 + eps;
				vy3 = vy3 + eps;
				vz3 = vz3 + eps;
				
				geom_f_face_X[i_dev][j + 0*n_faces_a[i_dev]] = vx1;
				geom_f_face_X[i_dev][j + 1*n_faces_a[i_dev]] = vy1;
				geom_f_face_X[i_dev][j + 2*n_faces_a[i_dev]] = vz1;
				geom_f_face_X[i_dev][j + 3*n_faces_a[i_dev]] = vx2;
				geom_f_face_X[i_dev][j + 4*n_faces_a[i_dev]] = vy2;
				geom_f_face_X[i_dev][j + 5*n_faces_a[i_dev]] = vz2;
				geom_f_face_X[i_dev][j + 6*n_faces_a[i_dev]] = vx3;
				geom_f_face_X[i_dev][j + 7*n_faces_a[i_dev]] = vy3;
				geom_f_face_X[i_dev][j + 8*n_faces_a[i_dev]] = vz3;
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

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_UpdateCounts(int i_dev)
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
