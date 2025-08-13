/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_g_t, int N_DIM>
int RecursiveRefineFace
(
	int Nref,
	const ufloat_g_t &l_spec,
	const ufloat_g_t vx1,
	const ufloat_g_t vy1,
	const ufloat_g_t vz1,
	const ufloat_g_t vx2,
	const ufloat_g_t vy2,
	const ufloat_g_t vz2,
	const ufloat_g_t vx3,
	const ufloat_g_t vy3,
	const ufloat_g_t vz3,
	std::vector<ufloat_g_t> &v_geom_f_face_1_X,
	std::vector<ufloat_g_t> &v_geom_f_face_1_Y,
	std::vector<ufloat_g_t> &v_geom_f_face_1_Z,
	std::vector<ufloat_g_t> &v_geom_f_face_2_X,
	std::vector<ufloat_g_t> &v_geom_f_face_2_Y,
	std::vector<ufloat_g_t> &v_geom_f_face_2_Z,
	std::vector<ufloat_g_t> &v_geom_f_face_3_X,
	std::vector<ufloat_g_t> &v_geom_f_face_3_Y,
	std::vector<ufloat_g_t> &v_geom_f_face_3_Z,
	bool &replaced_first,
	const int &index_first
)
{
	if (N_DIM==2)
	{
		ufloat_g_t lM_p = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1));
		
		if (lM_p > l_spec)
		{
			ufloat_g_t mx = (ufloat_g_t)0.5 * (vx1+vx2);
			ufloat_g_t my = (ufloat_g_t)0.5 * (vy1+vy2);
			
			RecursiveRefineFace<ufloat_g_t,N_DIM>
			(
				Nref+1,l_spec,vx1,vy1,(ufloat_g_t)0.0,mx,my,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,
				v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
				v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
				v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
				replaced_first, index_first
			);
			RecursiveRefineFace<ufloat_g_t,N_DIM>
			(
				Nref+1,l_spec,mx,my,(ufloat_g_t)0.0,vx2,vy2,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,
				v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
				v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
				v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
				replaced_first, index_first
			);
		}
		if (lM_p <= l_spec && Nref > 0)
		{
			if (replaced_first==false)
			{
				v_geom_f_face_1_X[index_first] = vx1;
				v_geom_f_face_1_Y[index_first] = vy1;
				v_geom_f_face_1_Z[index_first] = (ufloat_g_t)0.0;
				v_geom_f_face_2_X[index_first] = vx2;
				v_geom_f_face_2_Y[index_first] = vy2;
				v_geom_f_face_2_Z[index_first] = (ufloat_g_t)0.0;
				v_geom_f_face_3_X[index_first] = (ufloat_g_t)0.0;
				v_geom_f_face_3_Y[index_first] = (ufloat_g_t)0.0;
				v_geom_f_face_3_Z[index_first] = (ufloat_g_t)0.0;
				replaced_first = true;
			}
			else
			{
				v_geom_f_face_1_X.push_back(vx1);
				v_geom_f_face_1_Y.push_back(vy1);
				v_geom_f_face_1_Z.push_back((ufloat_g_t)0.0);
				v_geom_f_face_2_X.push_back(vx2);
				v_geom_f_face_2_Y.push_back(vy2);
				v_geom_f_face_2_Z.push_back((ufloat_g_t)0.0);
				v_geom_f_face_3_X.push_back((ufloat_g_t)0.0);
				v_geom_f_face_3_Y.push_back((ufloat_g_t)0.0);
				v_geom_f_face_3_Z.push_back((ufloat_g_t)0.0);
			}
		}
	}
	else // N_DIM==3
	{
		ufloat_g_t l1 = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
		ufloat_g_t l2 = sqrt((vx3-vx2)*(vx3-vx2) + (vy3-vy2)*(vy3-vy2) + (vz3-vz2)*(vz3-vz2));
		ufloat_g_t l3 = sqrt((vx1-vx3)*(vx1-vx3) + (vy1-vy3)*(vy1-vy3) + (vz1-vz3)*(vz1-vz3));
		ufloat_g_t lM_p = std::max({l1,l2,l3});
		
		if (lM_p > l_spec)
		{
			ufloat_g_t mx1 = (ufloat_g_t)0.5 * (vx1+vx2);
			ufloat_g_t my1 = (ufloat_g_t)0.5 * (vy1+vy2);
			ufloat_g_t mz1 = (ufloat_g_t)0.5 * (vz1+vz2);
			ufloat_g_t mx2 = (ufloat_g_t)0.5 * (vx2+vx3);
			ufloat_g_t my2 = (ufloat_g_t)0.5 * (vy2+vy3);
			ufloat_g_t mz2 = (ufloat_g_t)0.5 * (vz2+vz3);
			ufloat_g_t mx3 = (ufloat_g_t)0.5 * (vx3+vx1);
			ufloat_g_t my3 = (ufloat_g_t)0.5 * (vy3+vy1);
			ufloat_g_t mz3 = (ufloat_g_t)0.5 * (vz3+vz1);
			
			RecursiveRefineFace<ufloat_g_t,N_DIM>
			(
				Nref+1,l_spec,vx1,vy1,vz1,mx1,my1,mz1,mx3,my3,mz3,
				v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
				v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
				v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
				replaced_first, index_first
			);
			RecursiveRefineFace<ufloat_g_t,N_DIM>
			(
				Nref+1,l_spec,mx1,my1,mz1,vx2,vy2,vz2,mx2,my2,mz2,
				v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
				v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
				v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
				replaced_first, index_first
			);
			RecursiveRefineFace<ufloat_g_t,N_DIM>
			(
				Nref+1,l_spec,mx3,my3,mz3,mx2,my2,mz2,vx3,vy3,vz3,
				v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
				v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
				v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
				replaced_first, index_first
			);
			RecursiveRefineFace<ufloat_g_t,N_DIM>
			(
				Nref+1,l_spec,mx1,my1,mz1,mx2,my2,mz2,mx3,my3,mz3,
				v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
				v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
				v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
				replaced_first, index_first
			);
		}
		if (lM_p <= l_spec && Nref > 0)
		{
			if (replaced_first==false)
			{
				v_geom_f_face_1_X[index_first] = vx1;
				v_geom_f_face_1_Y[index_first] = vy1;
				v_geom_f_face_1_Z[index_first] = vz1;
				v_geom_f_face_2_X[index_first] = vx2;
				v_geom_f_face_2_Y[index_first] = vy2;
				v_geom_f_face_2_Z[index_first] = vz2;
				v_geom_f_face_3_X[index_first] = vx3;
				v_geom_f_face_3_Y[index_first] = vy3;
				v_geom_f_face_3_Z[index_first] = vz3;
				replaced_first = true;
			}
			else
			{
				v_geom_f_face_1_X.push_back(vx1);
				v_geom_f_face_1_Y.push_back(vy1);
				v_geom_f_face_1_Z.push_back(vz1);
				v_geom_f_face_2_X.push_back(vx2);
				v_geom_f_face_2_Y.push_back(vy2);
				v_geom_f_face_2_Z.push_back(vz2);
				v_geom_f_face_3_X.push_back(vx3);
				v_geom_f_face_3_Y.push_back(vy3);
				v_geom_f_face_3_Z.push_back(vz3);
			}
		}
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_RefineFaces_Length(int i_dev)
{
	if (v_geom_f_face_1_X.size() > 0)
	{
		G_UpdateCounts(i_dev);
		constexpr int N_DIMg = AP->N_DIM;
		ufloat_g_t l_spec = (ufloat_g_t)(0.95*1.0) * ((ufloat_g_t)Lx/(ufloat_g_t)G_BIN_DENSITY);
		
		// This needs to be done before calling G_Init_Arrays_CoordsList_CPU to make sure that n_faces is finalized
		// before GPU memory is allocated.
		for (int p = n_faces[i_dev]-1; p >= 0; p--)
		{
			ufloat_g_t vx1 = v_geom_f_face_1_X[p];
			ufloat_g_t vy1 = v_geom_f_face_1_Y[p];
			ufloat_g_t vz1 = v_geom_f_face_1_Z[p];
			ufloat_g_t vx2 = v_geom_f_face_2_X[p];
			ufloat_g_t vy2 = v_geom_f_face_2_Y[p];
			ufloat_g_t vz2 = v_geom_f_face_2_Z[p];
			ufloat_g_t vx3 = v_geom_f_face_3_X[p];
			ufloat_g_t vy3 = v_geom_f_face_3_Y[p];
			ufloat_g_t vz3 = v_geom_f_face_3_Z[p];
			
			ufloat_g_t l1 = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
			ufloat_g_t l2 = sqrt((vx3-vx2)*(vx3-vx2) + (vy3-vy2)*(vy3-vy2) + (vz3-vz2)*(vz3-vz2));
			ufloat_g_t l3 = sqrt((vx1-vx3)*(vx1-vx3) + (vy1-vy3)*(vy1-vy3) + (vz1-vz3)*(vz1-vz3));
			ufloat_g_t lM_p = std::max({l1,l2,l3});
			
			bool replaced_first = false;
			RecursiveRefineFace<ufloat_g_t,N_DIMg>
			(
				0,l_spec,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3,
				v_geom_f_face_1_X,
				v_geom_f_face_1_Y,
				v_geom_f_face_1_Z,
				v_geom_f_face_2_X,
				v_geom_f_face_2_Y,
				v_geom_f_face_2_Z,
				v_geom_f_face_3_X,
				v_geom_f_face_3_Y,
				v_geom_f_face_3_Z,
				replaced_first,
				p
			);
		}
		G_UpdateCounts(i_dev);
		std::cout << "Refined. There are now: " << n_faces[i_dev] << " faces..." << std::endl;
	}
	else
	{
		std::cout << "ERROR: There are no faces to refine..." << std::endl;
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_UpdateCounts(int i_dev)
{
	if (init_index_lists)
	{
		n_nodes[i_dev] = v_geom_f_node_X.size();
		n_faces[i_dev] = v_geom_ID_face_1.size();
		n_nodes_a[i_dev] = n_nodes[i_dev] + 128-(n_nodes[i_dev]%128);
		n_faces_a[i_dev] = n_faces[i_dev] + 128-(n_faces[i_dev]%128);
	}
	if (init_coords_list)
	{
		n_faces[i_dev] = v_geom_f_face_1_X.size();
		n_nodes[i_dev] = 3*n_faces[i_dev];
		n_nodes_a[i_dev] = n_nodes[i_dev] + 128-(n_nodes[i_dev]%128);
		n_faces_a[i_dev] = n_faces[i_dev] + 128-(n_faces[i_dev]%128);
	}
	
	if (!init_index_lists && !init_coords_list)
		std::cout << "[-] Warning: neither set of lists has been loaded. Count is zero..." << std::endl;
	
	return 0;
}

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
	G_BIN_OVERLAP           = params_int["G_BIN_OVERLAP"];
	G_BIN_DENSITY           = params_int["G_BIN_DENSITY"];
	G_BIN_FRAC              = params_int["G_BIN_FRAC"];
	G_BIN_APPROACH          = params_int["G_BIN_APPROACH"];
	
	// Derived.
	dx = Lx/Nx;
	//dx = (ufloat_t)(dx / pow(2.0, (ufloat_t)(MAX_LEVELS_WALL-1)));
	
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
		// Constants.
		constexpr int N_DIM = AP->N_DIM;
		ufloat_g_t odenom = (ufloat_g_t)1.0 / (ufloat_g_t)N_DIM;
		ufloat_g_t eps = (ufloat_g_t)0.0;
		if (std::is_same<ufloat_g_t, float>::value) eps = FLT_EPSILON;
		if (std::is_same<ufloat_g_t, double>::value) eps = DBL_EPSILON;
		
		// Update tracker variable and face count.
		init_coords_list = 1;
		G_RefineFaces_Length(i_dev);
		//G_UpdateCounts(i_dev);
		
		std::cout << "[-] Initializing the CPU coords list array..." << std::endl;
		geom_f_face_X[i_dev] = new ufloat_g_t[16*n_faces_a[i_dev]];
		geom_f_face_Xt[i_dev] = new ufloat_g_t[16*n_faces_a[i_dev]];
		
		for (int j = 0; j < n_faces_a[i_dev]; j++)
		{
			for (int p = 0; p < 16; p++) geom_f_face_X[i_dev][j + p*n_faces_a[i_dev]] = 0.0;
			for (int p = 0; p < 16; p++) geom_f_face_Xt[i_dev][p + j*16] = 0.0;
			if (j < n_faces[i_dev])
			{
				// Load vertices from vectors.
				ufloat_g_t vx1 = v_geom_f_face_1_X[j];
				ufloat_g_t vy1 = v_geom_f_face_1_Y[j];
				ufloat_g_t vz1 = v_geom_f_face_1_Z[j];
				ufloat_g_t vx2 = v_geom_f_face_2_X[j];
				ufloat_g_t vy2 = v_geom_f_face_2_Y[j];
				ufloat_g_t vz2 = v_geom_f_face_2_Z[j];
				ufloat_g_t vx3 = v_geom_f_face_3_X[j];
				ufloat_g_t vy3 = v_geom_f_face_3_Y[j];
				ufloat_g_t vz3 = v_geom_f_face_3_Z[j];
				
				
				
				// Compute face normals.
				ufloat_g_t nx = (ufloat_g_t)0.0;
				ufloat_g_t ny = (ufloat_g_t)0.0;
				ufloat_g_t nz = (ufloat_g_t)0.0;
				ufloat_g_t tmp = (ufloat_g_t)0.0;
				ufloat_g_t ex1 = vx2-vx1;
				ufloat_g_t ey1 = vy2-vy1;
				ufloat_g_t ez1 = vz2-vz1;
				ufloat_g_t ex2 = vx3-vx1;
				ufloat_g_t ey2 = vy3-vy1;
				ufloat_g_t ez2 = vz3-vz1;
				ufloat_g_t ex3 = vx1-vx3;
				ufloat_g_t ey3 = vy1-vy3;
				ufloat_g_t ez3 = vz1-vz3;
				if (N_DIM==2)
				{
					nx = vy2-vy1;
					ny = vx1-vx2;
				}
				else
				{
					nx = ey1*ez2 - ez1*ey2;
					ny = ez1*ex2 - ex1*ez2;
					nz = ex1*ey2 - ey1*ex2;
				}
				ex2 = vx3-vx2;
				ey2 = vy3-vy2;
				ez2 = vz3-vz2;
				tmp = sqrt(nx*nx + ny*ny + nz*nz);
				nx /= tmp;
				ny /= tmp;
				nz /= tmp;
				tmp = sqrt(ex1*ex1 + ey1*ey1 + ez1*ez1);
				ex1 /= tmp;
				ey1 /= tmp;
				ez1 /= tmp;
				tmp = sqrt(ex2*ex2 + ey2*ey2 + ez2*ez2);
				ex2 /= tmp;
				ey2 /= tmp;
				ez2 /= tmp;
				tmp = sqrt(ex3*ex3 + ey3*ey3 + ez3*ez3);
				ex3 /= tmp;
				ey3 /= tmp;
				ez3 /= tmp;
				
				// Compute edge normals.
				ufloat_g_t enx1 = ey1*nz - ez1*ny;
				ufloat_g_t eny1 = ez1*nx - ex1*nz;
				ufloat_g_t enz1 = ex1*ny - ey1*nx;
				ufloat_g_t enx2 = ey2*nz - ez2*ny;
				ufloat_g_t eny2 = ez2*nx - ex2*nz;
				ufloat_g_t enz2 = ex2*ny - ey2*nx;
				ufloat_g_t enx3 = ey3*nz - ez3*ny;
				ufloat_g_t eny3 = ez3*nx - ex3*nz;
				ufloat_g_t enz3 = ex3*ny - ey3*nx;
				tmp = sqrt(enx1*enx1 + eny1*eny1 + enz1*enz1);
				enx1 /= tmp;
				eny1 /= tmp;
				enz1 /= tmp;
				tmp = sqrt(enx2*enx2 + eny2*eny2 + enz2*enz2);
				enx2 /= tmp;
				eny2 /= tmp;
				enz2 /= tmp;
				tmp = sqrt(enx3*enx3 + eny3*eny3 + enz3*enz3);
				enx3 /= tmp;
				eny3 /= tmp;
				enz3 /= tmp;
				
				// Adjust vertices so that cell-face links don't suffer from roundoff problems.
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
				vx1 = vx1 - eps*vxe;
				vy1 = vy1 - eps*vye;
				vz1 = vz1 - eps*vze;
					// Vertex 2.
				vxe = vx2 - vxc;
				vye = vy2 - vyc;
				vze = vz2 - vzc;
				norm = sqrt(vxe*vxe + vye*vye + vze*vze);
				vxe /= norm; vye /= norm; vze /= norm;
				vx2 = vx2 - eps*vxe;
				vy2 = vy2 - eps*vye;
				vz2 = vz2 - eps*vze;
					// Vertex 3.
				vxe = vx3 - vxc;
				vye = vy3 - vyc;
				vze = vz3 - vzc;
				norm = sqrt(vxe*vxe + vye*vye + vze*vze);
				vxe /= norm; vye /= norm; vze /= norm;
				vx3 = vx3 - eps*vxe;
				vy3 = vy3 - eps*vye;
				vz3 = vz3 - eps*vze;
				
				
				
				// Write vertices.
				geom_f_face_X[i_dev][j + 0*n_faces_a[i_dev]] = vx1;
				geom_f_face_X[i_dev][j + 1*n_faces_a[i_dev]] = vy1;
				geom_f_face_X[i_dev][j + 2*n_faces_a[i_dev]] = vz1;
				geom_f_face_X[i_dev][j + 3*n_faces_a[i_dev]] = vx2;
				geom_f_face_X[i_dev][j + 4*n_faces_a[i_dev]] = vy2;
				geom_f_face_X[i_dev][j + 5*n_faces_a[i_dev]] = vz2;
				geom_f_face_X[i_dev][j + 6*n_faces_a[i_dev]] = vx3;
				geom_f_face_X[i_dev][j + 7*n_faces_a[i_dev]] = vy3;
				geom_f_face_X[i_dev][j + 8*n_faces_a[i_dev]] = vz3;
				//
				geom_f_face_Xt[i_dev][0 + j*16] = vx1;
				geom_f_face_Xt[i_dev][1 + j*16] = vy1;
				geom_f_face_Xt[i_dev][2 + j*16] = vz1;
				geom_f_face_Xt[i_dev][3 + j*16] = vx2;
				geom_f_face_Xt[i_dev][4 + j*16] = vy2;
				geom_f_face_Xt[i_dev][5 + j*16] = vz2;
				geom_f_face_Xt[i_dev][6 + j*16] = vx3;
				geom_f_face_Xt[i_dev][7 + j*16] = vy3;
				geom_f_face_Xt[i_dev][8 + j*16] = vz3;
				
				// Write normals.
				geom_f_face_X[i_dev][j + 9*n_faces_a[i_dev]] = nx;
				geom_f_face_X[i_dev][j + 10*n_faces_a[i_dev]] = ny;
				geom_f_face_X[i_dev][j + 11*n_faces_a[i_dev]] = nz;
				//
				geom_f_face_Xt[i_dev][9 + j*16] = nx;
				geom_f_face_Xt[i_dev][10 + j*16] = ny;
				geom_f_face_Xt[i_dev][11 + j*16] = nz;
				
				// Write edges.
// 				geom_f_face_X[i_dev][j + 12*n_faces_a[i_dev]] = ex1;
// 				geom_f_face_X[i_dev][j + 13*n_faces_a[i_dev]] = ey1;
// 				geom_f_face_X[i_dev][j + 14*n_faces_a[i_dev]] = ez1;
// 				geom_f_face_X[i_dev][j + 15*n_faces_a[i_dev]] = ex2;
// 				geom_f_face_X[i_dev][j + 16*n_faces_a[i_dev]] = ey2;
// 				geom_f_face_X[i_dev][j + 17*n_faces_a[i_dev]] = ez2;
// 				geom_f_face_X[i_dev][j + 18*n_faces_a[i_dev]] = ex3;
// 				geom_f_face_X[i_dev][j + 19*n_faces_a[i_dev]] = ey3;
// 				geom_f_face_X[i_dev][j + 20*n_faces_a[i_dev]] = ez3;
				//
// 				geom_f_face_X[i_dev][12 + j*32] = ex1;
// 				geom_f_face_X[i_dev][13 + j*32] = ey1;
// 				geom_f_face_X[i_dev][14 + j*32] = ez1;
// 				geom_f_face_X[i_dev][15 + j*32] = ex2;
// 				geom_f_face_X[i_dev][16 + j*32] = ey2;
// 				geom_f_face_X[i_dev][17 + j*32] = ez2;
// 				geom_f_face_X[i_dev][18 + j*32] = ex3;
// 				geom_f_face_X[i_dev][19 + j*32] = ey3;
// 				geom_f_face_X[i_dev][20 + j*32] = ez3;
				
				// Write edge normals.
// 				geom_f_face_X[i_dev][j + 21*n_faces_a[i_dev]] = enx1;
// 				geom_f_face_X[i_dev][j + 22*n_faces_a[i_dev]] = eny1;
// 				geom_f_face_X[i_dev][j + 23*n_faces_a[i_dev]] = enz1;
// 				geom_f_face_X[i_dev][j + 24*n_faces_a[i_dev]] = enx2;
// 				geom_f_face_X[i_dev][j + 25*n_faces_a[i_dev]] = eny2;
// 				geom_f_face_X[i_dev][j + 26*n_faces_a[i_dev]] = enz2;
// 				geom_f_face_X[i_dev][j + 27*n_faces_a[i_dev]] = enx3;
// 				geom_f_face_X[i_dev][j + 28*n_faces_a[i_dev]] = eny3;
// 				geom_f_face_X[i_dev][j + 29*n_faces_a[i_dev]] = enz3;
				//
// 				geom_f_face_X[i_dev][21 + j*32] = enx1;
// 				geom_f_face_X[i_dev][22 + j*32] = eny1;
// 				geom_f_face_X[i_dev][23 + j*32] = enz1;
// 				geom_f_face_X[i_dev][24 + j*32] = enx2;
// 				geom_f_face_X[i_dev][25 + j*32] = eny2;
// 				geom_f_face_X[i_dev][26 + j*32] = enz2;
// 				geom_f_face_X[i_dev][27 + j*32] = enx3;
// 				geom_f_face_X[i_dev][28 + j*32] = eny3;
// 				geom_f_face_X[i_dev][29 + j*32] = enz3;
			}
		}
		
		// Allocate memory on the GPU to store geometry data and copy the CPU data.
		std::cout << "[-] Initializing the GPU coords list array..." << std::endl;
		gpuErrchk( cudaMalloc((void **)&c_geom_f_face_X[i_dev], 16*n_faces_a[i_dev]*sizeof(ufloat_g_t)) );
		gpuErrchk( cudaMalloc((void **)&c_geom_f_face_Xt[i_dev], 16*n_faces_a[i_dev]*sizeof(ufloat_g_t)) );
		gpuErrchk( cudaMemcpy(c_geom_f_face_X[i_dev], geom_f_face_X[i_dev], 16*n_faces_a[i_dev]*sizeof(ufloat_g_t), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_geom_f_face_Xt[i_dev], geom_f_face_Xt[i_dev], 16*n_faces_a[i_dev]*sizeof(ufloat_g_t), cudaMemcpyHostToDevice) );
		std::cout << "[-] Finished copying the coords list array to the GPU..." << std::endl;
		cudaDeviceSynchronize();
	}
	else
		std::cout << "[-] Coords list already built, reset first to rebuild..." << std::endl;
	
	return 0;
}
