#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_ComputeRefCriteria_NearWall_Geometry_Binned
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dx_L, int L, int MAX_LEVELS_WALL,
	int *cells_ID_mask, int *cblock_ID_ref, int *cblock_ID_onb, int *cblock_ID_mask, ufloat_t *cblock_f_X, int *cblock_ID_nbr,
	int n_faces, int n_faces_a, ufloat_g_t *geom_f_face_X, ufloat_g_t R, ufloat_g_t R2,
	int *binned_face_ids_n, int *binned_face_ids_N, int *binned_face_ids, int G_BIN_DENSITY, int n_bins
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int N_Q_max = AP->N_Q_max;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	//constexpr int M_CBLOCK = AP->M_CBLOCK;
	constexpr int M_LBLOCK = AP->M_LBLOCK;
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_D[M_TBLOCK];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int I_kap = threadIdx.x % 4;
	int J_kap = (threadIdx.x / 4) % 4;
	int K_kap = (threadIdx.x / 4) / 4;
	int i_kap_b = -1;
	int global_bin_id;
	ufloat_g_t vx1 = (ufloat_g_t)0.0;
	ufloat_g_t vy1 = (ufloat_g_t)0.0;
	ufloat_g_t vz1 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vx2 = (ufloat_g_t)0.0;
	ufloat_g_t vy2 = (ufloat_g_t)0.0;
	ufloat_g_t vz2 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vx3 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vy3 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t vz3 __attribute__((unused)) = (ufloat_g_t)0.0;
	ufloat_g_t nx = (ufloat_g_t)0.0;
	ufloat_g_t ny = (ufloat_g_t)0.0;
	ufloat_g_t nz = (ufloat_g_t)0.0;
	ufloat_g_t ex1 = (ufloat_g_t)0.0;
	ufloat_g_t ey1 = (ufloat_g_t)0.0;
	ufloat_g_t ez1 = (ufloat_g_t)0.0;
	ufloat_g_t ex2 = (ufloat_g_t)0.0;
	ufloat_g_t ey2 = (ufloat_g_t)0.0;
	ufloat_g_t ez2 = (ufloat_g_t)0.0;
	ufloat_g_t ex3 = (ufloat_g_t)0.0;
	ufloat_g_t ey3 = (ufloat_g_t)0.0;
	ufloat_g_t ez3 = (ufloat_g_t)0.0;
	ufloat_g_t vxp = (ufloat_g_t)0.0;
	ufloat_g_t vyp = (ufloat_g_t)0.0;
	ufloat_g_t vzp = (ufloat_g_t)0.0;
	ufloat_g_t tmp = (ufloat_g_t)0.0;
	ufloat_g_t min_d = (ufloat_g_t)0.0;
	bool in_region = false;
	bool C1 = false;
	bool C2 __attribute__((unused)) = false;
	bool C3 __attribute__((unused)) = false;
	bool C4 __attribute__((unused)) = false;
	bool C5 __attribute__((unused)) = false;
	bool C6 __attribute__((unused)) = false;
	bool eligible = true;
	//int intersect_counter __attribute__((unused)) = 0;
	
	s_ID_cblock[threadIdx.x] = -1;
	s_D[threadIdx.x] = 0;
	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	__syncthreads();
	
	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && cblock_ID_ref[i_kap_b] == V_REF_ID_UNREFINED)
		{
			vxp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I_kap*dx_L + 0.5*dx_L;
			vyp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J_kap*dx_L + 0.5*dx_L;
			if (N_DIM==3)
				vzp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K_kap*dx_L + 0.5*dx_L;
			
			// For each face, check if the current cell is within the appropriate bounds. If at least
			// one condition is satisfied, exit the loop and make a note.
			s_D[threadIdx.x] = 0;
			int bin_id_x __attribute__((unused)) = (int)(vxp*G_BIN_DENSITY);
			int bin_id_y = (int)(vyp*G_BIN_DENSITY);
			int bin_id_z __attribute__((unused)) = (int)(vzp*G_BIN_DENSITY);
			if (N_DIM==2)
				global_bin_id = bin_id_y;
				//global_bin_id = bin_id_x + G_BIN_DENSITY*bin_id_y;
			else
				global_bin_id = bin_id_y + G_BIN_DENSITY*bin_id_z;
				//global_bin_id = bin_id_x + G_BIN_DENSITY*bin_id_y + G_BIN_DENSITY*G_BIN_DENSITY*bin_id_z;
						
			int n_f = binned_face_ids_n[global_bin_id];
			int N_f = 0;
			if (n_f > 0)
				N_f = binned_face_ids_N[global_bin_id];
			min_d = (ufloat_g_t)1.0;
			C6 = 0;
			for (int p = 0; p < n_f; p++)
			{
				int f_p = binned_face_ids[N_f+p];
				vx1 = geom_f_face_X[f_p + 0*n_faces_a];
				vy1 = geom_f_face_X[f_p + 1*n_faces_a];
				vx2 = geom_f_face_X[f_p + 3*n_faces_a];
				vy2 = geom_f_face_X[f_p + 4*n_faces_a];
				
				if (N_DIM==2)
				{
					nx = vy2-vy1;
					ny = vx1-vx2;
					tmp = Tsqrt(nx*nx + ny*ny);
					nx /= tmp;
					ny /= tmp;
					ex1 = vx2-vx1;
					ey1 = vy2-vy1;
					tmp = Tsqrt(ex1*ex1 + ey1*ey1);
					ex1 /= tmp;
					ey1 /= tmp;
					
					// Check adjacency.
					{
						tmp = (vx1-vxp)*nx+(vy1-vyp)*ny; // Now stores d.
						ex3 = vxp + tmp*nx; // Now stores xi_x.
						ey3 = vyp + tmp*ny; // Now stores xi_y.
						C1 = CheckPointInLine(ex3, ey3, vx1, vy1, vx2, vy2);
						if (C1) // Use the normal distance for the check.
						{
							if (Tabs(tmp) < min_d)
							{
								min_d = Tabs(tmp);
								if (tmp < 0)
									C6 = 0;
								else
									C6 = 1;
							}
						}
						else // Use the shortest of the two vertex-links for the check.
						{
							// tmp stores the length of the shortest link.
							// ez1 stores the angle between the link (with respect to the vertex) and the edge normal.
							GetShortestLinkVertex2D(vxp, vyp, vx1, vy1, vx2, vy2, ex1, ey1, nx, ny, tmp, ez1);
							if (tmp < min_d)
							{
								min_d = tmp;
								if (ez1 >= 0 && ez1 < (ufloat_g_t)M_PI*(ufloat_g_t)0.5)
									C6 = 1;
								else
									C6 = 0;
							}
						}
						if (min_d < R)
							in_region = true;
					}
				}
				else // N_DIM==3
				{
					vz1 = geom_f_face_X[f_p + 2*n_faces_a];
					vz2 = geom_f_face_X[f_p + 5*n_faces_a];
					vx3 = geom_f_face_X[f_p + 6*n_faces_a];
					vy3 = geom_f_face_X[f_p + 7*n_faces_a];
					vz3 = geom_f_face_X[f_p + 8*n_faces_a];

					ex1 = vx2-vx1;
					ey1 = vy2-vy1;
					ez1 = vz2-vz1;
					ex2 = vx3-vx1;
					ey2 = vy3-vy1;
					ez2 = vz3-vz1;
					Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
					tmp = Tsqrt(nx*nx + ny*ny + nz*nz);
					nx /= tmp;
					ny /= tmp;
					nz /= tmp;
					
					// Adjust vertices ever-so slightly to avoid race condition.
					ex1 = (ufloat_g_t)(0.3333333333333333)*(vx1+vx2+vx3);
					ey1 = (ufloat_g_t)(0.3333333333333333)*(vy1+vy2+vy3);
					ez1 = (ufloat_g_t)(0.3333333333333333)*(vz1+vz2+vz3);
					tmp = Tsqrt( (ex1-vx1)*(ex1-vx1) + (ey1-vy1)*(ey1-vy1) + (ez1-vz1)*(ez1-vz1) );
					vx1 = vx1 + (ex1-vx1)/tmp*(ufloat_g_t)1e-6;
					vy1 = vy1 + (ey1-vy1)/tmp*(ufloat_g_t)1e-6;
					vz1 = vz1 + (ez1-vz1)/tmp*(ufloat_g_t)1e-6;
					tmp = Tsqrt( (ex1-vx2)*(ex1-vx2) + (ey1-vy2)*(ey1-vy2) + (ez1-vz2)*(ez1-vz2) );
					vx2 = vx2 + (ex1-vx2)/tmp*(ufloat_g_t)1e-6;
					vy2 = vy2 + (ey1-vy2)/tmp*(ufloat_g_t)1e-6;
					vz2 = vz2 + (ez1-vz2)/tmp*(ufloat_g_t)1e-6;
					tmp = Tsqrt( (ex1-vx3)*(ex1-vx3) + (ey1-vy3)*(ey1-vy3) + (ez1-vz3)*(ez1-vz3) );
					vx3 = vx3 + (ex1-vx3)/tmp*(ufloat_g_t)1e-6;
					vy3 = vy3 + (ey1-vy3)/tmp*(ufloat_g_t)1e-6;
					vz3 = vz3 + (ez1-vz3)/tmp*(ufloat_g_t)1e-6;
					
					// Check adjacency.
					if (!in_region)
					{
						tmp = (vx1-vxp)*nx+(vy1-vyp)*ny+(vz1-vzp)*nz; // Now stores d.
						ex3 = vxp + tmp*nx; // Now stores xi_x.
						ey3 = vyp + tmp*ny; // Now stores xi_y.
						ez3 = vzp + tmp*nz; // Now stores xi_y.
						C1 = CheckPointInTriangle(ex3, ey3, ez3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2);
						if (C1) // Use the normal distance for the check.
						{
							if (Tabs(tmp) < min_d)
							{
								min_d = Tabs(tmp);
								if (tmp < 0)
									C6 = 0;
								else
									C6 = 1;
							}
						}
						else // Use the shortest of the two vertex-links for the check.
						{
							// ex2 stores the link length.
							// ey2 stores the angle between the link and the tri normal.
							// C1 stores if the link intersects the edge.
							C2 = false;
							
							// Edge 1.
							GetShortestLinkEdge3D(vxp, vyp, vzp, vx1, vy1, vz1, vx2, vy2, vz2, ex1, ey1, ez1, nx, ny, nz, ex2, ey2, ez2, C1);
							C2 = C2 or C1;
							if (C1 && ex2 < min_d)
							{
								min_d = ex2;
								if (ey2 >= 0 && ey2 < (ufloat_g_t)M_PI*(ufloat_g_t)0.5)
									C6 = 0;
								else
									C6 = 1;
							}
							
							// Edge 2.
							GetShortestLinkEdge3D(vxp, vyp, vzp, vx2, vy2, vz2, vx3, vy3, vz3, ex1, ey1, ez1, nx, ny, nz, ex2, ey2, ez2, C1);
							C2 = C2 or C1;
							if (C1 && ex2 < min_d)
							{
								min_d = ex2;
								if (ey2 >= 0 && ey2 < (ufloat_g_t)M_PI*(ufloat_g_t)0.5)
									C6 = 0;
								else
									C6 = 1;
							}
							
							// Edge 3.
							GetShortestLinkEdge3D(vxp, vyp, vzp, vx3, vy3, vz3, vx1, vy1, vz1, ex1, ey1, ez1, nx, ny, nz, ex2, ey2, ez2, C1);
							C2 = C2 or C1;
							if (C1 && ex2 < min_d)
							{
								min_d = ex2;
								if (ey2 >= 0 && ey2 < (ufloat_g_t)M_PI*(ufloat_g_t)0.5)
									C6 = 0;
								else
									C6 = 1;
							}
							
							
							// If none of the links intersect the edges, check the vertices.
							if (!C2)
							{
								GetShortedLinkVertex3D(vxp, vyp, vzp, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex2, ey2, ez2);
								if (ex2 < min_d)
								{
									min_d = ex2;
									if (ey2 >= 0 && ey2 < (ufloat_g_t)M_PI*(ufloat_g_t)0.5)
										C6 = 0;
									else
										C6 = 1;
								}
							}
						}
						if (min_d < R)
							in_region = true;
					}
				}
			}
			
			// If the cell was detected in the near-wall region, update its mask.
			if (in_region)
			{
				s_D[threadIdx.x] = 1;
				//cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = -1;
			}
			//if (C6)
			//	cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = -2;
			__syncthreads();
			
			// Consider refinement if not on the finest grid.
			if (L < MAX_LEVELS_WALL-1)
			{
				// Block reduction for maximum.
				for (int s=blockDim.x/2; s>0; s>>=1)
				{
					if (threadIdx.x < s)
					{
						s_D[threadIdx.x] = max( s_D[threadIdx.x],s_D[threadIdx.x + s] );
					}
					__syncthreads();
				}
				
				// Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
				for (int p = 0; p < N_Q_max; p++)
				{
					if (cblock_ID_nbr[i_kap_b + p*n_maxcblocks] == N_SKIPID)
						eligible = false;
				}
				
				// Mark for refinement.
				if (threadIdx.x == 0)
				{
					if (eligible && s_D[threadIdx.x] == 1)
						cblock_ID_ref[i_kap_b] = V_REF_ID_MARK_REFINE;
				}
			}
			
			// Reset some parameters.
			in_region = false;
			eligible = true;
			//intersect_counter = 0;
		}
	}
}

template <const ArgsPack *AP>
__global__
void Cu_PropagateMarks_S1
(
	int id_max_curr, int n_maxcblocks, 
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int M_BLOCK = AP->M_BLOCK;
	__shared__ int s_ID_nbr[M_BLOCK*(2*N_DIM+1)];
	int D = (2*N_DIM+1);
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Initialize shared memory.
	for (int p = 0; p < D; p++)
		s_ID_nbr[p + threadIdx.x*D] = -1;
	__syncthreads();
	
	// First, read neighbor Ids and place in shared memory. Arrange for contiguity.
	if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_MARK_REFINE)
	{
		
		for (int p = 0; p < D; p++)
			s_ID_nbr[p + threadIdx.x*D] = cblock_ID_nbr[kap + p*n_maxcblocks];
	}
	__syncthreads();
	
	// Replace neighbor Ids with their respective marks.
	for (int p = 0; p < D; p++)
	{
		int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
		if (i_p > -1 && cblock_ID_ref[i_p] == V_REF_ID_UNREFINED)
			cblock_ID_ref[i_p] = V_REF_ID_INDETERMINATE;
	}
}

template <const ArgsPack *AP>
__global__
void Cu_PropagateMarks_S2
(
	int id_max_curr, int n_maxcblocks, 
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	constexpr int N_Q_max = AP->N_Q_max;
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_INDETERMINATE)
	{
		// Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
		bool eligible = true;
		for (int p = 1; p < N_Q_max; p++)
		{
			if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
				eligible = false;
		}
		
		if (eligible)
			cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
	}
}





template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_ComputeRefCriteria_Geometry_Binned(int i_dev, int L)
{
	if (n_ids[i_dev][L] > 0)
	{
		ufloat_g_t R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(ufloat_g_t)L);
		ufloat_g_t R2 = R*R;
		int N_prop = (int)(R/(ufloat_g_t)(4.0*sqrt(2.0)*dxf_vec[L])) + 1;
		//if (!use_cpu)
		{
			Cu_ComputeRefCriteria_NearWall_Geometry_Binned<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L, MAX_LEVELS_WALL,
				c_cells_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_ID_mask[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
				geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], R, R2,
				geometry->c_binned_face_ids_n_v[i_dev], geometry->c_binned_face_ids_N_v[i_dev], geometry->c_binned_face_ids_v[i_dev], geometry->G_BIN_DENSITY, geometry->n_bins_v
			);
			for (int j = 0; j < N_prop; j++)
			{
				std::cout << "Propagation iteration " << j << "..." << std::endl;
				Cu_PropagateMarks_S1<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
					id_max[i_dev][L], n_maxcblocks,
					c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
				);
				Cu_PropagateMarks_S2<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
					id_max[i_dev][L], n_maxcblocks,
					c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
				);
			}
		}

	}
	
	return 0;
}
