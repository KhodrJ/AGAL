#include "mesh.h"


void ComputeRefCriteria_NearWall_Geometry_Binned
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dx_L, int L,
	int *cells_ID_mask, int *cblock_ID_ref, int *cblock_ID_onb, ufloat_t *cblock_f_X, int *cblock_ID_nbr,
	int n_faces, int n_faces_a, ufloat_g_t *geom_f_face_X, ufloat_g_t R, ufloat_g_t R2,
	int *binned_face_ids_n, int *binned_face_ids_N, int *binned_face_ids, int G_BIN_DENSITY, int n_bins
)
{
	int s_D[M_TBLOCK];
	ufloat_g_t vx1 = 0.0;
	ufloat_g_t vy1 = 0.0;
	ufloat_g_t vz1 = 0.0;
	ufloat_g_t vx2 = 0.0;
	ufloat_g_t vy2 = 0.0;
	ufloat_g_t vz2 = 0.0;
	ufloat_g_t vx3 = 0.0;
	ufloat_g_t vy3 = 0.0;
	ufloat_g_t vz3 = 0.0;
	ufloat_g_t nx = 0.0;
	ufloat_g_t ny = 0.0;
	ufloat_g_t nz = 0.0;
	ufloat_g_t ex1 = 0.0;
	ufloat_g_t ey1 = 0.0;
	ufloat_g_t ez1 = 0.0;
	ufloat_g_t ex2 = 0.0;
	ufloat_g_t ey2 = 0.0;
	ufloat_g_t ez2 = 0.0;
	ufloat_g_t v_xp = 0.0;
	ufloat_g_t v_yp = 0.0;
	ufloat_g_t v_zp = 0.0;
	ufloat_g_t tmp = 0.0;
	ufloat_g_t tmp1 = 0.0;
	ufloat_g_t tmp2 = 0.0;
	ufloat_g_t tmp3 = 0.0;
	bool in_region = false;
	bool C1 = false;
	bool C2 = false;
	bool C3 = false;
	bool C4 = false;
	bool C5 = false;
	bool C6 = false;
	bool eligible = true;
	int intersect_counter = 0;
	
	#pragma omp parallel for
	for (int k = 0; k < n_ids_idev_L; k++)
	{
		int i_kap_b = id_set_idev_L[k];
		std::cout << "Doing block no. " << i_kap_b << std::endl;
		if (i_kap_b > -1 && cblock_ID_ref[i_kap_b] == V_REF_ID_UNREFINED)
		{
			for (int t = 0; t < M_TBLOCK; t++)
			{
				int I_kap = t % Nbx;
				int J_kap = (t / Nbx) % Nbx;
#if (N_DIM==3)
				int K_kap = (t / Nbx) / Nbx;
#endif


				v_xp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I_kap*dx_L + 0.5*dx_L;
				v_yp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J_kap*dx_L + 0.5*dx_L;
#if (N_DIM==3)
				v_zp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K_kap*dx_L + 0.5*dx_L;
#endif
				
				// For each face, check if the current cell is within the appropriate bounds. If at least
				// one condition is satisfied, exit the loop and make a note.
				s_D[t] = 0;
#if (N_DIM==2)
				int global_bin_id = ((int)(v_xp*G_BIN_DENSITY)) + G_BIN_DENSITY*( (int)(v_yp*G_BIN_DENSITY) );
#else
				int global_bin_id = ((int)(v_xp*G_BIN_DENSITY)) + G_BIN_DENSITY*( (int)(v_yp*G_BIN_DENSITY) ) + G_BIN_DENSITY*G_BIN_DENSITY*( (int)(v_zp*G_BIN_DENSITY) );
#endif
				int n_f = binned_face_ids_n[global_bin_id];
				int N_f = 0;
				if (n_f > 0)
					N_f = binned_face_ids_N[global_bin_id];
				for (int p = 0; p < n_f; p++)
				{
					int f_p = binned_face_ids[N_f+p];
					vx1 = geom_f_face_X[f_p + 0*n_faces_a];
					vy1 = geom_f_face_X[f_p + 1*n_faces_a];
					vz1 = geom_f_face_X[f_p + 2*n_faces_a];
					vx2 = geom_f_face_X[f_p + 3*n_faces_a];
					vy2 = geom_f_face_X[f_p + 4*n_faces_a];
					vz2 = geom_f_face_X[f_p + 5*n_faces_a];
					vx3 = geom_f_face_X[f_p + 6*n_faces_a];
					vy3 = geom_f_face_X[f_p + 7*n_faces_a];
					vz3 = geom_f_face_X[f_p + 8*n_faces_a];
					
#if (N_DIM==2)
					nx = vy2-vy1;
					ny = vx1-vx2;
					tmp = sqrt(nx*nx + ny*ny);
					nx /= tmp;
					ny /= tmp;
					
					// Checking in circles.
					if ( (v_xp-vx1)*(v_xp-vx1) + (v_yp-vy1)*(v_yp-vy1) < R2 )
						in_region = true;
					if ( (v_xp-vx2)*(v_xp-vx2) + (v_yp-vy2)*(v_yp-vy2) < R2 )
						in_region = true;
					
					// Checking in rectangle.
					C1 = -(   (vx1+nx*R-v_xp)*(vx2-vx1) + (vy1+ny*R-v_yp)*(vy2-vy1)   )   > 0;
					C2 = (   (vx2+nx*R-v_xp)*(vx2-vx1) + (vy2+ny*R-v_yp)*(vy2-vy1)   ) > 0;
					C3 = -(   (vx1-nx*R-v_xp)*(nx) + (vy1-ny*R-v_yp)*(ny)   ) > 0;
					C4 = (   (vx1+nx*R-v_xp)*(nx) + (vy1+ny*R-v_yp)*(ny)   ) > 0;
					if (C1 && C2 && C3 && C4)
						in_region = true;
					
					// Check if the cell lies in the solid region.
					tmp = ( (vx1-v_xp)*(nx) + (vy1-v_yp)*(ny) ) / nx;
					tmp1 = v_xp + tmp;
					tmp2 = v_yp;
					C1 = -(   (vx1-tmp1)*(vx2-vx1) + (vy1-tmp2)*(vy2-vy1)   )   > 0;
					C2 = (   (vx2-tmp1)*(vx2-vx1) + (vy2-tmp2)*(vy2-vy1)   ) > 0;
					if (tmp > 0 && C1 && C2)
						intersect_counter++;
					
#else // N_DIM==3
					ex1 = vx2-vx1;
					ey1 = vy2-vy1;
					ez1 = vz2-vz1;
					ex2 = vx3-vx1;
					ey2 = vy3-vy1;
					ez2 = vz3-vz1;
					nx = ey1*ez2 - ez1*ey2;
					ny = ez1*ex2 - ex1*ez2;
					nz = ex1*ey2 - ey1*ex2;
					tmp = sqrt(nx*nx + ny*ny + nz*nz);
					nx /= tmp;
					ny /= tmp;
					nz /= tmp;
					
					// Check in spheres.
					if ( (v_xp-vx1)*(v_xp-vx1) + (v_yp-vy1)*(v_yp-vy1) + (v_zp-vz1)*(v_zp-vz1) < R2 )
						in_region = true;
					if ( (v_xp-vx2)*(v_xp-vx2) + (v_yp-vy2)*(v_yp-vy2) + (v_zp-vz2)*(v_zp-vz2) < R2 )
						in_region = true;
					if ( (v_xp-vx3)*(v_xp-vx3) + (v_yp-vy3)*(v_yp-vy3) + (v_zp-vz3)*(v_zp-vz3) < R2 )
						in_region = true;
					
					// Check in cylinders.
					ex1 = vx2-vx1;
					ey1 = vy2-vy1;
					ez1 = vz2-vz1;
					ex2 = vx1-v_xp;
					ey2 = vy1-v_yp;
					ez2 = vz1-v_zp;
					tmp1 = ey1*ez2 - ez1*ey2;
					tmp2 = ez1*ex2 - ex1*ez2;
					tmp3 = ex1*ey2 - ey1*ex2;
					tmp = tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;
					tmp = tmp / (ex1*ex1 + ey1*ey1 + ez1*ez1);
					C1 = -(   (vx1-v_xp)*(ex1) + (vy1-v_yp)*(ey1) + (vz1-v_zp)*(ez1)  )   > 0;
					C2 = (   (vx2-v_xp)*(ex1) + (vy2-v_yp)*(ey1) + (vz2-v_zp)*(ez1)  ) > 0;
					if (tmp < R2 && C1 && C2)
						in_region = true;
					ex1 = vx3-vx2;
					ey1 = vy3-vy2;
					ez1 = vz3-vz2;
					ex2 = vx2-v_xp;
					ey2 = vy2-v_yp;
					ez2 = vz2-v_zp;
					tmp1 = ey1*ez2 - ez1*ey2;
					tmp2 = ez1*ex2 - ex1*ez2;
					tmp3 = ex1*ey2 - ey1*ex2;
					tmp = tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;
					tmp = tmp / (ex1*ex1 + ey1*ey1 + ez1*ez1);
					C1 = -(   (vx2-v_xp)*(ex1) + (vy2-v_yp)*(ey1) + (vz2-v_zp)*(ez1)  )   > 0;
					C2 = (   (vx3-v_xp)*(ex1) + (vy3-v_yp)*(ey1) + (vz3-v_zp)*(ez1)  ) > 0;
					if (tmp < R2 && C1 && C2)
						in_region = true;
					ex1 = vx1-vx3;
					ey1 = vy1-vy3;
					ez1 = vz1-vz3;
					ex2 = vx3-v_xp;
					ey2 = vy3-v_yp;
					ez2 = vz3-v_zp;
					tmp1 = ey1*ez2 - ez1*ey2;
					tmp2 = ez1*ex2 - ex1*ez2;
					tmp3 = ex1*ey2 - ey1*ex2;
					tmp = tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;
					tmp = tmp / (ex1*ex1 + ey1*ey1 + ez1*ez1);
					C1 = -(   (vx3-v_xp)*(ex1) + (vy3-v_yp)*(ey1) + (vz3-v_zp)*(ez1)  )   > 0;
					C2 = (   (vx1-v_xp)*(ex1) + (vy1-v_yp)*(ey1) + (vz1-v_zp)*(ez1)  ) > 0;
					if (tmp < R2 && C1 && C2)
						in_region = true;
					
					// Check in trianglular prism.
					C1 = -(   (vx1-nx*R-v_xp)*(nx) + (vy1-ny*R-v_yp)*(ny) + (vz1-nz*R-v_zp)*(nz)  )   > 0;
					C2 = (   (vx1+nx*R-v_xp)*(nx) + (vy1+ny*R-v_yp)*(ny) + (vz1+nz*R-v_zp)*(nz)  ) > 0;
					tmp1 = -( (vy2-vy1)*nz - (vz2-vz1)*ny );
					tmp2 = -( (vz2-vz1)*nx - (vx2-vx1)*nz );
					tmp3 = -( (vx2-vx1)*ny - (vy2-vy1)*nx );
					C3 = -(   (vx1-v_xp)*(tmp1) + (vy1-v_yp)*(tmp2) + (vz1-v_zp)*(tmp3)  )   > 0;
					tmp1 = -( (vy3-vy2)*nz - (vz3-vz2)*ny );
					tmp2 = -( (vz3-vz2)*nx - (vx3-vx2)*nz );
					tmp3 = -( (vx3-vx2)*ny - (vy3-vy2)*nx );
					C4 = -(   (vx2-v_xp)*(tmp1) + (vy2-v_yp)*(tmp2) + (vz2-v_zp)*(tmp3)  )   > 0;
					tmp1 = -( (vy1-vy3)*nz - (vz1-vz3)*ny );
					tmp2 = -( (vz1-vz3)*nx - (vx1-vx3)*nz );
					tmp3 = -( (vx1-vx3)*ny - (vy1-vy3)*nx );
					C5 = -(   (vx3-v_xp)*(tmp1) + (vy3-v_yp)*(tmp2) + (vz3-v_zp)*(tmp3)  )   > 0;
					if (C1 && C2 && C3 && C4 && C5)
						in_region = true;
					
					// Check if the cell lies in the solid region.
					tmp = ( (vx1-(v_xp))*(nx) + (vy1-v_yp)*(ny) + (vz1-v_zp)*(nz) ) / nx;
					tmp1 = -( (vy2-vy1)*nz - (vz2-vz1)*ny );
					tmp2 = -( (vz2-vz1)*nx - (vx2-vx1)*nz );
					tmp3 = -( (vx2-vx1)*ny - (vy2-vy1)*nx );
					C1 = -(   (vx1-(v_xp + tmp))*(tmp1) + (vy1-v_yp)*(tmp2) + (vz1-v_zp)*(tmp3)  )   > 0;
					tmp1 = -( (vy3-vy2)*nz - (vz3-vz2)*ny );
					tmp2 = -( (vz3-vz2)*nx - (vx3-vx2)*nz );
					tmp3 = -( (vx3-vx2)*ny - (vy3-vy2)*nx );
					C2 = -(   (vx2-(v_xp + tmp))*(tmp1) + (vy2-v_yp)*(tmp2) + (vz2-v_zp)*(tmp3)  )   > 0;
					tmp1 = -( (vy1-vy3)*nz - (vz1-vz3)*ny );
					tmp2 = -( (vz1-vz3)*nx - (vx1-vx3)*nz );
					tmp3 = -( (vx1-vx3)*ny - (vy1-vy3)*nx );
					C3 = -(   (vx3-(v_xp + tmp))*(tmp1) + (vy3-v_yp)*(tmp2) + (vz3-v_zp)*(tmp3)  )   > 0;
					if (tmp > 0 && C1 && C2 && C3)
						intersect_counter++;
#endif
				}
				
				// If the cell was detected in the near-wall region, update its mask.
				if (in_region)
				{
					s_D[t] = 1;
					cells_ID_mask[i_kap_b*M_CBLOCK + t] = -1;
				}
				if (intersect_counter%2 == 1)
					cells_ID_mask[i_kap_b*M_CBLOCK + t] = -2;
				
				// Reset some parameters.
				in_region = false;
				intersect_counter = 0;
			}
			
			// Block reduction for maximum.
			s_D[0] = *std::max_element(s_D, s_D+M_TBLOCK);
			
			// Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
			for (int p = 0; p < N_Q_max; p++)
			{
				if (cblock_ID_nbr[i_kap_b + p*n_maxcblocks] == N_SKIPID)
					eligible = false;
			}
			
			// Mark for refinement.
			if (eligible && s_D[0] == 1)
				cblock_ID_ref[i_kap_b] = V_REF_ID_MARK_REFINE;
			eligible = true;
		}
	}
}












































__global__
void Cu_ComputeRefCriteria_NearWall_Geometry_Binned
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dx_L, int L,
	int *cells_ID_mask, int *cblock_ID_ref, int *cblock_ID_onb, ufloat_t *cblock_f_X, int *cblock_ID_nbr,
	int n_faces, int n_faces_a, ufloat_g_t *geom_f_face_X, ufloat_g_t R, ufloat_g_t R2,
	int *binned_face_ids_n, int *binned_face_ids_N, int *binned_face_ids, int G_BIN_DENSITY, int n_bins, int rad
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_D[M_TBLOCK];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
#if (N_DIM==3)
	int K_kap = (threadIdx.x / Nbx) / Nbx;
#endif
	int i_kap_b = -1;
	ufloat_g_t vx1 = 0.0;
	ufloat_g_t vy1 = 0.0;
	ufloat_g_t vz1 = 0.0;
	ufloat_g_t vx2 = 0.0;
	ufloat_g_t vy2 = 0.0;
	ufloat_g_t vz2 = 0.0;
	ufloat_g_t vx3 = 0.0;
	ufloat_g_t vy3 = 0.0;
	ufloat_g_t vz3 = 0.0;
	ufloat_g_t nx = 0.0;
	ufloat_g_t ny = 0.0;
	ufloat_g_t nz = 0.0;
	ufloat_g_t ex1 = 0.0;
	ufloat_g_t ey1 = 0.0;
	ufloat_g_t ez1 = 0.0;
	ufloat_g_t ex2 = 0.0;
	ufloat_g_t ey2 = 0.0;
	ufloat_g_t ez2 = 0.0;
	ufloat_g_t v_xp = 0.0;
	ufloat_g_t v_yp = 0.0;
	ufloat_g_t v_zp = 0.0;
	ufloat_g_t tmp = 0.0;
	ufloat_g_t tmp1 = 0.0;
	ufloat_g_t tmp2 = 0.0;
	ufloat_g_t tmp3 = 0.0;
	bool in_region = false;
	bool C1 = false;
	bool C2 = false;
	bool C3 = false;
	bool C4 = false;
	bool C5 = false;
	bool C6 = false;
	bool eligible = true;
	int intersect_counter = 0;
	
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
			v_xp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I_kap*dx_L + 0.5*dx_L;
			v_yp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J_kap*dx_L + 0.5*dx_L;
#if (N_DIM==3)
			v_zp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K_kap*dx_L + 0.5*dx_L;
#endif
			
			// For each face, check if the current cell is within the appropriate bounds. If at least
			// one condition is satisfied, exit the loop and make a note.
			s_D[threadIdx.x] = 0;
// 			int rad_k = 0;
// #if (N_DIM==3)
// 			for (rad_k = -rad; rad_k < rad+1; rad_k++)
// #endif
// 			{
// 				for (int rad_j = -rad; rad_j < rad+1; rad_j++)
// 				{
// 					for (int rad_i = -rad; rad_i < rad+1; rad_i++)
// 					{
						int bin_id_x = (int)(v_xp*G_BIN_DENSITY);
						int bin_id_y = (int)(v_yp*G_BIN_DENSITY);
						int bin_id_z = (int)(v_zp*G_BIN_DENSITY);
// 						if (bin_id_x < 0)                bin_id_x = 0;
// 						if (bin_id_x >= G_BIN_DENSITY)   bin_id_x = G_BIN_DENSITY-1;
// 						if (bin_id_y < 0)                bin_id_y = 0;
// 						if (bin_id_y >= G_BIN_DENSITY)   bin_id_y = G_BIN_DENSITY-1;
#if (N_DIM==2)
						//int global_bin_id = ( (int)(v_xp*G_BIN_DENSITY)+rad_i ) + G_BIN_DENSITY*( (int)(v_yp*G_BIN_DENSITY) +rad_j );
						int global_bin_id = bin_id_x + G_BIN_DENSITY*bin_id_y;
#else
						if (bin_id_z < 0)                bin_id_z = 0;
						if (bin_id_z >= G_BIN_DENSITY)   bin_id_z = G_BIN_DENSITY-1;
						//int global_bin_id = ( (int)(v_xp*G_BIN_DENSITY)+rad_i ) + G_BIN_DENSITY*( (int)(v_yp*G_BIN_DENSITY)+rad_j ) + G_BIN_DENSITY*G_BIN_DENSITY*( (int)(v_zp*G_BIN_DENSITY)+rad_k );
						int global_bin_id = bin_id_x + G_BIN_DENSITY*bin_id_y + G_BIN_DENSITY*G_BIN_DENSITY*bin_id_z;
#endif
						
						int n_f = binned_face_ids_n[global_bin_id];
						int N_f = 0;
						if (n_f > 0)
							N_f = binned_face_ids_N[global_bin_id];
						for (int p = 0; p < n_f; p++)
						{
							int f_p = binned_face_ids[N_f+p];
							vx1 = geom_f_face_X[f_p + 0*n_faces_a];
							vy1 = geom_f_face_X[f_p + 1*n_faces_a];
							vz1 = geom_f_face_X[f_p + 2*n_faces_a];
							vx2 = geom_f_face_X[f_p + 3*n_faces_a];
							vy2 = geom_f_face_X[f_p + 4*n_faces_a];
							vz2 = geom_f_face_X[f_p + 5*n_faces_a];
							vx3 = geom_f_face_X[f_p + 6*n_faces_a];
							vy3 = geom_f_face_X[f_p + 7*n_faces_a];
							vz3 = geom_f_face_X[f_p + 8*n_faces_a];
							
#if (N_DIM==2)
							nx = vy2-vy1;
							ny = vx1-vx2;
							tmp = sqrt(nx*nx + ny*ny);
							nx /= tmp;
							ny /= tmp;
							
							// Checking in circles.
							if ( (v_xp-vx1)*(v_xp-vx1) + (v_yp-vy1)*(v_yp-vy1) < R2 )
								in_region = true;
							if ( (v_xp-vx2)*(v_xp-vx2) + (v_yp-vy2)*(v_yp-vy2) < R2 )
								in_region = true;
							
							// Checking in rectangle.
							C1 = -(   (vx1+nx*R-v_xp)*(vx2-vx1) + (vy1+ny*R-v_yp)*(vy2-vy1)   )   > 0;
							C2 = (   (vx2+nx*R-v_xp)*(vx2-vx1) + (vy2+ny*R-v_yp)*(vy2-vy1)   ) > 0;
							C3 = -(   (vx1-nx*R-v_xp)*(nx) + (vy1-ny*R-v_yp)*(ny)   ) > 0;
							C4 = (   (vx1+nx*R-v_xp)*(nx) + (vy1+ny*R-v_yp)*(ny)   ) > 0;
							if (C1 && C2 && C3 && C4)
								in_region = true;
							
							// Check if the cell lies in the solid region.
							tmp = ( (vx1-v_xp)*(nx) + (vy1-v_yp)*(ny) ) / nx;
							tmp1 = v_xp + tmp;
							tmp2 = v_yp;
							C1 = -(   (vx1-tmp1)*(vx2-vx1) + (vy1-tmp2)*(vy2-vy1)   )   > 0;
							C2 = (   (vx2-tmp1)*(vx2-vx1) + (vy2-tmp2)*(vy2-vy1)   ) > 0;
							if (tmp > 0 && C1 && C2)
								intersect_counter++;
							
#else // N_DIM==3
							ex1 = vx2-vx1;
							ey1 = vy2-vy1;
							ez1 = vz2-vz1;
							ex2 = vx3-vx1;
							ey2 = vy3-vy1;
							ez2 = vz3-vz1;
							nx = ey1*ez2 - ez1*ey2;
							ny = ez1*ex2 - ex1*ez2;
							nz = ex1*ey2 - ey1*ex2;
							tmp = sqrt(nx*nx + ny*ny + nz*nz);
							nx /= tmp;
							ny /= tmp;
							nz /= tmp;
							
							// Check in spheres.
							if ( (v_xp-vx1)*(v_xp-vx1) + (v_yp-vy1)*(v_yp-vy1) + (v_zp-vz1)*(v_zp-vz1) < R2 )
								in_region = true;
							if ( (v_xp-vx2)*(v_xp-vx2) + (v_yp-vy2)*(v_yp-vy2) + (v_zp-vz2)*(v_zp-vz2) < R2 )
								in_region = true;
							if ( (v_xp-vx3)*(v_xp-vx3) + (v_yp-vy3)*(v_yp-vy3) + (v_zp-vz3)*(v_zp-vz3) < R2 )
								in_region = true;
							
							// Check in cylinders.
							ex1 = vx2-vx1;
							ey1 = vy2-vy1;
							ez1 = vz2-vz1;
							ex2 = vx1-v_xp;
							ey2 = vy1-v_yp;
							ez2 = vz1-v_zp;
							tmp1 = ey1*ez2 - ez1*ey2;
							tmp2 = ez1*ex2 - ex1*ez2;
							tmp3 = ex1*ey2 - ey1*ex2;
							tmp = tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;
							tmp = tmp / (ex1*ex1 + ey1*ey1 + ez1*ez1);
							C1 = -(   (vx1-v_xp)*(ex1) + (vy1-v_yp)*(ey1) + (vz1-v_zp)*(ez1)  )   > 0;
							C2 = (   (vx2-v_xp)*(ex1) + (vy2-v_yp)*(ey1) + (vz2-v_zp)*(ez1)  ) > 0;
							if (tmp < R2 && C1 && C2)
								in_region = true;
							ex1 = vx3-vx2;
							ey1 = vy3-vy2;
							ez1 = vz3-vz2;
							ex2 = vx2-v_xp;
							ey2 = vy2-v_yp;
							ez2 = vz2-v_zp;
							tmp1 = ey1*ez2 - ez1*ey2;
							tmp2 = ez1*ex2 - ex1*ez2;
							tmp3 = ex1*ey2 - ey1*ex2;
							tmp = tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;
							tmp = tmp / (ex1*ex1 + ey1*ey1 + ez1*ez1);
							C1 = -(   (vx2-v_xp)*(ex1) + (vy2-v_yp)*(ey1) + (vz2-v_zp)*(ez1)  )   > 0;
							C2 = (   (vx3-v_xp)*(ex1) + (vy3-v_yp)*(ey1) + (vz3-v_zp)*(ez1)  ) > 0;
							if (tmp < R2 && C1 && C2)
								in_region = true;
							ex1 = vx1-vx3;
							ey1 = vy1-vy3;
							ez1 = vz1-vz3;
							ex2 = vx3-v_xp;
							ey2 = vy3-v_yp;
							ez2 = vz3-v_zp;
							tmp1 = ey1*ez2 - ez1*ey2;
							tmp2 = ez1*ex2 - ex1*ez2;
							tmp3 = ex1*ey2 - ey1*ex2;
							tmp = tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;
							tmp = tmp / (ex1*ex1 + ey1*ey1 + ez1*ez1);
							C1 = -(   (vx3-v_xp)*(ex1) + (vy3-v_yp)*(ey1) + (vz3-v_zp)*(ez1)  )   > 0;
							C2 = (   (vx1-v_xp)*(ex1) + (vy1-v_yp)*(ey1) + (vz1-v_zp)*(ez1)  ) > 0;
							if (tmp < R2 && C1 && C2)
								in_region = true;
							
							// Check in trianglular prism.
							C1 = -(   (vx1-nx*R-v_xp)*(nx) + (vy1-ny*R-v_yp)*(ny) + (vz1-nz*R-v_zp)*(nz)  )   > 0;
							C2 = (   (vx1+nx*R-v_xp)*(nx) + (vy1+ny*R-v_yp)*(ny) + (vz1+nz*R-v_zp)*(nz)  ) > 0;
							tmp1 = -( (vy2-vy1)*nz - (vz2-vz1)*ny );
							tmp2 = -( (vz2-vz1)*nx - (vx2-vx1)*nz );
							tmp3 = -( (vx2-vx1)*ny - (vy2-vy1)*nx );
							C3 = -(   (vx1-v_xp)*(tmp1) + (vy1-v_yp)*(tmp2) + (vz1-v_zp)*(tmp3)  )   > 0;
							tmp1 = -( (vy3-vy2)*nz - (vz3-vz2)*ny );
							tmp2 = -( (vz3-vz2)*nx - (vx3-vx2)*nz );
							tmp3 = -( (vx3-vx2)*ny - (vy3-vy2)*nx );
							C4 = -(   (vx2-v_xp)*(tmp1) + (vy2-v_yp)*(tmp2) + (vz2-v_zp)*(tmp3)  )   > 0;
							tmp1 = -( (vy1-vy3)*nz - (vz1-vz3)*ny );
							tmp2 = -( (vz1-vz3)*nx - (vx1-vx3)*nz );
							tmp3 = -( (vx1-vx3)*ny - (vy1-vy3)*nx );
							C5 = -(   (vx3-v_xp)*(tmp1) + (vy3-v_yp)*(tmp2) + (vz3-v_zp)*(tmp3)  )   > 0;
							if (C1 && C2 && C3 && C4 && C5)
								in_region = true;
							
							// Check if the cell lies in the solid region.
							tmp = ( (vx1-(v_xp))*(nx) + (vy1-v_yp)*(ny) + (vz1-v_zp)*(nz) ) / nx;
							tmp1 = -( (vy2-vy1)*nz - (vz2-vz1)*ny );
							tmp2 = -( (vz2-vz1)*nx - (vx2-vx1)*nz );
							tmp3 = -( (vx2-vx1)*ny - (vy2-vy1)*nx );
							C1 = -(   (vx1-(v_xp + tmp))*(tmp1) + (vy1-v_yp)*(tmp2) + (vz1-v_zp)*(tmp3)  )   > 0;
							tmp1 = -( (vy3-vy2)*nz - (vz3-vz2)*ny );
							tmp2 = -( (vz3-vz2)*nx - (vx3-vx2)*nz );
							tmp3 = -( (vx3-vx2)*ny - (vy3-vy2)*nx );
							C2 = -(   (vx2-(v_xp + tmp))*(tmp1) + (vy2-v_yp)*(tmp2) + (vz2-v_zp)*(tmp3)  )   > 0;
							tmp1 = -( (vy1-vy3)*nz - (vz1-vz3)*ny );
							tmp2 = -( (vz1-vz3)*nx - (vx1-vx3)*nz );
							tmp3 = -( (vx1-vx3)*ny - (vy1-vy3)*nx );
							C3 = -(   (vx3-(v_xp + tmp))*(tmp1) + (vy3-v_yp)*(tmp2) + (vz3-v_zp)*(tmp3)  )   > 0;
							if (tmp > 0 && C1 && C2 && C3)
								intersect_counter++;
#endif
						}
			//		}
			//	}
			//}
			
			// If the cell was detected in the near-wall region, update its mask.
			if (in_region)
			{
				s_D[threadIdx.x] = 1;
				cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = -1;
			}
			if (intersect_counter%2 == 1)
				cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = -2;
			__syncthreads();
			
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
			
			// Reset some parameters.
			in_region = false;
			eligible = true;
			intersect_counter = 0;
		}
	}
}

__global__
void Cu_PropagateMarks_S1
(
	int id_max_curr, int n_maxcblocks, 
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	__shared__ int s_ID_nbr[M_BLOCK*9];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Initialize shared memory.
	for (int p = 0; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	
#if (N_DIM==3)
	for (int k = 0; k < 3; k++)
	{
#else
	int k = 0;
#endif
	
	// First, read neighbor Ids and place in shared memory. Arrange for contiguity.
	if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_MARK_REFINE)
	{
		
		for (int p = 1; p < 9; p++)
			s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
	}
	__syncthreads();
	
	// Replace neighbor Ids with their respective marks.
	for (int p = 1; p < 9; p++)
	{
		int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
		if (i_p > -1)
			s_ID_nbr[threadIdx.x + p*M_BLOCK] = cblock_ID_ref[i_p];
	}
	__syncthreads();
	
	// Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
	if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_MARK_REFINE)
	{
		for (int p = 1; p < 9; p++)
		{
			if ((k*9+p) > 0 && s_ID_nbr[p + threadIdx.x*9] == V_REF_ID_UNREFINED)
				cblock_ID_nbr_child[kap + (k*9+p)*n_maxcblocks] = V_REF_ID_INDETERMINATE;
		}
	}
	
#if (N_DIM==3)
	__syncthreads();
	
	for (int p = 1; p < 9; p++)
		s_ID_nbr[p + threadIdx.x*9] = -1;
	__syncthreads();
	}
#endif
}

__global__
void Cu_PropagateMarks_S2
(
	int id_max_curr, int n_maxcblocks, 
	int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_INDETERMINATE)
		cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
}









int Mesh::M_ComputeRefCriteria_Geometry_Binned(int i_dev, int L)
{
	if (n_ids[i_dev][L] > 0)
	{
		ufloat_g_t R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(ufloat_g_t)L);
		ufloat_g_t R2 = R*R;
		int N_prop = (int)(R/(ufloat_g_t)(4.0*dxf_vec[L]));
		Cu_ComputeRefCriteria_NearWall_Geometry_Binned<<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
			n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L,
			c_cells_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
			geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], R, R2,
			geometry->c_binned_face_ids_n[i_dev], geometry->c_binned_face_ids_N[i_dev], geometry->c_binned_face_ids[i_dev], geometry->G_BIN_DENSITY, geometry->n_bins, 0
		);
		for (int j = 0; j < N_prop; j++)
		{
			std::cout << "Propagation iteration " << j << "..." << std::endl;
			Cu_PropagateMarks_S1<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
			Cu_PropagateMarks_S2<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK>>>(
				id_max[i_dev][MAX_LEVELS], n_maxcblocks,
				c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
			);
		}
		
		/*
		ComputeRefCriteria_NearWall_Geometry_Binned(
			n_ids[i_dev][L], &id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L,
			cells_ID_mask[i_dev], cblock_ID_ref[i_dev], cblock_ID_onb[i_dev], cblock_f_X[i_dev], cblock_ID_nbr[i_dev],
			geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->geom_f_face_X[i_dev], R, R2,
			geometry->binned_face_ids_n[i_dev], geometry->binned_face_ids_N[i_dev], geometry->binned_face_ids[i_dev], geometry->G_BIN_DENSITY, geometry->n_bins
		);
		*/
	}
	
	return 0;
}
