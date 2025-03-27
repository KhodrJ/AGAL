#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
void ComputeRefCriteria_NearWall_Geometry_Naive
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dx_L, int L,
	int *cells_ID_mask, int *cblock_ID_ref, int *cblock_ID_onb, ufloat_t *cblock_f_X, int *cblock_ID_nbr,
	int n_faces, int n_faces_a, ufloat_g_t *geom_f_face_X, ufloat_g_t R, ufloat_g_t R2
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int N_Q_max = AP->N_Q_max;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	constexpr int M_CBLOCK = AP->M_CBLOCK;
	int s_D[M_TBLOCK];
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
	bool C6 __attribute__((unused)) = false;
	bool eligible = true;
	int intersect_counter __attribute__((unused)) = 0;
	
	#pragma omp parallel for
	for (int k = 0; k < n_ids_idev_L; k++)
	{
		i_kap_b = id_set_idev_L[k];
		//std::cout << "Doing block no. " << i_kap_b << std::endl;
		if (i_kap_b > -1 && cblock_ID_ref[i_kap_b] == V_REF_ID_UNREFINED)
		{
			for (int t = 0; t < M_TBLOCK; t++)
			{
				int I_kap = t % 4;
				int J_kap = (t / 4) % 4;
				int K_kap = (t / 4) / 4;
				v_xp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I_kap*dx_L + 0.5*dx_L;
				v_yp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J_kap*dx_L + 0.5*dx_L;
				v_zp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K_kap*dx_L + 0.5*dx_L;
				
				// For each face, check if the current cell is within the appropriate bounds. If at least
				// one condition is satisfied, exit the loop and make a note.
				s_D[t] = 0;
				for (int p = 0; p < n_faces; p++)
				{
					vx1 = geom_f_face_X[p + 0*n_faces_a];
					vy1 = geom_f_face_X[p + 1*n_faces_a];
					vz1 = geom_f_face_X[p + 2*n_faces_a];
					vx2 = geom_f_face_X[p + 3*n_faces_a];
					vy2 = geom_f_face_X[p + 4*n_faces_a];
					vz2 = geom_f_face_X[p + 5*n_faces_a];
					vx3 = geom_f_face_X[p + 6*n_faces_a];
					vy3 = geom_f_face_X[p + 7*n_faces_a];
					vz3 = geom_f_face_X[p + 8*n_faces_a];
					
if (N_DIM==2)
{
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
					
}
else // N_DIM==3
{
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
}
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









template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_ComputeRefCriteria_NearWall_Geometry_Naive
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dx_L, int L,
	int *cells_ID_mask, int *cblock_ID_ref, int *cblock_ID_onb, ufloat_t *cblock_f_X, int *cblock_ID_nbr,
	int n_faces, int n_faces_a, ufloat_g_t *geom_f_face_X, ufloat_g_t R, ufloat_g_t R2
)
{
	constexpr int N_DIM = AP->N_DIM;
	constexpr int N_Q_max = AP->N_Q_max;
	constexpr int M_TBLOCK = AP->M_TBLOCK;
	constexpr int M_CBLOCK = AP->M_CBLOCK;
	constexpr int M_LBLOCK = AP->M_LBLOCK;
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ int s_D[M_TBLOCK];
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
	int I_kap = threadIdx.x % 4;
	int J_kap = (threadIdx.x / 4) % 4;
	int K_kap = (threadIdx.x / 4) / 4;
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
	bool C6 __attribute__((unused)) = false;
	bool eligible = true;
	int intersect_counter __attribute__((unused)) = 0;
	
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
if (N_DIM==3)
			v_zp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K_kap*dx_L + 0.5*dx_L;

			
			// For each face, check if the current cell is within the appropriate bounds. If at least
			// one condition is satisfied, exit the loop and make a note.
			s_D[threadIdx.x] = 0;
			for (int p = 0; p < n_faces; p++)
			{
				vx1 = geom_f_face_X[p + 0*n_faces_a];
				vy1 = geom_f_face_X[p + 1*n_faces_a];
				vz1 = geom_f_face_X[p + 2*n_faces_a];
				vx2 = geom_f_face_X[p + 3*n_faces_a];
				vy2 = geom_f_face_X[p + 4*n_faces_a];
				vz2 = geom_f_face_X[p + 5*n_faces_a];
				vx3 = geom_f_face_X[p + 6*n_faces_a];
				vy3 = geom_f_face_X[p + 7*n_faces_a];
				vz3 = geom_f_face_X[p + 8*n_faces_a];
				
if (N_DIM==2)
{
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
				
}
else // N_DIM==3
{
				ex1 = vx2-vx1;
				ey1 = vy2-vy1;
				ez1 = vz2-vz1;
				ex2 = vx3-vx1;
				ey2 = vy3-vy1;
				ez2 = vz3-vz1;
				nx = ey1*ez2 - ez1*ey2;
				ny = ez1*ex2 - ex1*ez2;
				nz = ex1*ey2 - ey1*ex2;
// 				Cu_CrossProduct(vx2-vx1,vy2-vy1,vz2-vz1,vx3-vx1,vy3-vy1,vz3-vz1,   nx,ny,nz);
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
}
			}
			
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









template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_ComputeRefCriteria_Geometry_Naive(int i_dev, int L)
{
	ufloat_g_t R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(ufloat_g_t)L);
	ufloat_g_t R2 = R*R;
	if (n_ids[i_dev][L] > 0)
	{
		if (!use_cpu)
		{
			Cu_ComputeRefCriteria_NearWall_Geometry_Naive<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L,
				c_cells_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
				geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], R, R2
			);
		}
		else
		{
			std::cout << "USING CPU VERSION" << std::endl;
			ComputeRefCriteria_NearWall_Geometry_Naive<ufloat_t,ufloat_g_t,AP>(
				n_ids[i_dev][L], &id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L,
				cells_ID_mask[i_dev], cblock_ID_ref[i_dev], cblock_ID_onb[i_dev], cblock_f_X[i_dev], cblock_ID_nbr[i_dev],
				geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->geom_f_face_X[i_dev], R, R2
			);
		}
	}
	
	return 0;
}
