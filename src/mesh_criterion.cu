/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

template <class T>
__device__
void Cu_CrossProduct(T a1, T a2, T a3, T b1, T b2, T b3, T &s1, T &s2, T &s3)
{
	s1 = a2*b3 - a3*b2;
	s2 = a3*b1 - a1*b3;
	s3 = a1*b2 - a2*b1;
}

__global__
void Cu_ComputeRefCriteria_NearWall_Cases
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dxb_L, int L,
	int *cblock_ID_ref, int *cblock_ID_onb, ufloat_t *cblock_f_X
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		// Evaluate only if current cell-block is not refined already.
		if (cblock_ID_ref[i_kap] == V_REF_ID_UNREFINED)
		{
			// Get the coordinates of the block.
			ufloat_t x_k_plus = cblock_f_X[i_kap + 0*n_maxcblocks] + N_Pf(0.5)*dxb_L;
			ufloat_t y_k_plus = cblock_f_X[i_kap + 1*n_maxcblocks] + N_Pf(0.5)*dxb_L;
#if (N_DIM==3)
			ufloat_t z_k_plus = cblock_f_X[i_kap + 2*n_maxcblocks] + N_Pf(0.5)*dxb_L;
#endif
			
			
			
#if (N_CASE==0 || N_CASE==2)
			// Loop over cavity walls and identify the closest one.
			// If this closest wall is within a certain threshhold, mark for refinement.
			ufloat_t dist_min = N_Pf(1.0);
			ufloat_t dist_tmp = N_Pf(1.0);
				// xM
			//dist_min = x_k_plus - N_Pf(0.0);
				// xP
			//dist_tmp = N_Pf(1.0) - x_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
#if (N_DIM==2)
				// yM
			//dist_tmp = y_k_plus - N_Pf(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yP
			//dist_tmp = N_Pf(1.0) - y_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
#else
				// zM
			//dist_tmp = z_k_plus - N_Pf(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// zP
			//dist_tmp = N_Pf(1.0) - z_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
#endif
			
			// Evaluate criterion based on dist_min.
			//    + '(cblock_ID_onb[i_kap] == 1)' only refines near boundary.
			//    + 'dist_min <= N_Pf(d_spec)/( (ufloat_t)(1<<L) )' refined by specified distance d_spec.
			if (cblock_ID_onb[i_kap] == 1)
			//if ( dist_min <= N_Pf(0.2)/( (ufloat_t)(1<<L) ) )
			//if (dist_min < dxb_L)
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
#endif
			
			
			
#if (N_DIM==2 && N_CASE==1)	
			ufloat_t D = N_Pf(1.0)/N_Pf(32.0);
			ufloat_t rad = N_Pf(2.5)*D/( (ufloat_t)(1<<L) ); // Old value was 1.5
			//if (x_k_plus >= N_Pf(0.3125)-N_Pf(0.5)*D - rad   &&   x_k_plus <= N_Pf(0.3125)+N_Pf(0.5)*D + rad   &&   y_k_plus >= N_Pf(0.5)*N_Pf(1.0)-N_Pf(0.5)*D - rad   &&   y_k_plus <= N_Pf(0.5)*N_Pf(1.0)+N_Pf(0.5)*D + rad)
			if (x_k_plus >= N_Pf(0.3125)-N_Pf(0.5)*D - rad  &&   y_k_plus >= N_Pf(0.5)*N_Pf(1.0)-N_Pf(0.5)*D - rad   &&   y_k_plus <= N_Pf(0.5)*N_Pf(1.0)+N_Pf(0.5)*D + rad)
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
			
			
			
			ufloat_t dist_min = N_Pf(1.0);
			ufloat_t dist_tmp = N_Pf(1.0);
				// xM
			//dist_min = x_k_plus - N_Pf(0.0);
				// xP
			dist_tmp = N_Pf(1.0) - x_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yM
			//dist_tmp = y_k_plus - N_Pf(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yP
			//dist_tmp = L_fy - y_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
			
			// Evaluate criterion based on dist_min. (old value 0.05).
			//if ( dist_min <= N_Pf(0.05)/( (ufloat_t)(1<<L) )  ||  cblock_ID_onb[i_kap]==1 )
			if (cblock_ID_onb[i_kap] == 1)
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
#endif	
			
			
			
			
			// DEBUG
			//if (L == 0 && x_k_plus > N_Pf(0.3) && x_k_plus < N_Pf(0.7) && y_k_plus > N_Pf(0.3) && y_k_plus < N_Pf(0.7) && z_k_plus > N_Pf(0.3) && z_k_plus < N_Pf(0.8))
			//	cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
			//if (L == 0 && x_k_plus > N_Pf(0.3) && x_k_plus < N_Pf(0.7) && y_k_plus > N_Pf(0.3) && y_k_plus <= N_Pf(0.85))
			//	cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
		}
	}
}





__global__
void Cu_ComputeRefCriteria_NearWall_Geometry
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dx_L, int L,
	int *cells_ID_mask, int *cblock_ID_ref, int *cblock_ID_onb, ufloat_t *cblock_f_X, int *cblock_ID_nbr,
	int n_faces, int n_faces_a, double *geom_f_face_X, double R, double R2
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
	double vx1 = 0.0;
	double vy1 = 0.0;
	double vz1 = 0.0;
	double vx2 = 0.0;
	double vy2 = 0.0;
	double vz2 = 0.0;
	double vx3 = 0.0;
	double vy3 = 0.0;
	double vz3 = 0.0;
	double nx = 0.0;
	double ny = 0.0;
	double nz = 0.0;
	double ex1 = 0.0;
	double ey1 = 0.0;
	double ez1 = 0.0;
	double ex2 = 0.0;
	double ey2 = 0.0;
	double ez2 = 0.0;
	double v_xp = 0.0;
	double v_yp = 0.0;
	double v_zp = 0.0;
	double tmp = 0.0;
	double tmp1 = 0.0;
	double tmp2 = 0.0;
	double tmp3 = 0.0;
// 	double ax = 0.0;
// 	double ay = 0.0;
// 	double az = 0.0;
// 	double bx = 0.0;
// 	double by = 0.0;
// 	double bz = 0.0;
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
				ay = v_yp;
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
// 				tmp = ( (vx1-v_xp)*(nx) + (vy1-v_yp)*(ny) + (vz1-v_zp)*(nz) ) / nx;
// 				tmp1 = v_xp + tmp;
// 				tmp2 = v_yp;
// 				tmp3 = v_zp;
// 				C1 = -(   (vx1-tmp1)*(vx2-vx1) + (vy1-tmp2)*(vy2-vy1)   )   > 0;
// 				C2 = (   (vx2-tmp1)*(vx2-vx1) + (vy2-tmp2)*(vy2-vy1)   ) > 0;
// 				C3 = ;
// 				if (tmp > 0 && C1 && C2 && C3)
// 					intersect_counter++;
#endif
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


/*
__global__
void Cu_ComputeRefCriteria_NearWall_Geometry
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dx_L, int L,
	int *cblock_ID_ref, int *cblock_ID_onb, ufloat_t *cblock_f_X, int *cblock_ID_nbr,
	int n_nodes_idev, int n_faces_idev, double *geom_f_node_X, int *geom_ID_face
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
	int f1 = -1;
	int f2 = -1;
	int f3 = -1;
	double vx1 = 0.0;
	double vy1 = 0.0;
	double vz1 = 0.0;
	double vx2 = 0.0;
	double vy2 = 0.0;
	double vz2 = 0.0;
	double vx3 = 0.0;
	double vy3 = 0.0;
	double vz3 = 0.0;
	double vxc = 0.0;
	double vyc = 0.0;
	double vzc = 0.0;
	double xi = 0.0;
	double yi = 0.0;
	double zi = 0.0;
	double nx = 0.0;
	double ny = 0.0;
	double nz = 0.0;
	bool marked_for_refinement;
	bool eligible;
	
	s_ID_cblock[threadIdx.x] = -1;
	s_D[threadIdx.x] = dx_L;
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
			// Reset mark.
			marked_for_refinement = false;
			
			// Reset eligibility.
			eligible = true;
			
			// Compute cell coordinates.
			vxc = (double)( cblock_f_X[i_kap_b+0*n_maxcblocks] + dx_L*(0.5 + I_kap) );
			vyc = (double)( cblock_f_X[i_kap_b+1*n_maxcblocks] + dx_L*(0.5 + J_kap) );
#if (N_DIM==3)
			vzx = (double)( cblock_f_X[i_kap_b+2*n_maxcblocks] + dx_L*(0.5 + K_kap) );
#endif
			
			// Loop over all faces (or edges, in 2D) and identify the ones where streamed DDFs would bounce off a boundary.
			for (int m = 0; m < n_faces_idev; m++)
			{
				f1 = geom_ID_face[m+0*n_faces_idev];
				f2 = geom_ID_face[m+1*n_faces_idev];
				f3 = geom_ID_face[m+2*n_faces_idev];
				if (f1 > -1)
				{
					vx1 = geom_f_node_X[f1+0*n_nodes_idev];
					vy1 = geom_f_node_X[f1+1*n_nodes_idev];
					vz1 = geom_f_node_X[f1+2*n_nodes_idev];
					vx2 = geom_f_node_X[f2+0*n_nodes_idev];
					vy2 = geom_f_node_X[f2+1*n_nodes_idev];
					vz2 = geom_f_node_X[f2+2*n_nodes_idev];
					vx3 = geom_f_node_X[f3+0*n_nodes_idev];
					vy3 = geom_f_node_X[f3+1*n_nodes_idev];
					vz3 = geom_f_node_X[f3+2*n_nodes_idev];
					
					s_D[threadIdx.x] = 0;
#if (N_DIM==2) // Distance from point to line.
					nx = (vx2-vx1)/sqrt( (vy2-vy1)*(vy2-vy1) + (vx2-vx1)*(vx2-vx1) );
					ny = (vy2-vy1)/sqrt( (vy2-vy1)*(vy2-vy1) + (vx2-vx1)*(vx2-vx1) );
					xi = vx1 - (nx*(vx1-vxc)+ny*(vy1-vyc))*nx;
					yi = vy1 - (nx*(vx1-vxc)+ny*(vy1-vyc))*ny;
					
					// if (sqrt((xi-vxc)*(xi-vxc)+(yi-vyc)*(yi-vyc)) < 4*dx_L && abs(xi-vx1)+abs(vx2-xi) <= abs(vx2-vx1) && abs(yi-vy1)+abs(vy2-yi) <= abs(vy2-vy1))
					if (sqrt((xi-vxc)*(xi-vxc)+(yi-vyc)*(yi-vyc)) < 8*dx_L && xi > min(vx1,vx2)-8*dx_L && xi < max(vx1,vx2)+8*dx_L && yi > min(vy1,vy2)-8*dx_L && yi < max(vy1,vy2)+8*dx_L)
						s_D[threadIdx.x] = 1;
					
					//tmp_1 = abs( (vy2-vy1)*vxc - (vx2-vx1)*vyc + vx2*vy1 - vy2*vx1 );
					//tmp_2 = sqrt( (vy2-vy1)*(vy2-vy1) + (vx2-vx1)*(vx2-vx1) );
					//if (tmp_2 > 0)
					//	s_D[threadIdx.x] = tmp_1 / tmp_2;
#else // Distance from point to plane.
					//tmp_1 = abs( (vy2-vy1)*vxc - (vx2-vx1)*vyc + vx2*vy1 - vy2*vx1 );
					//tmp_2 = sqrt( (vy2-vy1)*(vy2-vy1) + (vx2-vx1)*(vx2-vx1) );
					//s_D[threadIdx.x] = tmp1 / tmp2;
#endif
					__syncthreads();
					
					// Block reduction for minimum.
					for (int s=blockDim.x/2; s>0; s>>=1)
					{
						if (threadIdx.x < s)
						{
							s_D[threadIdx.x] = max( s_D[threadIdx.x],s_D[threadIdx.x + s] );
						}
						__syncthreads();
					}
					
					// If minimum is smaller than dx_L/2, at least on cell's DDFs will bounce off of the face.
					if (threadIdx.x==0)
					{
						if (s_D[0] == 1)
							marked_for_refinement = true;
					}
					__syncthreads();
				}
			}
			
			// Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
			for (int p = 0; p < N_Q_max; p++)
			{
				if (cblock_ID_nbr[i_kap_b + p*n_maxcblocks] == N_SKIPID)
					eligible = false;
			}
			
			// If at least one suitable face was identified, mark for refinement.
			if (eligible && threadIdx.x==0 && marked_for_refinement)
				cblock_ID_ref[i_kap_b] = V_REF_ID_MARK_REFINE;
			__syncthreads();
		}
	}
}
*/



__global__
void Cu_ComputeRefCriteria_Uniform
(
	int n_ids_idev_L, int *id_set_idev_L, int n_maxcblocks, ufloat_t dxb_L, int L,
	int *cblock_ID_ref
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < n_ids_idev_L)
	{
		int i_kap = id_set_idev_L[kap];
		
		// Evaluate only if current cell-block is not refined already.
		if (cblock_ID_ref[i_kap] == V_REF_ID_UNREFINED)
			cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
	}
}



int Mesh::M_ComputeRefCriteria(int i_dev, int L, int var)
{
	if (var == V_MESH_REF_NW_CASES) // Near-wall distance criterion.
	{
		if (n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_NearWall_Cases<<<(M_BLOCK+n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L]*(Nbx*Nqx), L,
				c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_f_X[i_dev]
			);
		}
	}
	if (var == V_MESH_REF_NW_GEOMETRY) // Complex geometry.
	{
		double R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(double)L);
		double R2 = R*R;
		if (n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_NearWall_Geometry<<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], L,
				c_cells_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_onb[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
				geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], R, R2
			);
		}
	}
	if (var == V_MESH_REF_UNIFORM) // Refine the whole level.
	{
		if (n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_Uniform<<<(M_BLOCK+n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
				n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L]*Nbx, L,
				c_cblock_ID_ref[i_dev]
			);
		}
	}
	if (var == V_MESH_REF_SOLUTION) // Refine the whole level.
	{	
		if (n_ids[i_dev][L] > 0)
			solver->S_ComputeRefCriteria(i_dev, L, var);
	}
	
	return 0;
}
