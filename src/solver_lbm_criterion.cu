/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"
#include "solver.h"

__global__
void Cu_ComputeRefCriteria_V1
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
			
			
			
#if (N_CASE==0)
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
			ufloat_t rad = N_Pf(1.5)*D/( (ufloat_t)(1<<L) );
			if (x_k_plus >= N_Pf(0.3125)-N_Pf(0.5)*D - rad   &&   x_k_plus <= N_Pf(0.3125)+N_Pf(0.5)*D + rad   &&   y_k_plus >= N_Pf(0.5)*L_fy-N_Pf(0.5)*D - rad   &&   y_k_plus <= N_Pf(0.5)*L_fy+N_Pf(0.5)*D + rad)
				cblock_ID_ref[i_kap] = V_REF_ID_MARK_REFINE;
			
			
			
			ufloat_t dist_min = N_Pf(1.0);
			ufloat_t dist_tmp = N_Pf(1.0);
				// xM
			//dist_min = x_k_plus - N_Pf(0.0);
				// xP
			dist_tmp = L_c - x_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yM
			//dist_tmp = y_k_plus - N_Pf(0.0); if (dist_min > dist_tmp) dist_min = dist_tmp;
				// yP
			//dist_tmp = L_fy - y_k_plus; if (dist_min > dist_tmp) dist_min = dist_tmp;
			
			// Evaluate criterion based on dist_min.
			if ( dist_min <= N_Pf(0.05)/( (ufloat_t)(1<<L) ) )
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
void Cu_ComputeRefCriteria_V1_All
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

__global__
void Cu_ComputeRefCriteria_V2
(
	int id_max_curr, int n_maxcells, int n_maxcblocks, ufloat_t dx_L,
	int *cblock_ID_ref, int *cblock_level, int *cblock_ID_nbr,
	ufloat_t *cells_f_F, int *cells_ID_mask,
	int N_REFINE_START, int N_REFINE_INC, int MAX_LEVELS_INTERIOR
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_Ui[M_CBLOCK];
#if (N_DIM==3)
	__shared__ ufloat_t s_Ui2[M_CBLOCK];
#endif
	__shared__ ufloat_t s_W[M_CBLOCK];
	__shared__ ufloat_t s_Wmax[M_CBLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
#if (N_DIM==3)
	int K_kap = (threadIdx.x / Nbx) / Nbx;
#endif
	
	// DDFs and macroscopic properties.
	ufloat_t f_i = N_Pf(0.0);
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
#if (N_DIM==3)
	ufloat_t w_kap = N_Pf(0.0);
#endif

	// Intermediate vorticity variables.
#if (N_DIM==3)
	ufloat_t vortX = N_Pf(0.0);
	ufloat_t vortY = N_Pf(0.0);
#endif
	ufloat_t vortZ = N_Pf(0.0);
	
	bool eligible = true;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	s_Wmax[threadIdx.x] = -1;
	if (kap < id_max_curr)
	{
		//int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		if (i_kap_b > -1 && cblock_ID_ref[i_kap_b] != V_REF_ID_INACTIVE)
		{
			// Reset variables.
			u_kap = N_Pf(0.0);
			v_kap = N_Pf(0.0);
#if (N_DIM==3)
			w_kap = N_Pf(0.0);
#endif
			
			// Compute local vorticity magnitudes and place in shared memory.
			//
			//
			//
#if (N_Q==9)
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; 
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; 
#elif (N_Q==19)
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
#else // (N_Q==27)
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
			f_i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
#endif
			
#if (N_DIM==3)
			vortX = N_Pf(0.0);
			vortY = N_Pf(0.0);
#endif
			vortZ = N_Pf(0.0);
#if (N_DIM==2)
			// X
			s_Ui[threadIdx.x] = v_kap;
			__syncthreads();
			if (I_kap < Nbx-1)
				vortZ += s_Ui[(I_kap+1)+Nbx*(J_kap)] - s_Ui[threadIdx.x];
			else
				vortZ += s_Ui[threadIdx.x] - s_Ui[(I_kap-1)+Nbx*(J_kap)];
			
			// Y
			s_Ui[threadIdx.x] = u_kap;
			__syncthreads();
			if (J_kap < Nbx-1)
				vortZ -= s_Ui[(I_kap)+Nbx*(J_kap+1)] - s_Ui[threadIdx.x];
			else
				vortZ -= s_Ui[threadIdx.x] - s_Ui[(I_kap)+Nbx*(J_kap-1)];
			
			vortZ /= dx_L;
			s_W[threadIdx.x] = floor(log2( sqrt(vortZ*vortZ) ));
			__syncthreads();
#else
			// X
			s_Ui[threadIdx.x] = v_kap;
			s_Ui2[threadIdx.x] = w_kap;
			__syncthreads();
			if (I_kap < Nbx-1)
			{
				vortY -= s_Ui2[(I_kap+1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap)] - s_Ui2[threadIdx.x];
				vortZ += s_Ui[(I_kap+1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap)] - s_Ui[threadIdx.x];
			}
			else
			{
				vortY -= s_Ui2[threadIdx.x] - s_Ui2[(I_kap-1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap)];
				vortZ += s_Ui[threadIdx.x] - s_Ui[(I_kap-1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap)];
			}
			
			// Y
			s_Ui[threadIdx.x] = u_kap;
			s_Ui2[threadIdx.x] = w_kap;
			__syncthreads();
			if (J_kap < Nbx-1)
			{
				vortX += s_Ui2[(I_kap)+Nbx*(J_kap+1)+Nbx*Nbx*(K_kap)] - s_Ui2[threadIdx.x];
				vortZ -= s_Ui[(I_kap)+Nbx*(J_kap+1)+Nbx*Nbx*(K_kap)] - s_Ui[threadIdx.x];
			}
			else
			{
				vortX += s_Ui2[threadIdx.x] - s_Ui2[(I_kap)+Nbx*(J_kap-1)+Nbx*Nbx*(K_kap)];
				vortZ -= s_Ui[threadIdx.x] - s_Ui[(I_kap)+Nbx*(J_kap-1)+Nbx*Nbx*(K_kap)];
			}
			
			// Z
			s_Ui[threadIdx.x] = u_kap;
			s_Ui2[threadIdx.x] = v_kap;
			__syncthreads();
			if (K_kap < Nbx-1)
			{
				vortX -= s_Ui2[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap+1)] - s_Ui2[threadIdx.x];
				vortY += s_Ui[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap+1)] - s_Ui[threadIdx.x];
			}
			else
			{
				vortX -= s_Ui2[threadIdx.x] - s_Ui2[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap-1)];
				vortY += s_Ui[threadIdx.x] - s_Ui[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap-1)];
			}
			
			vortX /= dx_L;
			vortY /= dx_L;
			vortZ /= dx_L;
			s_W[threadIdx.x] = floor(log2( sqrt(vortX*vortX + vortY*vortY + vortZ*vortZ) ));
			__syncthreads();
#endif
			//
			//
			//
			
			// Set vorticity to zero if the cell is a ghost cell (it has wrong values).
			if (cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] == 2)
				s_W[threadIdx.x] = N_Pf(0.0);
			
			// Block reduction for maximum.
			for (int s=blockDim.x/2; s>0; s>>=1)
			{
				if (threadIdx.x < s)
				{
					s_W[threadIdx.x] = max( s_W[threadIdx.x],s_W[threadIdx.x + s] );
				}
				__syncthreads();
			}
			
			// Store maximum.
			if (threadIdx.x == 0)
				s_Wmax[k] = s_W[0];
			__syncthreads();
			
			//cells_f_W[i_kap_b*M_CBLOCK + threadIdx.x] = s_W[0];
			__syncthreads();
		}
	}
	__syncthreads();
	
	// Evaluate criterion.
	if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
	{
		// If vorticity is very large, cap at 1.0 to indicate maximum needed refinement.
		if (s_Wmax[threadIdx.x] > N_Pf(1.0))
			s_Wmax[threadIdx.x] = N_Pf(1.0);
		
		int ref_kap = cblock_ID_ref[kap];
		int level_kap = cblock_level[kap];
		
#if (N_DIM==2)
		int L_desired = MAX_LEVELS_INTERIOR-1;
		for (int p = 1; p <= MAX_LEVELS_INTERIOR-1; p++)
		{
			if (s_Wmax[threadIdx.x] < N_REFINE_START-N_REFINE_INC*p)
				L_desired = (MAX_LEVELS_INTERIOR-1)-p;
		}
#else
		int L_desired = MAX_LEVELS_INTERIOR-1;
		for (int p = 1; p <= MAX_LEVELS_INTERIOR-1; p++)
		{
			if (s_Wmax[threadIdx.x] < N_REFINE_START-N_REFINE_INC*p)
				L_desired = (MAX_LEVELS_INTERIOR-1)-p;
		}
#endif
		
		// Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
		for (int p = 0; p < N_Q_max; p++)
		{
			if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
				eligible = false;
		}
		
		// If cell-block is unrefined but desired level is higher than current, mark for refinement.
		if (eligible && level_kap != MAX_LEVELS_INTERIOR-1 && ref_kap == V_REF_ID_UNREFINED && L_desired > level_kap)
			cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
		
		// If cell-block is refined (leaf) but desired level is less than current, mark for coarsening.
		if (ref_kap == V_REF_ID_REFINED && L_desired < level_kap+1)
			cblock_ID_ref[kap] = V_REF_ID_MARK_COARSEN;
	}
}

int Solver_LBM::S_ComputeRefCriteria(int i_dev, int L, int var)
{
	if (var == 0)
	{
		if (mesh->n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_V1<<<(M_BLOCK+mesh->n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,mesh->streams[i_dev]>>>(
				mesh->n_ids[i_dev][L], &mesh->c_id_set[i_dev][L * mesh->n_maxcblocks], mesh->n_maxcblocks, mesh->dxf_vec[L]*Nbx, L,
				mesh->c_cblock_ID_ref[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_f_X[i_dev]
			);
		}
	}
	if (var == 1)
	{
		Cu_ComputeRefCriteria_V2<<<(M_CBLOCK+mesh->id_max[i_dev][MAX_LEVELS]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->id_max[i_dev][MAX_LEVELS], mesh->n_maxcells, mesh->n_maxcblocks, mesh->dxf_vec[L],
			mesh->c_cblock_ID_ref[i_dev], mesh->c_cblock_level[i_dev], mesh->c_cblock_ID_nbr[i_dev],
			mesh->c_cells_f_F[i_dev], mesh->c_cells_ID_mask[i_dev],
			mesh->N_REFINE_START, mesh->N_REFINE_INC, mesh->MAX_LEVELS_INTERIOR
		);
	}
	if (var == 2)
	{
		if (mesh->n_ids[i_dev][L] > 0)
		{
			Cu_ComputeRefCriteria_V1_All<<<(M_BLOCK+mesh->n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,mesh->streams[i_dev]>>>(
				mesh->n_ids[i_dev][L], &mesh->c_id_set[i_dev][L * mesh->n_maxcblocks], mesh->n_maxcblocks, mesh->dxf_vec[L]*Nbx, L,
				mesh->c_cblock_ID_ref[i_dev]
			);
		}
	}
	
	return 0;
}
