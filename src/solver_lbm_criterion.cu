/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"
#include "solver.h"


__global__
void Cu_ComputeRefCriteria
(
	int id_max_curr, int n_maxcells, int n_maxcblocks, ufloat_t dx_L,
	int *cblock_ID_ref, int *cblock_level, int *cblock_ID_nbr,
	ufloat_t *cells_f_F, int *cells_ID_mask,
	int S_COLLISION, int S_CRITERION, ufloat_t N_REFINE_START, ufloat_t N_REFINE_INC, ufloat_t N_REFINE_MAX, int MAX_LEVELS_INTERIOR
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
#if (N_DIM==2)
	__shared__ ufloat_t s_u[(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_v[(Nbx+2)*(Nbx+2)];
#else
	__shared__ ufloat_t s_u[(Nbx+2)*(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_v[(Nbx+2)*(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_w[(Nbx+2)*(Nbx+2)*(Nbx+2)];
#endif
	__shared__ ufloat_t s_W[M_TBLOCK];
	__shared__ ufloat_t s_Wmax[M_TBLOCK];
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
	ufloat_t uX = N_Pf(0.0);
	ufloat_t uY = N_Pf(0.0);
	ufloat_t uZ = N_Pf(0.0);
	ufloat_t vX = N_Pf(0.0);
	ufloat_t vY = N_Pf(0.0);
	ufloat_t vZ = N_Pf(0.0);
	ufloat_t wX = N_Pf(0.0);
	ufloat_t wY = N_Pf(0.0);
	ufloat_t wZ = N_Pf(0.0);
	ufloat_t tmp = N_Pf(0.0);
	bool eligible = true;
	
	// Keep in mind that each ID represents a block, not just a cell.
	s_ID_cblock[threadIdx.x] = -1;
	s_Wmax[threadIdx.x] = -1000;
	if (kap < id_max_curr)
	{
		//int i_kap = id_set_idev_L[kap];
		
		s_ID_cblock[threadIdx.x] = kap;
	}
	__syncthreads();
	
	// Now we loop over all cell-blocks and operate on the cells.
	for (int k = 0; k < M_TBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		if (i_kap_b > -1 && cblock_ID_ref[i_kap_b] != V_REF_ID_INACTIVE)
		{
			for (int i_Q = 0; i_Q < N_QUADS; i_Q++)
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
				if (S_COLLISION==0)
				{
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; 
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; 
				}
				else
				{
					rho_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells];
					u_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells] / rho_kap;
					v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells] / rho_kap;
				}
#elif (N_Q==19)
				if (S_COLLISION==0)
				{
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 10*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 9*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 12*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 11*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 14*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 13*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 16*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 15*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 18*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 17*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
				}
				else
				{
					rho_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells];
					u_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells] / rho_kap;
					v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells] / rho_kap;
					v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells] / rho_kap;
				}
#else // (N_Q==27)
				if (S_COLLISION==0)
				{
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 10*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 9*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 12*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 11*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 14*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 13*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 16*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 15*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 18*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 17*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 20*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 19*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 22*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 21*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 24*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 23*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 26*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
					f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 25*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
				}
				else
				{
					rho_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells];
					u_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells] / rho_kap;
					v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells] / rho_kap;
					v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells] / rho_kap;
				}
#endif
				
				
				
#if (N_DIM==2)
				s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)] = u_kap;
				s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)] = v_kap;
				__syncthreads();
				if (I_kap==0)
				{
					s_u[(0)+(Nbx+2)*(J_kap+1)] = N_Pf(4.0)*s_u[(1)+(Nbx+2)*(J_kap+1)] - N_Pf(6.0)*s_u[(2)+(Nbx+2)*(J_kap+1)] + N_Pf(4.0)*s_u[(3)+(Nbx+2)*(J_kap+1)] - N_Pf(1.0)*s_u[(4)+(Nbx+2)*(J_kap+1)];
					s_u[(5)+(Nbx+2)*(J_kap+1)] = N_Pf(4.0)*s_u[(4)+(Nbx+2)*(J_kap+1)] - N_Pf(6.0)*s_u[(3)+(Nbx+2)*(J_kap+1)] + N_Pf(4.0)*s_u[(2)+(Nbx+2)*(J_kap+1)] - N_Pf(1.0)*s_u[(1)+(Nbx+2)*(J_kap+1)];
					s_v[(0)+(Nbx+2)*(J_kap+1)] = N_Pf(4.0)*s_v[(1)+(Nbx+2)*(J_kap+1)] - N_Pf(6.0)*s_v[(2)+(Nbx+2)*(J_kap+1)] + N_Pf(4.0)*s_v[(3)+(Nbx+2)*(J_kap+1)] - N_Pf(1.0)*s_v[(4)+(Nbx+2)*(J_kap+1)];
					s_v[(5)+(Nbx+2)*(J_kap+1)] = N_Pf(4.0)*s_v[(4)+(Nbx+2)*(J_kap+1)] - N_Pf(6.0)*s_v[(3)+(Nbx+2)*(J_kap+1)] + N_Pf(4.0)*s_v[(2)+(Nbx+2)*(J_kap+1)] - N_Pf(1.0)*s_v[(1)+(Nbx+2)*(J_kap+1)];
				}
				if (J_kap==0)
				{
					s_u[(I_kap+1)+(Nbx+2)*(0)] = N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(1)] - N_Pf(6.0)*s_u[(I_kap+1)+(Nbx+2)*(2)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(3)] - N_Pf(1.0)*s_u[(I_kap+1)+(Nbx+2)*(4)];
					s_u[(I_kap+1)+(Nbx+2)*(5)] = N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(4)] - N_Pf(6.0)*s_u[(I_kap+1)+(Nbx+2)*(3)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(2)] - N_Pf(1.0)*s_u[(I_kap+1)+(Nbx+2)*(1)];
					s_v[(I_kap+1)+(Nbx+2)*(0)] = N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(1)] - N_Pf(6.0)*s_v[(I_kap+1)+(Nbx+2)*(2)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(3)] - N_Pf(1.0)*s_v[(I_kap+1)+(Nbx+2)*(4)];
					s_v[(I_kap+1)+(Nbx+2)*(5)] = N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(4)] - N_Pf(6.0)*s_v[(I_kap+1)+(Nbx+2)*(3)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(2)] - N_Pf(1.0)*s_v[(I_kap+1)+(Nbx+2)*(1)];
				}
				__syncthreads();
				
				uX = (s_u[(I_kap+1 +1)+(Nbx+2)*(J_kap+1)] - s_u[(I_kap+1 -1)+(Nbx+2)*(J_kap+1)])/(2.0*dx_L);
				uY = (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1 +1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1 -1)])/(2.0*dx_L);
				uZ = N_Pf(0.0);
				vX = (s_v[(I_kap+1 +1)+(Nbx+2)*(J_kap+1)] - s_v[(I_kap+1 -1)+(Nbx+2)*(J_kap+1)])/(2.0*dx_L);
				vY = (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1 +1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1 -1)])/(2.0*dx_L);
				vZ = N_Pf(0.0);
				wX = N_Pf(0.0);
				wY = N_Pf(0.0);
				wZ = N_Pf(0.0);
#else
				s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = u_kap;
				s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = v_kap;
				s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = w_kap;
				__syncthreads();
				if (I_kap==0)
				{
					s_u[(0)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_u[(1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_u[(2)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_u[(3)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_u[(4)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_u[(5)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_u[(4)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_u[(3)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_u[(2)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_u[(1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					
					s_v[(0)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_v[(1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_v[(2)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_v[(3)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_v[(4)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_v[(5)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_v[(4)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_v[(3)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_v[(2)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_v[(1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_w[(0)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_w[(1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_w[(2)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_w[(3)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_w[(4)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_w[(5)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_w[(4)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_w[(3)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_w[(2)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_w[(1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				}
				if (J_kap==0)
				{
					s_u[(I_kap+1)+(Nbx+2)*(0)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_u[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_u[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_u[(I_kap+1)+(Nbx+2)*(5)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_u[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_u[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_v[(I_kap+1)+(Nbx+2)*(0)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_v[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_v[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_v[(I_kap+1)+(Nbx+2)*(5)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_v[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_v[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_w[(I_kap+1)+(Nbx+2)*(0)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_w[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_w[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
					s_w[(I_kap+1)+(Nbx+2)*(5)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(6.0)*s_w[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - N_Pf(1.0)*s_w[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				}
				if (K_kap==0)
				{
					s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(0)] = N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(1)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(2)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(3)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(4)];
					s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(5)] = N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(4)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(3)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(2)] + N_Pf(4.0)*s_u[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(1)];
					s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(0)] = N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(1)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(2)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(3)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(4)];
					s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(5)] = N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(4)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(3)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(2)] + N_Pf(4.0)*s_v[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(1)];
					s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(0)] = N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(1)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(2)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(3)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(4)];
					s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(5)] = N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(4)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(3)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(2)] + N_Pf(4.0)*s_w[(I_kap+1)+(Nbx+2)*(J_kap)+(Nbx+2)*(Nbx+2)*(1)];
				}
				__syncthreads();
				
				uX = (s_u[(I_kap+1 +1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1 -1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])/(2*dx_L);
				uY = (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1 +1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1 -1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])/(2*dx_L);
				uZ = (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1 +1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1 -1)])/(2*dx_L);
				vX = (s_v[(I_kap+1 +1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1 -1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])/(2*dx_L);
				vY = (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1 +1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1 -1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])/(2*dx_L);
				vZ = (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1 +1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1 -1)])/(2*dx_L);
				wX = (s_w[(I_kap+1 +1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1 -1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])/(2*dx_L);
				wY = (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1 +1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1 -1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])/(2*dx_L);
				wZ = (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1 +1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1 -1)])/(2*dx_L);
#endif
				__syncthreads();
				
				
				
				s_W[threadIdx.x] = 0;
				if (S_CRITERION == 0)
					s_W[threadIdx.x] = floor(log2( sqrt((wY-vZ)*(wY-vZ) + (uZ-wX)*(uZ-wX) + (vX-uY)*(vX-uY)) ));
				if (S_CRITERION == 1)
					s_W[threadIdx.x] = floor(log2( abs((uX*vY+vY*wZ+wZ*uX)-(uY*vX+vZ*wY+uZ*wX)) ));
				//
				//
				//
				
				// Set vorticity to zero if the cell is a ghost cell (it has wrong values).
				if (cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] == 2)
					s_W[threadIdx.x] = N_Pf(0.0);
				__syncthreads();
				
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
				{
					//s_Wmax[k] = s_W[0];
					if (s_W[0] > s_Wmax[k])
						s_Wmax[k] = s_W[0];
				}
				__syncthreads();
			}
		}
	}
	__syncthreads();
	
	// Evaluate criterion.
	if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
	{
		// If vorticity is very large, cap at 1.0 to indicate maximum needed refinement.
		if (s_Wmax[threadIdx.x] > N_REFINE_MAX)
			s_Wmax[threadIdx.x] = N_REFINE_MAX;
		
		int ref_kap = cblock_ID_ref[kap];
		int level_kap = cblock_level[kap];
		int L_desired = MAX_LEVELS_INTERIOR-1;
		for (int p = 1; p <= MAX_LEVELS_INTERIOR-1; p++)
		{
			if (s_Wmax[threadIdx.x] < N_REFINE_START-N_REFINE_INC*p)
				L_desired = (MAX_LEVELS_INTERIOR-1)-p;
		}
		
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
	// Solution-based criterion. Only one type implemented for current solver.
	{
		Cu_ComputeRefCriteria<<<(M_TBLOCK+mesh->id_max[i_dev][MAX_LEVELS]-1)/M_TBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(
			mesh->id_max[i_dev][MAX_LEVELS], mesh->n_maxcells, mesh->n_maxcblocks, mesh->dxf_vec[L],
			mesh->c_cblock_ID_ref[i_dev], mesh->c_cblock_level[i_dev], mesh->c_cblock_ID_nbr[i_dev],
			mesh->c_cells_f_F[i_dev], mesh->c_cells_ID_mask[i_dev],
			S_COLLISION, S_CRITERION, mesh->N_REFINE_START, mesh->N_REFINE_INC, mesh->N_REFINE_MAX, mesh->MAX_LEVELS_INTERIOR
		);
	}
	
	return 0;
}
