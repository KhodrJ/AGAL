#include "solver.h"

#if (N_Q==19)

__global__
void Cu_Collide_d3q19
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_L, ufloat_t tau_L, ufloat_t tau_ratio,
	int *cblock_ID_onb, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int n_maxcblocks,
	int *cells_ID_mask, ufloat_t *cells_f_F
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
#if (B_TYPE==1||S_LES==1)
	__shared__ ufloat_t s_u[(Nbx+2)*(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_v[(Nbx+2)*(Nbx+2)*(Nbx+2)];
	__shared__ ufloat_t s_w[(Nbx+2)*(Nbx+2)*(Nbx+2)];
#endif
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int block_on_boundary = 0;
	int i_kap = -1;
	int i_kap_bc = -1;
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int K_kap = (threadIdx.x / Nbx) / Nbx;
	ufloat_t f_0 = N_Pf(0.0);
	ufloat_t f_1 = N_Pf(0.0);
	ufloat_t f_2 = N_Pf(0.0);
	ufloat_t f_3 = N_Pf(0.0);
	ufloat_t f_4 = N_Pf(0.0);
	ufloat_t f_5 = N_Pf(0.0);
	ufloat_t f_6 = N_Pf(0.0);
	ufloat_t f_7 = N_Pf(0.0);
	ufloat_t f_8 = N_Pf(0.0);
	ufloat_t f_9 = N_Pf(0.0);
	ufloat_t f_10 = N_Pf(0.0);
	ufloat_t f_11 = N_Pf(0.0);
	ufloat_t f_12 = N_Pf(0.0);
	ufloat_t f_13 = N_Pf(0.0);
	ufloat_t f_14 = N_Pf(0.0);
	ufloat_t f_15 = N_Pf(0.0);
	ufloat_t f_16 = N_Pf(0.0);
	ufloat_t f_17 = N_Pf(0.0);
	ufloat_t f_18 = N_Pf(0.0);
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
	ufloat_t w_kap = N_Pf(0.0);
	ufloat_t tmp_i = N_Pf(0.0);
#if (S_LES==1)
	ufloat_t tmp_j = N_Pf(0.0);
	ufloat_t tmp_k = N_Pf(0.0);
#endif
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = N_Pf(0.0);
	ufloat_t omeg = dx_L / tau_L;
	ufloat_t omegp = N_Pf(1.0) - omeg;

	s_ID_cblock[threadIdx.x] = -1;
	if (kap < n_ids_idev_L)
	{
		i_kap = id_set_idev_L[kap];
		s_ID_cblock[threadIdx.x] = i_kap;
	}
	__syncthreads();

	// Loop over block Ids.
	for (int k = 0; k < M_CBLOCK; k++)
	{
		int i_kap_b = s_ID_cblock[k];
		int nbr_kap_b = -1;
		i_kap_bc = -1;
		block_on_boundary = 0;

		if (i_kap_b > -1)
		{
			i_kap_bc = cblock_ID_nbr_child[i_kap_b];
			block_on_boundary = cblock_ID_mask[i_kap_b];
		}

		if ( i_kap_b > -1 && (i_kap_bc < 0 || block_on_boundary == 1) )
		{
			// Load DDFs and compute macroscopic properties.
			f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			f_9 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells];
			f_10 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells];
			f_11 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells];
			f_12 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells];
			f_13 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells];
			f_14 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells];
			f_15 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells];
			f_16 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells];
			f_17 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells];
			f_18 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells];
			rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18;
			u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / rho_kap;
			v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / rho_kap;
			w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
#if (B_TYPE==1||S_LES==1)
			s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = u_kap;
			s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = v_kap;
			s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = w_kap;
			__syncthreads();
#endif

			// Get turublent viscosity for Large Eddy Simulation.
#if (S_LES==1)
			// Extrapolate macroscopic properties to block edges.
			if (I_kap==0)
			{
				s_u[0+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_u[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_u[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_u[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_v[0+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_v[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_v[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_v[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_w[0+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_w[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_w[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_w[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_u[5+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_u[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_u[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_u[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_v[5+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_v[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_v[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_v[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_w[5+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_w[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_w[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_w[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
			}
			if (J_kap==0)
			{
				s_u[(I_kap+1)+(Nbx+2)*(0)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_u[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_u[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_u[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_v[(I_kap+1)+(Nbx+2)*(0)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_v[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_v[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_v[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_w[(I_kap+1)+(Nbx+2)*(0)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_w[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_w[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_w[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_u[(I_kap+1)+(Nbx+2)*(5)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_u[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_u[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_u[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_v[(I_kap+1)+(Nbx+2)*(5)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_v[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_v[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_v[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
				s_w[(I_kap+1)+(Nbx+2)*(5)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*s_w[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*s_w[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*s_w[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)];
			}
			if (K_kap==0)
			{
				s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(0)] = 4*s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)] - 6*s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] + 4*s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)];
				s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(0)] = 4*s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)] - 6*s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] + 4*s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)];
				s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(0)] = 4*s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)] - 6*s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] + 4*s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)];
				s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(5)] = 4*s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)] - 6*s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] + 4*s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)];
				s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(5)] = 4*s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)] - 6*s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] + 4*s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)];
				s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(5)] = 4*s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)] - 6*s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] + 4*s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)];
			}
			// Compute turbulent viscosity, storing S_{ij}^d in tmp_j, S_{ij} in tmp_k.
			tmp_i = N_Pf(0.0);
			tmp_j = N_Pf(0.0);
			tmp_k = N_Pf(0.0);
				// (11,22)
			tmp_i += (( + (s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]))*( + (s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])));
				// (11,33)
			tmp_i += (( + (s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]))*( + (s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])));
				// (22,33)
			tmp_i += (( + (s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]))*( + (s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])));
				// (12,12)
			tmp_j += (( + (s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]))*( + (s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])));
				// (13,13)
			tmp_j += (( + (s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]))*( + (s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])));
				// (23,23)
			tmp_j += (( + (s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]))*( + (s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]) + (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])));
			tmp_j = N_Pf(0.25)*(tmp_i - tmp_j);

			// Denominator.
			tmp_k += (s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_u[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]);
			tmp_k += (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_u[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]);
			tmp_k += (s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_u[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]);
			tmp_k += (s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]);
			tmp_k += (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]);
			tmp_k += (s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_v[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]);
			tmp_k += (s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap-1+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]);
			tmp_k += (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap-1+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]);
			tmp_k += (s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)])*(s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1+1)] - s_w[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap-1+1)]);

			// Compute t_eff.
			tmp_k = tmp_k/(N_Pf(4.0)*dx_L*dx_L);
			tmp_i = (N_Pf(0.070000000000000))*sqrt(tmp_j/tmp_k);
			if (isnan(tmp_i))
			{
				tmp_i = N_Pf(0.0);
			}
			omeg = dx_L / (   N_Pf(3.0)*(v0 + tmp_i) + N_Pf(0.5)*dx_L   );
			omegp = N_Pf(1.0) - omeg;
			tau_ratio = N_Pf(0.25) + (N_Pf(0.75)*tau_L - N_Pf(0.25)*dx_L)*(omeg/dx_L);
#endif

			// Collision step.
			cdotu = N_Pf(0.0);
			f_0 = f_0*omegp + ( N_Pf(0.333333333333333)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +u_kap;
			f_1 = f_1*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -u_kap;
			f_2 = f_2*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +v_kap;
			f_3 = f_3*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -v_kap;
			f_4 = f_4*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +w_kap;
			f_5 = f_5*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -w_kap;
			f_6 = f_6*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +u_kap+v_kap;
			f_7 = f_7*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -u_kap-v_kap;
			f_8 = f_8*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +u_kap+w_kap;
			f_9 = f_9*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -u_kap-w_kap;
			f_10 = f_10*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +v_kap+w_kap;
			f_11 = f_11*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -v_kap-w_kap;
			f_12 = f_12*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +u_kap-v_kap;
			f_13 = f_13*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -u_kap+v_kap;
			f_14 = f_14*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +u_kap-w_kap;
			f_15 = f_15*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -u_kap+w_kap;
			f_16 = f_16*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = +v_kap-w_kap;
			f_17 = f_17*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = -v_kap+w_kap;
			f_18 = f_18*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;

			// Impose boundary conditions.
			block_on_boundary = cblock_ID_onb[i_kap_b];
			if (block_on_boundary == 1)
			{
				// nbr 1
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
					// p = 1
				if ((I_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 7
				if ((I_kap+1==Nbx) && (J_kap+1< Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 9
				if ((I_kap+1==Nbx) && (K_kap+1< Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 13
				if ((I_kap+1==Nbx) && (J_kap-1>= 0))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 15
				if ((I_kap+1==Nbx) && (K_kap-1>= 0))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 2
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
					// p = 2
				if ((I_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 8
				if ((I_kap-1==-1) && (J_kap-1>= 0))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 10
				if ((I_kap-1==-1) && (K_kap-1>= 0))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 14
				if ((I_kap-1==-1) && (J_kap+1< Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 16
				if ((I_kap-1==-1) && (K_kap+1< Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 3
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
					// p = 3
				if ((J_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 7
				if ((I_kap+1< Nbx) && (J_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 11
				if ((J_kap+1==Nbx) && (K_kap+1< Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 14
				if ((I_kap-1>= 0) && (J_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 17
				if ((J_kap+1==Nbx) && (K_kap-1>= 0))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 4
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
					// p = 4
				if ((J_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 8
				if ((I_kap-1>= 0) && (J_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 12
				if ((J_kap-1==-1) && (K_kap-1>= 0))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 13
				if ((I_kap+1< Nbx) && (J_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 18
				if ((J_kap-1==-1) && (K_kap+1< Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 5
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
					// p = 5
				if ((K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 9
				if ((I_kap+1< Nbx) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 11
				if ((J_kap+1< Nbx) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 16
				if ((I_kap-1>= 0) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 18
				if ((J_kap-1>= 0) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 6
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
					// p = 6
				if ((K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 10
				if ((I_kap-1>= 0) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 12
				if ((J_kap-1>= 0) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 15
				if ((I_kap+1< Nbx) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
					// p = 17
				if ((J_kap+1< Nbx) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 7
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
					// p = 7
				if ((I_kap+1==Nbx) && (J_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 8
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
					// p = 8
				if ((I_kap-1==-1) && (J_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 9
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
					// p = 9
				if ((I_kap+1==Nbx) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 10
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
					// p = 10
				if ((I_kap-1==-1) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 11
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
					// p = 11
				if ((J_kap+1==Nbx) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 12
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
					// p = 12
				if ((J_kap-1==-1) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 13
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
					// p = 13
				if ((I_kap+1==Nbx) && (J_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 14
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
					// p = 14
				if ((I_kap-1==-1) && (J_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 15
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
					// p = 15
				if ((I_kap+1==Nbx) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.050000000000000);
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 16
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
					// p = 16
				if ((I_kap-1==-1) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(-0.050000000000000);
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 17
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
					// p = 17
				if ((J_kap+1==Nbx) && (K_kap-1==-1))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
				// nbr 18
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
					// p = 18
				if ((J_kap-1==-1) && (K_kap+1==Nbx))
				{
					if (nbr_kap_b == -6)
					{
						cdotu = N_Pf(0.000000000000000);
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
					}
				}
			}

			// Write fi* to global memory.
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_1;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_2;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_3;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_4;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_5;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_6;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_7;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_8;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_9;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_10;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_11;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_12;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_13;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_14;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_15;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_16;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_17;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_18;
			__syncthreads();
		}
	}
}


int Solver_LBM::S_Collide_d3q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collide_d3q19<<<(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]>>>
		(
			mesh->n_ids[i_dev][L], mesh->c_id_set[i_dev][L], mesh->n_maxcells, dx_vec[L], tau_vec[L], tau_ratio_vec_C2F[L],
			mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->n_maxcblocks,
			mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev]
		);
	}

	return 0;
}

#endif
