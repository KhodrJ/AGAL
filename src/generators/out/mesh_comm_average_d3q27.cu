#include "mesh.h"

#if (N_Q==27)

template <int ave_type = 0>
__global__
void Cu_Average_d3q27
(
	int n_ids_idev_L, int *id_set_idev_L, long int n_maxcells, ufloat_t dx_Lm1, ufloat_t dx_L, ufloat_t tau_L, ufloat_t tau_ratio,
	int *cblock_ID_onb, int *cblock_ID_mask, int *cblock_ID_nbr_child, int n_maxcblocks,
	int *cells_ID_mask, ufloat_t *cells_f_F
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ int s_ID_mask_child[M_CBLOCK];
	__shared__ ufloat_t s_Fc[M_CBLOCK];
#if (S_LES==1)
	__shared__ ufloat_t s_Feq[M_CBLOCK];
	__shared__ ufloat_t s_tau[M_CBLOCK];
#endif
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
	int child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2)) + Nbx*Nbx*(2*(((threadIdx.x / Nbx) / Nbx)%2));
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
	ufloat_t f_19 = N_Pf(0.0);
	ufloat_t f_20 = N_Pf(0.0);
	ufloat_t f_21 = N_Pf(0.0);
	ufloat_t f_22 = N_Pf(0.0);
	ufloat_t f_23 = N_Pf(0.0);
	ufloat_t f_24 = N_Pf(0.0);
	ufloat_t f_25 = N_Pf(0.0);
	ufloat_t f_26 = N_Pf(0.0);
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
		i_kap_bc = -1;
		block_on_boundary = 0;

		if (i_kap_b > -1)
		{
			i_kap_bc = cblock_ID_nbr_child[i_kap_b];
			block_on_boundary = cblock_ID_mask[i_kap_b];
		}

		if ( i_kap_bc > -1 && (ave_type == 2 || block_on_boundary == 1) )
		{
			for (int xc_k = 0; xc_k < 2; xc_k++)
			{
				for (int xc_j = 0; xc_j < 2; xc_j++)
				{
					for (int xc_i = 0; xc_i < 2; xc_i++)
					{
						int xc = xc_i + 2*xc_j + 4*xc_k;

						// Load DDFs and compute macroscopic properties.
						f_0 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 0*n_maxcells];
						f_1 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 2*n_maxcells];
						f_2 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 1*n_maxcells];
						f_3 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 4*n_maxcells];
						f_4 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 3*n_maxcells];
						f_5 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 6*n_maxcells];
						f_6 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 5*n_maxcells];
						f_7 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 8*n_maxcells];
						f_8 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 7*n_maxcells];
						f_9 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 10*n_maxcells];
						f_10 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 9*n_maxcells];
						f_11 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 12*n_maxcells];
						f_12 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 11*n_maxcells];
						f_13 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 14*n_maxcells];
						f_14 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 13*n_maxcells];
						f_15 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 16*n_maxcells];
						f_16 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 15*n_maxcells];
						f_17 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 18*n_maxcells];
						f_18 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 17*n_maxcells];
						f_19 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 20*n_maxcells];
						f_20 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 19*n_maxcells];
						f_21 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 22*n_maxcells];
						f_22 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 21*n_maxcells];
						f_23 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 24*n_maxcells];
						f_24 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 23*n_maxcells];
						f_25 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 26*n_maxcells];
						f_26 = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + 25*n_maxcells];
						rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26;
						u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16 +f_19 -f_20 +f_21 -f_22 +f_23 -f_24 -f_25 +f_26) / rho_kap;
						v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18 +f_19 -f_20 +f_21 -f_22 -f_23 +f_24 +f_25 -f_26) / rho_kap;
						w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18 +f_19 -f_20 -f_21 +f_22 +f_23 -f_24 +f_25 -f_26) / rho_kap;
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
						__syncthreads();

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
						s_tau[threadIdx.x] = N_Pf(3.0)*(v0 + tmp_i) + N_Pf(0.5)*dx_L;
#endif

						// Average rescaled fi to parent if applicable.
						s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x];
						if (ave_type > 0 && s_ID_mask_child[threadIdx.x] < 2)
						{
							s_ID_mask_child[threadIdx.x] = 1;
						}

							// p = 0
						cdotu = N_Pf(0.0);
						tmp_i = N_Pf(0.296296296296296)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_0 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_0;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 1
						cdotu = +u_kap;
						tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_1 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_1;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 2
						cdotu = -u_kap;
						tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_2 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_2;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 3
						cdotu = +v_kap;
						tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_3 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_3;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 4
						cdotu = -v_kap;
						tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_4 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_4;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 5
						cdotu = +w_kap;
						tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_5 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_5;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 6
						cdotu = -w_kap;
						tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_6 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_6;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 7
						cdotu = +u_kap+v_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_7 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_7;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 8
						cdotu = -u_kap-v_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_8 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_8;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 9
						cdotu = +u_kap+w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_9 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_9;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 10*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 10
						cdotu = -u_kap-w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_10 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_10;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 9*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 11
						cdotu = +v_kap+w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_11 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_11;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 12*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 12
						cdotu = -v_kap-w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_12 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_12;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 11*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 13
						cdotu = +u_kap-v_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_13 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_13;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 14*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 14
						cdotu = -u_kap+v_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_14 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_14;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 13*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 15
						cdotu = +u_kap-w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_15 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_15;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 16*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 16
						cdotu = -u_kap+w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_16 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_16;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 15*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 17
						cdotu = +v_kap-w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_17 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_17;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 18*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 18
						cdotu = -v_kap+w_kap;
						tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_18 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_18;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 17*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 19
						cdotu = +u_kap+v_kap+w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_19 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_19;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 20
						cdotu = -u_kap-v_kap-w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_20 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_20;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 21
						cdotu = +u_kap+v_kap-w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_21 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_21;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 22
						cdotu = -u_kap-v_kap+w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_22 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_22;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 23
						cdotu = +u_kap-v_kap+w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_23 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_23;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 24
						cdotu = -u_kap+v_kap-w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_24 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_24;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 25
						cdotu = -u_kap+v_kap+w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_25 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_25;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();

							// p = 26
						cdotu = +u_kap-v_kap-w_kap;
						tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#if (S_LES==0)
						s_Fc[threadIdx.x] = tmp_i + (f_26 - tmp_i)*tau_ratio;
#else
						s_Fc[threadIdx.x] = f_26;
						s_Feq[threadIdx.x] = tmp_i;
#endif
						__syncthreads();
#if (S_LES==0)
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
						}
#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.
						if (s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k)
						{
							tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
							cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k );
						}
#endif
						__syncthreads();
					}
				}
			}
		}
	}
}


int Mesh::M_Average_d3q27(int i_dev, int L, int var, ufloat_t Cscale, ufloat_t Cscale2)
{
	if (n_ids[i_dev][L] > 0 && var == V_AVERAGE_INTERFACE)
	{
		Cu_Average_d3q27<0><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>
		(
			n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, dxf_vec[L+1], dxf_vec[L], Cscale, Cscale2,
			c_cblock_ID_onb[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr_child[i_dev], n_maxcblocks,
			c_cells_ID_mask[i_dev], c_cells_f_F[i_dev]
		);
	}
	if (n_ids[i_dev][L] > 0 && var == V_AVERAGE_BLOCK)
	{
		Cu_Average_d3q27<1><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>
		(
			n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, dxf_vec[L+1], dxf_vec[L], Cscale, Cscale2,
			c_cblock_ID_onb[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr_child[i_dev], n_maxcblocks,
			c_cells_ID_mask[i_dev], c_cells_f_F[i_dev]
		);
	}
	if (n_ids[i_dev][L] > 0 && var == V_AVERAGE_GRID)
	{
		Cu_Average_d3q27<2><<<(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]>>>
		(
			n_ids[i_dev][L], c_id_set[i_dev][L], n_maxcells, dxf_vec[L+1], dxf_vec[L], Cscale, Cscale2,
			c_cblock_ID_onb[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr_child[i_dev], n_maxcblocks,
			c_cells_ID_mask[i_dev], c_cells_f_F[i_dev]
		);
	}

	return 0;
}

#endif
