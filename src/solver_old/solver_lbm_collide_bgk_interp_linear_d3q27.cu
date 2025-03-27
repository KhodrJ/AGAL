/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

#if (N_Q==27)

__global__
void Cu_Collide_BGK_Interpolate_Linear_d3q27
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t dx_L, ufloat_t tau_L, ufloat_t tau_ratio, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb, ufloat_t *cblock_f_X
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ ufloat_t s_F[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	int K_kap = (threadIdx.x / Nbx) / Nbx;
	ufloat_t x_kap = N_Pf(0.0);
	ufloat_t y_kap = N_Pf(0.0);
	ufloat_t z_kap = N_Pf(0.0);
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int nbr_kap_b = -1;
	int block_on_boundary = -1;
	int block_on_interface = -1;
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
	ufloat_t tmp_i = N_Pf(0.0);
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
	ufloat_t w_kap = N_Pf(0.0);
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = N_Pf(0.0);
	ufloat_t omeg = dx_L / tau_L;
	ufloat_t omegp = N_Pf(1.0) - omeg;
	int kap = blockIdx.x*M_LBLOCK + threadIdx.x;

	s_ID_cblock[threadIdx.x] = -1;
	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
	{
		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
	}
	__syncthreads();

	// Loop over block Ids.
	for (int k = 0; k < M_LBLOCK; k += 1)
	{
		i_kap_b = s_ID_cblock[k];

		// This part is included if n>0 only.
		if (i_kap_b > -1)
		{
			i_kap_bc=cblock_ID_nbr_child[i_kap_b];
			block_on_interface=cblock_ID_mask[i_kap_b];
		}

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && ((i_kap_bc<0)||(block_on_interface==1)))
		{
			// Retrieve DDFs and compute macroscopic properties.
			block_on_boundary = cblock_ID_onb[i_kap_b];
			x_kap = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*(N_Pf(0.5) + I_kap);
			y_kap = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*(N_Pf(0.5) + J_kap);
			z_kap = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*(N_Pf(0.5) + K_kap);
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
			f_19 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells];
			f_20 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells];
			f_21 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells];
			f_22 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells];
			f_23 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells];
			f_24 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells];
			f_25 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells];
			f_26 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells];
			rho_kap = f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26 ;
			u_kap = (N_Pf(0.0)*f_0+N_Pf(1.0)*f_1+N_Pf(-1.0)*f_2+N_Pf(0.0)*f_3+N_Pf(0.0)*f_4+N_Pf(0.0)*f_5+N_Pf(0.0)*f_6+N_Pf(1.0)*f_7+N_Pf(-1.0)*f_8+N_Pf(1.0)*f_9+N_Pf(-1.0)*f_10+N_Pf(0.0)*f_11+N_Pf(0.0)*f_12+N_Pf(1.0)*f_13+N_Pf(-1.0)*f_14+N_Pf(1.0)*f_15+N_Pf(-1.0)*f_16+N_Pf(0.0)*f_17+N_Pf(0.0)*f_18+N_Pf(1.0)*f_19+N_Pf(-1.0)*f_20+N_Pf(1.0)*f_21+N_Pf(-1.0)*f_22+N_Pf(1.0)*f_23+N_Pf(-1.0)*f_24+N_Pf(-1.0)*f_25+N_Pf(1.0)*f_26) / rho_kap;
			v_kap = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(0.0)*f_2+N_Pf(1.0)*f_3+N_Pf(-1.0)*f_4+N_Pf(0.0)*f_5+N_Pf(0.0)*f_6+N_Pf(1.0)*f_7+N_Pf(-1.0)*f_8+N_Pf(0.0)*f_9+N_Pf(0.0)*f_10+N_Pf(1.0)*f_11+N_Pf(-1.0)*f_12+N_Pf(-1.0)*f_13+N_Pf(1.0)*f_14+N_Pf(0.0)*f_15+N_Pf(0.0)*f_16+N_Pf(1.0)*f_17+N_Pf(-1.0)*f_18+N_Pf(1.0)*f_19+N_Pf(-1.0)*f_20+N_Pf(1.0)*f_21+N_Pf(-1.0)*f_22+N_Pf(-1.0)*f_23+N_Pf(1.0)*f_24+N_Pf(1.0)*f_25+N_Pf(-1.0)*f_26) / rho_kap;
			w_kap = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(0.0)*f_2+N_Pf(0.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(0.0)*f_7+N_Pf(0.0)*f_8+N_Pf(1.0)*f_9+N_Pf(-1.0)*f_10+N_Pf(1.0)*f_11+N_Pf(-1.0)*f_12+N_Pf(0.0)*f_13+N_Pf(0.0)*f_14+N_Pf(-1.0)*f_15+N_Pf(1.0)*f_16+N_Pf(-1.0)*f_17+N_Pf(1.0)*f_18+N_Pf(1.0)*f_19+N_Pf(-1.0)*f_20+N_Pf(-1.0)*f_21+N_Pf(1.0)*f_22+N_Pf(1.0)*f_23+N_Pf(-1.0)*f_24+N_Pf(1.0)*f_25+N_Pf(-1.0)*f_26) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;

			// Interpolate data to children if on an interface at this stage.
			if (block_on_interface==1)
			{
				//
				// DDF 0.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.296296296296296)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_0 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 0*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 1.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_1 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 2*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 2.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_2 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 1*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 3.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_3 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 4*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 4.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_4 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 3*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 5.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_5 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 6*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 6.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_6 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 5*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 7.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_7 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 8*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 8.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_8 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 7*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 9.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_9 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 10*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 10.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_10 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 9*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 11.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_11 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 12*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 12.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_12 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 11*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 13.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_13 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 14*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 14.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_14 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 13*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 15.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_15 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 16*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 16.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_16 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 15*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 17.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_17 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 18*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 18.
				//
				cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_18 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 17*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 19.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_19 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 20*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 20.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_20 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 19*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 21.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_21 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 22*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 22.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_22 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 21*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 23.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_23 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 24*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 24.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_24 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 23*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 25.
				//
				cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_25 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 26*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

				//
				// DDF 26.
				//
				cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(-1.0)*w_kap;
				tmp_i = N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
				s_F[threadIdx.x] = tmp_i + (f_26 - tmp_i)*(tau_ratio);
				__syncthreads();
				//	Child 0.
				if ((cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[0] +  (s_F[1] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[4] - s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[16] - s_F[0])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[5] - s_F[1] - s_F[4] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[17] - s_F[1] - s_F[16] + s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[20] - s_F[4] - s_F[16] + s_F[0])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[21] + s_F[1] + s_F[4] + s_F[16] - s_F[5] - s_F[17] - s_F[20] - s_F[0])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 1.
				if ((cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[2] +  (s_F[3] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[6] - s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[18] - s_F[2])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[7] - s_F[3] - s_F[6] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[19] - s_F[3] - s_F[18] + s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[22] - s_F[6] - s_F[18] + s_F[2])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[23] + s_F[3] + s_F[6] + s_F[18] - s_F[7] - s_F[19] - s_F[22] - s_F[2])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 2.
				if ((cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[8] +  (s_F[9] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[12] - s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[24] - s_F[8])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[13] - s_F[9] - s_F[12] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[25] - s_F[9] - s_F[24] + s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[28] - s_F[12] - s_F[24] + s_F[8])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[29] + s_F[9] + s_F[12] + s_F[24] - s_F[13] - s_F[25] - s_F[28] - s_F[8])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 3.
				if ((cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[10] +  (s_F[11] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[14] - s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[26] - s_F[10])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[15] - s_F[11] - s_F[14] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[27] - s_F[11] - s_F[26] + s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[30] - s_F[14] - s_F[26] + s_F[10])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[31] + s_F[11] + s_F[14] + s_F[26] - s_F[15] - s_F[27] - s_F[30] - s_F[10])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 4.
				if ((cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[32] +  (s_F[33] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[36] - s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[48] - s_F[32])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[37] - s_F[33] - s_F[36] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[49] - s_F[33] - s_F[48] + s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[52] - s_F[36] - s_F[48] + s_F[32])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[53] + s_F[33] + s_F[36] + s_F[48] - s_F[37] - s_F[49] - s_F[52] - s_F[32])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 5.
				if ((cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[34] +  (s_F[35] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[38] - s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[50] - s_F[34])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[39] - s_F[35] - s_F[38] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[51] - s_F[35] - s_F[50] + s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[54] - s_F[38] - s_F[50] + s_F[34])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[55] + s_F[35] + s_F[38] + s_F[50] - s_F[39] - s_F[51] - s_F[54] - s_F[34])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 6.
				if ((cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[40] +  (s_F[41] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[44] - s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[56] - s_F[40])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[45] - s_F[41] - s_F[44] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[57] - s_F[41] - s_F[56] + s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[60] - s_F[44] - s_F[56] + s_F[40])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[61] + s_F[41] + s_F[44] + s_F[56] - s_F[45] - s_F[57] - s_F[60] - s_F[40])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				//	Child 7.
				if ((cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2))
				{
					cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 25*n_maxcells] =  s_F[42] +  (s_F[43] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) +  (s_F[46] - s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[58] - s_F[42])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[47] - s_F[43] - s_F[46] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) +  (s_F[59] - s_F[43] - s_F[58] + s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[62] - s_F[46] - s_F[58] + s_F[42])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) +  (s_F[63] + s_F[43] + s_F[46] + s_F[58] - s_F[47] - s_F[59] - s_F[62] - s_F[42])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
				}
				__syncthreads();

			}

			// Eddy viscosity calculation.

			// Collision step.
			cdotu = N_Pf(0.0);
			f_0 = f_0*omegp + ( N_Pf(0.296296296296296)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap;
			f_1 = f_1*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap);
			f_2 = f_2*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = v_kap;
			f_3 = f_3*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-v_kap);
			f_4 = f_4*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = w_kap;
			f_5 = f_5*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-w_kap);
			f_6 = f_6*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+v_kap;
			f_7 = f_7*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+(-v_kap);
			f_8 = f_8*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+w_kap;
			f_9 = f_9*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+(-w_kap);
			f_10 = f_10*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = v_kap+w_kap;
			f_11 = f_11*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-v_kap)+(-w_kap);
			f_12 = f_12*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+(-v_kap);
			f_13 = f_13*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+v_kap;
			f_14 = f_14*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+(-w_kap);
			f_15 = f_15*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+w_kap;
			f_16 = f_16*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = v_kap+(-w_kap);
			f_17 = f_17*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-v_kap)+w_kap;
			f_18 = f_18*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+v_kap+w_kap;
			f_19 = f_19*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+(-v_kap)+(-w_kap);
			f_20 = f_20*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+v_kap+(-w_kap);
			f_21 = f_21*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+(-v_kap)+w_kap;
			f_22 = f_22*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+(-v_kap)+w_kap;
			f_23 = f_23*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+v_kap+(-w_kap);
			f_24 = f_24*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = (-u_kap)+v_kap+w_kap;
			f_25 = f_25*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
			cdotu = u_kap+(-v_kap)+(-w_kap);
			f_26 = f_26*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;

			// Impose boundary conditions.
			if (block_on_boundary == 1)
			{
				nbr_kap_b = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
				if ((I_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap;
						f_1 = -f_1 + N_Pf(2.0)*N_Pf(0.074074074074074)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 0.0*0 );
						f_1 = f_1 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap;
						f_7 = -f_7 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+w_kap;
						f_9 = -f_9 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap);
						f_13 = -f_13 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-w_kap);
						f_15 = -f_15 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap<Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+w_kap;
						f_19 = -f_19 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap<Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+(-w_kap);
						f_21 = -f_21 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap>0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+w_kap;
						f_23 = -f_23 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap>0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+(-w_kap);
						f_26 = -f_26 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
				if ((I_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap);
						f_2 = -f_2 + N_Pf(2.0)*N_Pf(0.074074074074074)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 0.0*0 );
						f_2 = f_2 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap);
						f_8 = -f_8 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-w_kap);
						f_10 = -f_10 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap;
						f_14 = -f_14 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+w_kap;
						f_16 = -f_16 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap>0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+(-w_kap);
						f_20 = -f_20 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap>0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+w_kap;
						f_22 = -f_22 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap<Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+(-w_kap);
						f_24 = -f_24 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap<Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+w_kap;
						f_25 = -f_25 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
				if ((J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap;
						f_3 = -f_3 + N_Pf(2.0)*N_Pf(0.074074074074074)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 0.0*0 );
						f_3 = f_3 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap;
						f_7 = -f_7 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap==Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap+w_kap;
						f_11 = -f_11 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap;
						f_14 = -f_14 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap==Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap+(-w_kap);
						f_17 = -f_17 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+w_kap;
						f_19 = -f_19 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+(-w_kap);
						f_21 = -f_21 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+(-w_kap);
						f_24 = -f_24 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+w_kap;
						f_25 = -f_25 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 4*n_maxcblocks];
				if ((J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap);
						f_4 = -f_4 + N_Pf(2.0)*N_Pf(0.074074074074074)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 0.0*0 );
						f_4 = f_4 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap);
						f_8 = -f_8 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap==0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap)+(-w_kap);
						f_12 = -f_12 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap);
						f_13 = -f_13 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap==0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap)+w_kap;
						f_18 = -f_18 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+(-w_kap);
						f_20 = -f_20 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+w_kap;
						f_22 = -f_22 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+w_kap;
						f_23 = -f_23 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+(-w_kap);
						f_26 = -f_26 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 5*n_maxcblocks];
				if ((K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 1.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = w_kap;
						f_5 = -f_5 + N_Pf(2.0)*N_Pf(0.074074074074074)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + 1.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + 1.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + 1.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + 1.0*0 );
						f_5 = f_5 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+w_kap;
						f_9 = -f_9 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap<Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap+w_kap;
						f_11 = -f_11 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+w_kap;
						f_16 = -f_16 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap>0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap)+w_kap;
						f_18 = -f_18 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap<Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+w_kap;
						f_19 = -f_19 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap>0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+w_kap;
						f_22 = -f_22 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap>0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+w_kap;
						f_23 = -f_23 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap<Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+w_kap;
						f_25 = -f_25 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 6*n_maxcblocks];
				if ((K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + -1.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-w_kap);
						f_6 = -f_6 + N_Pf(2.0)*N_Pf(0.074074074074074)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + -1.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + -1.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + -1.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 0.0*0 + -1.0*0 );
						f_6 = f_6 - N_Pf(2.0)*N_Pf(0.074074074074074)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-w_kap);
						f_10 = -f_10 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap>0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap)+(-w_kap);
						f_12 = -f_12 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-w_kap);
						f_15 = -f_15 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((J_kap<Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap+(-w_kap);
						f_17 = -f_17 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap>0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+(-w_kap);
						f_20 = -f_20 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap<Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+(-w_kap);
						f_21 = -f_21 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap<Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+(-w_kap);
						f_24 = -f_24 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap>0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+(-w_kap);
						f_26 = -f_26 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 7*n_maxcblocks];
				if ((I_kap==Nbx-1)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap;
						f_7 = -f_7 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 0.0*0 );
						f_7 = f_7 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap==Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+w_kap;
						f_19 = -f_19 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap==Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+(-w_kap);
						f_21 = -f_21 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 8*n_maxcblocks];
				if ((I_kap==0)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap);
						f_8 = -f_8 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 0.0*0 );
						f_8 = f_8 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap==0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+(-w_kap);
						f_20 = -f_20 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap==0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+w_kap;
						f_22 = -f_22 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 9*n_maxcblocks];
				if ((I_kap==Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+w_kap;
						f_9 = -f_9 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + 1.0*0 );
						f_9 = f_9 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap<Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+w_kap;
						f_19 = -f_19 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap>0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+w_kap;
						f_23 = -f_23 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 10*n_maxcblocks];
				if ((I_kap==0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-w_kap);
						f_10 = -f_10 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + -1.0*0 );
						f_10 = f_10 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap>0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+(-w_kap);
						f_20 = -f_20 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap<Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+(-w_kap);
						f_24 = -f_24 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 11*n_maxcblocks];
				if ((J_kap==Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap+w_kap;
						f_11 = -f_11 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + 1.0*0 );
						f_11 = f_11 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+w_kap;
						f_19 = -f_19 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+w_kap;
						f_25 = -f_25 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 12*n_maxcblocks];
				if ((J_kap==0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap)+(-w_kap);
						f_12 = -f_12 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + -1.0*0 );
						f_12 = f_12 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+(-w_kap);
						f_20 = -f_20 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+(-w_kap);
						f_26 = -f_26 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 13*n_maxcblocks];
				if ((I_kap==Nbx-1)and(J_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap);
						f_13 = -f_13 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 0.0*0 );
						f_13 = f_13 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap==0)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+w_kap;
						f_23 = -f_23 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap==0)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+(-w_kap);
						f_26 = -f_26 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 14*n_maxcblocks];
				if ((I_kap==0)and(J_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap;
						f_14 = -f_14 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 0.0*0 );
						f_14 = f_14 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap==Nbx-1)and(K_kap>0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+(-w_kap);
						f_24 = -f_24 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap==Nbx-1)and(K_kap<Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+w_kap;
						f_25 = -f_25 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 15*n_maxcblocks];
				if ((I_kap==Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-w_kap);
						f_15 = -f_15 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 0.0*0 + -1.0*0 );
						f_15 = f_15 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap<Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+(-w_kap);
						f_21 = -f_21 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==Nbx-1)and(J_kap>0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+(-w_kap);
						f_26 = -f_26 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 16*n_maxcblocks];
				if ((I_kap==0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+w_kap;
						f_16 = -f_16 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 0.0*0 + 1.0*0 );
						f_16 = f_16 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap>0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+w_kap;
						f_22 = -f_22 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap==0)and(J_kap<Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+w_kap;
						f_25 = -f_25 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 17*n_maxcblocks];
				if ((J_kap==Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = v_kap+(-w_kap);
						f_17 = -f_17 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + 1.0*0 + -1.0*0 );
						f_17 = f_17 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+(-w_kap);
						f_21 = -f_21 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+(-w_kap);
						f_24 = -f_24 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 18*n_maxcblocks];
				if ((J_kap==0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 0.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-v_kap)+w_kap;
						f_18 = -f_18 + N_Pf(2.0)*N_Pf(0.018518518518519)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 0.0*0 + -1.0*0 + 1.0*0 );
						f_18 = f_18 - N_Pf(2.0)*N_Pf(0.018518518518519)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap>0)and(J_kap==0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+w_kap;
						f_22 = -f_22 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}
				if ((I_kap<Nbx-1)and(J_kap==0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+w_kap;
						f_23 = -f_23 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 19*n_maxcblocks];
				if ((I_kap==Nbx-1)and(J_kap==Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+w_kap;
						f_19 = -f_19 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + 1.0*0 );
						f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 20*n_maxcblocks];
				if ((I_kap==0)and(J_kap==0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+(-w_kap);
						f_20 = -f_20 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + -1.0*0 );
						f_20 = f_20 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 21*n_maxcblocks];
				if ((I_kap==Nbx-1)and(J_kap==Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+v_kap+(-w_kap);
						f_21 = -f_21 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + 1.0*0 + -1.0*0 );
						f_21 = f_21 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 22*n_maxcblocks];
				if ((I_kap==0)and(J_kap==0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+(-v_kap)+w_kap;
						f_22 = -f_22 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + -1.0*0 + 1.0*0 );
						f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 23*n_maxcblocks];
				if ((I_kap==Nbx-1)and(J_kap==0)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+w_kap;
						f_23 = -f_23 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + 1.0*0 );
						f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 24*n_maxcblocks];
				if ((I_kap==0)and(J_kap==Nbx-1)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+(-w_kap);
						f_24 = -f_24 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + -1.0*0 );
						f_24 = f_24 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 25*n_maxcblocks];
				if ((I_kap==0)and(J_kap==Nbx-1)and(K_kap==Nbx-1))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( -1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = (-u_kap)+v_kap+w_kap;
						f_25 = -f_25 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( -1.0*0 + 1.0*0 + 1.0*0 );
						f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
				}

				nbr_kap_b = cblock_ID_nbr[i_kap_b + 26*n_maxcblocks];
				if ((I_kap==Nbx-1)and(J_kap==0)and(K_kap==0))
				{
					if (nbr_kap_b == -1)
					{
						cdotu = (ufloat_t)( 1.0*(16*y_kap)*(1.0-4*y_kap)*0.05 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -2)
					{
						cdotu = u_kap+(-v_kap)+(-w_kap);
						f_26 = -f_26 + N_Pf(2.0)*N_Pf(0.004629629629630)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
					}
					if (nbr_kap_b == -3)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -4)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -6)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
					}
					if (nbr_kap_b == -8)
					{
						cdotu = (ufloat_t)( 1.0*0 + -1.0*0 + -1.0*0 );
						f_26 = f_26 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
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
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 19*n_maxcells] = f_19;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 20*n_maxcells] = f_20;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 21*n_maxcells] = f_21;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 22*n_maxcells] = f_22;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 23*n_maxcells] = f_23;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 24*n_maxcells] = f_24;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 25*n_maxcells] = f_25;
			cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 26*n_maxcells] = f_26;
			__syncthreads();
		}
	}
}

int Solver_LBM::S_Collide_BGK_Interpolate_Linear_d3q27(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Collide_BGK_Interpolate_Linear_d3q27<<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, dxf_vec[L], tau_vec[L], tau_ratio_vec_C2F[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_f_X[i_dev]);
	}

	return 0;
}

#endif