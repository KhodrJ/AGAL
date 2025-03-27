/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

#if (N_Q==9)

template <int interp_type=0>
__global__
void Cu_Interpolate_Cubic_d2q9
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t dx_L, ufloat_t tau_L, ufloat_t tau_ratio, ufloat_t v0, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_CBLOCK];
	__shared__ ufloat_t s_F[M_CBLOCK];
	ufloat_t x_kap = N_Pf(-0.083333333333333)+N_Pf(0.166666666666667)*(threadIdx.x % Nbx);
	ufloat_t y_kap = N_Pf(-0.083333333333333)+N_Pf(0.166666666666667)*((threadIdx.x / Nbx) % Nbx);
	int i_kap_b = -1;
	int i_kap_bc = -1;
	int block_on_boundary = -1;
	ufloat_t f_0 = N_Pf(0.0);
	ufloat_t f_1 = N_Pf(0.0);
	ufloat_t f_2 = N_Pf(0.0);
	ufloat_t f_3 = N_Pf(0.0);
	ufloat_t f_4 = N_Pf(0.0);
	ufloat_t f_5 = N_Pf(0.0);
	ufloat_t f_6 = N_Pf(0.0);
	ufloat_t f_7 = N_Pf(0.0);
	ufloat_t f_8 = N_Pf(0.0);
	ufloat_t tmp_i = N_Pf(0.0);
	ufloat_t rho_kap = N_Pf(0.0);
	ufloat_t u_kap = N_Pf(0.0);
	ufloat_t v_kap = N_Pf(0.0);
	ufloat_t w_kap = N_Pf(0.0);
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = N_Pf(0.0);
	ufloat_t alpha = N_Pf(0.0);
	ufloat_t S_beta_0 = N_Pf(0.0);
	ufloat_t S_res_0 = N_Pf(0.0);
	ufloat_t S_beta_1 = N_Pf(0.0);
	ufloat_t S_res_1 = N_Pf(0.0);
	ufloat_t S_beta_2 = N_Pf(0.0);
	ufloat_t S_res_2 = N_Pf(0.0);
	ufloat_t S_beta_3 = N_Pf(0.0);
	ufloat_t S_res_3 = N_Pf(0.0);
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
			block_on_boundary=cblock_ID_mask[i_kap_b];
		}

		// Latter condition is added only if n>0.
		if (i_kap_b > -1 && (((interp_type==0)and(block_on_boundary==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE))))
		{
			// Load DDFs and compute macroscopic properties.
			f_0 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
			f_1 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
			f_2 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 4*n_maxcells];
			f_3 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
			f_4 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
			f_5 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 7*n_maxcells];
			f_6 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 8*n_maxcells];
			f_7 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 5*n_maxcells];
			f_8 = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + 6*n_maxcells];
			rho_kap = f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 ;
			u_kap = (N_Pf(0.0)*f_0+N_Pf(1.0)*f_1+N_Pf(0.0)*f_2+N_Pf(-1.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(1.0)*f_8) / rho_kap;
			v_kap = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(1.0)*f_2+N_Pf(0.0)*f_3+N_Pf(-1.0)*f_4+N_Pf(1.0)*f_5+N_Pf(1.0)*f_6+N_Pf(-1.0)*f_7+N_Pf(-1.0)*f_8) / rho_kap;
			udotu = u_kap*u_kap + v_kap*v_kap;

			//
			// DDF 0.
			//
			cdotu = N_Pf(0.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_0 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 1.
			//
			cdotu = N_Pf(1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_1 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 2.
			//
			cdotu = N_Pf(0.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_2 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 3.
			//
			cdotu = N_Pf(-1.0)*u_kap + N_Pf(0.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_3 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 4.
			//
			cdotu = N_Pf(0.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_4 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 5.
			//
			cdotu = N_Pf(1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_5 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 6.
			//
			cdotu = N_Pf(-1.0)*u_kap + N_Pf(1.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_6 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 7.
			//
			cdotu = N_Pf(-1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_7 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = S_res_3;
			}
			__syncthreads();

			//
			// DDF 8.
			//
			cdotu = N_Pf(1.0)*u_kap + N_Pf(-1.0)*v_kap + N_Pf(0.0)*w_kap;
			tmp_i = N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
			s_F[threadIdx.x] = tmp_i + (f_8 - tmp_i)*(tau_ratio);
			__syncthreads();
			S_res_0 = N_Pf(0.0);
			S_res_1 = N_Pf(0.0);
			S_res_2 = N_Pf(0.0);
			S_res_3 = N_Pf(0.0);
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,3).
			alpha = N_Pf(20.250000000000000)*s_F[0]+N_Pf(-60.750000000000000)*s_F[1]+N_Pf(60.750000000000000)*s_F[2]+N_Pf(-20.250000000000000)*s_F[3]+N_Pf(-60.750000000000000)*s_F[4]+N_Pf(182.250000000000000)*s_F[5]+N_Pf(-182.250000000000000)*s_F[6]+N_Pf(60.750000000000000)*s_F[7]+N_Pf(60.750000000000000)*s_F[8]+N_Pf(-182.250000000000000)*s_F[9]+N_Pf(182.250000000000000)*s_F[10]+N_Pf(-60.750000000000000)*s_F[11]+N_Pf(-20.250000000000000)*s_F[12]+N_Pf(60.750000000000000)*s_F[13]+N_Pf(-60.750000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,3).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(101.250000000000000)*s_F[1]+N_Pf(-81.000000000000000)*s_F[2]+N_Pf(20.250000000000000)*s_F[3]+N_Pf(121.500000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(243.000000000000000)*s_F[6]+N_Pf(-60.750000000000000)*s_F[7]+N_Pf(-121.500000000000000)*s_F[8]+N_Pf(303.750000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(60.750000000000000)*s_F[11]+N_Pf(40.500000000000000)*s_F[12]+N_Pf(-101.250000000000000)*s_F[13]+N_Pf(81.000000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,3).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-40.500000000000000)*s_F[1]+N_Pf(20.250000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(-74.250000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-60.750000000000000)*s_F[6]+N_Pf(13.500000000000000)*s_F[7]+N_Pf(74.250000000000000)*s_F[8]+N_Pf(-121.500000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-13.500000000000000)*s_F[11]+N_Pf(-24.750000000000000)*s_F[12]+N_Pf(40.500000000000000)*s_F[13]+N_Pf(-20.250000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,3).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(13.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-13.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,2).
			alpha = N_Pf(-40.500000000000000)*s_F[0]+N_Pf(121.500000000000000)*s_F[1]+N_Pf(-121.500000000000000)*s_F[2]+N_Pf(40.500000000000000)*s_F[3]+N_Pf(101.250000000000000)*s_F[4]+N_Pf(-303.750000000000000)*s_F[5]+N_Pf(303.750000000000000)*s_F[6]+N_Pf(-101.250000000000000)*s_F[7]+N_Pf(-81.000000000000000)*s_F[8]+N_Pf(243.000000000000000)*s_F[9]+N_Pf(-243.000000000000000)*s_F[10]+N_Pf(81.000000000000000)*s_F[11]+N_Pf(20.250000000000000)*s_F[12]+N_Pf(-60.750000000000000)*s_F[13]+N_Pf(60.750000000000000)*s_F[14]+N_Pf(-20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,2).
			alpha = N_Pf(81.000000000000000)*s_F[0]+N_Pf(-202.500000000000000)*s_F[1]+N_Pf(162.000000000000000)*s_F[2]+N_Pf(-40.500000000000000)*s_F[3]+N_Pf(-202.500000000000000)*s_F[4]+N_Pf(506.250000000000000)*s_F[5]+N_Pf(-405.000000000000000)*s_F[6]+N_Pf(101.250000000000000)*s_F[7]+N_Pf(162.000000000000000)*s_F[8]+N_Pf(-405.000000000000000)*s_F[9]+N_Pf(324.000000000000000)*s_F[10]+N_Pf(-81.000000000000000)*s_F[11]+N_Pf(-40.500000000000000)*s_F[12]+N_Pf(101.250000000000000)*s_F[13]+N_Pf(-81.000000000000000)*s_F[14]+N_Pf(20.250000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,2).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(81.000000000000000)*s_F[1]+N_Pf(-40.500000000000000)*s_F[2]+N_Pf(9.000000000000000)*s_F[3]+N_Pf(123.750000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(101.250000000000000)*s_F[6]+N_Pf(-22.500000000000000)*s_F[7]+N_Pf(-99.000000000000000)*s_F[8]+N_Pf(162.000000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(18.000000000000000)*s_F[11]+N_Pf(24.750000000000000)*s_F[12]+N_Pf(-40.500000000000000)*s_F[13]+N_Pf(20.250000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,2).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(-22.500000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(18.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,1).
			alpha = N_Pf(24.750000000000000)*s_F[0]+N_Pf(-74.250000000000000)*s_F[1]+N_Pf(74.250000000000000)*s_F[2]+N_Pf(-24.750000000000000)*s_F[3]+N_Pf(-40.500000000000000)*s_F[4]+N_Pf(121.500000000000000)*s_F[5]+N_Pf(-121.500000000000000)*s_F[6]+N_Pf(40.500000000000000)*s_F[7]+N_Pf(20.250000000000000)*s_F[8]+N_Pf(-60.750000000000000)*s_F[9]+N_Pf(60.750000000000000)*s_F[10]+N_Pf(-20.250000000000000)*s_F[11]+N_Pf(-4.500000000000000)*s_F[12]+N_Pf(13.500000000000000)*s_F[13]+N_Pf(-13.500000000000000)*s_F[14]+N_Pf(4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,1).
			alpha = N_Pf(-49.500000000000000)*s_F[0]+N_Pf(123.750000000000000)*s_F[1]+N_Pf(-99.000000000000000)*s_F[2]+N_Pf(24.750000000000000)*s_F[3]+N_Pf(81.000000000000000)*s_F[4]+N_Pf(-202.500000000000000)*s_F[5]+N_Pf(162.000000000000000)*s_F[6]+N_Pf(-40.500000000000000)*s_F[7]+N_Pf(-40.500000000000000)*s_F[8]+N_Pf(101.250000000000000)*s_F[9]+N_Pf(-81.000000000000000)*s_F[10]+N_Pf(20.250000000000000)*s_F[11]+N_Pf(9.000000000000000)*s_F[12]+N_Pf(-22.500000000000000)*s_F[13]+N_Pf(18.000000000000000)*s_F[14]+N_Pf(-4.500000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,1).
			alpha = N_Pf(30.250000000000000)*s_F[0]+N_Pf(-49.500000000000000)*s_F[1]+N_Pf(24.750000000000000)*s_F[2]+N_Pf(-5.500000000000000)*s_F[3]+N_Pf(-49.500000000000000)*s_F[4]+N_Pf(81.000000000000000)*s_F[5]+N_Pf(-40.500000000000000)*s_F[6]+N_Pf(9.000000000000000)*s_F[7]+N_Pf(24.750000000000000)*s_F[8]+N_Pf(-40.500000000000000)*s_F[9]+N_Pf(20.250000000000000)*s_F[10]+N_Pf(-4.500000000000000)*s_F[11]+N_Pf(-5.500000000000000)*s_F[12]+N_Pf(9.000000000000000)*s_F[13]+N_Pf(-4.500000000000000)*s_F[14]+N_Pf(1.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,1).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(9.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(-4.500000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(1.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;
			S_beta_0 = N_Pf(0.0);
			S_beta_1 = N_Pf(0.0);
			S_beta_2 = N_Pf(0.0);
			S_beta_3 = N_Pf(0.0);
			// Compute weight alpha_ijk (3,0).
			alpha = N_Pf(-4.500000000000000)*s_F[0]+N_Pf(13.500000000000000)*s_F[1]+N_Pf(-13.500000000000000)*s_F[2]+N_Pf(4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(-0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (2,0).
			alpha = N_Pf(9.000000000000000)*s_F[0]+N_Pf(-22.500000000000000)*s_F[1]+N_Pf(18.000000000000000)*s_F[2]+N_Pf(-4.500000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(-0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (1,0).
			alpha = N_Pf(-5.500000000000000)*s_F[0]+N_Pf(9.000000000000000)*s_F[1]+N_Pf(-4.500000000000000)*s_F[2]+N_Pf(1.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			// Compute weight alpha_ijk (0,0).
			alpha = N_Pf(1.000000000000000)*s_F[0]+N_Pf(-0.000000000000000)*s_F[1]+N_Pf(0.000000000000000)*s_F[2]+N_Pf(0.000000000000000)*s_F[3]+N_Pf(0.000000000000000)*s_F[4]+N_Pf(0.000000000000000)*s_F[5]+N_Pf(0.000000000000000)*s_F[6]+N_Pf(0.000000000000000)*s_F[7]+N_Pf(0.000000000000000)*s_F[8]+N_Pf(0.000000000000000)*s_F[9]+N_Pf(0.000000000000000)*s_F[10]+N_Pf(0.000000000000000)*s_F[11]+N_Pf(0.000000000000000)*s_F[12]+N_Pf(0.000000000000000)*s_F[13]+N_Pf(0.000000000000000)*s_F[14]+N_Pf(0.000000000000000)*s_F[15];
			S_beta_0 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_0;
			S_beta_1 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_1;
			S_beta_2 = alpha + (x_kap+N_Pf(0.000000000000000))*S_beta_2;
			S_beta_3 = alpha + (x_kap+N_Pf(0.666666666666667))*S_beta_3;
			S_res_0 = S_beta_0 + (y_kap+N_Pf(0.000000000000000))*S_res_0;
			S_res_1 = S_beta_1 + (y_kap+N_Pf(0.000000000000000))*S_res_1;
			S_res_2 = S_beta_2 + (y_kap+N_Pf(0.666666666666667))*S_res_2;
			S_res_3 = S_beta_3 + (y_kap+N_Pf(0.666666666666667))*S_res_3;

			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = S_res_0;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = S_res_1;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = S_res_2;
			}
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = S_res_3;
			}
			__syncthreads();

		}
	}
}

int Solver_LBM::S_Interpolate_Cubic_d2q9(int i_dev, int L, int var, ufloat_t tau_L, ufloat_t tau_ratio_L)
{
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_INTERFACE)
	{
		Cu_Interpolate_Cubic_d2q9<0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->dxf_vec[L], tau_L, tau_ratio_L, v0, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_ADDED)
	{
		Cu_Interpolate_Cubic_d2q9<1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->dxf_vec[L], tau_L, tau_ratio_L, v0, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif