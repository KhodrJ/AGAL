/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "solver.h"
#include "mesh.h"

#if (N_Q==19)

template <int interp_type=0>
__global__
void Cu_Interpolate_Cubic_d3q19
(
	int n_ids_idev_L, long int n_maxcells, int n_maxcblocks, ufloat_t v0, ufloat_t dx_L, ufloat_t tau_Lp1, ufloat_t tau_ratio, int *id_set_idev_L, int *cells_ID_mask, ufloat_t *cells_f_F, int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
	__shared__ int s_ID_cblock[M_TBLOCK];
	__shared__ ufloat_t s_rho[M_TBLOCK];
	__shared__ ufloat_t s_u[M_TBLOCK];
	__shared__ ufloat_t s_v[M_TBLOCK];
	__shared__ ufloat_t s_w[M_TBLOCK];
	int I_kap = threadIdx.x % Nbx;
	int J_kap = (threadIdx.x / Nbx) % Nbx;
	ufloat_t x_kap = N_Pf(-0.25) + N_Pf(0.5)*I_kap;
	ufloat_t y_kap = N_Pf(-0.25) + N_Pf(0.5)*J_kap;
	int K_kap = (threadIdx.x / Nbx) / Nbx;
	ufloat_t z_kap = N_Pf(-0.25) + N_Pf(0.5)*K_kap;
	int i_kap_b = -1;
	int i_kap_bc = -1;
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
	ufloat_t cdotu = N_Pf(0.0);
	ufloat_t udotu = N_Pf(0.0);
	ufloat_t tmp_i = N_Pf(0.0);
	ufloat_t rho = N_Pf(0.0);
	ufloat_t rho_x = N_Pf(0.0);
	ufloat_t rho_y = N_Pf(0.0);
	ufloat_t rho_z = N_Pf(0.0);
	ufloat_t u = N_Pf(0.0);
	ufloat_t u_x = N_Pf(0.0);
	ufloat_t u_y = N_Pf(0.0);
	ufloat_t u_z = N_Pf(0.0);
	ufloat_t u_xx = N_Pf(0.0);
	ufloat_t u_xy = N_Pf(0.0);
	ufloat_t u_zx = N_Pf(0.0);
	ufloat_t u_yy = N_Pf(0.0);
	ufloat_t u_yz = N_Pf(0.0);
	ufloat_t u_zz = N_Pf(0.0);
	ufloat_t v = N_Pf(0.0);
	ufloat_t v_x = N_Pf(0.0);
	ufloat_t v_y = N_Pf(0.0);
	ufloat_t v_z = N_Pf(0.0);
	ufloat_t v_xx = N_Pf(0.0);
	ufloat_t v_xy = N_Pf(0.0);
	ufloat_t v_xz = N_Pf(0.0);
	ufloat_t v_yy = N_Pf(0.0);
	ufloat_t v_yz = N_Pf(0.0);
	ufloat_t v_zz = N_Pf(0.0);
	ufloat_t w = N_Pf(0.0);
	ufloat_t w_x = N_Pf(0.0);
	ufloat_t w_y = N_Pf(0.0);
	ufloat_t w_z = N_Pf(0.0);
	ufloat_t w_xx = N_Pf(0.0);
	ufloat_t w_xy = N_Pf(0.0);
	ufloat_t w_xz = N_Pf(0.0);
	ufloat_t w_yy = N_Pf(0.0);
	ufloat_t w_yz = N_Pf(0.0);
	ufloat_t w_zz = N_Pf(0.0);
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
		if (i_kap_b > -1 && (((interp_type==0)and(block_on_interface==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE))))
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
			rho = f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 ;
			u = (N_Pf(0.0)*f_0 +N_Pf(1.0)*f_1 +N_Pf(-1.0)*f_2 +N_Pf(0.0)*f_3 +N_Pf(0.0)*f_4 +N_Pf(0.0)*f_5 +N_Pf(0.0)*f_6 +N_Pf(1.0)*f_7 +N_Pf(-1.0)*f_8 +N_Pf(1.0)*f_9 +N_Pf(-1.0)*f_10 +N_Pf(0.0)*f_11 +N_Pf(0.0)*f_12 +N_Pf(1.0)*f_13 +N_Pf(-1.0)*f_14 +N_Pf(1.0)*f_15 +N_Pf(-1.0)*f_16 +N_Pf(0.0)*f_17 +N_Pf(0.0)*f_18 ) / rho;
			v = (N_Pf(0.0)*f_0 +N_Pf(0.0)*f_1 +N_Pf(0.0)*f_2 +N_Pf(1.0)*f_3 +N_Pf(-1.0)*f_4 +N_Pf(0.0)*f_5 +N_Pf(0.0)*f_6 +N_Pf(1.0)*f_7 +N_Pf(-1.0)*f_8 +N_Pf(0.0)*f_9 +N_Pf(0.0)*f_10 +N_Pf(1.0)*f_11 +N_Pf(-1.0)*f_12 +N_Pf(-1.0)*f_13 +N_Pf(1.0)*f_14 +N_Pf(0.0)*f_15 +N_Pf(0.0)*f_16 +N_Pf(1.0)*f_17 +N_Pf(-1.0)*f_18 ) / rho;
			w = (N_Pf(0.0)*f_0+N_Pf(0.0)*f_1+N_Pf(0.0)*f_2+N_Pf(0.0)*f_3+N_Pf(0.0)*f_4+N_Pf(1.0)*f_5+N_Pf(-1.0)*f_6+N_Pf(0.0)*f_7+N_Pf(0.0)*f_8+N_Pf(1.0)*f_9+N_Pf(-1.0)*f_10+N_Pf(1.0)*f_11+N_Pf(-1.0)*f_12+N_Pf(0.0)*f_13+N_Pf(0.0)*f_14+N_Pf(-1.0)*f_15+N_Pf(1.0)*f_16+N_Pf(-1.0)*f_17+N_Pf(1.0)*f_18) / rho;
			s_u[threadIdx.x] = u;
			s_v[threadIdx.x] = v;
			s_rho[threadIdx.x] = rho;
			__syncthreads();
			//	Child 0.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+0)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+0)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			//	Child 1.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+1)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+1)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			//	Child 2.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+2)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+2)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			//	Child 3.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+3)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+3)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			//	Child 4.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+4)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+4)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			//	Child 5.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+5)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+5)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			//	Child 6.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+6)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(0.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+6)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			//	Child 7.
			if ((interp_type == 0 && cells_ID_mask[(i_kap_bc+7)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
			{
				x_kap = N_Pf(-0.083333333333333)+I_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);
				y_kap = N_Pf(-0.083333333333333)+J_kap*N_Pf(0.166666666666667)+N_Pf(1.000000000000000)*N_Pf(0.666666666666667);

				//
				// u
				//
				A00 = N_Pf(1.0000000000)*s_u[0];
				A10 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[1] + N_Pf(-4.5000000000)*s_u[2] + N_Pf(1.0000000000)*s_u[3];
				A20 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[1] + N_Pf(18.0000000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3];
				A30 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[1] + N_Pf(-13.5000000000)*s_u[2] + N_Pf(4.5000000000)*s_u[3];
				A01 = N_Pf(-5.5000000000)*s_u[0] + N_Pf(9.0000000000)*s_u[4] + N_Pf(-4.5000000000)*s_u[8] + N_Pf(1.0000000000)*s_u[12];
				A11 = N_Pf(30.2500000000)*s_u[0] + N_Pf(-49.5000000000)*s_u[1] + N_Pf(24.7500000000)*s_u[2] + N_Pf(-5.5000000000)*s_u[3] + N_Pf(-49.5000000000)*s_u[4] + N_Pf(81.0000000000)*s_u[5] + N_Pf(-40.5000000000)*s_u[6] + N_Pf(9.0000000000)*s_u[7] + N_Pf(24.7500000000)*s_u[8] + N_Pf(-40.5000000000)*s_u[9] + N_Pf(20.2500000000)*s_u[10] + N_Pf(-4.5000000000)*s_u[11] + N_Pf(-5.5000000000)*s_u[12] + N_Pf(9.0000000000)*s_u[13] + N_Pf(-4.5000000000)*s_u[14] + N_Pf(1.0000000000)*s_u[15];
				A21 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(123.7500000000)*s_u[1] + N_Pf(-99.0000000000)*s_u[2] + N_Pf(24.7500000000)*s_u[3] + N_Pf(81.0000000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(162.0000000000)*s_u[6] + N_Pf(-40.5000000000)*s_u[7] + N_Pf(-40.5000000000)*s_u[8] + N_Pf(101.2500000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(20.2500000000)*s_u[11] + N_Pf(9.0000000000)*s_u[12] + N_Pf(-22.5000000000)*s_u[13] + N_Pf(18.0000000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A31 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-74.2500000000)*s_u[1] + N_Pf(74.2500000000)*s_u[2] + N_Pf(-24.7500000000)*s_u[3] + N_Pf(-40.5000000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-121.5000000000)*s_u[6] + N_Pf(40.5000000000)*s_u[7] + N_Pf(20.2500000000)*s_u[8] + N_Pf(-60.7500000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-20.2500000000)*s_u[11] + N_Pf(-4.5000000000)*s_u[12] + N_Pf(13.5000000000)*s_u[13] + N_Pf(-13.5000000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A02 = N_Pf(9.0000000000)*s_u[0] + N_Pf(-22.5000000000)*s_u[4] + N_Pf(18.0000000000)*s_u[8] + N_Pf(-4.5000000000)*s_u[12];
				A12 = N_Pf(-49.5000000000)*s_u[0] + N_Pf(81.0000000000)*s_u[1] + N_Pf(-40.5000000000)*s_u[2] + N_Pf(9.0000000000)*s_u[3] + N_Pf(123.7500000000)*s_u[4] + N_Pf(-202.5000000000)*s_u[5] + N_Pf(101.2500000000)*s_u[6] + N_Pf(-22.5000000000)*s_u[7] + N_Pf(-99.0000000000)*s_u[8] + N_Pf(162.0000000000)*s_u[9] + N_Pf(-81.0000000000)*s_u[10] + N_Pf(18.0000000000)*s_u[11] + N_Pf(24.7500000000)*s_u[12] + N_Pf(-40.5000000000)*s_u[13] + N_Pf(20.2500000000)*s_u[14] + N_Pf(-4.5000000000)*s_u[15];
				A22 = N_Pf(81.0000000000)*s_u[0] + N_Pf(-202.5000000000)*s_u[1] + N_Pf(162.0000000000)*s_u[2] + N_Pf(-40.5000000000)*s_u[3] + N_Pf(-202.5000000000)*s_u[4] + N_Pf(506.2500000000)*s_u[5] + N_Pf(-405.0000000000)*s_u[6] + N_Pf(101.2500000000)*s_u[7] + N_Pf(162.0000000000)*s_u[8] + N_Pf(-405.0000000000)*s_u[9] + N_Pf(324.0000000000)*s_u[10] + N_Pf(-81.0000000000)*s_u[11] + N_Pf(-40.5000000000)*s_u[12] + N_Pf(101.2500000000)*s_u[13] + N_Pf(-81.0000000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				A32 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(121.5000000000)*s_u[1] + N_Pf(-121.5000000000)*s_u[2] + N_Pf(40.5000000000)*s_u[3] + N_Pf(101.2500000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(303.7500000000)*s_u[6] + N_Pf(-101.2500000000)*s_u[7] + N_Pf(-81.0000000000)*s_u[8] + N_Pf(243.0000000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(81.0000000000)*s_u[11] + N_Pf(20.2500000000)*s_u[12] + N_Pf(-60.7500000000)*s_u[13] + N_Pf(60.7500000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A03 = N_Pf(-4.5000000000)*s_u[0] + N_Pf(13.5000000000)*s_u[4] + N_Pf(-13.5000000000)*s_u[8] + N_Pf(4.5000000000)*s_u[12];
				A13 = N_Pf(24.7500000000)*s_u[0] + N_Pf(-40.5000000000)*s_u[1] + N_Pf(20.2500000000)*s_u[2] + N_Pf(-4.5000000000)*s_u[3] + N_Pf(-74.2500000000)*s_u[4] + N_Pf(121.5000000000)*s_u[5] + N_Pf(-60.7500000000)*s_u[6] + N_Pf(13.5000000000)*s_u[7] + N_Pf(74.2500000000)*s_u[8] + N_Pf(-121.5000000000)*s_u[9] + N_Pf(60.7500000000)*s_u[10] + N_Pf(-13.5000000000)*s_u[11] + N_Pf(-24.7500000000)*s_u[12] + N_Pf(40.5000000000)*s_u[13] + N_Pf(-20.2500000000)*s_u[14] + N_Pf(4.5000000000)*s_u[15];
				A23 = N_Pf(-40.5000000000)*s_u[0] + N_Pf(101.2500000000)*s_u[1] + N_Pf(-81.0000000000)*s_u[2] + N_Pf(20.2500000000)*s_u[3] + N_Pf(121.5000000000)*s_u[4] + N_Pf(-303.7500000000)*s_u[5] + N_Pf(243.0000000000)*s_u[6] + N_Pf(-60.7500000000)*s_u[7] + N_Pf(-121.5000000000)*s_u[8] + N_Pf(303.7500000000)*s_u[9] + N_Pf(-243.0000000000)*s_u[10] + N_Pf(60.7500000000)*s_u[11] + N_Pf(40.5000000000)*s_u[12] + N_Pf(-101.2500000000)*s_u[13] + N_Pf(81.0000000000)*s_u[14] + N_Pf(-20.2500000000)*s_u[15];
				A33 = N_Pf(20.2500000000)*s_u[0] + N_Pf(-60.7500000000)*s_u[1] + N_Pf(60.7500000000)*s_u[2] + N_Pf(-20.2500000000)*s_u[3] + N_Pf(-60.7500000000)*s_u[4] + N_Pf(182.2500000000)*s_u[5] + N_Pf(-182.2500000000)*s_u[6] + N_Pf(60.7500000000)*s_u[7] + N_Pf(60.7500000000)*s_u[8] + N_Pf(-182.2500000000)*s_u[9] + N_Pf(182.2500000000)*s_u[10] + N_Pf(-60.7500000000)*s_u[11] + N_Pf(-20.2500000000)*s_u[12] + N_Pf(60.7500000000)*s_u[13] + N_Pf(-60.7500000000)*s_u[14] + N_Pf(20.2500000000)*s_u[15];
				u = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				u_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				u_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				u_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				u_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// v
				A00 = N_Pf(1.0000000000)*s_v[0];
				A10 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[1] + N_Pf(-4.5000000000)*s_v[2] + N_Pf(1.0000000000)*s_v[3];
				A20 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[1] + N_Pf(18.0000000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3];
				A30 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[1] + N_Pf(-13.5000000000)*s_v[2] + N_Pf(4.5000000000)*s_v[3];
				A01 = N_Pf(-5.5000000000)*s_v[0] + N_Pf(9.0000000000)*s_v[4] + N_Pf(-4.5000000000)*s_v[8] + N_Pf(1.0000000000)*s_v[12];
				A11 = N_Pf(30.2500000000)*s_v[0] + N_Pf(-49.5000000000)*s_v[1] + N_Pf(24.7500000000)*s_v[2] + N_Pf(-5.5000000000)*s_v[3] + N_Pf(-49.5000000000)*s_v[4] + N_Pf(81.0000000000)*s_v[5] + N_Pf(-40.5000000000)*s_v[6] + N_Pf(9.0000000000)*s_v[7] + N_Pf(24.7500000000)*s_v[8] + N_Pf(-40.5000000000)*s_v[9] + N_Pf(20.2500000000)*s_v[10] + N_Pf(-4.5000000000)*s_v[11] + N_Pf(-5.5000000000)*s_v[12] + N_Pf(9.0000000000)*s_v[13] + N_Pf(-4.5000000000)*s_v[14] + N_Pf(1.0000000000)*s_v[15];
				A21 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(123.7500000000)*s_v[1] + N_Pf(-99.0000000000)*s_v[2] + N_Pf(24.7500000000)*s_v[3] + N_Pf(81.0000000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(162.0000000000)*s_v[6] + N_Pf(-40.5000000000)*s_v[7] + N_Pf(-40.5000000000)*s_v[8] + N_Pf(101.2500000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(20.2500000000)*s_v[11] + N_Pf(9.0000000000)*s_v[12] + N_Pf(-22.5000000000)*s_v[13] + N_Pf(18.0000000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A31 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-74.2500000000)*s_v[1] + N_Pf(74.2500000000)*s_v[2] + N_Pf(-24.7500000000)*s_v[3] + N_Pf(-40.5000000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-121.5000000000)*s_v[6] + N_Pf(40.5000000000)*s_v[7] + N_Pf(20.2500000000)*s_v[8] + N_Pf(-60.7500000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-20.2500000000)*s_v[11] + N_Pf(-4.5000000000)*s_v[12] + N_Pf(13.5000000000)*s_v[13] + N_Pf(-13.5000000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A02 = N_Pf(9.0000000000)*s_v[0] + N_Pf(-22.5000000000)*s_v[4] + N_Pf(18.0000000000)*s_v[8] + N_Pf(-4.5000000000)*s_v[12];
				A12 = N_Pf(-49.5000000000)*s_v[0] + N_Pf(81.0000000000)*s_v[1] + N_Pf(-40.5000000000)*s_v[2] + N_Pf(9.0000000000)*s_v[3] + N_Pf(123.7500000000)*s_v[4] + N_Pf(-202.5000000000)*s_v[5] + N_Pf(101.2500000000)*s_v[6] + N_Pf(-22.5000000000)*s_v[7] + N_Pf(-99.0000000000)*s_v[8] + N_Pf(162.0000000000)*s_v[9] + N_Pf(-81.0000000000)*s_v[10] + N_Pf(18.0000000000)*s_v[11] + N_Pf(24.7500000000)*s_v[12] + N_Pf(-40.5000000000)*s_v[13] + N_Pf(20.2500000000)*s_v[14] + N_Pf(-4.5000000000)*s_v[15];
				A22 = N_Pf(81.0000000000)*s_v[0] + N_Pf(-202.5000000000)*s_v[1] + N_Pf(162.0000000000)*s_v[2] + N_Pf(-40.5000000000)*s_v[3] + N_Pf(-202.5000000000)*s_v[4] + N_Pf(506.2500000000)*s_v[5] + N_Pf(-405.0000000000)*s_v[6] + N_Pf(101.2500000000)*s_v[7] + N_Pf(162.0000000000)*s_v[8] + N_Pf(-405.0000000000)*s_v[9] + N_Pf(324.0000000000)*s_v[10] + N_Pf(-81.0000000000)*s_v[11] + N_Pf(-40.5000000000)*s_v[12] + N_Pf(101.2500000000)*s_v[13] + N_Pf(-81.0000000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				A32 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(121.5000000000)*s_v[1] + N_Pf(-121.5000000000)*s_v[2] + N_Pf(40.5000000000)*s_v[3] + N_Pf(101.2500000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(303.7500000000)*s_v[6] + N_Pf(-101.2500000000)*s_v[7] + N_Pf(-81.0000000000)*s_v[8] + N_Pf(243.0000000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(81.0000000000)*s_v[11] + N_Pf(20.2500000000)*s_v[12] + N_Pf(-60.7500000000)*s_v[13] + N_Pf(60.7500000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A03 = N_Pf(-4.5000000000)*s_v[0] + N_Pf(13.5000000000)*s_v[4] + N_Pf(-13.5000000000)*s_v[8] + N_Pf(4.5000000000)*s_v[12];
				A13 = N_Pf(24.7500000000)*s_v[0] + N_Pf(-40.5000000000)*s_v[1] + N_Pf(20.2500000000)*s_v[2] + N_Pf(-4.5000000000)*s_v[3] + N_Pf(-74.2500000000)*s_v[4] + N_Pf(121.5000000000)*s_v[5] + N_Pf(-60.7500000000)*s_v[6] + N_Pf(13.5000000000)*s_v[7] + N_Pf(74.2500000000)*s_v[8] + N_Pf(-121.5000000000)*s_v[9] + N_Pf(60.7500000000)*s_v[10] + N_Pf(-13.5000000000)*s_v[11] + N_Pf(-24.7500000000)*s_v[12] + N_Pf(40.5000000000)*s_v[13] + N_Pf(-20.2500000000)*s_v[14] + N_Pf(4.5000000000)*s_v[15];
				A23 = N_Pf(-40.5000000000)*s_v[0] + N_Pf(101.2500000000)*s_v[1] + N_Pf(-81.0000000000)*s_v[2] + N_Pf(20.2500000000)*s_v[3] + N_Pf(121.5000000000)*s_v[4] + N_Pf(-303.7500000000)*s_v[5] + N_Pf(243.0000000000)*s_v[6] + N_Pf(-60.7500000000)*s_v[7] + N_Pf(-121.5000000000)*s_v[8] + N_Pf(303.7500000000)*s_v[9] + N_Pf(-243.0000000000)*s_v[10] + N_Pf(60.7500000000)*s_v[11] + N_Pf(40.5000000000)*s_v[12] + N_Pf(-101.2500000000)*s_v[13] + N_Pf(81.0000000000)*s_v[14] + N_Pf(-20.2500000000)*s_v[15];
				A33 = N_Pf(20.2500000000)*s_v[0] + N_Pf(-60.7500000000)*s_v[1] + N_Pf(60.7500000000)*s_v[2] + N_Pf(-20.2500000000)*s_v[3] + N_Pf(-60.7500000000)*s_v[4] + N_Pf(182.2500000000)*s_v[5] + N_Pf(-182.2500000000)*s_v[6] + N_Pf(60.7500000000)*s_v[7] + N_Pf(60.7500000000)*s_v[8] + N_Pf(-182.2500000000)*s_v[9] + N_Pf(182.2500000000)*s_v[10] + N_Pf(-60.7500000000)*s_v[11] + N_Pf(-20.2500000000)*s_v[12] + N_Pf(60.7500000000)*s_v[13] + N_Pf(-60.7500000000)*s_v[14] + N_Pf(20.2500000000)*s_v[15];
				v = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				v_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				v_xx = ((N_Pf(2.0)*A20 + x_kap*(N_Pf(6.0)*A30)) + y_kap*((N_Pf(2.0)*A21 + x_kap*(N_Pf(6.0)*A31)) + y_kap*((N_Pf(2.0)*A22 + x_kap*(N_Pf(6.0)*A32)) + y_kap*((N_Pf(2.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));
				v_xy = ((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((N_Pf(2.0)*A12 + x_kap*(N_Pf(4.0)*A22 + x_kap*(N_Pf(6.0)*A32))) + y_kap*((N_Pf(3.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(9.0)*A33))))));
				v_yy = ((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(6.0)*A03 + x_kap*(N_Pf(6.0)*A13 + x_kap*(N_Pf(6.0)*A23 + x_kap*(N_Pf(6.0)*A33))))));

				// rho
				A00 = N_Pf(1.0000000000)*s_rho[0];
				A10 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[1] + N_Pf(-4.5000000000)*s_rho[2] + N_Pf(1.0000000000)*s_rho[3];
				A20 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[1] + N_Pf(18.0000000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3];
				A30 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[1] + N_Pf(-13.5000000000)*s_rho[2] + N_Pf(4.5000000000)*s_rho[3];
				A01 = N_Pf(-5.5000000000)*s_rho[0] + N_Pf(9.0000000000)*s_rho[4] + N_Pf(-4.5000000000)*s_rho[8] + N_Pf(1.0000000000)*s_rho[12];
				A11 = N_Pf(30.2500000000)*s_rho[0] + N_Pf(-49.5000000000)*s_rho[1] + N_Pf(24.7500000000)*s_rho[2] + N_Pf(-5.5000000000)*s_rho[3] + N_Pf(-49.5000000000)*s_rho[4] + N_Pf(81.0000000000)*s_rho[5] + N_Pf(-40.5000000000)*s_rho[6] + N_Pf(9.0000000000)*s_rho[7] + N_Pf(24.7500000000)*s_rho[8] + N_Pf(-40.5000000000)*s_rho[9] + N_Pf(20.2500000000)*s_rho[10] + N_Pf(-4.5000000000)*s_rho[11] + N_Pf(-5.5000000000)*s_rho[12] + N_Pf(9.0000000000)*s_rho[13] + N_Pf(-4.5000000000)*s_rho[14] + N_Pf(1.0000000000)*s_rho[15];
				A21 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(123.7500000000)*s_rho[1] + N_Pf(-99.0000000000)*s_rho[2] + N_Pf(24.7500000000)*s_rho[3] + N_Pf(81.0000000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(162.0000000000)*s_rho[6] + N_Pf(-40.5000000000)*s_rho[7] + N_Pf(-40.5000000000)*s_rho[8] + N_Pf(101.2500000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(20.2500000000)*s_rho[11] + N_Pf(9.0000000000)*s_rho[12] + N_Pf(-22.5000000000)*s_rho[13] + N_Pf(18.0000000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A31 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-74.2500000000)*s_rho[1] + N_Pf(74.2500000000)*s_rho[2] + N_Pf(-24.7500000000)*s_rho[3] + N_Pf(-40.5000000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-121.5000000000)*s_rho[6] + N_Pf(40.5000000000)*s_rho[7] + N_Pf(20.2500000000)*s_rho[8] + N_Pf(-60.7500000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-20.2500000000)*s_rho[11] + N_Pf(-4.5000000000)*s_rho[12] + N_Pf(13.5000000000)*s_rho[13] + N_Pf(-13.5000000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A02 = N_Pf(9.0000000000)*s_rho[0] + N_Pf(-22.5000000000)*s_rho[4] + N_Pf(18.0000000000)*s_rho[8] + N_Pf(-4.5000000000)*s_rho[12];
				A12 = N_Pf(-49.5000000000)*s_rho[0] + N_Pf(81.0000000000)*s_rho[1] + N_Pf(-40.5000000000)*s_rho[2] + N_Pf(9.0000000000)*s_rho[3] + N_Pf(123.7500000000)*s_rho[4] + N_Pf(-202.5000000000)*s_rho[5] + N_Pf(101.2500000000)*s_rho[6] + N_Pf(-22.5000000000)*s_rho[7] + N_Pf(-99.0000000000)*s_rho[8] + N_Pf(162.0000000000)*s_rho[9] + N_Pf(-81.0000000000)*s_rho[10] + N_Pf(18.0000000000)*s_rho[11] + N_Pf(24.7500000000)*s_rho[12] + N_Pf(-40.5000000000)*s_rho[13] + N_Pf(20.2500000000)*s_rho[14] + N_Pf(-4.5000000000)*s_rho[15];
				A22 = N_Pf(81.0000000000)*s_rho[0] + N_Pf(-202.5000000000)*s_rho[1] + N_Pf(162.0000000000)*s_rho[2] + N_Pf(-40.5000000000)*s_rho[3] + N_Pf(-202.5000000000)*s_rho[4] + N_Pf(506.2500000000)*s_rho[5] + N_Pf(-405.0000000000)*s_rho[6] + N_Pf(101.2500000000)*s_rho[7] + N_Pf(162.0000000000)*s_rho[8] + N_Pf(-405.0000000000)*s_rho[9] + N_Pf(324.0000000000)*s_rho[10] + N_Pf(-81.0000000000)*s_rho[11] + N_Pf(-40.5000000000)*s_rho[12] + N_Pf(101.2500000000)*s_rho[13] + N_Pf(-81.0000000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				A32 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(121.5000000000)*s_rho[1] + N_Pf(-121.5000000000)*s_rho[2] + N_Pf(40.5000000000)*s_rho[3] + N_Pf(101.2500000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(303.7500000000)*s_rho[6] + N_Pf(-101.2500000000)*s_rho[7] + N_Pf(-81.0000000000)*s_rho[8] + N_Pf(243.0000000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(81.0000000000)*s_rho[11] + N_Pf(20.2500000000)*s_rho[12] + N_Pf(-60.7500000000)*s_rho[13] + N_Pf(60.7500000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A03 = N_Pf(-4.5000000000)*s_rho[0] + N_Pf(13.5000000000)*s_rho[4] + N_Pf(-13.5000000000)*s_rho[8] + N_Pf(4.5000000000)*s_rho[12];
				A13 = N_Pf(24.7500000000)*s_rho[0] + N_Pf(-40.5000000000)*s_rho[1] + N_Pf(20.2500000000)*s_rho[2] + N_Pf(-4.5000000000)*s_rho[3] + N_Pf(-74.2500000000)*s_rho[4] + N_Pf(121.5000000000)*s_rho[5] + N_Pf(-60.7500000000)*s_rho[6] + N_Pf(13.5000000000)*s_rho[7] + N_Pf(74.2500000000)*s_rho[8] + N_Pf(-121.5000000000)*s_rho[9] + N_Pf(60.7500000000)*s_rho[10] + N_Pf(-13.5000000000)*s_rho[11] + N_Pf(-24.7500000000)*s_rho[12] + N_Pf(40.5000000000)*s_rho[13] + N_Pf(-20.2500000000)*s_rho[14] + N_Pf(4.5000000000)*s_rho[15];
				A23 = N_Pf(-40.5000000000)*s_rho[0] + N_Pf(101.2500000000)*s_rho[1] + N_Pf(-81.0000000000)*s_rho[2] + N_Pf(20.2500000000)*s_rho[3] + N_Pf(121.5000000000)*s_rho[4] + N_Pf(-303.7500000000)*s_rho[5] + N_Pf(243.0000000000)*s_rho[6] + N_Pf(-60.7500000000)*s_rho[7] + N_Pf(-121.5000000000)*s_rho[8] + N_Pf(303.7500000000)*s_rho[9] + N_Pf(-243.0000000000)*s_rho[10] + N_Pf(60.7500000000)*s_rho[11] + N_Pf(40.5000000000)*s_rho[12] + N_Pf(-101.2500000000)*s_rho[13] + N_Pf(81.0000000000)*s_rho[14] + N_Pf(-20.2500000000)*s_rho[15];
				A33 = N_Pf(20.2500000000)*s_rho[0] + N_Pf(-60.7500000000)*s_rho[1] + N_Pf(60.7500000000)*s_rho[2] + N_Pf(-20.2500000000)*s_rho[3] + N_Pf(-60.7500000000)*s_rho[4] + N_Pf(182.2500000000)*s_rho[5] + N_Pf(-182.2500000000)*s_rho[6] + N_Pf(60.7500000000)*s_rho[7] + N_Pf(60.7500000000)*s_rho[8] + N_Pf(-182.2500000000)*s_rho[9] + N_Pf(182.2500000000)*s_rho[10] + N_Pf(-60.7500000000)*s_rho[11] + N_Pf(-20.2500000000)*s_rho[12] + N_Pf(60.7500000000)*s_rho[13] + N_Pf(-60.7500000000)*s_rho[14] + N_Pf(20.2500000000)*s_rho[15];
				rho = ((A00 + x_kap*(A10 + x_kap*(A20 + x_kap*(A30)))) + y_kap*((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((A02 + x_kap*(A12 + x_kap*(A22 + x_kap*(A32)))) + y_kap*((A03 + x_kap*(A13 + x_kap*(A23 + x_kap*(A33))))))));
				rho_x = ((A10 + x_kap*(N_Pf(2.0)*A20 + x_kap*(N_Pf(3.0)*A30))) + y_kap*((A11 + x_kap*(N_Pf(2.0)*A21 + x_kap*(N_Pf(3.0)*A31))) + y_kap*((A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(3.0)*A32))) + y_kap*((A13 + x_kap*(N_Pf(2.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));
				rho_y = ((A01 + x_kap*(A11 + x_kap*(A21 + x_kap*(A31)))) + y_kap*((N_Pf(2.0)*A02 + x_kap*(N_Pf(2.0)*A12 + x_kap*(N_Pf(2.0)*A22 + x_kap*(N_Pf(2.0)*A32)))) + y_kap*((N_Pf(3.0)*A03 + x_kap*(N_Pf(3.0)*A13 + x_kap*(N_Pf(3.0)*A23 + x_kap*(N_Pf(3.0)*A33)))))));

				// aux
				udotu = u*u + v*v;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_0 = (N_Pf(0.333333333333333)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.333333333333333)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 0*n_maxcells] = f_0;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_1 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 2*n_maxcells] = f_1;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_2 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 1*n_maxcells] = f_2;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_3 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 4*n_maxcells] = f_3;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_4 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 3*n_maxcells] = f_4;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_5 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 6*n_maxcells] = f_5;

				cdotu = N_Pf(0.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_6 = (N_Pf(0.055555555555556)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.055555555555556)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 5*n_maxcells] = f_6;

				cdotu = N_Pf(1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_7 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 8*n_maxcells] = f_7;

				cdotu = N_Pf(-1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_8 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 7*n_maxcells] = f_8;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_9 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 10*n_maxcells] = f_9;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_10 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 9*n_maxcells] = f_10;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_11 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 12*n_maxcells] = f_11;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_12 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 11*n_maxcells] = f_12;

				cdotu = N_Pf(1.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_13 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 14*n_maxcells] = f_13;

				cdotu = N_Pf(-1.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(1.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_14 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 13*n_maxcells] = f_14;

				cdotu = N_Pf(1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_15 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 16*n_maxcells] = f_15;

				cdotu = N_Pf(-1.0)*u + N_Pf(0.0)*v;
				aux_1xx = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(0.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(-1.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*u_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(-1.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*v_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(0.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(-1.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(0.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_16 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 15*n_maxcells] = f_16;

				cdotu = N_Pf(0.0)*u + N_Pf(1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(1.0)*N_Pf(1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_17 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 18*n_maxcells] = f_17;

				cdotu = N_Pf(0.0)*u + N_Pf(-1.0)*v;
				aux_1xx = (   (N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*rho*u_x + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(0.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x) + N_Pf(-1.0)*(rho_y*u*u + rho*u_y*u + rho*u*u_y))   ) - (N_Pf(0.0)*(rho_x*u*u + rho*u_x*u + rho*u*u_x));
				aux_1xy = (   (N_Pf(0.0)*N_Pf(-1.0))*rho*u_y + N_Pf(1.5)*(N_Pf(0.0)*N_Pf(-1.0))*(N_Pf(0.0)*(rho_x*u*v + rho*u_x*v + rho*u*v_x) + N_Pf(-1.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y))   ) - (N_Pf(0.0)*(rho_y*u*v + rho*u_y*v + rho*u*v_y));
				aux_1yx = (   (N_Pf(-1.0)*N_Pf(0.0))*rho*v_x + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(0.0))*(N_Pf(0.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x) + N_Pf(-1.0)*(rho_y*v*u + rho*v_y*u + rho*v*u_y))   ) - (N_Pf(-1.0)*(rho_x*v*u + rho*v_x*u + rho*v*u_x));
				aux_1yy = (   (N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*rho*v_y + N_Pf(1.5)*(N_Pf(-1.0)*N_Pf(-1.0)-N_Pf(0.3333333333333333))*(N_Pf(0.0)*(rho_x*v*v + rho*v_x*v + rho*v*v_x) + N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y))   ) - (N_Pf(-1.0)*(rho_y*v*v + rho*v_y*v + rho*v*v_y));
				f_18 = (N_Pf(0.027777777777778)*rho*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)) - tau_Lp1*N_Pf(0.027777777777778)*N_Pf(3.0)*(aux_1xx + aux_1xy + aux_1yx + aux_1yy); 
				cells_f_F[(i_kap_bc+7)*M_CBLOCK + threadIdx.x + 17*n_maxcells] = f_18;

			}
			__syncthreads();
		}
	}
}

int Solver_LBM::S_Interpolate_Cubic_d3q19(int i_dev, int L, int var, ufloat_t tau_L, ufloat_t tau_ratio_L)
{
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_INTERFACE)
	{
		Cu_Interpolate_Cubic_d3q19<0><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, v0, dxf_vec[L], tau_vec[L+1], tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}
	if (mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_ADDED)
	{
		Cu_Interpolate_Cubic_d3q19<1><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, v0, dxf_vec[L], tau_vec[L+1], tau_ratio_L, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cblock_ID_ref[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev]);
	}

	return 0;
}

#endif