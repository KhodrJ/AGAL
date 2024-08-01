#include "mesh.h"
#include "solver.h"

int Solver_LBM::S_ComputeProperties(int i_dev, int i_kap, ufloat_t dx_L, double *out)
{
	// Density and velocity computations.	
	for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
	{
#if (N_Q==9)
		ufloat_t f_0 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0L*n_maxcells];
		ufloat_t f_1 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3L*n_maxcells];
		ufloat_t f_2 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4L*n_maxcells];
		ufloat_t f_3 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1L*n_maxcells];
		ufloat_t f_4 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2L*n_maxcells];
		ufloat_t f_5 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7L*n_maxcells];
		ufloat_t f_6 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8L*n_maxcells];
		ufloat_t f_7 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5L*n_maxcells];
		ufloat_t f_8 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6L*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8;
		ufloat_t u_kap = ( +f_1 -f_3 +f_5 -f_6 -f_7 +f_8) / rho_kap;
		ufloat_t v_kap = ( +f_2 -f_4 +f_5 +f_6 -f_7 -f_8) / rho_kap;
		
		out[kap_i + 0*M_CBLOCK] = (double)rho_kap;
		out[kap_i + 1*M_CBLOCK] = (double)u_kap;
		out[kap_i + 2*M_CBLOCK] = (double)v_kap;
		out[kap_i + 3*M_CBLOCK] = 0.0;
#elif (N_Q==19)
		ufloat_t f_0 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0L*n_maxcells];
		ufloat_t f_1 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2L*n_maxcells];
		ufloat_t f_2 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1L*n_maxcells];
		ufloat_t f_3 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4L*n_maxcells];
		ufloat_t f_4 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3L*n_maxcells];
		ufloat_t f_5 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6L*n_maxcells];
		ufloat_t f_6 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5L*n_maxcells];
		ufloat_t f_7 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8L*n_maxcells];
		ufloat_t f_8 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7L*n_maxcells];
		ufloat_t f_9 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10L*n_maxcells];
		ufloat_t f_10 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9L*n_maxcells];
		ufloat_t f_11 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12L*n_maxcells];
		ufloat_t f_12 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11L*n_maxcells];
		ufloat_t f_13 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14L*n_maxcells];
		ufloat_t f_14 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13L*n_maxcells];
		ufloat_t f_15 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16L*n_maxcells];
		ufloat_t f_16 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15L*n_maxcells];
		ufloat_t f_17 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18L*n_maxcells];
		ufloat_t f_18 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17L*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18;
		ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / rho_kap;
		ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / rho_kap;
		ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / rho_kap;
		
		out[kap_i + 0*M_CBLOCK] = (double)rho_kap;
		out[kap_i + 1*M_CBLOCK] = (double)u_kap;
		out[kap_i + 2*M_CBLOCK] = (double)v_kap;
		out[kap_i + 3*M_CBLOCK] = (double)w_kap;
#else // (N_Q==27)
		ufloat_t f_0 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0L*n_maxcells];
		ufloat_t f_1 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2L*n_maxcells];
		ufloat_t f_2 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1L*n_maxcells];
		ufloat_t f_3 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4L*n_maxcells];
		ufloat_t f_4 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3L*n_maxcells];
		ufloat_t f_5 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6L*n_maxcells];
		ufloat_t f_6 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5L*n_maxcells];
		ufloat_t f_7 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8L*n_maxcells];
		ufloat_t f_8 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7L*n_maxcells];
		ufloat_t f_9 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10L*n_maxcells];
		ufloat_t f_10 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9L*n_maxcells];
		ufloat_t f_11 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12L*n_maxcells];
		ufloat_t f_12 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11L*n_maxcells];
		ufloat_t f_13 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14L*n_maxcells];
		ufloat_t f_14 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13L*n_maxcells];
		ufloat_t f_15 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16L*n_maxcells];
		ufloat_t f_16 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15L*n_maxcells];
		ufloat_t f_17 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18L*n_maxcells];
		ufloat_t f_18 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17L*n_maxcells];
		ufloat_t f_19 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 20L*n_maxcells];
		ufloat_t f_20 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 19L*n_maxcells];
		ufloat_t f_21 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 22L*n_maxcells];
		ufloat_t f_22 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 21L*n_maxcells];
		ufloat_t f_23 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 24L*n_maxcells];
		ufloat_t f_24 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 23L*n_maxcells];
		ufloat_t f_25 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 26L*n_maxcells];
		ufloat_t f_26 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 25L*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26;
		ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16 +f_19 -f_20 +f_21 -f_22 +f_23 -f_24 -f_25 +f_26) / rho_kap;
		ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18 +f_19 -f_20 +f_21 -f_22 -f_23 +f_24 +f_25 -f_26) / rho_kap;
		ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18 +f_19 -f_20 -f_21 +f_22 +f_23 -f_24 +f_25 -f_26) / rho_kap;
		
		out[kap_i + 0*M_CBLOCK] = (double)rho_kap;
		out[kap_i + 1*M_CBLOCK] = (double)u_kap;
		out[kap_i + 2*M_CBLOCK] = (double)v_kap;
		out[kap_i + 3*M_CBLOCK] = (double)w_kap;
#endif
	}

	return 0;
}

int Solver_LBM::S_ComputeOutputProperties(int i_dev, int i_kap, ufloat_t dx_L, double *out)
{
	if (S_LES == 0)
	{
		S_ComputeProperties(i_dev, i_kap, dx_L, out);
	}
	else
	{
		// Density and velocity identification.
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
			out[kap_i + 0*M_CBLOCK] = mesh->cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 0*mesh->n_ids[i_dev][0]*M_CBLOCK];
			out[kap_i + 1*M_CBLOCK] = mesh->cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 1*mesh->n_ids[i_dev][0]*M_CBLOCK];
			out[kap_i + 2*M_CBLOCK] = mesh->cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*mesh->n_ids[i_dev][0]*M_CBLOCK];
			out[kap_i + 3*M_CBLOCK] = mesh->cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 3*mesh->n_ids[i_dev][0]*M_CBLOCK];
		}
		
		// Vorticity calculation.
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
			int I_kap = kap_i % Nbx;
			int J_kap = (kap_i / Nbx) % Nbx;
	#if (N_DIM==3)
			int K_kap = (kap_i / Nbx) / Nbx;
	#endif
			//ufloat_t dU = N_Pf(0.0);
			//ufloat_t dV = N_Pf(0.0);
	#if (N_DIM==3)
			//ufloat_t dW = N_Pf(0.0);
	#endif
			
			
			
			out[kap_i + 4*M_CBLOCK] = 0.0;
			out[kap_i + 5*M_CBLOCK] = 0.0;
			out[kap_i + 6*M_CBLOCK] = 0.0;
	#if (N_DIM==2)
			// X
			if (I_kap < Nbx-1)
				out[kap_i + 6*M_CBLOCK] += (double)(out[(I_kap+1)+Nbx*(J_kap) + 2*M_CBLOCK] - out[kap_i + 2*M_CBLOCK])/dx_L;
			else
				out[kap_i + 6*M_CBLOCK] += (double)(out[kap_i + 2*M_CBLOCK] - out[(I_kap-1)+Nbx*(J_kap) + 2*M_CBLOCK])/dx_L;
			
			// Y
			if (J_kap < Nbx-1)
				out[kap_i + 6*M_CBLOCK] -= (double)(out[(I_kap)+Nbx*(J_kap+1) + 1*M_CBLOCK] - out[kap_i + 1*M_CBLOCK])/dx_L;
			else
				out[kap_i + 6*M_CBLOCK] -= (double)(out[kap_i + 1*M_CBLOCK] - out[(I_kap)+Nbx*(J_kap-1) + 1*M_CBLOCK])/dx_L;
	#else
			// X
			if (I_kap < Nbx-1)
			{
				out[kap_i + 5*M_CBLOCK] -= (double)(out[(I_kap+1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap) + 3*M_CBLOCK] - out[kap_i + 3*M_CBLOCK])/dx_L;
				out[kap_i + 6*M_CBLOCK] += (double)(out[(I_kap+1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap) + 2*M_CBLOCK] - out[kap_i + 2*M_CBLOCK])/dx_L;
			}
			else
			{
				out[kap_i + 5*M_CBLOCK] -= (double)(out[kap_i + 3*M_CBLOCK] - out[(I_kap-1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap) + 3*M_CBLOCK])/dx_L;
				out[kap_i + 6*M_CBLOCK] += (double)(out[kap_i + 2*M_CBLOCK] - out[(I_kap-1)+Nbx*(J_kap)+Nbx*Nbx*(K_kap) + 2*M_CBLOCK])/dx_L;
			}
			
			// Y
			if (J_kap < Nbx-1)
			{
				out[kap_i + 4*M_CBLOCK] += (double)(out[(I_kap)+Nbx*(J_kap+1)+Nbx*Nbx*(K_kap) + 3*M_CBLOCK] - out[kap_i + 3*M_CBLOCK])/dx_L;
				out[kap_i + 6*M_CBLOCK] -= (double)(out[(I_kap)+Nbx*(J_kap+1)+Nbx*Nbx*(K_kap) + 1*M_CBLOCK] - out[kap_i + 1*M_CBLOCK])/dx_L;
			}
			else
			{
				out[kap_i + 4*M_CBLOCK] += (double)(out[kap_i + 3*M_CBLOCK] - out[(I_kap)+Nbx*(J_kap-1)+Nbx*Nbx*(K_kap) + 3*M_CBLOCK])/dx_L;
				out[kap_i + 6*M_CBLOCK] -= (double)(out[kap_i + 1*M_CBLOCK] - out[(I_kap)+Nbx*(J_kap-1)+Nbx*Nbx*(K_kap) + 1*M_CBLOCK])/dx_L;
			}
			
			// Z
			if (K_kap < Nbx-1)
			{
				out[kap_i + 4*M_CBLOCK] -= (double)(out[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap+1) + 2*M_CBLOCK] - out[kap_i + 2*M_CBLOCK])/dx_L;
				out[kap_i + 5*M_CBLOCK] += (double)(out[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap+1) + 1*M_CBLOCK] - out[kap_i + 1*M_CBLOCK])/dx_L;
			}
			else
			{
				out[kap_i + 4*M_CBLOCK] -= (double)(out[kap_i + 2*M_CBLOCK] - out[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap-1) + 2*M_CBLOCK])/dx_L;
				out[kap_i + 5*M_CBLOCK] += (double)(out[kap_i + 1*M_CBLOCK] - out[(I_kap)+Nbx*(J_kap)+Nbx*Nbx*(K_kap-1) + 1*M_CBLOCK])/dx_L;
			}
	#endif
		}
		
		// Calculation of y+.
		if (mesh->cblock_ID_onb[i_dev][i_kap] == 1)
		{
			for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
			{
				int I_kap = kap_i % Nbx;
				int J_kap = (kap_i / Nbx) % Nbx;
#if (N_DIM==3)
				int K_kap = (kap_i / Nbx) / Nbx;
#endif
				double tau_w = 0.0;
				double yplus = 1000.0;
				
#if (N_DIM==2)
				if (mesh->cblock_ID_nbr[i_dev][i_kap + 1*n_maxcblocks] < 0)
				{
					tau_w = ( -N_Pf(71.0)*out[4*J_kap+3 + 2*M_CBLOCK] + N_Pf(141.0)*out[4*J_kap+2 + 2*M_CBLOCK] - N_Pf(93.0)*out[4*J_kap+1 + 2*M_CBLOCK] + N_Pf(23.0)*out[4*J_kap + 2*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
					yplus = std::min( (0.5*dx_L + (3-I_kap)*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
				}
				if (mesh->cblock_ID_nbr[i_dev][i_kap + 2*n_maxcblocks] < 0)
				{
					tau_w = ( -N_Pf(71.0)*out[I_kap+12 + 1*M_CBLOCK] + N_Pf(141.0)*out[I_kap+8 + 1*M_CBLOCK] - N_Pf(93.0)*out[I_kap+4 + 1*M_CBLOCK] + N_Pf(23.0)*out[I_kap + 1*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
					yplus = std::min( (0.5*dx_L + (3-J_kap)*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
				}
				if (mesh->cblock_ID_nbr[i_dev][i_kap + 3*n_maxcblocks] < 0)
				{
					tau_w = ( -N_Pf(71.0)*out[4*J_kap + 2*M_CBLOCK] + N_Pf(141.0)*out[4*J_kap+1 + 2*M_CBLOCK] - N_Pf(93.0)*out[4*J_kap+2 + 2*M_CBLOCK] + N_Pf(23.0)*out[4*J_kap+3 + 2*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
					yplus = std::min( (0.5*dx_L + I_kap*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
				}
				if (mesh->cblock_ID_nbr[i_dev][i_kap + 4*n_maxcblocks] < 0)
				{
					tau_w = ( -N_Pf(71.0)*out[I_kap + 1*M_CBLOCK] + N_Pf(141.0)*out[I_kap+4 + 1*M_CBLOCK] - N_Pf(93.0)*out[I_kap+8 + 1*M_CBLOCK] + N_Pf(23.0)*out[I_kap+12 + 1*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
					yplus = std::min( (0.5*dx_L + J_kap*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
				}
				out[kap_i + 7*M_CBLOCK] = yplus;
#else
				if (mesh->cblock_ID_nbr[i_dev][i_kap + 1*n_maxcblocks] < 0)
				{
					
				}
				
				
				//if (mesh->cblock_ID_nbr[i_dev][i_kap + 1*n_maxcblocks] < 0)
				//{
				//	yplus[kap_i] = ( -N_Pf(71.0)*out_kap[4*J_kap+3 + 1*M_CBLOCK] + N_Pf(141.0)*s_u[4*J_kap+2 + 1*M_CBLOCK] - N_Pf(93.0)*s_u[4*J_kap+1 + 1*M_CBLOCK] + N_Pf(23.0)*s_u[4*J_kap + 1*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
				//}
				//yplus[kap_i] = (0.5*dx_L + J_kap*dx_L)*sqrt(yplus[kap_i] / v0); // y+
#endif
			}
		}
		else
		{
			for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++) out[kap_i + 7*M_CBLOCK] = -1.0;
		}
	}
	
	return 0;
}

int Solver_LBM::S_ComputeForces(int i_dev, int L, std::ofstream *out)
{
	ufloat_t Fxp = N_Pf(0.0);
	ufloat_t Fyp = N_Pf(0.0);
	ufloat_t Fxm = N_Pf(0.0);
	ufloat_t Fym = N_Pf(0.0);	
	ufloat_t dx_L = mesh->dxf_vec[L];
	int total_blocks = 0;
	
	for (int kap = 0; kap < mesh->n_ids[i_dev][L]; kap++)
	{
		int i_kap_b = mesh->id_set[i_dev][L * n_maxcblocks + kap];
		bool block_involved = false;
		
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
			int I_kap = kap_i % Nbx;
			int J_kap = (kap_i / Nbx) % Nbx;
			
			//ufloat_t f_0 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_2 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_3 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_4 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_5 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_6 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_7 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_8 = mesh->cells_f_F[i_dev][i_kap_b*M_CBLOCK + kap_i + 6*n_maxcells];
	
			int nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 1*n_maxcblocks];
			// nbr 1.
				// p = 3.
			if (((I_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_3);
				Fyp += (0)*(2.0*f_3);
			}
				// p = 7.
			if (((I_kap+1==Nbx) && (J_kap+1< Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_7);
				Fym += ((+1.0))*(2.0*f_7);
			}
				// p = 6.
			if (((I_kap+1==Nbx) && (J_kap-1>= 0)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_6);
				Fyp += (1.0)*(2.0*f_6);
			}
			// nbr 2.
			nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 2*n_maxcblocks];
				// p = 4.
			if (((J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxp += (0)*(2.0*f_4);
				Fym += ((+1.0))*(2.0*f_4);
			}
				// p = 7.
			if (((I_kap+1< Nbx) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_7);
				Fym += ((+1.0))*(2.0*f_7);
			}
				// p = 8.
			if (((I_kap-1>= 0) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_8);
				Fym += ((+1.0))*(2.0*f_8);
			}
			// nbr 3.
			nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 3*n_maxcblocks];
				// p = 1.
			if (((I_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_1);
				Fyp += (0)*(2.0*f_1);
			}
				// p = 8.
			if (((I_kap-1==-1) && (J_kap+1< Nbx)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_8);
				Fym += ((+1.0))*(2.0*f_8);
			}
				// p = 5.
			if (((I_kap-1==-1) && (J_kap-1>= 0)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_5);
				Fyp += (1.0)*(2.0*f_5);
			}
			// nbr 4.
			nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 4*n_maxcblocks];
				// p = 2.
			if (((J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxp += (0)*(2.0*f_2);
				Fyp += (1.0)*(2.0*f_2);
			}
				// p = 5.
			if (((I_kap-1>= 0) && (J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_5);
				Fyp += (1.0)*(2.0*f_5);
			}
				// p = 6.
			if (((I_kap+1< Nbx) && (J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_6);
				Fyp += (1.0)*(2.0*f_6);
			}
			// nbr 5.
			nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 5*n_maxcblocks];
				// p = 7.
			if (((I_kap+1==Nbx) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_7);
				Fym += ((+1.0))*(2.0*f_7);
			}
			// nbr 6.
			nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 6*n_maxcblocks];
				// p = 8.
			if (((I_kap-1==-1) && (J_kap+1==Nbx)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_8);
				Fym += ((+1.0))*(2.0*f_8);
			}
			// nbr 7.
			nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 7*n_maxcblocks];
				// p = 5.
			if (((I_kap-1==-1) && (J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxp += (1.0)*(2.0*f_5);
				Fyp += (1.0)*(2.0*f_5);
			}
			// nbr 8.
			nbr_kap_b = mesh->cblock_ID_nbr[i_dev][i_kap_b + 8*n_maxcblocks];
				// p = 6.
			if (((I_kap+1==Nbx) && (J_kap-1==-1)) && nbr_kap_b == -8)
			{
				Fxm += ((+1.0))*(2.0*f_6);
				Fyp += (1.0)*(2.0*f_6);
			}
			
			for (int p = 0; p < 9; p++)
			{
				if (mesh->cblock_ID_nbr[i_dev][i_kap_b + p*n_maxcblocks] == -8)
					block_involved = true;
			}
		}
		
		if (block_involved)
			total_blocks++;
	}
	
	Fxp *= dx_L;
	Fyp *= dx_L;
	Fxm *= dx_L;
	Fym *= dx_L;
	ufloat_t Fx = Fxp - Fxm;
	ufloat_t Fy = Fyp - Fym;
	
	ufloat_t Dp = 1.0/32.0;
	//std::cout << std::setprecision(16) << "Fxm: " << Fxm << ", Fxp: " << Fxp << std::endl;
	//std::cout << std::setprecision(16) << "Fym: " << Fym << ", Fyp: " << Fyp << std::endl;
	//std::cout << "Fx: " << Fx << ", CD: " << 2.0*Fx / (0.05*0.05*(Dp)) << std::endl;
	//std::cout << "Fy: " << Fy << ", CL: " << 2.0*Fy / (0.05*0.05*(Dp)) << std::endl;
	*out << 2.0*Fx / (0.05*0.05*(Dp)) << " " << 2.0*Fy / (0.05*0.05*(Dp)) << std::endl;
	
	return 0;
}
