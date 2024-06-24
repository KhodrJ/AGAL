/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

/*
         .d88888b.           888                      888             
        d88P" "Y88b          888                      888             
        888     888          888                      888             
        888     888 888  888 888888 88888b.  888  888 888888          
        888     888 888  888 888    888 "88b 888  888 888             
        888     888 888  888 888    888  888 888  888 888             
        Y88b. .d88P Y88b 888 Y88b.  888 d88P Y88b 888 Y88b.           
88888888 "Y88888P"   "Y88888  "Y888 88888P"   "Y88888  "Y888 88888888 
                                    888                               
                                    888                               
                                    888                               
*/

// NOTE: Only works with N_LEVEL_START=0 for now.
int Mesh::M_UpdateMeanVelocities(int i_dev, int N_iters_ave)
{
	for (int kap = 0; kap < n_ids[i_dev][0]; kap++)
	{
		int i_kap = id_set[i_dev][0][kap];
		
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
#if (N_Q==9)
			ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
			ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8;
			ufloat_t u_kap = ( +f_1 -f_3 +f_5 -f_6 -f_7 +f_8) / rho_kap;
			ufloat_t v_kap = ( +f_2 -f_4 +f_5 +f_6 -f_7 -f_8) / rho_kap;
#elif (N_Q==19)
			ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17*n_maxcells];
			ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18;
			ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / rho_kap;
			ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / rho_kap;
			ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / rho_kap;
#else // (N_Q==27)
			ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17*n_maxcells];
			ufloat_t f_19 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 20*n_maxcells];
			ufloat_t f_20 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 19*n_maxcells];
			ufloat_t f_21 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 22*n_maxcells];
			ufloat_t f_22 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 21*n_maxcells];
			ufloat_t f_23 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 24*n_maxcells];
			ufloat_t f_24 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 23*n_maxcells];
			ufloat_t f_25 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 26*n_maxcells];
			ufloat_t f_26 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 25*n_maxcells];
			ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26;
			ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16 +f_19 -f_20 +f_21 -f_22 +f_23 -f_24 -f_25 +f_26) / rho_kap;
			ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18 +f_19 -f_20 +f_21 -f_22 -f_23 +f_24 +f_25 -f_26) / rho_kap;
			ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18 +f_19 -f_20 -f_21 +f_22 +f_23 -f_24 +f_25 -f_26) / rho_kap;
#endif
			
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_ids[i_dev][0]*M_CBLOCK] = 
				((ufloat_t)N_iters_ave*cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_ids[i_dev][0]*M_CBLOCK] + u_kap) / ((ufloat_t)N_iters_ave+1.0);
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_ids[i_dev][0]*M_CBLOCK] = 
				((ufloat_t)N_iters_ave*cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_ids[i_dev][0]*M_CBLOCK] + v_kap) / ((ufloat_t)N_iters_ave+1.0);
#if (N_DIM==3)
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_ids[i_dev][0]*M_CBLOCK] = 
				((ufloat_t)N_iters_ave*cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_ids[i_dev][0]*M_CBLOCK] + w_kap) / ((ufloat_t)N_iters_ave+1.0);
#else
			cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_ids[i_dev][0]*M_CBLOCK] = N_Pf(0.0);
#endif
		}
	}
	
	return 0;
}

ufloat_t Mesh::M_CheckConvergence(int i_dev)
{
	ufloat_t sum = N_Pf(0.0);
	ufloat_t norm = N_Pf(0.0);
	
	for (int kap = 0; kap < n_ids_probed[i_dev]; kap++)
	{
		int i_kap = id_set_probed[i_dev][kap];
	
		// Density and velocity computations.
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
		{
#if (N_Q==9)
			ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
			ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8;
			ufloat_t u_kap = ( +f_1 -f_3 +f_5 -f_6 -f_7 +f_8) / rho_kap;
			ufloat_t v_kap = ( +f_2 -f_4 +f_5 +f_6 -f_7 -f_8) / rho_kap;
#elif (N_Q==19)
			ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17*n_maxcells];
			ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18;
			ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / rho_kap;
			ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / rho_kap;
			ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / rho_kap;
#else // (N_Q==27)
			ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
			ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
			ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
			ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
			ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
			ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
			ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
			ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
			ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
			ufloat_t f_9 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10*n_maxcells];
			ufloat_t f_10 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9*n_maxcells];
			ufloat_t f_11 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12*n_maxcells];
			ufloat_t f_12 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11*n_maxcells];
			ufloat_t f_13 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14*n_maxcells];
			ufloat_t f_14 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13*n_maxcells];
			ufloat_t f_15 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16*n_maxcells];
			ufloat_t f_16 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15*n_maxcells];
			ufloat_t f_17 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18*n_maxcells];
			ufloat_t f_18 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17*n_maxcells];
			ufloat_t f_19 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 20*n_maxcells];
			ufloat_t f_20 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 19*n_maxcells];
			ufloat_t f_21 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 22*n_maxcells];
			ufloat_t f_22 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 21*n_maxcells];
			ufloat_t f_23 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 24*n_maxcells];
			ufloat_t f_24 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 23*n_maxcells];
			ufloat_t f_25 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 26*n_maxcells];
			ufloat_t f_26 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 25*n_maxcells];
			ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26;
			ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16 +f_19 -f_20 +f_21 -f_22 +f_23 -f_24 -f_25 +f_26) / rho_kap;
			ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18 +f_19 -f_20 +f_21 -f_22 -f_23 +f_24 +f_25 -f_26) / rho_kap;
			ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18 +f_19 -f_20 -f_21 +f_22 +f_23 -f_24 +f_25 -f_26) / rho_kap;
#endif
		
#if (N_DIM==2)
			ufloat_t u_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK];
			ufloat_t v_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK];
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK] = u_kap;
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK] = v_kap;
			sum += (u_kap-u_kap_prev)*(u_kap-u_kap_prev) + (v_kap-v_kap_prev)*(v_kap-v_kap_prev);
			norm += u_kap*u_kap + v_kap*v_kap;
			
			
#else
			ufloat_t u_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK];
			ufloat_t v_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK];
			ufloat_t w_kap_prev = cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 2*n_ids_probed[i_dev]*M_CBLOCK];
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 0*n_ids_probed[i_dev]*M_CBLOCK] = u_kap;
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 1*n_ids_probed[i_dev]*M_CBLOCK] = v_kap;
			cells_f_U_probed_tn[i_dev][kap*M_CBLOCK + kap_i + 2*n_ids_probed[i_dev]*M_CBLOCK] = w_kap;
			sum += (u_kap-u_kap_prev)*(u_kap-u_kap_prev) + (v_kap-v_kap_prev)*(v_kap-v_kap_prev) + (w_kap-w_kap_prev)*(w_kap-w_kap_prev);
			norm += u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
#endif
		}
	}

	sum = sqrt(sum/norm);
	std::cout << "Residue: " << sum << std::endl; 
	return sum;
}

int Mesh::M_ComputeProperties(int i_dev, int i_kap, ufloat_t dx_L, double *out, double *out2)
{	
	// Density and velocity computations.
	for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
	{
#if (N_Q==9)
		ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
		ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
		ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
		ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
		ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
		ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
		ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
		ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
		ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8;
		ufloat_t u_kap = ( +f_1 -f_3 +f_5 -f_6 -f_7 +f_8) / rho_kap;
		ufloat_t v_kap = ( +f_2 -f_4 +f_5 +f_6 -f_7 -f_8) / rho_kap;
		
		out[kap_i + 0*M_CBLOCK] = (double)rho_kap;
		out[kap_i + 1*M_CBLOCK] = (double)u_kap;
		out[kap_i + 2*M_CBLOCK] = (double)v_kap;
		out[kap_i + 3*M_CBLOCK] = 0.0;
#elif (N_Q==19)
		ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
		ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
		ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
		ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
		ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
		ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
		ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
		ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
		ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
		ufloat_t f_9 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10*n_maxcells];
		ufloat_t f_10 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9*n_maxcells];
		ufloat_t f_11 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12*n_maxcells];
		ufloat_t f_12 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11*n_maxcells];
		ufloat_t f_13 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14*n_maxcells];
		ufloat_t f_14 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13*n_maxcells];
		ufloat_t f_15 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16*n_maxcells];
		ufloat_t f_16 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15*n_maxcells];
		ufloat_t f_17 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18*n_maxcells];
		ufloat_t f_18 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18;
		ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / rho_kap;
		ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / rho_kap;
		ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / rho_kap;
		
		out[kap_i + 0*M_CBLOCK] = (double)rho_kap;
		out[kap_i + 1*M_CBLOCK] = (double)u_kap;
		out[kap_i + 2*M_CBLOCK] = (double)v_kap;
		out[kap_i + 3*M_CBLOCK] = (double)w_kap;
#else // (N_Q==27)
		ufloat_t f_0 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_maxcells];
		ufloat_t f_1 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_maxcells];
		ufloat_t f_2 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_maxcells];
		ufloat_t f_3 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 4*n_maxcells];
		ufloat_t f_4 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 3*n_maxcells];
		ufloat_t f_5 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 6*n_maxcells];
		ufloat_t f_6 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 5*n_maxcells];
		ufloat_t f_7 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 8*n_maxcells];
		ufloat_t f_8 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 7*n_maxcells];
		ufloat_t f_9 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 10*n_maxcells];
		ufloat_t f_10 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 9*n_maxcells];
		ufloat_t f_11 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 12*n_maxcells];
		ufloat_t f_12 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 11*n_maxcells];
		ufloat_t f_13 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 14*n_maxcells];
		ufloat_t f_14 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 13*n_maxcells];
		ufloat_t f_15 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 16*n_maxcells];
		ufloat_t f_16 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 15*n_maxcells];
		ufloat_t f_17 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 18*n_maxcells];
		ufloat_t f_18 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 17*n_maxcells];
		ufloat_t f_19 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 20*n_maxcells];
		ufloat_t f_20 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 19*n_maxcells];
		ufloat_t f_21 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 22*n_maxcells];
		ufloat_t f_22 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 21*n_maxcells];
		ufloat_t f_23 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 24*n_maxcells];
		ufloat_t f_24 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 23*n_maxcells];
		ufloat_t f_25 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 26*n_maxcells];
		ufloat_t f_26 = cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + 25*n_maxcells];
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
	
	// Vorticity computations (now that velocity is known for all cells in the cell-block).
	for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
	{
		int I_kap = kap_i % Nbx;
		int J_kap = (kap_i / Nbx) % Nbx;
#if (N_DIM==3)
		int K_kap = (kap_i / Nbx) / Nbx;
#endif
		ufloat_t dU = N_Pf(0.0);
		ufloat_t dV = N_Pf(0.0);
#if (N_DIM==3)
		ufloat_t dW = N_Pf(0.0);
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
	
#if (S_LES==1)
	// Calculation of y+.
	if (cblock_ID_onb[i_dev][i_kap] == 1)
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
			if (cblock_ID_nbr[i_dev][i_kap + 1*n_maxcblocks] < 0)
			{
				tau_w = ( -N_Pf(71.0)*out[4*J_kap+3 + 2*M_CBLOCK] + N_Pf(141.0)*out[4*J_kap+2 + 2*M_CBLOCK] - N_Pf(93.0)*out[4*J_kap+1 + 2*M_CBLOCK] + N_Pf(23.0)*out[4*J_kap + 2*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
				yplus = std::min( (0.5*dx_L + (3-I_kap)*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
			}
			if (cblock_ID_nbr[i_dev][i_kap + 2*n_maxcblocks] < 0)
			{
				tau_w = ( -N_Pf(71.0)*out[I_kap+12 + 1*M_CBLOCK] + N_Pf(141.0)*out[I_kap+8 + 1*M_CBLOCK] - N_Pf(93.0)*out[I_kap+4 + 1*M_CBLOCK] + N_Pf(23.0)*out[I_kap + 1*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
				yplus = std::min( (0.5*dx_L + (3-J_kap)*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
			}
			if (cblock_ID_nbr[i_dev][i_kap + 3*n_maxcblocks] < 0)
			{
				tau_w = ( -N_Pf(71.0)*out[4*J_kap + 2*M_CBLOCK] + N_Pf(141.0)*out[4*J_kap+1 + 2*M_CBLOCK] - N_Pf(93.0)*out[4*J_kap+2 + 2*M_CBLOCK] + N_Pf(23.0)*out[4*J_kap+3 + 2*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
				yplus = std::min( (0.5*dx_L + I_kap*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
			}
			if (cblock_ID_nbr[i_dev][i_kap + 4*n_maxcblocks] < 0)
			{
				tau_w = ( -N_Pf(71.0)*out[I_kap + 1*M_CBLOCK] + N_Pf(141.0)*out[I_kap+4 + 1*M_CBLOCK] - N_Pf(93.0)*out[I_kap+8 + 1*M_CBLOCK] + N_Pf(23.0)*out[I_kap+12 + 1*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
				yplus = std::min( (0.5*dx_L + J_kap*dx_L)*sqrt(abs(tau_w) / v0), yplus ); // y+
			}
			out2[kap_i] = yplus;
#else
			//if (cblock_ID_nbr[i_dev][i_kap + 1*n_maxcblocks] < 0)
			//{
			//	yplus[kap_i] = ( -N_Pf(71.0)*out_kap[4*J_kap+3 + 1*M_CBLOCK] + N_Pf(141.0)*s_u[4*J_kap+2 + 1*M_CBLOCK] - N_Pf(93.0)*s_u[4*J_kap+1 + 1*M_CBLOCK] + N_Pf(23.0)*s_u[4*J_kap + 1*M_CBLOCK] ) / (N_Pf(24.0)*dx_L); // Wall shear stress.
			//}
			//yplus[kap_i] = (0.5*dx_L + J_kap*dx_L)*sqrt(yplus[kap_i] / v0); // y+
#endif
		}
	}
	else
	{
		for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++) out2[kap_i] = -1.0;
	}
#endif
	
	return 0;
}

int Mesh::M_Print_VTHB(int i_dev, int iter)
{
	// New variables.
	vtkNew<vtkOverlappingAMR> data;
	int blocks_per_level[N_PRINT_LEVELS];
	for (int L = 0; L < N_PRINT_LEVELS; L++)
		blocks_per_level[L] = n_ids[i_dev][L];
	double global_origin[3] = {0,0,0};
	
	// Get the number of levels in which there are a non-zero number of blocks.
	int n_levels_nonzero_blocks = 1;
#if (MAX_LEVELS>1)
	for (int L = 1; L < N_PRINT_LEVELS; L++)
	{
		if (n_ids[i_dev][L] > 0)
			n_levels_nonzero_blocks++;
	}
#endif
	
	// Initialize AMR object.
	data->Initialize(n_levels_nonzero_blocks, blocks_per_level);
	data->SetOrigin(global_origin);
	data->SetGridDescription(N_DIM==2?VTK_XY_PLANE:VTK_XYZ_GRID);
	
	// Parameters.
	//int n_dim_box[3] = {Nbx, Nbx, N_DIM==2?1:Nbx};
	int n_dim_lattice[3] = {Nbx+1, Nbx+1, N_DIM==2?1:Nbx+1};
	double origin_kap[3] = {0,0,0};
	double u_kap[M_CBLOCK*(6+1)]; for (int i = 0; i < M_CBLOCK*(6+1); i++) u_kap[i] = 0.0; 
	double yplus_kap[M_CBLOCK]; for (int i = 0; i < M_CBLOCK; i++) yplus_kap[i] = 0.0;
	
	// For each level, insert all existing blocks.
	for (int L = 0; L < std::min(n_levels_nonzero_blocks, N_PRINT_LEVELS); L++)
	{
		// Construct spacing array for level L.
		double dxf_L = (double)dxf_vec[L];
		double h_L_kap[3] = {dxf_L, dxf_L, N_DIM==2?(double)dx:dxf_L};
		
		// Only insert spacing array if there are actual blocks to insert on this level.
		if (n_ids[i_dev][L] > 0)
			data->SetSpacing(L, h_L_kap);
		
		// For all blocks on level L, build vtkAMRBox and insert in AMR object.
		int kap_counter = 0;
		for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
		{
			// ID of the kap'th cblock.
			int i_kap = id_set[i_dev][L][kap];
			
			//if (cblock_ID_ref[i_dev][i_kap] == V_REF_ID_UNREFINED)
			{
				// Set origin of block (based on lower-left corner stored in cblock_f_X).
				for (int d = 0; d < N_DIM; d++)
					origin_kap[d] = cblock_f_X[i_dev][i_kap + d*n_maxcblocks];
				
				// Initialize uniform grid defining the block.
				vtkNew<vtkUniformGrid> grid_kap;
				grid_kap->Initialize();
				grid_kap->SetOrigin(origin_kap);
				grid_kap->SetSpacing(h_L_kap);
				grid_kap->SetDimensions(n_dim_lattice);
				
				// Fill data in the uniform grid defining the block.
					// Debug.
				vtkNew<vtkDoubleArray> data_kap_dbg;
				data_kap_dbg->SetName("debug - ref ID");
				data_kap_dbg->SetNumberOfComponents(1);
				data_kap_dbg->SetNumberOfTuples(M_CBLOCK);
				int min_nbr_id = 0;
				#pragma unroll
				for (int p = 0; p < N_Q_max; p++)
				{
					int nbr_id_p = cblock_ID_nbr[i_dev][i_kap + p*n_maxcblocks];
					//if (nbr_id_p == N_SKIPID)
					//	nbr_id_p = -1;
					if (nbr_id_p < 0 && min_nbr_id == 0)
						min_nbr_id = nbr_id_p;
					if (nbr_id_p > min_nbr_id && nbr_id_p < 0)
						min_nbr_id = nbr_id_p;
				}
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_ID_ref[i_dev][i_kap]);
					data_kap_dbg->SetTuple1(kap_i, (double)cells_ID_mask[i_dev][i_kap*M_CBLOCK + kap_i]);
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_ID_mask[i_dev][i_kap]);
					//data_kap_dbg->SetTuple1(kap_i, (double)min_nbr_id);
					//data_kap_dbg->SetTuple1(kap_i, (double)cblock_f_X[i_dev][i_kap + 0*n_maxcblocks] + (kap_i + 0.5)*dx);
				}
				grid_kap->GetCellData()->AddArray(data_kap_dbg);
					// Compute macroscopic properties.
				for (int i = 0; i < M_CBLOCK*(6+1); i++) u_kap[i] = 0.0;
				for (int i = 0; i < M_CBLOCK; i++) yplus_kap[i] = 0.0;
				M_ComputeProperties(i_dev, i_kap, dxf_vec[L], u_kap, yplus_kap);
					// Density.
				vtkNew<vtkDoubleArray> data_kap_sc;
				data_kap_sc->SetName("Density");
				data_kap_sc->SetNumberOfComponents(1);
				data_kap_sc->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_sc->SetTuple1(kap_i,
						u_kap[kap_i + 0*M_CBLOCK]
					);
				}
				grid_kap->GetCellData()->AddArray(data_kap_sc);
					// Velocity.
				vtkNew<vtkDoubleArray> data_kap_v;
				data_kap_v->SetName("Velocity");
				data_kap_v->SetNumberOfComponents(3);
				data_kap_v->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					if (L > 0 || (L == 0 && N_PROBE_AVE == 0))
					{
						data_kap_v->SetTuple3(kap_i,
							u_kap[kap_i + 1*M_CBLOCK],
							u_kap[kap_i + 2*M_CBLOCK],
							u_kap[kap_i + 3*M_CBLOCK]
						);
					}
					else
					{
						data_kap_v->SetTuple3(kap_i,
							(double)cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 0*n_ids[i_dev][0]*M_CBLOCK],
							(double)cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 1*n_ids[i_dev][0]*M_CBLOCK],
							(double)cells_f_U_mean[i_dev][i_kap*M_CBLOCK + kap_i + 2*n_ids[i_dev][0]*M_CBLOCK]
						);
					}
				}
				grid_kap->GetCellData()->AddArray(data_kap_v);
					// Vorticity.
				vtkNew<vtkDoubleArray> data_kap_w;
				data_kap_w->SetName("Vorticity");
				data_kap_w->SetNumberOfComponents(3);
				data_kap_w->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_w->SetTuple3(kap_i, 
						u_kap[kap_i + 4*M_CBLOCK],
						u_kap[kap_i + 5*M_CBLOCK],
						u_kap[kap_i + 6*M_CBLOCK]
					);
				}
				grid_kap->GetCellData()->AddArray(data_kap_w);
#if (S_LES==1)
					// Y+
				vtkNew<vtkDoubleArray> data_kap_yp;
				data_kap_yp->SetName("Y+");
				data_kap_yp->SetNumberOfComponents(1);
				data_kap_yp->SetNumberOfTuples(M_CBLOCK);
				for (int kap_i = 0; kap_i < M_CBLOCK; kap_i++)
				{
					data_kap_yp->SetTuple1(kap_i, 
						yplus_kap[kap_i]
					);
				}
				grid_kap->GetCellData()->AddArray(data_kap_yp);
#endif
				
				// Create the vtkAMRBox and insert it into AMR object.
				vtkAMRBox box_kap(origin_kap, n_dim_lattice, h_L_kap, data->GetOrigin(), data->GetGridDescription());
				data->SetAMRBox(L, kap_counter, box_kap);
				data->SetDataSet(L, kap_counter, grid_kap);
				kap_counter++;
			}
		}
	}
	
	// Write the AMR object.
	std::cout << "Finished building VTK dataset, writing..." << std::endl;
	std::string fileName = P_DIR_NAME + std::string("out_") + std::to_string(iter+1) + ".vthb";
	vtkNew<vtkXMLUniformGridAMRWriter> writer;
	writer->SetInputData(data);
	writer->SetFileName(fileName.c_str());
	writer->Write();
	std::cout << "Finished writing VTK dataset..." << std::endl;
	
	return 0;
}

int Mesh::M_PrintConnectivity(int i_dev)
{
	std::ofstream out_conn; out_conn.open("out_conn.txt");
	
	for (int L = 0; L < MAX_LEVELS; L++)
	{
		for (int kap = 0; kap < n_ids[i_dev][L]; kap++)
		{
			int i_kap =  id_set[i_dev][L][kap];
			
			out_conn << "[ID: " << i_kap << "] Block " << kap << ", Level: " << L << std::endl;
			
			out_conn << "X: ";
			for (int d = 0; d < N_DIM; d++)
				out_conn << "x_" << d << "|" << cblock_f_X[i_dev][i_kap + d*n_maxcblocks] << " ";
			out_conn << std::endl;
			
			/*
			for (int xkap = 0; xkap < M_CBLOCK; xkap++)
			{
				out_conn << "Cell " << xkap << ": ";
				for (int p = 0; p < N_Q; p++)
					out_conn << "F(p=" << p << ")|" << cells_f_F[i_dev][i_kap*M_CBLOCK + xkap + p*n_maxcells] << "  ";
				out_conn << std::endl;
			}
			*/
		
			/*
			out_conn << "Nbrs: ";
			for (int p = 0; p < N_Q_max; p++)
				out_conn << "p=" << p << "|" << cblock_ID_nbr[i_dev][i_kap + p*n_maxcblocks] << " ";
			out_conn << std::endl;
			
			out_conn << "Child nbrs: ";
			for (int p = 0; p < N_Q_max; p++)
				out_conn << "p=" << p << "|" << cblock_ID_nbr_child[i_dev][i_kap + p*n_maxcblocks] << " ";
			out_conn << std::endl;
			
			out_conn << std::endl;
			*/
		}
	}
	
	out_conn.close();
	
	return 0;
}





int Mesh::M_LoadToGPU()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		int cblocks_id_max = id_max[i_dev][MAX_LEVELS];
		long int cells_id_max = id_max[i_dev][MAX_LEVELS]*M_CBLOCK;


		// Floating point arrays.
		for (int p = 0; p < N_Q; p++)
			gpuErrchk( cudaMemcpy(&c_cells_f_F[i_dev][p*n_maxcells], &cells_f_F[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );
		for (int d = 0; d < N_DIM; d++)
			gpuErrchk( cudaMemcpy(&c_cblock_f_X[i_dev][d*n_maxcblocks], &cblock_f_X[i_dev][d*n_maxcblocks], cblocks_id_max*sizeof(ufloat_t), cudaMemcpyHostToDevice) );	


		// Connectivity arrays.
		for (int p = 0; p < N_Q_max; p++)
		{
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr[i_dev][p*n_maxcblocks], &cblock_ID_nbr[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
			gpuErrchk( cudaMemcpy(&c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &cblock_ID_nbr_child[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		}
		gpuErrchk( cudaMemcpy(c_cblock_ID_onb[i_dev], cblock_ID_onb[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );

		
		// Metadata arrays.
		gpuErrchk( cudaMemcpy(c_cells_ID_mask[i_dev], cells_ID_mask[i_dev], cells_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_ID_mask[i_dev], cblock_ID_mask[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_cblock_level[i_dev], cblock_level[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyHostToDevice) );

		for (int L = 0; L < MAX_LEVELS; L++)
			gpuErrchk( cudaMemcpy(c_id_set[i_dev][L], id_set[i_dev][L], n_ids[i_dev][L]*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_gap_set[i_dev], gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyHostToDevice) );
	}
	
	return 0;
}

int Mesh::M_RetrieveFromGPU()
{
	for (int i_dev = 0; i_dev < N_DEV; i_dev++)
	{
		int cblocks_id_max = id_max[i_dev][MAX_LEVELS];
		long int cells_id_max = id_max[i_dev][MAX_LEVELS]*M_CBLOCK;


		// Floating point arrays.
		for (int p = 0; p < N_Q; p++)
			gpuErrchk( cudaMemcpy(&cells_f_F[i_dev][p*n_maxcells], &c_cells_f_F[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );
		for (int d = 0; d < N_DIM; d++)
			gpuErrchk( cudaMemcpy(&cblock_f_X[i_dev][d*n_maxcblocks], &c_cblock_f_X[i_dev][d*n_maxcblocks], cblocks_id_max*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );


		// Connectivity arrays.
		for (int p = 0; p < N_Q_max; p++)
		{
			gpuErrchk( cudaMemcpy(&cblock_ID_nbr[i_dev][p*n_maxcblocks], &c_cblock_ID_nbr[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
			gpuErrchk( cudaMemcpy(&cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &c_cblock_ID_nbr_child[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		}
		gpuErrchk( cudaMemcpy(cblock_ID_onb[i_dev], c_cblock_ID_onb[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );


		// Metadata arrays.
		gpuErrchk( cudaMemcpy(cells_ID_mask[i_dev], c_cells_ID_mask[i_dev], cells_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_ID_mask[i_dev], c_cblock_ID_mask[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_ID_ref[i_dev], c_cblock_ID_ref[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(cblock_level[i_dev], c_cblock_level[i_dev], cblocks_id_max*sizeof(int), cudaMemcpyDeviceToHost) );
		
		for (int L = 0; L < MAX_LEVELS; L++)
			gpuErrchk( cudaMemcpy(id_set[i_dev][L], c_id_set[i_dev][L], n_ids[i_dev][L]*sizeof(int), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(gap_set[i_dev], c_gap_set[i_dev], n_gaps[i_dev]*sizeof(int), cudaMemcpyDeviceToHost) );
	}

	return 0;
}
