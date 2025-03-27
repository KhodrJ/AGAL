#include "solver_lbm.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeMacroProperties(int i_dev, int i_kap, int i_Q, int kap_i, ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w)
{
	if (VS == VS_D2Q9)
	{
		ufloat_t f_0 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 0L*n_maxcells];
		ufloat_t f_1 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 3L*n_maxcells];
		ufloat_t f_2 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 4L*n_maxcells];
		ufloat_t f_3 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 1L*n_maxcells];
		ufloat_t f_4 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 2L*n_maxcells];
		ufloat_t f_5 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 7L*n_maxcells];
		ufloat_t f_6 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 8L*n_maxcells];
		ufloat_t f_7 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 5L*n_maxcells];
		ufloat_t f_8 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 6L*n_maxcells];
		rho = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8;
		u = ( +f_1 -f_3 +f_5 -f_6 -f_7 +f_8) / rho;
		v = ( +f_2 -f_4 +f_5 +f_6 -f_7 -f_8) / rho;
		w = 0;
	}
	if (VS == VS_D3Q19)
	{
		ufloat_t f_0 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 0L*n_maxcells];
		ufloat_t f_1 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 2L*n_maxcells];
		ufloat_t f_2 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 1L*n_maxcells];
		ufloat_t f_3 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 4L*n_maxcells];
		ufloat_t f_4 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 3L*n_maxcells];
		ufloat_t f_5 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 6L*n_maxcells];
		ufloat_t f_6 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 5L*n_maxcells];
		ufloat_t f_7 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 8L*n_maxcells];
		ufloat_t f_8 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 7L*n_maxcells];
		ufloat_t f_9 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 10L*n_maxcells];
		ufloat_t f_10 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 9L*n_maxcells];
		ufloat_t f_11 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 12L*n_maxcells];
		ufloat_t f_12 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 11L*n_maxcells];
		ufloat_t f_13 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 14L*n_maxcells];
		ufloat_t f_14 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 13L*n_maxcells];
		ufloat_t f_15 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 16L*n_maxcells];
		ufloat_t f_16 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 15L*n_maxcells];
		ufloat_t f_17 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 18L*n_maxcells];
		ufloat_t f_18 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 17L*n_maxcells];
		rho = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18;
		u = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / rho;
		v = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / rho;
		w = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / rho;
	}
	if (VS == VS_D3Q27)
	{
		ufloat_t f_0 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 0L*n_maxcells];
		ufloat_t f_1 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 2L*n_maxcells];
		ufloat_t f_2 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 1L*n_maxcells];
		ufloat_t f_3 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 4L*n_maxcells];
		ufloat_t f_4 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 3L*n_maxcells];
		ufloat_t f_5 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 6L*n_maxcells];
		ufloat_t f_6 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 5L*n_maxcells];
		ufloat_t f_7 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 8L*n_maxcells];
		ufloat_t f_8 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 7L*n_maxcells];
		ufloat_t f_9 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 10L*n_maxcells];
		ufloat_t f_10 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 9L*n_maxcells];
		ufloat_t f_11 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 12L*n_maxcells];
		ufloat_t f_12 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 11L*n_maxcells];
		ufloat_t f_13 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 14L*n_maxcells];
		ufloat_t f_14 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 13L*n_maxcells];
		ufloat_t f_15 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 16L*n_maxcells];
		ufloat_t f_16 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 15L*n_maxcells];
		ufloat_t f_17 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 18L*n_maxcells];
		ufloat_t f_18 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 17L*n_maxcells];
		ufloat_t f_19 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 20L*n_maxcells];
		ufloat_t f_20 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 19L*n_maxcells];
		ufloat_t f_21 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 22L*n_maxcells];
		ufloat_t f_22 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 21L*n_maxcells];
		ufloat_t f_23 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 24L*n_maxcells];
		ufloat_t f_24 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 23L*n_maxcells];
		ufloat_t f_25 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 26L*n_maxcells];
		ufloat_t f_26 = mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + i_Q*M_TBLOCK + kap_i + 25L*n_maxcells];
		rho = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26;
		u = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16 +f_19 -f_20 +f_21 -f_22 +f_23 -f_24 -f_25 +f_26) / rho;
		v = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18 +f_19 -f_20 +f_21 -f_22 -f_23 +f_24 +f_25 -f_26) / rho;
		w = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18 +f_19 -f_20 -f_21 +f_22 +f_23 -f_24 +f_25 -f_26) / rho;
	}
	
	return 0;
}
