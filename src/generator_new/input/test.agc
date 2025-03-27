# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
#FILE_NAME testname
#FILE_DIR ./output/

ROUTINE_NAME Average
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var
ROUTINE_REQUIRE ufloat_t tau_L
ROUTINE_REQUIRE ufloat_t tau_ratio_L

KERNEL_REQUIRE int n_ids_idev_L           | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t tau_L
KERNEL_REQUIRE ufloat_t tau_ratio         | tau_ratio_L
KERNEL_REQUIRE int *id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]

ROUTINE_TEMPLATE_PARAMS int ave_type
ROUTINE_TEMPLATE_VALS 0
ROUTINE_TEMPLATE_VALS 1
ROUTINE_TEMPLATE_VALS 2
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_INTERFACE
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_BLOCK
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_AVERAGE_GRID





# Kernel definition.

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG int I_kap = threadIdx.x % Nbx;
REG int J_kap = (threadIdx.x / Nbx) % Nbx;
INIF Ldim==3
    int K_kap = (threadIdx.x / Nbx) / Nbx;
END_INIF
REG ufloat_t x_kap = N_Pf(0.0);
REG ufloat_t y_kap = N_Pf(0.0);
INIF Ldim==3
	REG ufloat_t z_kap = N_Pf(0.0);
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int nbr_kap_b = -1;
REG int block_on_boundary = -1;
INFOR p 1   0 Lsize 1
    REG ufloat_t f_<p> = N_Pf(0.0);
END_INFOR
REG ufloat_t rho_kap = N_Pf(0.0);
REG ufloat_t u_kap = N_Pf(0.0);
REG ufloat_t v_kap = N_Pf(0.0);
INIF Ldim==3
    REG ufloat_t w_kap = N_Pf(0.0);
END_INIF
REG ufloat_t cdotu = N_Pf(0.0);
REG ufloat_t udotu = N_Pf(0.0);
REG ufloat_t omeg = dx_L / tau_L;
REG ufloat_t omegp = N_Pf(1.0) - omeg;
<

TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(block_on_boundary==1)

// Retrieve DDFs and compute macroscopic properties.
REG block_on_boundary = cblock_ID_onb[i_kap_b];
REG x_kap = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*(N_Pf(0.5) + I_kap);
REG y_kap = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*(N_Pf(0.5) + J_kap);
INIF Ldim==3
    REG z_kap = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*(N_Pf(0.5) + K_kap);
END_INIF
INFOR p 1   0 Lsize 1
    REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
END_INFOR
REG-vs rho_kap = gSUM(i,0 Lsize 1,f_<i>);
REG-vsvz u_kap = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc0(<i>))*f_<i>))) / rho_kap;
REG-vsvz v_kap = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc1(<i>))*f_<i>))) / rho_kap;
INIF Ldim==2
    REG-vmz udotu = u_kap*u_kap + v_kap*v_kap;
INELSE
    REG-vsvz w_kap = (gNz(gSUM(i,0 Lsize 1,N_Pf(Lc2(<i>))*f_<i>))) / rho_kap;
    REG udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
END_INIF

TEMPLATE NEW_BLOCK
END_TEMPLATE










