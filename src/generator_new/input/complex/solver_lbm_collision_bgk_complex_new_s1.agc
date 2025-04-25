# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_collision_new_s1
FILE_DIR ../solver_lbm/

ROUTINE_NAME Collision_New_S1
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L

KERNEL_REQUIRE int n_ids_idev_L                        | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t dx_L                           | dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L                          | tau_vec[L]
KERNEL_REQUIRE int *__restrict__ id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *__restrict__ cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F_aux    | mesh->c_cells_f_F_aux[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cblock_f_X       | mesh->c_cblock_f_X[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]



#
# Kernel definition.
#

REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG __shared__ int s_ID_cblock[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int valid_block = -1;
REG ufloat_t f_p = (ufloat_t)(0.0);
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t up = (ufloat_t)(0.0);
REG ufloat_t um = (ufloat_t)(0.0);
REG ufloat_t vp = (ufloat_t)(0.0);
REG ufloat_t vm = (ufloat_t)(0.0);
REG ufloat_t wp = (ufloat_t)(0.0);
REG ufloat_t wm = (ufloat_t)(0.0);



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(valid_block>-3)
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG valid_block=cblock_ID_mask[i_kap_b];


// Retrieve DDFs one by one and compute macroscopic properties.
// I'm using up to store the positive contributions and um to store the negative ones. Same for v and w.
REG rho = (ufloat_t)(0.0);
REG up = (ufloat_t)(0.0);
REG um = (ufloat_t)(0.0);
REG vp = (ufloat_t)(0.0);
REG vm = (ufloat_t)(0.0);
INIF Ldim==3
    REG wp = (ufloat_t)(0.0);
    REG wm = (ufloat_t)(0.0);
END_INIF
INFOR p 1   0 Lsize 1
    REG-v f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
    REG rho += f_p;
    INIF (Lc0(<p>) > 0)
        REG-v up += f_p;
    END_INIF
    INIF (Lc0(<p>) < 0)
        REG-v um += f_p;
    END_INIF
    INIF (Lc1(<p>) > 0)
        REG-v vp += f_p;
    END_INIF
    INIF (Lc1(<p>) < 0)
        REG-v vm += f_p;
    END_INIF
    INIF (Lc2(<p>) > 0)
        REG-v wp += f_p;
    END_INIF
    INIF (Lc2(<p>) < 0)
        REG-v wm += f_p;
    END_INIF
END_INFOR
REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = rho;
REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = (up-um)/rho;
REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = (vp-vm)/rho;
INIF Ldim==3
    REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = (wp-wm)/rho;
END_INIF

# Macroscopic variables are now available on this grid level.

TEMPLATE NEW_BLOCK
END_TEMPLATE

