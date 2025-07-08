# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_collision_new
FILE_DIR ../solver_lbm/

ROUTINE_NAME Collision_New
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
REG int valid_mask = -1;
INFOR p 1   0 Lsize 1
    REG ufloat_t f_<p> = (ufloat_t)(0.0);
END_INFOR
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t u = (ufloat_t)(0.0);
REG ufloat_t v = (ufloat_t)(0.0);
INIF Ldim==3
    REG ufloat_t w = (ufloat_t)(0.0);
END_INIF
REG ufloat_t omeg = dx_L / tau_L;
REG ufloat_t omegp = (ufloat_t)(1.0) - omeg;



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(valid_block>-3)
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG valid_block=cblock_ID_mask[i_kap_b];


// o====================================================================================
// | BATCH #<K>
// o====================================================================================
<

// Retrieve DDFs one by one and compute macroscopic properties.
// I'm using up to store the positive contributions and um to store the negative ones. Same for v and w.
REG valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
REG rho = (ufloat_t)(0.0);
INFOR p 1   0 Lsize 1
REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
END_INFOR
REG-vsvz rho = gNz(gSUM(i,0 Lsize 1,f_<i>));
REG-vsvz u = ( gNz(gSUM(i,0 Lsize 1,Lc0(<i>)*f_<i>)) )/rho;
REG-vsvz v = ( gNz(gSUM(i,0 Lsize 1,Lc1(<i>)*f_<i>)) )/rho;
INIF Ldim==3
REG-vsvz w = ( gNz(gSUM(i,0 Lsize 1,Lc2(<i>)*f_<i>)) )/rho;
REG udotu = u*u + v*v + w*w;
INELSE
REG udotu = u*u + v*v;
END_INIF
REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells] = rho;
REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells] = u;
REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells] = v;
INIF Ldim==3
REG cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells] = w;
END_INIF

// Collision step.
INFOR p 1   0 Lsize 1
REG-vz cdotu = gNz(Lc0(<p>)*u + Lc1(<p>)*v + Lc2(<p>)*w);
REG-vm f_<p> = f_<p>*omegp + ( (ufloat_t)(gD(Lw(<p>)))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
END_INFOR
<

// Write fi* to global memory.
OUTIF (valid_mask != -1)
    INFOR p 1   0 Lsize 1
    REG cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + <p>*n_maxcells] = f_<p>;
    END_INFOR
END_OUTIF
<

TEMPLATE NEW_BLOCK
END_TEMPLATE

