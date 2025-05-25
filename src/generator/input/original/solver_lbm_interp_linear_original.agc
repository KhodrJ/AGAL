# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_interp_linear_original
FILE_DIR ../solver_lbm/

ROUTINE_NAME Interpolate_Linear_Original
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var

KERNEL_REQUIRE int n_ids_idev_L           | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t tau_ratio         | (tau_vec[L+1]/tau_vec[L])
KERNEL_REQUIRE int *id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask         | mesh->c_cells_ID_mask[i_dev] mesh->c_cblock_ID_ref[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]

ROUTINE_TEMPLATE_PARAMS int interp_type
ROUTINE_TEMPLATE_VALS 0
ROUTINE_TEMPLATE_VALS 1
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_INTERP_INTERFACE
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==V_INTERP_ADDED



#
# Kernel definition.
#

TEXTSUBS sQ         2*<xc_i>+2*4*<xc_j>+2*4*4*<xc_k>

REG constexpr int Nqx = AP->Nqx;
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ ufloat_t s_F[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int J = (threadIdx.x / 4) % 4;
INIF Ldim==3
    REG int K = (threadIdx.x / 4) / 4;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int block_on_boundary = -1;
INFOR p 1   0 Lsize 1
    REG ufloat_t f_<p> = (ufloat_t)(0.0);
END_INFOR
REG ufloat_t tmp_i = (ufloat_t)(0.0);
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t u = (ufloat_t)(0.0);
REG ufloat_t v = (ufloat_t)(0.0);
INIF Ldim==3
	REG ufloat_t w = (ufloat_t)(0.0);
END_INIF
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG ((interp_type==0)and(block_on_boundary==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE))
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG block_on_boundary=cblock_ID_mask[i_kap_b];

// Load DDFs and compute macroscopic properties.
INFOR p 1   0 Lsize 1
    REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
END_INFOR
REG-vs rho = gSUM(i,0 Lsize 1,f_<i>);
REG-vsvz u = (gNz(gSUM(i,0 Lsize 1,(ufloat_t)(Lc0(<i>))*f_<i>))) / rho;
REG-vsvz v = (gNz(gSUM(i,0 Lsize 1,(ufloat_t)(Lc1(<i>))*f_<i>))) / rho;
INIF Ldim==2
    REG udotu = u*u + v*v;
INELSE
    REG-vsvz w = (gNz(gSUM(i,0 Lsize 1,(ufloat_t)(Lc2(<i>))*f_<i>))) / rho;
    REG udotu = u*u + v*v + w*w;
END_INIF

<
// Interpolate rescaled fi to children if applicable.
INFOR p 1   0 Lsize 1
    //
    // DDF <p>.
    //
    REG-vz cdotu = gNz((Lc0(<p>))*u + (Lc1(<p>))*v + (Lc2(<p>))*w);
    REG-vm tmp_i = (ufloat_t)(gD(Lw(<p>)))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
    REG s_F[threadIdx.x] = tmp_i + (f_<p> - tmp_i)*(tau_ratio);
    REG __syncthreads();
    INFOR xc_ixc_jxc_k 4   0 2 1   0 2 1   0 2 1
        INIF ((<xc_k>==0 and Ldim==2) or (Ldim==3))
            REG-m //\tChild gI(<xc_i>+2*<xc_j>+4*<xc_k>).
            OUTIF ((interp_type == 0 && cells_ID_mask[(i_kap_bc+gI(<xc_i>+2*<xc_j>+4*<xc_k>))*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
                
                INIF Ldim==2
                    REG-tvm cells_f_F[(i_kap_bc+gI( <xc_i>+2*<xc_j>+4*<xc_k> ))*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = \
                    s_F[gI(0+sQ)] + \
                    (s_F[gI(1+sQ)]-s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + \
                    (s_F[gI(4+sQ)]-s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + \
                    (s_F[gI(5+sQ)]-s_F[gI(4+sQ)]-s_F[gI(1+sQ)]+s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5));
                INELSE
                    REG-tvm cells_f_F[(i_kap_bc+gI( <xc_i>+2*<xc_j>+4*<xc_k> ))*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = \
                    s_F[gI(0+sQ)] + \
                    (s_F[gI(1+sQ)] - s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5)) + \
                    (s_F[gI(4+sQ)] - s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + \
                    (s_F[gI(16+sQ)] - s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + \
                    (s_F[gI(5+sQ)] - s_F[gI(1+sQ)] - s_F[gI(4+sQ)] + s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5)) + \
                    (s_F[gI(17+sQ)] - s_F[gI(1+sQ)] - s_F[gI(16+sQ)] + s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + \
                    (s_F[gI(20+sQ)] - s_F[gI(4+sQ)] - s_F[gI(16+sQ)] + s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5)) + \
                    (s_F[gI(21+sQ)] + s_F[gI(1+sQ)] + s_F[gI(4+sQ)] + s_F[gI(16+sQ)] - s_F[gI(5+sQ)] - s_F[gI(17+sQ)] - s_F[gI(20+sQ)] - s_F[gI(0+sQ)])*((ufloat_t)(-0.25) + I*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + J*(ufloat_t)(0.5))*((ufloat_t)(-0.25) + K*(ufloat_t)(0.5));
                END_INIF
                
            END_OUTIF
        END_INIF
    END_INFOR 
    REG __syncthreads();
    <
END_INFOR

TEMPLATE NEW_BLOCK
END_TEMPLATE
