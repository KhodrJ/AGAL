# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_interp_cubic_original
FILE_DIR ../solver_lbm/

ROUTINE_NAME Interpolate_Cubic_Original
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

REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG __shared__ int s_ID_cblock[M_CBLOCK];
REG __shared__ ufloat_t s_F[M_CBLOCK];
REG-m ufloat_t x_kap = (ufloat_t)(gD(-1.0/12.0))+(ufloat_t)(gD(1.0/6.0))*(threadIdx.x % 4);
REG-m ufloat_t y_kap = (ufloat_t)(gD(-1.0/12.0))+(ufloat_t)(gD(1.0/6.0))*((threadIdx.x / 4) % 4);
INIF Ldim==3
    REG-m ufloat_t z_kap = (ufloat_t)(gD(-1.0/12.0))+(ufloat_t)(gD(1.0/6.0))*((threadIdx.x / 4) / 4);
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
REG ufloat_t w = (ufloat_t)(0.0);
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);

REG ufloat_t alpha = (ufloat_t)(0.0);
INFOR q 1   0 gI(2^ Ldim) 1
    REG ufloat_t S_beta_<q> = (ufloat_t)(0.0);
    INIF Ldim==3
        REG ufloat_t S_gamma_<q> = (ufloat_t)(0.0);
    END_INIF
    REG ufloat_t S_res_<q> = (ufloat_t)(0.0);
END_INFOR





TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG ((interp_type==0)and(block_on_boundary==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE))
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG block_on_boundary=cblock_ID_mask[i_kap_b];


// Load DDFs and compute macroscopic properties.
INFOR p 1   0 Lsize 1
    REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
END_INFOR
REG-vsvz rho = gSUM(i,0 Lsize 1,f_<i>);
REG-vsvz u = (gNz(gSUM(i,0 Lsize 1,(ufloat_t)(Lc0(<i>))*f_<i>))) / rho;
REG-vsvz v = (gNz(gSUM(i,0 Lsize 1,(ufloat_t)(Lc1(<i>))*f_<i>))) / rho;
INIF Ldim==2
    REG udotu = u*u + v*v;
INELSE
    REG-vsvz w = (gNz(gSUM(i,0 Lsize 1,(ufloat_t)(Lc2(<i>))*f_<i>))) / rho;
    REG udotu = u*u + v*v + w*w;
END_INIF

<
# For each DDF, compute sample points and place in shared memory. Then compute interpolation weights, and interpolate to each child block.
INFOR p 1   0 Lsize 1
    //
    // DDF <p>.
    //
    REG-v cdotu = (ufloat_t)(Lc0(<p>))*u + (ufloat_t)(Lc1(<p>))*v + (ufloat_t)(Lc2(<p>))*w;
    REG-v tmp_i = (ufloat_t)(Lw(<p>))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
    REG s_F[threadIdx.x] = tmp_i + (f_<p> - tmp_i)*(tau_ratio);
    REG __syncthreads();
    INIF Ldim==2
    
        # Reset result registers.
        INFOR q 1   0 4 1
            REG S_res_<q> = (ufloat_t)(0.0);
        END_INFOR
        
        # Loop in Y.
        INFOR j 1   0 4 1
            # Reset intermediate register for beta_jk.
            INFOR q 1   0 4 1
                REG S_beta_<q> = (ufloat_t)(0.0);
            END_INFOR
            INFOR i 1   0 4 1
                REG-m // Compute weight alpha_ijk (gI(3-<i>),gI(3-<j>)).
                REG-smvz alpha = (gSUM(m,0 16 1,(ufloat_t)(InterpM2D(gI(<m>+16*( (3-<i>)+4*(3-<j>) ))))*s_F[<m>]));
                
                # Update beta_jk.
                INFOR iijj 2   0 2 1   0 2 1
                    REG-mm S_beta_gI(<ii>+2*<jj>) = alpha + (x_kap+(ufloat_t)(gD(<ii>*(2.0/3.0))))*S_beta_gI(<ii>+2*<jj>);
                END_INFOR
            END_INFOR
    
            # Update the final result for all child blocks.
            INFOR iijj 2   0 2 1   0 2 1
                REG-mm S_res_gI(<ii>+2*<jj>) = S_beta_gI(<ii>+2*<jj>) + (y_kap+(ufloat_t)(gD(<jj>*(2.0/3.0))))*S_res_gI(<ii>+2*<jj>);
            END_INFOR
        END_INFOR
    
    INELSE
    
        # Reset result registers.
        INFOR q 1   0 8 1
            REG S_res_<q> = (ufloat_t)(0.0);
        END_INFOR
        
        # Loop over polynomial weight computation. Start in Z. Work in reverse for Horners' decomposition (3,2,1,0).
        INFOR k 1   0 4 1
            # Reset intermediate register for gamma_k.
            INFOR q 1   0 8 1
                REG S_gamma_<q> = (ufloat_t)(0.0);
            END_INFOR
            
            # Loop in Y.
            INFOR j 1   0 4 1
                # Reset intermediate register for beta_jk.
                INFOR q 1   0 8 1
                    REG S_beta_<q> = (ufloat_t)(0.0);
                END_INFOR
                INFOR i 1   0 4 1
                    REG-m // Compute weight alpha_ijk (gI(3-<i>),gI(3-<j>),gI(3-<k>)).
                    REG-smvz alpha = (gSUM(m,0 64 1,(ufloat_t)(InterpM3D(gI(<m>+64*( (3-<i>)+4*(3-<j>)+16*(3-<k>)))))*s_F[<m>]));
                    
                    # Update beta_jk.
                    INFOR iijjkk 2   0 2 1   0 2 1   0 2 1
                        REG-mm S_beta_gI(<ii>+2*<jj>+4*<kk>) = alpha + (x_kap+(ufloat_t)(gD(<ii>*(2.0/3.0))))*S_beta_gI(<ii>+2*<jj>+4*<kk>);
                    END_INFOR
                END_INFOR
                # Update gamma_k.
                INFOR iijjkk 2   0 2 1   0 2 1   0 2 1
                    REG-mm S_gamma_gI(<ii>+2*<jj>+4*<kk>) = S_beta_gI(<ii>+2*<jj>+4*<kk>) + (y_kap+(ufloat_t)(gD(<jj>*(2.0/3.0))))*S_gamma_gI(<ii>+2*<jj>+4*<kk>);
                END_INFOR
            END_INFOR
            
            # Update the final result for all child blocks.
            INFOR iijjkk 2   0 2 1   0 2 1   0 2 1
                REG-mm S_res_gI(<ii>+2*<jj>+4*<kk>) = S_gamma_gI(<ii>+2*<jj>+4*<kk>) + (z_kap+(ufloat_t)(gD(<kk>*(2.0/3.0))))*S_res_gI(<ii>+2*<jj>+4*<kk>);
            END_INFOR
        END_INFOR
        
    END_INIF
    
    <
    # Write result to child blocks.
    INFOR q 1   0 gI(2^ Ldim) 1
        OUTIF ((interp_type == 0 && cells_ID_mask[(i_kap_bc+<q>)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1))
            REG-v cells_f_F[(i_kap_bc+<q>)*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = S_res_<q>;
        END_OUTIF
    END_INFOR
    REG __syncthreads();
    <
    
END_INFOR


TEMPLATE NEW_BLOCK
END_TEMPLATE
