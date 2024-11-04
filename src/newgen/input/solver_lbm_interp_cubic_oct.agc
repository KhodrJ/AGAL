# Kernel name and includes.
NAME Interpolate_Cubic_LBM_name
NAME_FILE solver_lbm_interp_cubic_mq
INCLUDE "solver.h"
INCLUDE "mesh.h"
INCLUDE_GUARD N_Q==LBM_size

# Kernel parameters.
TEMPLATE int interp_type=0
TEMPLATE_VALS 0
TEMPLATE_VALS 1
KERNEL_REQUIRE int n_ids_idev_L ROUTINE mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t dx_L ROUTINE mesh->dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L
KERNEL_REQUIRE ufloat_t tau_ratio ROUTINE tau_ratio_L
KERNEL_REQUIRE ufloat_t v0
KERNEL_REQUIRE int *id_set_idev_L ROUTINE &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask ROUTINE mesh->c_cells_ID_mask[i_dev] mesh->c_cblock_ID_ref[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F ROUTINE mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr ROUTINE mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child ROUTINE mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask ROUTINE mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *cblock_ID_onb ROUTINE mesh->c_cblock_ID_onb[i_dev]

# Routine parameters.
ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var
ROUTINE_REQUIRE ufloat_t tau_L
ROUTINE_REQUIRE ufloat_t tau_ratio_L
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_INTERFACE
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_ADDED
ROUTINE_OBJECT Solver_LBM

# Import interpolation matrix.
IMPORT VECTOR mat_interp_2D_single.txt A_2D
IMPORT VECTOR mat_interp_3D_single.txt A_3D



#
# Kernel definition.
#

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ ufloat_t s_F[M_TBLOCK];
REG ufloat_t x_kap = N_Pf(^D< -1.0/12.0 >^)+N_Pf(^D< 1.0/6.0 >^)*(threadIdx.x % Nbx);
REG ufloat_t y_kap = N_Pf(^D< -1.0/12.0 >^)+N_Pf(^D< 1.0/6.0 >^)*((threadIdx.x / Nbx) % Nbx);
INIF LBM_dim==3
    ufloat_t z_kap = N_Pf(^D< -1.0/12.0 >^)+N_Pf(^D< 1.0/6.0 >^)*((threadIdx.x / Nbx) / Nbx);
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int i_Q = -1;
REG int i_Qc = -1;
REG int i_Qcp = -1;
REG int block_on_boundary = -1;
INFOR p 1   0 LBM_size 1
    REG ufloat_t f_<p> = N_Pf(0.0);
END_INFOR
REG ufloat_t tmp_i = N_Pf(0.0);
REG ufloat_t rho_kap = N_Pf(0.0);
REG ufloat_t u_kap = N_Pf(0.0);
REG ufloat_t v_kap = N_Pf(0.0);
REG ufloat_t w_kap = N_Pf(0.0);
REG ufloat_t cdotu = N_Pf(0.0);
REG ufloat_t udotu = N_Pf(0.0);

REG ufloat_t alpha = N_Pf(0.0);
INFOR q 1   0 ^I< 2^ LBM_dim >^ 1
    REG ufloat_t S_beta_<q> = N_Pf(0.0);
    INIF LBM_dim==3
        REG ufloat_t S_gamma_<q> = N_Pf(0.0);
    END_INIF
    REG ufloat_t S_res_<q> = N_Pf(0.0);
END_INFOR



LOOPBLOCKS 2\
    CONDITION (((interp_type==0)and(block_on_boundary==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE)))\
    REQUIRING 2 i_kap_bc=cblock_ID_nbr_child[i_kap_b] block_on_boundary=cblock_ID_mask[i_kap_b]




INIF LBM_dim==3
    OUTFOR k_q 3   0 Nqx/2 1
END_INIF
OUTFOR i_qj_q 3   0 Nqx/2 1   0 Nqx/2 1
    INFOR I_qJ_qK_q 3   0 2 1   0 2 1   0 ^I< LBM_dim-1 >^ 1
    
        //
        // Child block ^I< <I_q>+2*<J_q>+4*<K_q> >^.
        // 
        //
        DEFINE IKAPC ^I< <I_q>+2*<J_q>+4*<K_q> >^
    
        <
        // Load DDFs and compute macroscopic properties.
        INIF LBM_dim==2
            REG i_Q = (i_q+<I_q>*Nqx/2) + Nqx*(j_q+<J_q>*Nqx/2);
            REG i_Qc = 2*i_q + 2*Nqx*j_q;
        INELSE
            REG i_Q = (i_q+<I_q>*Nqx/2) + Nqx*(j_q+<J_q>*Nqx/2) + Nqx*Nqx*(k_q+<K_q>*Nqx/2);
            REG i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
        END_INIF
        INFOR p 1   0 LBM_size 1
            REG f_<p> = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells];
        END_INFOR
        REG rho_kap = SUM< i 0 LBM_size 1 f_<i> >END_SUM;
        REG u_kap = (SUM< i 0 LBM_size 1 LBM_c0(<i>)*f_<i> >END_SUM) / rho_kap;
        REG v_kap = (SUM< i 0 LBM_size 1 LBM_c1(<i>)*f_<i> >END_SUM) / rho_kap;
        INIF LBM_dim==2
            udotu = u_kap*u_kap + v_kap*v_kap;
        INELSE
            REG w_kap = (SUM< i 0 LBM_size 1 N_Pf(LBM_c2(<i>))*f_<i> >END_SUM) / rho_kap;
            REG udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
        END_INIF
        
        # For each DDF, compute sample points and place in shared memory. Then compute interpolation weights, and interpolate to each child block.
        INFOR p 1   0 LBM_size 1
            //
            // DDF <p>.
            //
            REG cdotu = LBM_c0(<p>)*u_kap + LBM_c1(<p>)*v_kap + LBM_c2(<p>)*w_kap;
            REG tmp_i = N_Pf(LBM_w(<p>))*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
            REG s_F[threadIdx.x] = tmp_i + (f_<p> - tmp_i)*(tau_ratio);
            REG __syncthreads();
            INIF LBM_dim==2
            
                # Reset result registers.
                INFOR q 1   0 4 1
                    REG S_res_<q> = N_Pf(0.0);
                END_INFOR
                
                # Loop in Y.
                INFOR j 1   0 4 1
                    # Reset intermediate register for beta_jk.
                    INFOR q 1   0 4 1
                        REG S_beta_<q> = N_Pf(0.0);
                    END_INFOR
                    INFOR i 1   0 4 1
                        // Compute weight alpha_ijk (^I< 3-<i> >^,^I< 3-<j> >^).
                        REG alpha = SUM<m 0 16 1 (IMP_VEC(A_2D,^I< <m>+16*( (3-<i>)+4*(3-<j>) ) >^)END_IMP)*s_F[<m>]>END_SUM;
                        
                        # Update beta_jk.
                        INFOR iijj 2   0 2 1   0 2 1
                            REG S_beta_^I<<ii>+2*<jj>>^ = alpha + (x_kap+N_Pf(^D< <ii>*(2.0/3.0) >^))*S_beta_^I<<ii>+2*<jj>>^;
                        END_INFOR
                    END_INFOR
            
                    # Update the final result for all child blocks.
                    INFOR iijj 2   0 2 1   0 2 1
                        REG S_res_^I<<ii>+2*<jj>>^ = S_beta_^I<<ii>+2*<jj>>^ + (y_kap+N_Pf(^D< <jj>*(2.0/3.0) >^))*S_res_^I<<ii>+2*<jj>>^;
                    END_INFOR
                END_INFOR
            
            INELSE
            
                # Reset result registers.
                INFOR q 1   0 8 1
                    REG S_res_<q> = N_Pf(0.0);
                END_INFOR
                
                # Loop over polynomial weight computation. Start in Z. Work in reverse for Horners' decomposition (3,2,1,0).
                INFOR k 1   0 4 1
                    # Reset intermediate register for gamma_k.
                    INFOR q 1   0 8 1
                        REG S_gamma_<q> = N_Pf(0.0);
                    END_INFOR
                    
                    # Loop in Y.
                    INFOR j 1   0 4 1
                        # Reset intermediate register for beta_jk.
                        INFOR q 1   0 8 1
                            REG S_beta_<q> = N_Pf(0.0);
                        END_INFOR
                        INFOR i 1   0 4 1
                            // Compute weight alpha_ijk (^I< 3-<i> >^,^I< 3-<j> >^,^I< 3-<k> >^).
                            REG alpha = SUM<m 0 64 1 (IMP_VEC(A_3D,^I< <m>+64*( (3-<i>)+4*(3-<j>)+16*(3-<k>)) >^)END_IMP)*s_F[<m>]>END_SUM;
                            
                            # Update beta_jk.
                            INFOR iijjkk 2   0 2 1   0 2 1   0 2 1
                                REG S_beta_^I<<ii>+2*<jj>+4*<kk>>^ = alpha + (x_kap+N_Pf(^D< <ii>*(2.0/3.0) >^))*S_beta_^I<<ii>+2*<jj>+4*<kk>>^;
                            END_INFOR
                        END_INFOR
                        # Update gamma_k.
                        INFOR iijjkk 2   0 2 1   0 2 1   0 2 1
                            REG S_gamma_^I<<ii>+2*<jj>+4*<kk>>^ = S_beta_^I<<ii>+2*<jj>+4*<kk>>^ + (y_kap+N_Pf(^D< <jj>*(2.0/3.0) >^))*S_gamma_^I<<ii>+2*<jj>+4*<kk>>^;
                        END_INFOR
                    END_INFOR
                    
                    # Update the final result for all child blocks.
                    INFOR iijjkk 2   0 2 1   0 2 1   0 2 1
                        REG S_res_^I<<ii>+2*<jj>+4*<kk>>^ = S_gamma_^I<<ii>+2*<jj>+4*<kk>>^ + (z_kap+N_Pf(^D< <kk>*(2.0/3.0) >^))*S_res_^I<<ii>+2*<jj>+4*<kk>>^;
                    END_INFOR
                END_INFOR
                
            END_INIF
            
            <
            # Write result to child blocks.
            INFOR xc_ixc_jxc_k 4   0 2 1   0 2 1   0 ^I< LBM_dim-1 >^ 1
                //\tChild quadrant ^I< <xc_i>+2*<xc_j>+4*<xc_k> >^.
                INIF LBM_dim==2
                    REG i_Qcp = i_Qc + (<xc_i>+Nqx*<xc_j>);
                INELSE
                    REG i_Qcp = i_Qc + (<xc_i>+Nqx*<xc_j>+Nqx*Nqx*<xc_k>);
                END_INIF
                OUTIF (interp_type == 0 && cells_ID_mask[(i_kap_bc+IKAPC)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x]==2) || (interp_type == 1)
                    REG cells_f_F[(i_kap_bc+IKAPC)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells] = S_res_^I<<xc_i> + 2*<xc_j> + 4*<xc_k> >^;
                END_OUTIF
                REG __syncthreads();
            END_INFOR
            
        END_INFOR
    END_INFOR
END_OUTFOR 2
INIF LBM_dim==3
    END_OUTFOR
END_INIF


END_LOOPBLOCKS
