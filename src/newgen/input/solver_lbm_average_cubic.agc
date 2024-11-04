# Kernel name and includes.
NAME Average_Cubic_LBM_name
NAME_FILE solver_lbm_average_cubic
INCLUDE "solver.h"
INCLUDE "mesh.h"
INCLUDE_GUARD N_Q==LBM_size

# Kernel parameters.
TEMPLATE int ave_type=1
TEMPLATE_VALS 1
TEMPLATE_VALS 2
KERNEL_REQUIRE int n_ids_idev_L ROUTINE mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t dx_L ROUTINE mesh->dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L
KERNEL_REQUIRE ufloat_t tau_ratio ROUTINE tau_ratio_L
KERNEL_REQUIRE ufloat_t v0
KERNEL_REQUIRE int *id_set_idev_L ROUTINE &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask ROUTINE mesh->c_cells_ID_mask[i_dev]
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
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_BLOCK
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_GRID
ROUTINE_OBJECT Solver_LBM

# Import interpolation matrix.
IMPORT VECTOR mat_interp_2D_double.txt A_2D
IMPORT VECTOR mat_interp_3D_double.txt A_3D



#
# Kernel definition.
#

REG __shared__ int s_ID_cblock[M_CBLOCK];
REG __shared__ int s_ID_mask[M_CBLOCK];
REG __shared__ ufloat_t s_Fc[M_CBLOCK];
REG int I_kap = threadIdx.x % Nbx;
REG int J_kap = (threadIdx.x / Nbx) % Nbx;
INIF LBM_dim==3
    int K_kap = (threadIdx.x / Nbx) / Nbx;
END_INIF
REG ufloat_t x_kap = N_Pf(^D< -1.0/2.0 >^)+N_Pf(^D< 2.0/3.0 >^)*(threadIdx.x % Nbx);
REG ufloat_t y_kap = N_Pf(^D< -1.0/2.0 >^)+N_Pf(^D< 2.0/3.0 >^)*((threadIdx.x / Nbx) % Nbx);
INIF LBM_dim==3
    ufloat_t z_kap = N_Pf(^D< -1.0/2.0 >^)+N_Pf(^D< 2.0/3.0 >^)*((threadIdx.x / Nbx) / Nbx);
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
#REG int mask_parent = -1;
#REG int child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2)) + Nbx*Nbx*(2*(((threadIdx.x / Nbx) / Nbx)%2));
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
REG ufloat_t S_beta = N_Pf(0.0);
INIF LBM_dim==3
    REG ufloat_t S_gamma = N_Pf(0.0);
END_INIF
REG ufloat_t S_res = N_Pf(0.0);



LOOPBLOCKS 2\
    CONDITION ((i_kap_bc>-1)and((ave_type==2)or(block_on_boundary==1)))\
    REQUIRING 2 i_kap_bc=cblock_ID_nbr_child[i_kap_b] block_on_boundary=cblock_ID_mask[i_kap_b]



INFOR xc_ixc_jxc_k 4   0 2 1   0 2 1   0 2 1
    INIF (<xc_k>==0 and LBM_dim==2)or(LBM_dim==3)
        DEFINE q ^I< <xc_i>+2*<xc_j>+4*<xc_k> >^

        // Check if child block is ghost or interior. Proceed only if interior.
        REG s_ID_mask[threadIdx.x] = cells_ID_mask[(i_kap_bc+q)*M_CBLOCK + threadIdx.x];
        REG __syncthreads();
        INIF LBM_dim==2
            #DEFINE MASK_COND (s_ID_mask[0]==0)and(s_ID_mask[3]==0)and(s_ID_mask[12]==0)and(s_ID_mask[15]==0)
            DEFINE PARENT_COND (I_kap >= ^I< 2*<xc_i> >^)and(I_kap < ^I< 2*<xc_i>+2 >^)and(J_kap >= ^I< 2*<xc_j> >^)and(J_kap < ^I< 2*<xc_j>+2 >^)
        INELSE
            #DEFINE MASK_COND (s_ID_mask[0]==0)and(s_ID_mask[3]==0)and(s_ID_mask[12]==0)and(s_ID_mask[15]==0)and(s_ID_mask[48]==0)and(s_ID_mask[51]==0)and(s_ID_mask[60]==0)and(s_ID_mask[63]==0)
            DEFINE PARENT_COND (I_kap >= ^I< 2*<xc_i> >^)and(I_kap < ^I< 2*<xc_i>+2 >^)and(J_kap >= ^I< 2*<xc_j> >^)and(J_kap < ^I< 2*<xc_j>+2 >^)and(K_kap >= ^I< 2*<xc_k> >^)and(K_kap < ^I< 2*<xc_k>+2 >^)
        END_INIF
        
        # If in the interior...
        OUTIF s_ID_mask[threadIdx.x]==0
        
        
            // Load DDFs and compute macroscopic properties.
            INFOR p 1   0 LBM_size 1
                REG f_<p> = cells_f_F[(i_kap_bc+q)*M_CBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells];
            END_INFOR
            REG rho_kap = SUM< i 0 LBM_size 1 f_<i> >END_SUM;
            REG u_kap = (SUM< i 0 LBM_size 1 LBM_c0(<i>)*f_<i> >END_SUM) / rho_kap;
            REG v_kap = (SUM< i 0 LBM_size 1 LBM_c1(<i>)*f_<i> >END_SUM) / rho_kap;
            REG w_kap = (SUM< i 0 LBM_size 1 LBM_c2(<i>)*f_<i> >END_SUM) / rho_kap;
            REG udotu = u_kap*u_kap + v_kap*v_kap + LBM_3D*w_kap*w_kap;
            REG x_kap = N_Pf(^D< (1.0/6.0) - <xc_i>*(4.0/3.0) >^) + N_Pf(^D< 2.0/3.0 >^)*I_kap;
            REG y_kap = N_Pf(^D< (1.0/6.0) - <xc_j>*(4.0/3.0) >^) + N_Pf(^D< 2.0/3.0 >^)*J_kap;
            INIF LBM_dim==3
                REG z_kap = N_Pf(^D< (1.0/6.0) - <xc_k>*(4.0/3.0) >^) + N_Pf(^D< 2.0/3.0 >^)*K_kap;
            END_INIF

            <
            # Now interpolate from selected child block to the parent block.
            INFOR p 1   0 LBM_size 1
                //
                // DDF <p>.
                //
                REG cdotu = LBM_c0(<p>)*u_kap + LBM_c1(<p>)*v_kap + LBM_c2(<p>)*w_kap;
                REG tmp_i = N_Pf(LBM_w(<p>))*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
                REG s_Fc[threadIdx.x] = tmp_i + (f_<p> - tmp_i)*(tau_ratio);
                REG __syncthreads();
                INIF LBM_dim==2
                
                    # Reset result registers.
                    REG S_res = N_Pf(0.0);
                    
                    # Loop in Y.
                    INFOR j 1   0 4 1
                        # Reset intermediate register for beta_jk.
                        S_beta = N_Pf(0.0);
                        INFOR i 1   0 4 1
                            // Compute weight alpha_ijk (^I< 3-<i> >^,^I< 3-<j> >^).
                            REG alpha = SUM<m 0 16 1 (IMP_VEC(A_2D,^I< <m>+16*( (3-<i>)+4*(3-<j>) ) >^)END_IMP)*s_Fc[<m>]>END_SUM;
                            
                            # Update beta_jk.
                            REG S_beta = alpha + x_kap*S_beta;
                        END_INFOR
                
                        # Update the final result for all child blocks.
                        REG S_res = S_beta + y_kap*S_res;
                    END_INFOR
                
                INELSE
                
                    # Reset result registers.
                    REG S_res = N_Pf(0.0);
                    
                    # Loop over polynomial weight computation. Start in Z. Work in reverse for Horners' decomposition (3,2,1,0).
                    INFOR k 1   0 4 1
                        # Reset intermediate register for gamma_k.
                        REG S_gamma = N_Pf(0.0);
                        
                        # Loop in Y.
                        INFOR j 1   0 4 1
                            # Reset intermediate register for beta_jk.
                            REG S_beta = N_Pf(0.0);
                            INFOR i 1   0 4 1
                                // Compute weight alpha_ijk (^I< 3-<i> >^,^I< 3-<j> >^,^I< 3-<k> >^).
                                REG alpha = SUM<m 0 64 1 (IMP_VEC(A_3D,^I< <m>+64*( (3-<i>)+4*(3-<j>)+16*(3-<k>)) >^)END_IMP)*s_F[<m>]>END_SUM;
                                
                                # Update beta_jk.
                                S_beta = alpha + x_kap*S_beta;
                            END_INFOR
                            # Update gamma_k.
                            S_gamma = S_beta + y_kap*S_gamma;
                        END_INFOR
                        
                        # Update the final result for all child blocks.
                        REG S_res = S_gamma + z_kap*S_res;
                    END_INFOR
                    
                END_INIF
                
                <
                // Write result to child blocks.
                OUTIF PARENT_COND
                    REG cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells] = S_res;
                END_OUTIF
                REG __syncthreads();
            END_INFOR
            <
            
            
        END_OUTIF
    END_INIF
END_INFOR



END_LOOPBLOCKS
