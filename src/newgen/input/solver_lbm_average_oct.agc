# Kernel name and includes.
NAME Average_LBM_name
NAME_FILE solver_lbm_average_mq
INCLUDE "solver.h"
INCLUDE "mesh.h"
INCLUDE_GUARD N_Q==LBM_size

# Kernel parameters.
TEMPLATE int ave_type=1
TEMPLATE_VALS 0
TEMPLATE_VALS 1
TEMPLATE_VALS 2
KERNEL_REQUIRE int n_ids_idev_L ROUTINE mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t tau_L
KERNEL_REQUIRE ufloat_t tau_ratio ROUTINE tau_ratio_L
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
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_INTERFACE
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_BLOCK
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_GRID
ROUTINE_OBJECT Solver_LBM



#
# Kernel definition.
#

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ int s_ID_mask_child[M_TBLOCK];
REG __shared__ ufloat_t s_Fc[M_TBLOCK];
REG int I_kap = threadIdx.x % Nbx;
REG int J_kap = (threadIdx.x / Nbx) % Nbx;
INIF LBM_dim==3
    int K_kap = (threadIdx.x / Nbx) / Nbx;
END_INIF
REG int i_Q = -1;
REG int i_Qc = -1;
REG int i_Qcp = -1;
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
#REG int mask_parent = -1;
INIF LBM_dim==2
    REG int child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2));
INELSE
    REG int child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2)) + Nbx*Nbx*(2*(((threadIdx.x / Nbx) / Nbx)%2));
END_INIF
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

LOOPBLOCKS 2\
    CONDITION ((i_kap_bc>-1)and((ave_type==2)or(block_on_boundary==1)))\
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
        INIF LBM_dim==2
            REG i_Q = (i_q+<I_q>*Nqx/2) + Nqx*(j_q+<J_q>*Nqx/2);
            REG i_Qc = 2*i_q + 2*Nqx*j_q;
        INELSE
            REG i_Q = (i_q+<I_q>*Nqx/2) + Nqx*(j_q+<J_q>*Nqx/2) + Nqx*Nqx*(k_q+<K_q>*Nqx/2);
            REG i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
        END_INIF
            INIF LBM_dim==3
                OUTFOR xc_k 4   0 2 1
            END_INIF
            OUTFOR xc_ixc_j 4   0 2 1   0 2 1

                INIF LBM_dim==2
                    #REG xc = xc_i + 2*xc_j;
                    i_Qcp = i_Qc + xc_i + Nqx*xc_j;
                INELSE
                    #REG xc = xc_i + 2*xc_j + 4*xc_k;
                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                END_INIF
                
                <
                // Load DDFs and compute macroscopic properties.
                INFOR p 1   0 LBM_size 1
                    REG f_<p> = cells_f_F[(i_kap_bc+IKAPC)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells];
                END_INFOR
                REG rho_kap = SUM<p 0 LBM_size 1 f_<p>>END_SUM;
                REG u_kap = (SUM<p 0 LBM_size 1 N_Pf(LBM_c0(<p>))*f_<p>>END_SUM) / rho_kap;
                REG v_kap = (SUM<p 0 LBM_size 1 N_Pf(LBM_c1(<p>))*f_<p>>END_SUM) / rho_kap;
                INIF LBM_dim==2
                    REG udotu = u_kap*u_kap + v_kap*v_kap;
                INELSE
                    REG w_kap = (SUM< i 0 LBM_size 1 N_Pf(LBM_c2(<i>))*f_<i>>END_SUM) / rho_kap;
                    REG udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
                END_INIF
                
                <
                // Average rescaled fi to parent if applicable.
                REG s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+IKAPC)*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                OUTIF (ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2)
                    REG s_ID_mask_child[threadIdx.x] = 1;
                END_OUTIF
                REG __syncthreads();
                
                    INFOR p 1   0 LBM_size 1
                        <
                        //\t p = <p>
                        REG cdotu = LBM_c0(<p>)*u_kap + LBM_c1(<p>)*v_kap + LBM_c2(<p>)*w_kap;
                        REG tmp_i = N_Pf(LBM_w(<p>))*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
                        REG s_Fc[threadIdx.x] = tmp_i + (f_<p> - tmp_i)*tau_ratio;
                        REG __syncthreads();
                        INIF LBM_dim==2
                            OUTIF (s_ID_mask_child[child0_IJK] == 1) and (I_kap >= 2*xc_i) and (I_kap < 2+2*xc_i) and (J_kap >= 2*xc_j) and (J_kap < 2+2*xc_j)
                        INELSE
                            OUTIF (s_ID_mask_child[child0_IJK] == 1) and (I_kap >= 2*xc_i) and (I_kap < 2+2*xc_i) and (J_kap >= 2*xc_j) and (J_kap < 2+2*xc_j) and (K_kap >= 2*xc_k) and (K_kap < 2+2*xc_k)
                        END_INIF
                        
                            INIF LBM_dim==2
                                REG cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] );
                            INELSE
                                REG cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] );
                            END_INIF
                            
                        END_OUTIF
                        REG __syncthreads();
                    END_INFOR

            END_OUTFOR 2
            INIF LBM_dim==3
                END_OUTFOR 1
            END_INIF
        #
        #
        #
        END_INFOR
    END_OUTFOR 2
    INIF LBM_dim==3
        END_OUTFOR
    END_INIF
    

END_LOOPBLOCKS

END_FILE
