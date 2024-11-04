# Kernel name and includes.
NAME UpdateCellsMasks
INCLUDE_GUARD N_DIM==LBM_dim
OUTPUT_DIR ./output/

# Kernel parameters.

TEMPLATE int ave_type=0
TEMPLATE_VALS 0
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
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_INTERFACE
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_BLOCK
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_AVERAGE_GRID
ROUTINE_OBJECT Solver_LBM



#
# Kernel definition.
#

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ int s_ID_mask[M_TBLOCK];
REG int kap = blockIdx.x*blockDim.x + threadIdx.x;
REG int I_kap = threadIdx.x % Nbx;
REG int J_kap = (threadIdx.x / Nbx) % Nbx;
INIF LBM_dim==3
    int K_kap = (threadIdx.x / Nbx) / Nbx;
END_INIF

<
// Keep in mind that each ID represents a block, not just a cell.
REG s_ID_cblock[threadIdx.x] = -1;
OUTIF kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE
    REG s_ID_cblock[threadIdx.x] = kap;
END_OUTIF
REG __syncthreads();

<
# Looping over blocks.
OUTFOR k 1   0 M_TBLOCK 1
    REG int i_kap_b = s_ID_cblock[k];
    <
    OUTIF i_kap_b>-1

        INIF LBM_dim==3
            OUTFOR k_q 3   0 Nqx 1
        END_INIF
        
        OUTFOR i_qj_q 3   0 Nqx 1   0 Nqx 1
            INIF LBM_dim==2
                REG int i_Q = i_q+Nqx*j_q+;
            INELSE
                REG int i_Q = i_q+Nqx*j_q+Nqx*Nqx*k_q;
            END_INIF
            REG s_ID_mask[threadIdx.x] = 0;
            
            INFOR p 1   1 LBM_size 1
                # Build the condition on the quadrant.
                DEFINE Q_COND 
                INIF (LBM_c0(<p>)==-1)
                    DEFINE DEF_PUSH and Q_COND (i_q==0)
                END_INIF
                INIF (LBM_c0(<p>)==1)
                    DEFINE DEF_PUSH and Q_COND (i_q==Nqx-1)
                END_INIF
                INIF (LBM_c1(<p>)==-1)
                    DEFINE DEF_PUSH and Q_COND (j_q==0)
                END_INIF
                INIF (LBM_c1(<p>)==1)
                    DEFINE DEF_PUSH and Q_COND (j_q==Nqx-1)
                END_INIF
                INIF (LBM_dim==3)and(LBM_c2(<p>)==-1)
                    DEFINE DEF_PUSH and Q_COND (k_q==0)
                END_INIF
                INIF (LBM_dim==3)and(LBM_c2(<p>)==1)
                    DEFINE DEF_PUSH and Q_COND (k_q==Nqx-1)
                END_INIF
                
                # If quadrant satisfies condition, set cells to 'interface'. Then, overlap with ghost cells.
                OUTIF_S (cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks]==N_SKIPID)and(Q_COND)
                    REG s_ID_mask[threadIdx.x] = 1;
                END_OUTIF_S
            END_INFOR
            INFOR p 1   1 LBM_size 1
                # Build the condition on the quadrant and cells.
                DEFINE Q_COND 
                DEFINE C_COND 
                INIF (LBM_c0(<p>)==-1)
                    DEFINE DEF_PUSH and Q_COND (i_q==0)
                    DEFINE DEF_PUSH and C_COND (I_kap < 2)
                END_INIF
                INIF (LBM_c0(<p>)==1)
                    DEFINE DEF_PUSH and Q_COND (i_q==Nqx-1)
                    DEFINE DEF_PUSH and C_COND (I_kap >= 2)
                END_INIF
                INIF (LBM_c1(<p>)==-1)
                    DEFINE DEF_PUSH and Q_COND (j_q==0)
                    DEFINE DEF_PUSH and C_COND (J_kap < 2)
                END_INIF
                INIF (LBM_c1(<p>)==1)
                    DEFINE DEF_PUSH and Q_COND (j_q==Nqx-1)
                    DEFINE DEF_PUSH and C_COND (J_kap >= 2)
                END_INIF
                INIF (LBM_dim==3)and(LBM_c2(<p>)==-1)
                    DEFINE DEF_PUSH and Q_COND (k_q==0)
                    DEFINE DEF_PUSH and C_COND (K_kap < 2)
                END_INIF
                INIF (LBM_dim==3)and(LBM_c2(<p>)==1)
                    DEFINE DEF_PUSH and Q_COND (k_q==Nqx-1)
                    DEFINE DEF_PUSH and C_COND (K_kap >= 2)
                END_INIF
            
                # Overlap interface cells with ghost cells.
                OUTIF (cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks]==N_SKIPID)and(Q_COND)
                    OUTIF_S C_COND
                        REG s_ID_mask[threadIdx.x] = 2;
                    END_OUTIF_S
                END_OUTIF
            END_INFOR
            
            REG cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] = s_ID_mask[threadIdx.x];
            REG __syncthreads();
        END_OUTFOR
            
        INIF LBM_dim==3
            END_OUTFOR
        END_INIF
         

    END_OUTIF
END_OUTFOR
