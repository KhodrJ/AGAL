# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_average_original
FILE_DIR ../solver_lbm/

ROUTINE_NAME Average_Original
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var

KERNEL_REQUIRE int n_ids_idev_L           | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t tau_ratio         | tau_vec[L]/tau_vec[L+1]
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


#
# Kernel definition.
#

REG constexpr int Nqx = AP->Nqx;
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ int s_ID_mask_child[M_TBLOCK];
REG __shared__ ufloat_t s_Fc[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int J = (threadIdx.x / 4) % 4;
INIF Ldim==3
    int K = (threadIdx.x / 4) / 4;
END_INIF
REG int i_Q = -1;
REG int i_Qc = -1;
REG int i_Qcp = -1;
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
#REG int mask_parent = -1;
INIF Ldim==2
    REG int child0_IJK = 2*((threadIdx.x % 4)%2) + 4*(2*(((threadIdx.x / 4) % 4)%2));
INELSE
    REG int child0_IJK = 2*((threadIdx.x % 4)%2) + 4*(2*(((threadIdx.x / 4) % 4)%2)) + 4*4*(2*(((threadIdx.x / 4) / 4)%2));
END_INIF
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



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc>-1)and((ave_type==2)or(block_on_boundary==1))
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG block_on_boundary=cblock_ID_mask[i_kap_b];
    
    INIF Ldim==3
        OUTFOR k_q 3   0 Nqx/2 1
    END_INIF   
    OUTFOR i_qj_q 3   0 Nqx/2 1   0 Nqx/2 1
        INFOR I_qJ_qK_q 3   0 2 1   0 2 1   0 gI(Ldim-1) 1
        //
        REG-m // Child block gI(<I_q>+2*<J_q>+4*<K_q>).
        // 
        //
        INIF Ldim==2
            REG i_Q = (i_q+<I_q>*Nqx/2) + Nqx*(j_q+<J_q>*Nqx/2);
            REG i_Qc = 2*i_q + 2*Nqx*j_q;
        INELSE
            REG i_Q = (i_q+<I_q>*Nqx/2) + Nqx*(j_q+<J_q>*Nqx/2) + Nqx*Nqx*(k_q+<K_q>*Nqx/2);
            REG i_Qc = 2*i_q + 2*Nqx*j_q + 2*Nqx*Nqx*k_q;
        END_INIF
            INIF Ldim==3
                OUTFOR xc_k 4   0 2 1
            END_INIF
            OUTFOR xc_ixc_j 4   0 2 1   0 2 1

                INIF Ldim==2
                    #REG xc = xc_i + 2*xc_j;
                    i_Qcp = i_Qc + xc_i + Nqx*xc_j;
                INELSE
                    #REG xc = xc_i + 2*xc_j + 4*xc_k;
                    i_Qcp = i_Qc + xc_i + Nqx*xc_j + Nqx*Nqx*xc_k;
                END_INIF
                
                <
                // Load DDFs and compute macroscopic properties.
                INFOR p 1   0 Lsize 1
                    REG-vm f_<p> = cells_f_F[(i_kap_bc+gI(<I_q>+2*<J_q>+4*<K_q>))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
                END_INFOR
                REG-vsvz rho = gNz(gSUM(p,0 Lsize 1,f_<p>));
                REG-vsvz u = (gNz(gSUM(p,0 Lsize 1,(Lc0(<p>))*f_<p>))) / rho;
                REG-vsvz v = (gNz(gSUM(p,0 Lsize 1,(Lc1(<p>))*f_<p>))) / rho;
                INIF Ldim==2
                    REG udotu = u*u + v*v;
                INELSE
                    REG-vsvz w = (gNz(gSUM(i,0 Lsize 1,(Lc2(<i>))*f_<i>))) / rho;
                    REG udotu = u*u + v*v + w*w;
                END_INIF
                
                <
                // Average rescaled fi to parent if applicable.
                REG-m s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc+gI(<I_q>+2*<J_q>+4*<K_q>))*M_CBLOCK + i_Qcp*M_TBLOCK + threadIdx.x];
                OUTIF ((ave_type > 0)and(s_ID_mask_child[threadIdx.x] < 2))
                    REG s_ID_mask_child[threadIdx.x] = 1;
                END_OUTIF
                REG __syncthreads();
                
                    INFOR p 1   0 Lsize 1
                        <
                        //\t p = <p>
                        REG-vz cdotu = gNz((Lc0(<p>))*u + (Lc1(<p>))*v + (Lc2(<p>))*w);
                        REG-vm tmp_i = (ufloat_t)(gD(Lw(<p>)))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu);
                        REG s_Fc[threadIdx.x] = tmp_i + (f_<p> - tmp_i)*tau_ratio;
                        REG __syncthreads();
                        INIF Ldim==2
                            OUTIF ((s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j))
                        INELSE
                            OUTIF ((s_ID_mask_child[child0_IJK] == 1) and (I >= 2*xc_i) and (I < 2+2*xc_i) and (J >= 2*xc_j) and (J < 2+2*xc_j) and (K >= 2*xc_k) and (K < 2+2*xc_k))
                        END_INIF
                        
                            INIF Ldim==2
                                REG-v cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = (ufloat_t)(0.25)*( s_Fc[(child0_IJK + 0 + 4*0)] + s_Fc[(child0_IJK + 1 + 4*0)] + s_Fc[(child0_IJK + 0 + 4*1)] + s_Fc[(child0_IJK + 1 + 4*1)] );
                            INELSE
                                REG-v cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = (ufloat_t)(0.125)*( s_Fc[(child0_IJK + 0 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*0)] + s_Fc[(child0_IJK + 0 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*0 + 4*4*1)] + s_Fc[(child0_IJK + 0 + 4*1 + 4*4*1)] + s_Fc[(child0_IJK + 1 + 4*1 + 4*4*1)] );
                            END_INIF
                            
                        END_OUTIF
                        REG __syncthreads();
                    END_INFOR

            END_OUTFOR 2
            INIF Ldim==3
                END_OUTFOR 1
            END_INIF
        #
        #
        #
        END_INFOR
    END_OUTFOR 2
    INIF Ldim==3
        END_OUTFOR
    END_INIF
    
TEMPLATE NEW_BLOCK
END_TEMPLATE
