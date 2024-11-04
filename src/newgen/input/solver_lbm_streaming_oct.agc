# Kernel name and includes.
NAME Stream_Inpl_LBM_name
NAME_FILE solver_lbm_stream_inpl
INCLUDE "solver.h"
INCLUDE "mesh.h"
INCLUDE_GUARD N_Q==LBM_size

# Kernel parameters.
KERNEL_REQUIRE int n_ids_idev_L ROUTINE mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
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
ROUTINE_COND mesh->n_ids[i_dev][L] > 0
ROUTINE_OBJECT Solver_LBM



#
# Kernel definition.
#

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ ufloat_t s_F_p[PROD<i 0 LBM_dim 1 (Nbx+2)>END_PROD];
REG __shared__ ufloat_t s_F_pb[PROD<i 0 LBM_dim 1 (Nbx+2)>END_PROD];
REG int I_kap = threadIdx.x % Nbx;
REG int J_kap = (threadIdx.x / Nbx) % Nbx;
INIF LBM_dim==3
    int K_kap = (threadIdx.x / Nbx) / Nbx;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int i_Q = -1;
REG int nbr_i_q = -1;
REG int nbr_j_q = -1;
INIF LBM_dim==3
    REG int nbr_k_q = -1;
END_INIF
REG int nbr_kap_b = -1;
REG int nbr_Q_b = -1;
INFOR q 1   1 LBM_size 1
    INIF  (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0)
        REG int nbr_id_global_<q> = -1;
    END_INIF
END_INFOR
INFOR q 1   1 LBM_size 1
    INIF  (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0)
        REG int nbr_id_<q> = -1;
    END_INIF
END_INFOR
INFOR q 1   1 LBM_size 1
    INIF  (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0)
        REG int nbr_Q_<q> = -1;
    END_INIF
END_INFOR
REG int block_on_boundary = -1;
REG size_t index;


LOOPBLOCKS 2\
    CONDITION ((i_kap_bc<0)||(block_on_boundary==1))\
    REQUIRING 2 i_kap_bc=cblock_ID_nbr_child[i_kap_b] block_on_boundary=cblock_ID_mask[i_kap_b]



INFOR q 1   1 LBM_size 1
    INIF  (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0)
        REG nbr_id_global_<q> = cblock_ID_nbr[i_kap_b + <q>*n_maxcblocks];
    END_INIF
END_INFOR

INIF LBM_dim==3
    OUTFOR k_q 3   0 Nqx 1
END_INIF
OUTFOR i_qj_q 3   0 Nqx 1   0 Nqx 1
    
    # Compute quadrant/octant Id.
    INIF LBM_dim==2
        REG i_Q = i_q + Nqx*j_q;
    INELSE
        REG i_Q = i_q + Nqx*j_q + Nqx*Nqx*k_q;
    END_INIF
    
    # Compute local (to quadrant) neighbor quadrants and blocks.
    INFOR q 1   1 LBM_size 1
        INIF  (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0)
            REG nbr_id_<q> = i_kap_b;
            REG nbr_i_q = i_q+^I< LBM_c0(<q>) >^; if (nbr_i_q == Nqx) nbr_i_q = 0; 
            REG nbr_j_q = j_q+^I< LBM_c1(<q>) >^; if (nbr_j_q == Nqx) nbr_j_q = 0; 
            INIF LBM_dim==3
                REG nbr_k_q = k_q+^I< LBM_c2(<q>) >^; if (nbr_k_q == Nqx) nbr_k_q = 0; 
            END_INIF
            
            INIF LBM_dim==2
                REG nbr_Q_<q> = nbr_i_q + Nqx*nbr_j_q;
            INELSE
                REG nbr_Q_<q> = nbr_i_q + Nqx*nbr_j_q + Nqx*Nqx*nbr_k_q;
            END_INIF
        END_INIF
    END_INFOR
    INFOR q 1   1 LBM_size 1
        INIF (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0)
            // Nbr <q>.
            INFOR lmn 1   0 2 1   0 2 1   0 2 1
                INIF (^I< <l>+2*<m>+4*<n> >^ > 0) and ( (LBM_c0(<q>)-<l> >= 0) and (LBM_c1(<q>)-<m> >= 0) and (LBM_c2(<q>)-<n> >= 0) )
                    // Consider nbr LBM_nbr(^I<<l>>^,^I<<m>>^,^I<<n>>^) (<l><m><n>).
                    DEFINE NBR_COND 
                    INIF <l>==1
                        DEFINE DEF_PUSH and NBR_COND (i_q==Nqx-1)
                    END_INIF
                    INIF <l>==0 and LBM_c0(<q>)!=0
                        DEFINE DEF_PUSH and NBR_COND (i_q<Nqx-1)
                    END_INIF
                    INIF <m>==1
                        DEFINE DEF_PUSH and NBR_COND (j_q==Nqx-1)
                    END_INIF
                    INIF <m>==0 and LBM_c1(<q>)!=0
                        DEFINE DEF_PUSH and NBR_COND (j_q<Nqx-1)
                    END_INIF
                    INIF LBM_dim==3
                        INIF <n>==1
                            DEFINE DEF_PUSH and NBR_COND (k_q==Nqx-1)
                        END_INIF
                        INIF <n>==0 and LBM_c2(<q>)!=0
                            DEFINE DEF_PUSH and NBR_COND (k_q<Nqx-1)
                        END_INIF
                    END_INIF
                    
                    OUTIF_S NBR_COND
                        REG nbr_id_<q> = nbr_id_global_LBM_nbr(^I<<l>>^,^I<<m>>^,^I<<n>>^);
                    END_OUTIF_S
                END_INIF
            END_INFOR
        END_INIF
    END_INFOR
    
    <
    # Loop over all DDFs for that quadrant/octant.
    INFOR p 1   1 LBM_size 1
        INIF  ( (LBM_dim==2)and((<p>==1)or((<p>+1)%3==0)) ) or ( (LBM_dim==3)and( (((<p>-1)%2==0)and(<p><25))or(<p>==26) ) ) 
        
            # o====================================================================================
            # | Reset shared memory halos, load DDFs in the interior.
            # o====================================================================================
            
            //
            // DDFs p=<p>, pb=LBM_pb(<p>).
            //
            INIF LBM_dim==2
                OUTFOR q 1   0 3 1
                        OUTIF threadIdx.x + q*16 < 36
                        REG s_F_p[threadIdx.x + q*16] = N_Pf(-1.0);
                        REG s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0);
                        END_OUTIF
                END_OUTFOR
            INELSE
                OUTFOR q 1   0 4 1
                        OUTIF threadIdx.x + q*64 < 216
                        REG s_F_p[threadIdx.x + q*64] = N_Pf(-1.0);
                        REG s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0);
                        END_OUTIF
                END_OUTFOR
            END_INIF
            REG __syncthreads();
            REG index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + <p>*n_maxcells;
            REG s_F_p[H_INDEX(LBM_dim,0,0,0)END_H_INDEX] = cells_f_F[index];
            REG index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells;
            REG s_F_pb[H_INDEX(LBM_dim,0,0,0)END_H_INDEX] = cells_f_F[index];
            
            # o====================================================================================
            # | Retrieve DDFs from neighbors and place in shared memory halo.
            # o====================================================================================
            
            INFOR q 1   1 LBM_size 1
                INIF  (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0) 
                
                    INIF  (LBM_c0(<p>)==LBM_c0(<q>) or LBM_c0(<q>)==0) and (LBM_c1(<p>)==LBM_c1(<q>) or LBM_c1(<q>)==0) and (LBM_c2(<p>)==LBM_c2(<q>) or LBM_c2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for p=<p>.
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF nbr_kap_b >= 0
                            #OUTIF LBM_COND_BC( LBM_c0(<p>),LBM_c0(LBM_pb(<q>)),  LBM_c1(<p>),LBM_c1(LBM_pb(<q>)),  LBM_c2(<p>),LBM_c2(LBM_pb(<q>)) )END_LBM_COND_BC
                            OUTIF LBM_COND_BC( 0,LBM_c0(LBM_pb(<q>)),  0,LBM_c1(LBM_pb(<q>)),  0,LBM_c2(LBM_pb(<q>)) )END_LBM_COND_BC
                                
                                REG index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells;
                                REG s_F_pb[H_LIMIT( LBM_dim,LBM_c0(<q>),LBM_c1(<q>),LBM_c2(<q>) )END_H_LIMIT] = cells_f_F[index];
                                
                            END_OUTIF
                        END_OUTIF
                    END_INIF
                    INIF  (LBM_c0(LBM_pb(<p>))==LBM_c0(<q>) or LBM_c0(<q>)==0) and (LBM_c1(LBM_pb(<p>))==LBM_c1(<q>) or LBM_c1(<q>)==0) and (LBM_c2(LBM_pb(<p>))==LBM_c2(<q>) or LBM_c2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for pb=LBM_pb(<p>).
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF nbr_kap_b >= 0
                            #OUTIF LBM_COND_BC( LBM_c0(<p>),LBM_c0(<q>),  LBM_c1(<p>),LBM_c1(<q>),  LBM_c2(<p>),LBM_c2(<q>) )END_LBM_COND_BC
                            OUTIF LBM_COND_BC( 0,LBM_c0(LBM_pb(<q>)),  0,LBM_c1(LBM_pb(<q>)),  0,LBM_c2(LBM_pb(<q>)) )END_LBM_COND_BC
                                
                                REG index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + <p>*n_maxcells;
                                REG s_F_p[H_LIMIT( LBM_dim,LBM_c0(<q>),LBM_c1(<q>),LBM_c2(<q>) )END_H_LIMIT] = cells_f_F[index];
                                
                            END_OUTIF
                        END_OUTIF
                    END_INIF
                    
                END_INIF
            END_INFOR
            REG __syncthreads();
            
            # o====================================================================================
            # | Write to global memory, exchanging DDFs between the halos while streaming.
            # o====================================================================================
            
            // Main writes.
            OUTIF s_F_pb[H_INDEX(LBM_dim,LBM_c0(<p>),LBM_c1(<p>),LBM_c2(<p>))END_H_INDEX] >= 0
                REG index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + <p>*n_maxcells;
                REG cells_f_F[index] = s_F_pb[H_INDEX(LBM_dim,LBM_c0(<p>),LBM_c1(<p>),LBM_c2(<p>))END_H_INDEX];
            END_OUTIF
            OUTIF s_F_p[H_INDEX(LBM_dim,LBM_c0(LBM_pb(<p>)),LBM_c1(LBM_pb(<p>)),LBM_c2(LBM_pb(<p>)))END_H_INDEX] >= 0
                REG index = i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells;
                REG cells_f_F[index] = s_F_p[H_INDEX(LBM_dim,LBM_c0(LBM_pb(<p>)),LBM_c1(LBM_pb(<p>)),LBM_c2(LBM_pb(<p>)))END_H_INDEX];
            END_OUTIF

            // Neighbor writes.
            INFOR q 1   1 LBM_size 1
                INIF  (LBM_c0(<q>) >= 0) and (LBM_c1(<q>) >= 0) and (LBM_c2(<q>) >= 0) 
                    
                    INIF  (LBM_c0(<p>)==LBM_c0(<q>) or LBM_c0(<q>)==0) and (LBM_c1(<p>)==LBM_c1(<q>) or LBM_c1(<q>)==0) and (LBM_c2(<p>)==LBM_c2(<q>) or LBM_c2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for p=<p>.
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF (nbr_kap_b >= 0) and LBM_COND_BC( 0,LBM_c0(LBM_pb(<q>)),  0,LBM_c1(LBM_pb(<q>)),  0,LBM_c2(LBM_pb(<q>)) )END_LBM_COND_BC
                            
                            OUTIF s_F_p[H_COMBO(LBM_dim,LBM_c0(LBM_pb(<p>)),LBM_c0(<q>),  LBM_c1(LBM_pb(<p>)),LBM_c1(<q>), LBM_c2(LBM_pb(<p>)),LBM_c2(<q>))END_H_COMBO] >= 0
                                REG index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells;
                                REG cells_f_F[index] = s_F_p[H_COMBO(LBM_dim,LBM_c0(LBM_pb(<p>)),LBM_c0(<q>),  LBM_c1(LBM_pb(<p>)),LBM_c1(<q>), LBM_c2(LBM_pb(<p>)),LBM_c2(<q>))END_H_COMBO];
                            END_OUTIF
                            
                        END_OUTIF
                    END_INIF
                    INIF  (LBM_c0(LBM_pb(<p>))==LBM_c0(<q>) or LBM_c0(<q>)==0) and (LBM_c1(LBM_pb(<p>))==LBM_c1(<q>) or LBM_c1(<q>)==0) and (LBM_c2(LBM_pb(<p>))==LBM_c2(<q>) or LBM_c2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for pb=LBM_pb(<p>).
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF (nbr_kap_b >= 0) and LBM_COND_BC( 0,LBM_c0(LBM_pb(<q>)),  0,LBM_c1(LBM_pb(<q>)),  0,LBM_c2(LBM_pb(<q>)) )END_LBM_COND_BC
                            
                            OUTIF s_F_pb[H_COMBO(LBM_dim,LBM_c0(<p>),LBM_c0(<q>),  LBM_c1(<p>),LBM_c1(<q>),  LBM_c2(<p>),LBM_c2(<q>))END_H_COMBO] >= 0
                                REG index = nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + <p>*n_maxcells;
                                REG cells_f_F[index] = s_F_pb[H_COMBO(LBM_dim,LBM_c0(<p>),LBM_c0(<q>),  LBM_c1(<p>),LBM_c1(<q>),  LBM_c2(<p>),LBM_c2(<q>))END_H_COMBO];
                            END_OUTIF
                            
                        END_OUTIF
                    END_INIF
                    
                END_INIF
            END_INFOR
            REG __syncthreads();
            <
            <
            <
        
        END_INIF
    END_INFOR
    
END_OUTFOR 2
INIF LBM_dim==3
    END_OUTFOR
END_INIF



END_LOOPBLOCKS

END_FILE
