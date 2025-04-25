# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_stream_original
FILE_DIR ../solver_lbm/

ROUTINE_NAME Stream_Original
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L

KERNEL_REQUIRE int n_ids_idev_L           | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE int *id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]



#
# Kernel definition.
#

# Declarations.
REG constexpr int Nqx = AP->Nqx;
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG __shared__ int s_ID_cblock[M_TBLOCK];
REG-vs __shared__ ufloat_t s_F_p[gPROD(i,0 Ldim 1,(4+2))];
REG-vs __shared__ ufloat_t s_F_pb[gPROD(i,0 Ldim 1,(4+2))];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int J = (threadIdx.x / 4) % 4;
INIF Ldim==3
    int K = (threadIdx.x / 4) / 4;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int i_Q = -1;
REG int nbr_i_q = -1;
REG int nbr_j_q = -1;
INIF Ldim==3
    REG int nbr_k_q = -1;
END_INIF
REG int nbr_kap_b = -1;
REG int nbr_Q_b = -1;
INFOR q 1   1 Lsize 1
    INIF  (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0)
        REG int nbr_id_global_<q> = -1;
    END_INIF
END_INFOR
INFOR q 1   1 Lsize 1
    INIF  (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0)
        REG int nbr_id_<q> = -1;
    END_INIF
END_INFOR
INFOR q 1   1 Lsize 1
    INIF  (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0)
        REG int nbr_Q_<q> = -1;
    END_INIF
END_INFOR
REG int block_on_boundary = -1;



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(block_on_boundary>-2)
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG block_on_boundary=cblock_ID_mask[i_kap_b];

INFOR q 1   1 Lsize 1
    INIF  (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0)
        REG nbr_id_global_<q> = cblock_ID_nbr[i_kap_b + <q>*n_maxcblocks];
    END_INIF
END_INFOR

INIF Ldim==3
    OUTFOR k_q 3   0 Nqx 1
END_INIF
OUTFOR i_qj_q 3   0 Nqx 1   0 Nqx 1
    
    # Compute quadrant/octant Id.
    INIF Ldim==2
        REG i_Q = i_q + Nqx*j_q;
    INELSE
        REG i_Q = i_q + Nqx*j_q + Nqx*Nqx*k_q;
    END_INIF
    
    # Compute local (to quadrant) neighbor quadrants and blocks.
    INFOR q 1   1 Lsize 1
        INIF  (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0)
            REG nbr_id_<q> = i_kap_b;
            REG-vm nbr_i_q = i_q + gI(Lc0(<q>)); if (nbr_i_q == Nqx) nbr_i_q = 0; 
            REG-vm nbr_j_q = j_q + gI(Lc1(<q>)); if (nbr_j_q == Nqx) nbr_j_q = 0; 
            INIF Ldim==3
                REG-vm nbr_k_q = k_q + gI(Lc2(<q>)); if (nbr_k_q == Nqx) nbr_k_q = 0; 
            END_INIF
            
            INIF Ldim==2
                REG nbr_Q_<q> = nbr_i_q + Nqx*nbr_j_q;
            INELSE
                REG nbr_Q_<q> = nbr_i_q + Nqx*nbr_j_q + Nqx*Nqx*nbr_k_q;
            END_INIF
        END_INIF
    END_INFOR
    INFOR q 1   1 Lsize 1
        INIF (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0)
            // Nbr <q>.
            INFOR p 1   1 Lsize 1
               INIF ( ( Lc0(<p>) >= 0) and (Lc1(<p>) >= 0) and (Lc2(<p>) >= 0 ) and ( Lc0(<q>)-Lc0(<p>)>=0 and Lc1(<q>)-Lc1(<p>)>=0 and Lc2(<q>)-Lc2(<p>)>=0 ) )
                    OUTIF (gNa( gCOND(Lc0(<p>):1,(i_q==Nqx-1),0,gCOND(Lc0(<q>):0,true,def.,(i_q<Nqx-1))) and gCOND(Lc1(<p>):1,(j_q==Nqx-1),0,gCOND(Lc1(<q>):0, true,def.,(j_q<Nqx-1))) and gCOND(Lc2(<p>):1,(k_q==Nqx-1),0,gCOND(Lc2(<q>):0,true,def.,(k_q<Nqx-1))) ))
                        REG nbr_id_<q> = nbr_id_global_<p>;
                    END_OUTIF
                END_INIF
            END_INFOR
        END_INIF
    END_INFOR
    
    <
    # Loop over all DDFs for that quadrant/octant.
    INFOR p 1   1 Lsize 1
        INIF  ( (Ldim==2)and((<p>==1)or((<p>+1)%3==0)) ) or ( (Ldim==3)and( (((<p>-1)%2==0)and(<p><25))or(<p>==26) ) ) 
        
            # o====================================================================================
            # | Reset shared memory halos, load DDFs in the interior.
            # o====================================================================================
            
            //
            // DDFs p=<p>, pb=Lpb(<p>).
            //
            INIF Ldim==2
                OUTFOR q 1   0 3 1
                        OUTIF (threadIdx.x + q*16 < 36)
                            REG s_F_p[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                            REG s_F_pb[threadIdx.x + q*16] = (ufloat_t)(-1.0);
                        END_OUTIF
                END_OUTFOR
            INELSE
                OUTFOR q 1   0 4 1
                        OUTIF (threadIdx.x + q*64 < 216)
                            REG s_F_p[threadIdx.x + q*64] = (ufloat_t)(-1.0);
                            REG s_F_pb[threadIdx.x + q*64] = (ufloat_t)(-1.0);
                        END_OUTIF
                END_OUTFOR
            END_INIF
            REG __syncthreads();
            REG-vc s_F_p[(I+1)+6*(J+1)+gCOND(Ldim: 2,0, def.,36*(K+1))] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + <p>*n_maxcells];
            REG-vc s_F_pb[(I+1)+6*(J+1)+gCOND(Ldim: 2,0, def.,36*(K+1))] = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
            
            # o====================================================================================
            # | Retrieve DDFs from neighbors and place in shared memory halo.
            # o====================================================================================
            
            INFOR q 1   1 Lsize 1
                INIF  (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0) 
                
                    INIF  (Lc0(<p>)==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for p=<p>.
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF (nbr_kap_b >= 0)
                            OUTIF (gNa( (gCOND(Lc0(Lpb(<q>)): 1,I==4-1,-1,I==0,def.,true)) and (gCOND(Lc1(Lpb(<q>)): 1,J==4-1,-1,J==0,def.,true)) and (gCOND(Lc2(Lpb(<q>)): 1,K==4-1,-1,K==0,def.,true)) ))
                            
                                REG-vc s_F_pb[(gCOND(Lc0(<q>): 1,4+1, def.,I+1)) + 6*(gCOND(Lc1(<q>): 1,4+1, def.,J+1)) + gCOND(Ldim: 2,0, 3,36*(gCOND(Lc2(<q>): 1,4+1, def.,K+1)))] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
                                
                            END_OUTIF
                        END_OUTIF
                    END_INIF
                    INIF  (Lc0(Lpb(<p>))==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(Lpb(<p>))==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(Lpb(<p>))==Lc2(<q>) or Lc2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for pb=Lpb(<p>).
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF (nbr_kap_b >= 0)
                            OUTIF (gNa( (gCOND(Lc0(Lpb(<q>)): 1,I==4-1,-1,I==0,def.,true)) and (gCOND(Lc1(Lpb(<q>)): 1,J==4-1,-1,J==0,def.,true)) and (gCOND(Lc2(Lpb(<q>)): 1,K==4-1,-1,K==0,def.,true)) ))
                            
                                REG-vc s_F_p[(gCOND(Lc0(<q>): 1,4+1, def.,I+1)) + 6*(gCOND(Lc1(<q>): 1,4+1, def.,J+1)) + gCOND(Ldim: 2,0, 3,36*(gCOND(Lc2(<q>): 1,4+1, def.,K+1)))] = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + <p>*n_maxcells];
                                
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
            OUTIF (cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] > -1)
                OUTIF (s_F_pb[(I+1+Lc0(<p>))+6*(J+1+Lc1(<p>))+gCOND(Ldim: 2,0, def.,36*(K+1+Lc2(<p>)))] >= 0)
                    REG-vc cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + <p>*n_maxcells] = s_F_pb[(I+1+Lc0(<p>))+6*(J+1+Lc1(<p>))+gCOND(Ldim: 2,0, def.,36*(K+1+Lc2(<p>)))];
                END_OUTIF
                OUTIF (s_F_p[(I+1+Lc0(Lpb(<p>)))+6*(J+1+Lc1(Lpb(<p>)))+gCOND(Ldim: 2,0, def.,36*(K+1+Lc2(Lpb(<p>))))] >= 0)
                    REG-vc cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = s_F_p[(I+1+Lc0(Lpb(<p>)))+6*(J+1+Lc1(Lpb(<p>)))+gCOND(Ldim: 2,0, def.,36*(K+1+Lc2(Lpb(<p>))))];
                END_OUTIF
            END_OUTIF
                
            // Neighbor writes.
            INFOR q 1   1 Lsize 1
                INIF  (Lc0(<q>) >= 0) and (Lc1(<q>) >= 0) and (Lc2(<q>) >= 0) 
                    
                    INIF  (Lc0(<p>)==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for p=<p>.
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF ((nbr_kap_b >=0) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and (gNa( (gCOND(Lc0(Lpb(<q>)): 1,I==4-1,-1,I==0,def.,true)) and (gCOND(Lc1(Lpb(<q>)): 1,J==4-1,-1,J==0,def.,true)) and (gCOND(Lc2(Lpb(<q>)): 1,K==4-1,-1,K==0,def.,true)) ))) 
                         
                            OUTIF (s_F_p[\
                                (I+1+Lc0(Lpb(<p>))+(Lc0(<q>)*4)) + 6*(J+1+Lc1(Lpb(<p>))+(Lc1(<q>)*4)) + gCOND(Ldim: 2,0, 3,36*(K+1+Lc2(Lpb(<p>))+(Lc2(<q>)*4)))\
                                ] >= 0)
                                REG-vc cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = s_F_p[\
                                    (I+1+Lc0(Lpb(<p>))+(Lc0(<q>)*4)) + 6*(J+1+Lc1(Lpb(<p>))+(Lc1(<q>)*4)) + gCOND(Ldim: 2,0, 3,36*(K+1+Lc2(Lpb(<p>))+(Lc2(<q>)*4)))\
                                ];
                            END_OUTIF
                            
                        END_OUTIF
                    END_INIF
                    INIF  (Lc0(Lpb(<p>))==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(Lpb(<p>))==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(Lpb(<p>))==Lc2(<q>) or Lc2(<q>)==0)
                        //\t Nbr <q> participates in an exchange for pb=Lpb(<p>).
                        REG nbr_Q_b = nbr_Q_<q>;
                        REG nbr_kap_b = nbr_id_<q>;
                        OUTIF ((nbr_kap_b >=0 ) and cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x]>-1 and (gNa( (gCOND(Lc0(Lpb(<q>)): 1,I==4-1,-1,I==0,def.,true)) and (gCOND(Lc1(Lpb(<q>)): 1,J==4-1,-1,J==0,def.,true)) and (gCOND(Lc2(Lpb(<q>)): 1,K==4-1,-1,K==0,def.,true)) )))
                        
                            OUTIF (s_F_pb[\
                                (I+1+Lc0(<p>)+(Lc0(<q>)*4)) + 6*(J+1+Lc1(<p>)+(Lc1(<q>)*4)) + gCOND(Ldim: 2,0, 3,36*(K+1+Lc2(<p>)+(Lc2(<q>)*4)))\
                                ] >= 0)
                                REG-vc cells_f_F[nbr_kap_b*M_CBLOCK + nbr_Q_b*M_TBLOCK + threadIdx.x + <p>*n_maxcells] = s_F_pb[\
                                    (I+1+Lc0(<p>)+(Lc0(<q>)*4)) + 6*(J+1+Lc1(<p>)+(Lc1(<q>)*4)) + gCOND(Ldim: 2,0, 3,36*(K+1+Lc2(<p>)+(Lc2(<q>)*4)))\
                                ];
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
    
    // Finished.
    
END_OUTFOR 2
INIF Ldim==3
    END_OUTFOR
END_INIF

TEMPLATE NEW_BLOCK
END_TEMPLATE
