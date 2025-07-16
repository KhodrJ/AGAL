# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_compute_forces_mea
FILE_DIR ../solver_lbm/

ROUTINE_NAME ComputeForces_MEA
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var

KERNEL_REQUIRE int n_ids_idev_L                        | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE int n_maxcells_b                        | mesh->n_maxcells_b
KERNEL_REQUIRE int n_maxblocks_b                       | mesh->n_solidb
KERNEL_REQUIRE ufloat_t dx_L                           | dxf_vec[L]
KERNEL_REQUIRE ufloat_t dv_L                           | dvf_vec[L]
KERNEL_REQUIRE int *__restrict__ id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *__restrict__ cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE ufloat_g_t *__restrict__ cells_f_X_b    | mesh->c_cells_f_X_b[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F_aux    | mesh->c_cells_f_F_aux[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cblock_f_X       | mesh->c_cblock_f_X[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_onb_solid   | mesh->c_cblock_ID_onb_solid[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cblock_f_Ff      | mesh->c_cblock_f_Ff[i_dev]
KERNEL_REQUIRE bool geometry_init                      | mesh->geometry_init
KERNEL_REQUIRE int order                               | S_FORCE_ORDER

ROUTINE_TEMPLATE_PARAMS int post_step
ROUTINE_TEMPLATE_VALS 0
ROUTINE_TEMPLATE_VALS 1
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==0
ROUTINE_TEMPLATE_ARGS mesh->n_ids[i_dev][L]>0 && var==1



#
# Kernel definition.
#

REG constexpr int Nqx = AP->Nqx;
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG constexpr int N_DIM = AP->N_DIM;
REG constexpr int N_Q_max = AP->N_Q_max;

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ int s_ID_nbr[N_Q_max];
REG __shared__ ufloat_t s_Fpx[M_TBLOCK];
REG __shared__ ufloat_t s_Fmx[M_TBLOCK];
REG __shared__ ufloat_t s_Fpy[M_TBLOCK];
REG __shared__ ufloat_t s_Fmy[M_TBLOCK];
REG __shared__ ufloat_t s_Fpz[M_TBLOCK];
REG __shared__ ufloat_t s_Fmz[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int Ip = I;
REG int J = (threadIdx.x / 4) % 4;
REG int Jp = J;
INIF Ldim==3
    REG int K = (threadIdx.x / 4) / 4;
    REG int Kp = K;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int nbr_kap_b = -1;
REG int nbr_kap_c = -1;
REG int valid_block = -1;
REG int block_mask = -1;
REG int valid_mask = -1;
REG ufloat_t f_p = (ufloat_t)(0.0);
REG ufloat_t f_q = (ufloat_t)(0.0);
REG ufloat_t f_m = (ufloat_t)(0.0);
REG ufloat_g_t dist_p = (ufloat_g_t)(0.0);



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (valid_block==1)
TEMPLATE ARG REG valid_block=cblock_ID_onb[i_kap_b];


// Compute cell coordinates and retrieve macroscopic properties.
REG valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
OUTIFL (geometry_init)
    REG block_mask = cblock_ID_onb_solid[i_kap_b];
END_OUTIFL
OUTIF (n_maxblocks_b > 0 && block_mask > -1)
    REG s_Fpx[threadIdx.x] = 0;
    REG s_Fmx[threadIdx.x] = 0;
    REG s_Fpy[threadIdx.x] = 0;
    REG s_Fmy[threadIdx.x] = 0;
    REG s_Fpz[threadIdx.x] = 0;
    REG s_Fmz[threadIdx.x] = 0;
END_OUTIF
OUTIF (valid_block == 1 && threadIdx.x == 0)
    INFOR p 1   1 gI(3^Ldim) 1
        REG s_ID_nbr[<p>] = cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks];
    END_INFOR
END_OUTIF
REG __syncthreads();
<


<
INFOR p 1   1 Lsize 1
    INIF (<p>==0 or ( (Ldim==2 and (<p>==1 or (<p>+1)%3==0)) or (Ldim==3 and ((((<p>-1)%2==0)and(<p><25))or(<p>==26))) ))
        //
        // p = <p>
        //
        <
        
        // Retrieve the DDF. Use correct order of access this time.
        REG-v f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + <p>*n_maxcells];
        INIF (<p> > 0)
            REG-v f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
        END_INIF
        <
        
        //
        // Find the right neighboring DDFs if the second-order Ginzburg and d'Humieres calculation is being used.
        //
        INIF (<p> > 0)
            OUTIF (valid_mask == -2)            
                # For p.
                // Check if DDF <p> is directed towards the solid object.
                REG-v dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + <p>*n_maxcells_b];
                OUTIF (dist_p > 0)
                    // Pick the right neighbor block for this cell (pb).
                    REG nbr_kap_b = i_kap_b;
                    REG-v Ip = I + Lc0(Lpb(<p>));
                    REG-v Jp = J + Lc1(Lpb(<p>));
                    INIF (Ldim==3)
                        REG-v Kp = K + Lc2(Lpb(<p>));
                    END_INIF
                    INFOR q 1   1 Lsize 1
                        INIF ((Lc0(Lpb(<p>))==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(Lpb(<p>))==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(Lpb(<p>))==Lc2(<q>) or Lc2(<q>)==0))
                            // Consider nbr <q>.
                            OUTIFL ( gNa( gCOND(Lc0(<q>): 1,(Ip==4),-1,(Ip==-1),def.,(Ip>=0)and(Ip<4)) and \
                                        gCOND(Lc1(<q>): 1,(Jp==4),-1,(Jp==-1),def.,(Jp>=0)and(Jp<4)) and \
                                        gCOND(Ldim: 3,gCOND(Lc2(<q>): 1,(Kp==4),-1,(Kp==-1),def.,(Kp>=0)and(Kp<4)), def.,true) )\
                            )
                                REG nbr_kap_b = s_ID_nbr[<q>];
                            END_OUTIFL
                        END_INIF
                    END_INFOR
                    
                    // Get the fluid node behind this boundary node.
                    INIF (Lc0(Lpb(<p>)) != 0)
                        REG Ip = (4 + (Ip % 4)) % 4;
                    END_INIF
                    INIF (Lc1(Lpb(<p>)) != 0)
                        REG Jp = (4 + (Jp % 4)) % 4;
                    END_INIF
                    INIF (Lc2(Lpb(<p>)) != 0)
                        REG Kp = (4 + (Kp % 4)) % 4;
                    END_INIF
                    INIF Ldim==2
                        REG nbr_kap_c = Ip + 4*Jp;
                    INELSE
                        REG nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    END_INIF
                    REG-v f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + <p>*n_maxcells];
                    REG dist_p /= dx_L;
                    
                    // Add force contributions.
                    OUTIF (order == 1)
                        INIF (Lc0(<p>) > 0)
                            REG s_Fpx[threadIdx.x] += f_p;
                        END_INIF
                        INIF (Lc0(<p>) < 0)
                            REG s_Fmx[threadIdx.x] += f_p;
                        END_INIF
                        INIF (Lc1(<p>) > 0)
                            REG s_Fpy[threadIdx.x] += f_p;
                        END_INIF
                        INIF (Lc1(<p>) < 0)
                            REG s_Fmy[threadIdx.x] += f_p;
                        END_INIF
                        INIF (Lc2(<p>) > 0)
                            REG s_Fpz[threadIdx.x] += f_p;
                        END_INIF
                        INIF (Lc2(<p>) < 0)
                            REG s_Fmz[threadIdx.x] += f_p;
                        END_INIF
                    OUTELSE
                        INIF (Lc0(<p>) > 0)
                            REG s_Fpx[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc0(<p>) < 0)
                            REG s_Fmx[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc1(<p>) > 0)
                            REG s_Fpy[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc1(<p>) < 0)
                            REG s_Fmy[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc2(<p>) > 0)
                            REG s_Fpz[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc2(<p>) < 0)
                            REG s_Fmz[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_p + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                    END_OUTIF
                END_OUTIF
                
                
                
                # For q.
                REG-v // Check if DDF Lpb(<p>) is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                REG-v dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells_b];
                OUTIF (dist_p > 0)
                    // Pick the right neighbor block for this cell (p).
                    REG nbr_kap_b = i_kap_b;
                    REG-v Ip = I + Lc0(<p>);
                    REG-v Jp = J + Lc1(<p>);
                    INIF (Ldim==3)
                        REG-v Kp = K + Lc2(<p>);
                    END_INIF
                    INFOR q 1   1 Lsize 1
                        INIF ((Lc0(<p>)==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<q>)==0))
                            // Consider nbr <q>.
                            OUTIFL ( gNa( gCOND(Lc0(<q>): 1,(Ip==4),-1,(Ip==-1),def.,(Ip>=0)and(Ip<4)) and \
                                        gCOND(Lc1(<q>): 1,(Jp==4),-1,(Jp==-1),def.,(Jp>=0)and(Jp<4)) and \
                                        gCOND(Ldim: 3,gCOND(Lc2(<q>): 1,(Kp==4),-1,(Kp==-1),def.,(Kp>=0)and(Kp<4)), def.,true) )\
                            )
                                REG nbr_kap_b = s_ID_nbr[<q>];
                            END_OUTIFL
                        END_INIF
                    END_INFOR
                    
                    // Get the fluid node behind this boundary node.
                    INIF (Lc0(<p>) != 0)
                        REG Ip = (4 + (Ip % 4)) % 4;
                    END_INIF
                    INIF (Lc1(<p>) != 0)
                        REG Jp = (4 + (Jp % 4)) % 4;
                    END_INIF
                    INIF (Lc2(<p>) != 0)
                        REG Kp = (4 + (Kp % 4)) % 4;
                    END_INIF
                    INIF Ldim==2
                        REG nbr_kap_c = Ip + 4*Jp;
                    INELSE
                        REG nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    END_INIF
                    REG-v f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + Lpb(<p>)*n_maxcells];
                    REG dist_p /= dx_L;

                    // Add force contributions.
                    OUTIF (n_maxblocks_b > 0 && dist_p > 0)
                        INIF (Lc0(Lpb(<p>)) > 0)
                            REG s_Fpx[threadIdx.x] += f_q;
                        END_INIF
                        INIF (Lc0(Lpb(<p>)) < 0)
                            REG s_Fmx[threadIdx.x] += f_q;
                        END_INIF
                        INIF (Lc1(Lpb(<p>)) > 0)
                            REG s_Fpy[threadIdx.x] += f_q;
                        END_INIF
                        INIF (Lc1(Lpb(<p>)) < 0)
                            REG s_Fmy[threadIdx.x] += f_q;
                        END_INIF
                        INIF (Lc2(Lpb(<p>)) > 0)
                            REG s_Fpz[threadIdx.x] += f_q;
                        END_INIF
                        INIF (Lc2(Lpb(<p>)) < 0)
                            REG s_Fmz[threadIdx.x] += f_q;
                        END_INIF
                    OUTELSE
                        INIF (Lc0(Lpb(<p>)) > 0)
                            REG s_Fpx[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc0(Lpb(<p>)) < 0)
                            REG s_Fmx[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc1(Lpb(<p>)) > 0)
                            REG s_Fpy[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc1(Lpb(<p>)) < 0)
                            REG s_Fmy[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc2(Lpb(<p>)) > 0)
                            REG s_Fpz[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                        INIF (Lc2(Lpb(<p>)) < 0)
                            REG s_Fmz[threadIdx.x] += (0.5+(ufloat_t)dist_p)*f_q + (0.5-(ufloat_t)dist_p)*f_m;
                        END_INIF
                    END_OUTIF
                END_OUTIF
            END_OUTIF
        END_INIF
        
        <
    END_INIF
END_INFOR

OUTIF (n_maxblocks_b > 0 && block_mask > -1)
    // Reductions for the sums of force contributions in this cell-block.
    __syncthreads();
    REG for (int s=blockDim.x/2; s>0; s>>=1)
    REG {
        OUTIF (threadIdx.x < s)
                REG s_Fpx[threadIdx.x] = s_Fpx[threadIdx.x] + s_Fpx[threadIdx.x + s];
                REG s_Fmx[threadIdx.x] = s_Fmx[threadIdx.x] + s_Fmx[threadIdx.x + s];
                REG s_Fpy[threadIdx.x] = s_Fpy[threadIdx.x] + s_Fpy[threadIdx.x + s];
                REG s_Fmy[threadIdx.x] = s_Fmy[threadIdx.x] + s_Fmy[threadIdx.x + s];
                INIF Ldim==3
                    REG s_Fpz[threadIdx.x] = s_Fpz[threadIdx.x] + s_Fpz[threadIdx.x + s];
                    REG s_Fmz[threadIdx.x] = s_Fmz[threadIdx.x] + s_Fmz[threadIdx.x + s];
                END_INIF
        END_OUTIF
        REG __syncthreads();
    REG }
    
    // Store the sums of contributions in global memory; this will be reduced further later.
    OUTIF (threadIdx.x == 0)
        OUTIF (post_step == 0)
            REG cblock_f_Ff[i_kap_b + 0*n_maxcblocks] = s_Fpx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 1*n_maxcblocks] = s_Fmx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 2*n_maxcblocks] = s_Fpy[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 3*n_maxcblocks] = s_Fmy[0]*dv_L;
            INIF Ldim==3
                REG cblock_f_Ff[i_kap_b + 4*n_maxcblocks] = s_Fpz[0]*dv_L;
                REG cblock_f_Ff[i_kap_b + 5*n_maxcblocks] = s_Fmz[0]*dv_L;
            END_INIF
        OUTELSE
            REG cblock_f_Ff[i_kap_b + 0*n_maxcblocks] += s_Fpx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 1*n_maxcblocks] += s_Fmx[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 2*n_maxcblocks] += s_Fpy[0]*dv_L;
            REG cblock_f_Ff[i_kap_b + 3*n_maxcblocks] += s_Fmy[0]*dv_L;
            INIF Ldim==3
                REG cblock_f_Ff[i_kap_b + 4*n_maxcblocks] += s_Fpz[0]*dv_L;
                REG cblock_f_Ff[i_kap_b + 5*n_maxcblocks] += s_Fmz[0]*dv_L;
            END_INIF
        END_OUTIF
    END_OUTIF
END_OUTIF


TEMPLATE NEW_BLOCK
END_TEMPLATE

