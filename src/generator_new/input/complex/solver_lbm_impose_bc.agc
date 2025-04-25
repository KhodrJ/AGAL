# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_imposebc
FILE_DIR ../solver_lbm/

ROUTINE_NAME ImposeBC
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L

KERNEL_REQUIRE int n_ids_idev_L                        | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE int n_maxcells_b                        | mesh->n_maxcells_b
KERNEL_REQUIRE int n_maxblocks_b                       | mesh->n_solidb
KERNEL_REQUIRE ufloat_t dx_L                           | dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L                          | tau_vec[L]
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
KERNEL_REQUIRE double *__restrict__ cblock_f_Ff_solid  | mesh->c_cblock_f_Ff_solid[i_dev]
KERNEL_REQUIRE bool geometry_init                      | mesh->geometry_init
KERNEL_REQUIRE bool compute_forces                     | compute_forces



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
REG __shared__ double s_Fpx[M_TBLOCK];
REG __shared__ double s_Fmx[M_TBLOCK];
REG __shared__ double s_Fpy[M_TBLOCK];
REG __shared__ double s_Fmy[M_TBLOCK];
REG __shared__ double s_Fpz[M_TBLOCK];
REG __shared__ double s_Fmz[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int Ip = I;
REG int J = (threadIdx.x / 4) % 4;
REG int Jp = J;
INIF Ldim==3
    REG int K = (threadIdx.x / 4) / 4;
    REG int Kp = K;
END_INIF
REG ufloat_t x __attribute__((unused)) = (ufloat_t)(0.0);
REG ufloat_t y __attribute__((unused)) = (ufloat_t)(0.0);
REG ufloat_t z __attribute__((unused)) = (ufloat_t)(0.0);
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
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t u = (ufloat_t)(0.0);
REG ufloat_t v = (ufloat_t)(0.0);
REG ufloat_t w = (ufloat_t)(0.0);
REG ufloat_t ub = (ufloat_t)(0.0);
REG ufloat_t vb = (ufloat_t)(0.0);
REG ufloat_t wb = (ufloat_t)(0.0);
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);
#REG ufloat_t omeg = dx_L / tau_L;
#REG ufloat_t omegp = (ufloat_t)(1.0) - omeg;



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (valid_block==1)
TEMPLATE ARG REG valid_block=cblock_ID_onb[i_kap_b];


// Compute cell coordinates and retrieve macroscopic properties.
REG valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
OUTIFL (geometry_init)
    REG block_mask = cblock_ID_onb_solid[i_kap_b];
END_OUTIFL
OUTIF (compute_forces && block_mask > -1)
    REG s_Fpx[threadIdx.x] = 0;
    REG s_Fmx[threadIdx.x] = 0;
    REG s_Fpy[threadIdx.x] = 0;
    REG s_Fmy[threadIdx.x] = 0;
    REG s_Fpz[threadIdx.x] = 0;
    REG s_Fmz[threadIdx.x] = 0;
END_OUTIF
INIF Ldim==2
    REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
    REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
INELSE
    REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
    REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
    REG z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
END_INIF
REG rho = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
REG u = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells]; ub = u;
REG v = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells]; vb = v;
REG w = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells]; wb = w;
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
        // Impose boundary conditions.
        // Do this only if on the boundary.
        //
        INIF (<p> > 0)
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
                        #REG nbr_kap_b = cblock_ID_nbr[i_kap_b + <q>*n_maxcblocks];
                    END_OUTIFL
                END_INIF
            END_INFOR
            OUTIFL (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                REG-v Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(Lw(<p>)), (ufloat_t)(Lc0(<p>)), (ufloat_t)(Lc1(<p>)), (ufloat_t)(Lc2(<p>)), cdotu);
            END_OUTIFL
            <
            
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
                        #REG nbr_kap_b = cblock_ID_nbr[i_kap_b + <q>*n_maxcblocks];
                    END_OUTIFL
                END_INIF
            END_INFOR
            OUTIFL (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                REG-v Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(Lw(Lpb(<p>))), (ufloat_t)(Lc0(Lpb(<p>))), (ufloat_t)(Lc1(Lpb(<p>))), (ufloat_t)(Lc2(Lpb(<p>))), cdotu);
            END_OUTIFL
            <
        END_INIF
        
        //
        // Impose curved boundary conditions.
        // Do this only if adjacent to a solid cell (by checking the cell mask).
        // Current assumption: only one DDFs from the pair will be altered at a time, stationary boundaries.
        //
        INIF (<p> > 0)
            OUTIF (valid_mask == -2)
                #OUTIF (cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + <p>*n_maxcells_b] > 0 && cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells_b] > 0)
                    #REG printf("BC ERROR\n");
                #END_OUTIF
            
                # For p.
                // Check if DDF <p> is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                REG-v dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + <p>*n_maxcells_b];
                OUTIF (compute_forces && dist_p > 0)
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
                END_OUTIF
            
                OUTIF (dist_p > 0 && dist_p < (ufloat_g_t)(0.5)*dx_L)
                    // Pick the right neighbor block for this cell (pb).
                    #REG printf("Cell %i is considering a curved boundary in direction %i (%17.15f %17.15f)\n", threadIdx.x, <p>, dist_p, dx_L);
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
                    REG f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + <p>*n_maxcells];
                    REG dist_p /= dx_L;
                    REG f_p = (ufloat_t)(2.0)*dist_p*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dist_p)*f_m;
                    REG dist_p *= dx_L;
                END_OUTIF
                
                OUTIF (dist_p >= (ufloat_g_t)(0.5)*dx_L && dist_p < dx_L)
                    #REG printf("Cell %i is considering a curved boundary in direction %i (%17.15f %17.15f)\n", threadIdx.x, <p>, dist_p, dx_L);
                    REG f_m = f_q;
                    REG dist_p /= dx_L;
                    REG f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dist_p))*f_p + (((ufloat_t)(2.0)*dist_p - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dist_p))*f_m;
                    REG dist_p *= dx_L;
                END_OUTIF
                OUTIF (compute_forces && dist_p > 0)
                    #REG printf("Added force contribution %17.15f for dist_p=%17.15f\n", f_p, dist_p);
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
                END_OUTIF
                
                
                
                # For q.
                REG-v // Check if DDF Lpb(<p>) is directed towards the solid object.
                // If computing forces, add the contributions of DDFs entering the geometry.
                REG-v dist_p = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells_b];
                OUTIF (compute_forces && dist_p > 0)
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
                END_OUTIF
            
                OUTIF (dist_p > 0 && dist_p < (ufloat_g_t)(0.5)*dx_L)
                    // Pick the right neighbor block for this cell (p).
                    #REG-v printf("Cell %i is considering a curved boundary in direction %i (%17.15f %17.15f)\n", threadIdx.x, Lpb(<p>), dist_p, dx_L);
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
                    REG f_q = (ufloat_t)(2.0)*dist_p*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dist_p)*f_m;
                    REG dist_p *= dx_L;
                END_OUTIF
                
                OUTIF (dist_p >= (ufloat_g_t)(0.5)*dx_L && dist_p < dx_L)
                    #REG-v printf("Cell %i is considering a curved boundary in direction %i (%17.15f %17.15f)\n", threadIdx.x, Lpb(<p>), dist_p, dx_L);
                    REG f_m = f_p;
                    REG dist_p /= dx_L;
                    REG f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dist_p))*f_q + (((ufloat_t)(2.0)*dist_p - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dist_p))*f_m;
                    REG dist_p *= dx_L;
                END_OUTIF
                OUTIF (compute_forces && dist_p > 0)
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
                END_OUTIF
                
            END_OUTIF
        END_INIF
        
        // Write fi* to global memory.
        OUTIF (valid_mask != -1)
            REG cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + <p>*n_maxcells] = f_p;
            INIF (<p> > 0)
                REG-v cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = f_q;
            END_INIF
        END_OUTIF
        <
    END_INIF
END_INFOR

OUTIF (compute_forces && block_mask > -1)
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
        REG cblock_f_Ff_solid[block_mask + 0*n_maxblocks_b] = s_Fpx[0];
        REG cblock_f_Ff_solid[block_mask + 1*n_maxblocks_b] = s_Fmx[0];
        REG cblock_f_Ff_solid[block_mask + 2*n_maxblocks_b] = s_Fpy[0];
        REG cblock_f_Ff_solid[block_mask + 3*n_maxblocks_b] = s_Fmy[0];
        INIF Ldim==3
            REG cblock_f_Ff_solid[block_mask + 4*n_maxblocks_b] = s_Fpz[0];
            REG cblock_f_Ff_solid[block_mask + 5*n_maxblocks_b] = s_Fmz[0];
        END_INIF
    END_OUTIF
END_OUTIF

# Post-collision DDFs (with BC imposed) are now available on this grid level.

TEMPLATE NEW_BLOCK
END_TEMPLATE

