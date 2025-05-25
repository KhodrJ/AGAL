# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_collision_new_s2
FILE_DIR ../solver_lbm/

ROUTINE_NAME Collision_New_S2
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L

KERNEL_REQUIRE int n_ids_idev_L                        | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t dx_L                           | dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L                          | tau_vec[L]
KERNEL_REQUIRE int *__restrict__ id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *__restrict__ cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cells_f_F_aux    | mesh->c_cells_f_F_aux[i_dev]
KERNEL_REQUIRE ufloat_t *__restrict__ cblock_f_X       | mesh->c_cblock_f_X[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *__restrict__ cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]
#KERNEL_REQUIRE int S_LES                               | S_LES
#KERNEL_REQUIRE ufloat_t S_LES_p1                       | S_LES_p1
#KERNEL_REQUIRE ufloat_t S_LES_p2                       | S_LES_p2



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
#REG __shared__ ufloat_t s_u[(6*6*6)];
#REG __shared__ ufloat_t s_v[(6*6*6)];
#REG __shared__ ufloat_t s_w[(6*6*6)];
#REG __shared__ int s_ID_nbr[N_Q_max];
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
REG int valid_block = -1;
REG int valid_mask = -1;
REG ufloat_t f_p = (ufloat_t)(0.0);
REG ufloat_t f_q = (ufloat_t)(0.0);
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t u = (ufloat_t)(0.0);
REG ufloat_t ub = (ufloat_t)(0.0);
REG ufloat_t v = (ufloat_t)(0.0);
REG ufloat_t vb = (ufloat_t)(0.0);
REG ufloat_t w = (ufloat_t)(0.0);
REG ufloat_t wb = (ufloat_t)(0.0);
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);
REG ufloat_t omeg = dx_L / tau_L;
REG ufloat_t omegp = (ufloat_t)(1.0) - omeg;




TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(valid_block>-3)
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG valid_block=cblock_ID_mask[i_kap_b];


// Compute cell coordinates and retrieve macroscopic properties.
REG valid_block = cblock_ID_onb[i_kap_b];
REG valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
#INIF Ldim==2
    #REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
    #REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
#INELSE
    #REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
    #REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
    #REG z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
#END_INIF
REG rho = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 0*n_maxcells];
REG u = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 1*n_maxcells];
REG v = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 2*n_maxcells];
INIF Ldim==3
    REG w = cells_f_F_aux[i_kap_b*M_CBLOCK + threadIdx.x + 3*n_maxcells];
END_INIF
REG udotu = u*u + v*v + w*w;
#OUTIF (valid_block == 1 && threadIdx.x == 0)
    #INFOR p 1   1 gI(3^Ldim) 1
        #REG s_ID_nbr[<p>] = cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks];
    #END_INFOR
#END_OUTIF
REG __syncthreads();
<

//
//
//
// First round of DDF loads to compute the macroscopic properties. Discard the DDFs to avoid large memory footprint for now.
// (Need to test if this really speeds anything up, convenient to have it all one kernel for now though).
//
//
//
#INFOR p 1   0 Lsize 1
    #REG-v f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
    #REG rho += f_p;
    #INIF (Lc0(<p>) > 0)
        #REG-v u += f_p;
    #END_INIF
    #INIF (Lc0(<p>) < 0)
        #REG-v ub += f_p;
    #END_INIF
    #INIF (Lc1(<p>) > 0)
        #REG-v v += f_p;
    #END_INIF
    #INIF (Lc1(<p>) < 0)
        #REG-v vb += f_p;
    #END_INIF
    #INIF (Lc2(<p>) > 0)
        #REG-v w += f_p;
    #END_INIF
    #INIF (Lc2(<p>) < 0)
        #REG-v wb += f_p;
    #END_INIF
#END_INFOR
#REG u = (u-ub)/rho;
#REG v = (v-vb)/rho;
#INIF Ldim==3
    #REG w = (w-wb)/rho;
#END_INIF
#REG udotu = u*u + v*v + w*w;
#REG ub = u;
#REG vb = v;
#REG wb = w;
#<

#// Compute and store the effective relaxation rate, if an SGS model has been selected.
#OUTIF (S_LES > 0)
    // Compute the velocity gradient.
    #REG valid_block = cblock_ID_onb_solid[i_kap];
    #REG dist_p = (ufloat_t)(1.0);
    #REG dist_q = (ufloat_t)(1.0);
    #// [-] +X
    #INIF Ldim==2
        #OUTIFL (valid_block>-1 && cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcblocks_b] > 0)
            #REG dist_p = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcblocks_b];
        #END_OUTIFL
        #OUTIFL (valid_block>-1 && cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcblocks_b] > 0)
            #REG dist_q = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 3*n_maxcblocks_b];
        #END_OUTIFL
    #INELSE
        #OUTIFL (valid_block>-1 && cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcblocks_b] > 0)
            #REG dist_p = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 1*n_maxcblocks_b];
        #END_OUTIFL
        #OUTIFL (valid_block>-1 && cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcblocks_b] > 0)
            #REG dist_q = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + 2*n_maxcblocks_b];
        #END_OUTIFL
    #END_INIF
    #REG um1 = s_u[(I+1 - 1)+6*(J+1)+36*(K+1)];
    #REG u0 = s_u[(I+1)+6*(J+1)+36*(K+1)];
    #REG up1 = s_u[(I+1 + 1)+6*(J+1)+36*(K+1)];

    #
    # Classical Smagorinsky
    #
    #OUTIF (S_LES==1)
        
    #END_OUTIF	
    #
    # WALE
    #
    #OUTIF (S_LES==1)
    
    #END_OUTIF
    #
    # Vreman
    #
    #OUTIF (S_LES==1)
    
    #END_OUTIF
#END_OUTIF

//
//
//
// Retrieve DDFs one by one again, but perform collision now and apply boundary conditions.
//
//
//
<
INFOR p 1   0 Lsize 1
    INIF (<p>==0 or ( (Ldim==2 and (<p>==1 or (<p>+1)%3==0)) or (Ldim==3 and ((((<p>-1)%2==0)and(<p><25))or(<p>==26))) ))
        //
        // p = <p>
        //
        <
        
        // Retrieve the DDF.
        REG-v f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
        INIF (<p> > 0)
            REG-v f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + <p>*n_maxcells];
        END_INIF
        <
        
        // Collision step.
        REG-vz cdotu = gNz(Lc0(<p>)*u + Lc1(<p>)*v + Lc2(<p>)*w);
        REG-vm f_p = f_p*omegp + ( (ufloat_t)(gD(Lw(<p>)))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
        INIF (<p> > 0)
            REG-vz cdotu = gNz(Lc0(Lpb(<p>))*u + Lc1(Lpb(<p>))*v + Lc2(Lpb(<p>))*w);
            REG-vm f_q = f_q*omegp + ( (ufloat_t)(gD(Lw(Lpb(<p>))))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
        END_INIF
        <
        
        #// Impose boundary conditions.
        #// Do this only if on the boundary.
        #INIF (<p> > 0)
            #OUTIF (valid_block==1)
                #// Pick the right neighbor block for this cell (p).
                #REG nbr_kap_b = i_kap_b;
                #REG-v Ip = I + Lc0(<p>);
                #REG-v Jp = J + Lc1(<p>);
                #INIF (Ldim==3)
                    #REG-v Kp = K + Lc2(<p>);
                #END_INIF
                #INFOR q 1   1 Lsize 1
                    #INIF ((Lc0(<p>)==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<q>)==0))
                        #// Consider nbr <q>.
                        #OUTIFL ( gNa( gCOND(Lc0(<q>): 1,(Ip==4),-1,(Ip==-1),def.,(Ip>=0)and(Ip<4)) and \
                                    #gCOND(Lc1(<q>): 1,(Jp==4),-1,(Jp==-1),def.,(Jp>=0)and(Jp<4)) and \
                                    #gCOND(Ldim: 3,gCOND(Lc2(<q>): 1,(Kp==4),-1,(Kp==-1),def.,(Kp>=0)and(Kp<4)), def.,true) )\
                        #)
                            #REG nbr_kap_b = cblock_ID_nbr[i_kap_b + <q>*n_maxcblocks];
                        #END_OUTIFL
                    #END_INIF
                #END_INFOR
                #OUTIFL (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    #REG-v Cu_ImposeBC(nbr_kap_b, f_p, rho, ub, vb, wb, x, y, z, (ufloat_t)(Lw(<p>)), (ufloat_t)(Lc0(<p>)), (ufloat_t)(Lc1(<p>)), (ufloat_t)(Lc2(<p>)), cdotu);
                #END_OUTIFL
                #<
                
                #// Pick the right neighbor block for this cell (pb).
                #REG nbr_kap_b = i_kap_b;
                #REG-v Ip = I + Lc0(Lpb(<p>));
                #REG-v Jp = J + Lc1(Lpb(<p>));
                #INIF (Ldim==3)
                    #REG-v Kp = K + Lc2(Lpb(<p>));
                #END_INIF
                #INFOR q 1   1 Lsize 1
                    #INIF ((Lc0(Lpb(<p>))==Lc0(<q>) or Lc0(<q>)==0) and (Lc1(Lpb(<p>))==Lc1(<q>) or Lc1(<q>)==0) and (Lc2(Lpb(<p>))==Lc2(<q>) or Lc2(<q>)==0))
                        #// Consider nbr <q>.
                        #OUTIFL ( gNa( gCOND(Lc0(<q>): 1,(Ip==4),-1,(Ip==-1),def.,(Ip>=0)and(Ip<4)) and \
                                    #gCOND(Lc1(<q>): 1,(Jp==4),-1,(Jp==-1),def.,(Jp>=0)and(Jp<4)) and \
                                    #gCOND(Ldim: 3,gCOND(Lc2(<q>): 1,(Kp==4),-1,(Kp==-1),def.,(Kp>=0)and(Kp<4)), def.,true) )\
                        #)
                            #REG nbr_kap_b = cblock_ID_nbr[i_kap_b + <q>*n_maxcblocks];
                        #END_OUTIFL
                    #END_INIF
                #END_INFOR
                #OUTIFL (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                    #REG-v Cu_ImposeBC(nbr_kap_b, f_q, rho, ub, vb, wb, x, y, z, (ufloat_t)(Lw(Lpb(<p>))), (ufloat_t)(Lc0(Lpb(<p>))), (ufloat_t)(Lc1(Lpb(<p>))), (ufloat_t)(Lc2(Lpb(<p>))), cdotu);
                #END_OUTIFL
            #END_OUTIF
            #<
        #END_INIF
        
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

# Post-collision DDFs (without BC imposed) are now available on this grid level.

TEMPLATE NEW_BLOCK
END_TEMPLATE

