# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_collision_original
FILE_DIR ../solver_lbm/

ROUTINE_NAME Collision_Original
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "solver_lbm.h"
ROUTINE_INCLUDE "mesh.h"

ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L

KERNEL_REQUIRE int n_ids_idev_L           | mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t dx_L              | dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L             | tau_vec[L]
KERNEL_REQUIRE int *id_set_idev_L         | &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask         | mesh->c_cells_ID_mask[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F        | mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE ufloat_t *cblock_f_X       | mesh->c_cblock_f_X[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr         | mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child   | mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]



#
# Kernel definition.
#

REG constexpr int Nqx = AP->Nqx;
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG __shared__ int s_ID_cblock[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int I = threadIdx.x % 4;
REG int J = (threadIdx.x / 4) % 4;
INIF Ldim==3
    int K = (threadIdx.x / 4) / 4;
END_INIF
REG ufloat_t x __attribute__((unused)) = (ufloat_t)(0.0);
REG ufloat_t y __attribute__((unused)) = (ufloat_t)(0.0);
INIF Ldim==3
	REG ufloat_t z __attribute__((unused)) = (ufloat_t)(0.0);
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int i_Q = -1;
REG int nbr_kap_b = -1;
REG int block_on_boundary = -1;
INFOR p 1   0 Lsize 1
    REG ufloat_t f_<p> = (ufloat_t)(0.0);
END_INFOR
REG ufloat_t rho = (ufloat_t)(0.0);
REG ufloat_t u = (ufloat_t)(0.0);
REG ufloat_t v = (ufloat_t)(0.0);
INIF Ldim==3
    REG ufloat_t w = (ufloat_t)(0.0);
END_INIF
REG ufloat_t cdotu = (ufloat_t)(0.0);
REG ufloat_t udotu = (ufloat_t)(0.0);
REG ufloat_t omeg = dx_L / tau_L;
REG ufloat_t omegp = (ufloat_t)(1.0) - omeg;
INFOR p 1   1 Lsize 1
    REG int nbr_id_<p> = (ufloat_t)(0.0);
END_INFOR



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL
TEMPLATE ARG (i_kap_bc<0)||(block_on_boundary==1)
TEMPLATE ARG REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];
TEMPLATE ARG REG block_on_boundary=cblock_ID_mask[i_kap_b];

# Loop over individual block quadrants.
REG block_on_boundary = cblock_ID_onb[i_kap_b];
OUTIF (block_on_boundary == 1)
    INFOR p 1   1 Lsize 1
        REG nbr_id_<p> = cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks];
    END_INFOR
END_OUTIF

INIF Ldim==3
    OUTFOR k_q 3   0 Nqx 1
END_INIF
OUTFOR i_qj_q 3   0 Nqx 1   0 Nqx 1
    // Retrieve DDFs and compute macroscopic properties.
    INIF Ldim==2
        REG i_Q = i_q + Nqx*j_q;
        REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I) + i_q*4*dx_L;
        REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J) + j_q*4*dx_L;
    INELSE
        REG i_Q = i_q + Nqx*j_q + Nqx*Nqx*k_q;
        REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I) + i_q*4*dx_L;
        REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J) + j_q*4*dx_L;
        REG z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K) + k_q*4*dx_L;
    END_INIF
    INFOR p 1   0 Lsize 1
        REG-v f_<p> = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells];
    END_INFOR
    REG-vsvz rho = gNz(gSUM(i,0 Lsize 1,f_<i>));
    REG-vsvz u = (gNz(gSUM(i,0 Lsize 1,(Lc0(<i>))*f_<i>))) / rho;
    REG-vsvz v = (gNz(gSUM(i,0 Lsize 1,(Lc1(<i>))*f_<i>))) / rho;
    INIF Ldim==2
        REG udotu = u*u + v*v;
    INELSE
        REG-vsvz w = (gNz(gSUM(i,0 Lsize 1,(Lc2(<i>))*f_<i>))) / rho;
        REG udotu = u*u + v*v + w*w;
    END_INIF

    <
    // Eddy viscosity calculation.

    <
    // Collision step.
    INFOR p 1   0 Lsize 1
        REG-vz cdotu = gNz(Lc0(<p>)*u + Lc1(<p>)*v + Lc2(<p>)*w);
        REG-vm f_<p> = f_<p>*omegp + ( (ufloat_t)(gD(Lw(<p>)))*rho*((ufloat_t)(1.0) + (ufloat_t)(3.0)*cdotu + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*udotu) )*omeg;
    END_INFOR

    <
    // Impose boundary conditions.
    OUTIF (block_on_boundary == 1)
        INFOR p 1   1 Lsize 1
            // nbr <p>
            OUTIF (gNa( (gCOND(Lc0(<p>): 1,(i_q==Nqx-1), -1,(i_q==0), def.,true)) and\
            (gCOND(Lc1(<p>): 1,(j_q==Nqx-1), -1,(j_q==0), def.,true)) and\
            (gCOND(Lc2(<p>): 1,(k_q==Nqx-1), -1,(k_q==0), def.,true)) ))
                REG nbr_kap_b = nbr_id_<p>;
                INFOR q 1   1 Lsize 1
                    INIF ((Lc0(<p>)==Lc0(<q>) or Lc0(<p>)==0.0) and (Lc1(<p>)==Lc1(<q>) or Lc1(<p>)==0.0) and (Lc2(<p>)==Lc2(<q>) or Lc2(<p>)==0.0))
                        # For each boundary edge, construct the conditional and loop over defined boundary conditions.
                        OUTIF (gNa( gCOND(Lc0(<p>): 1,(I+4*i_q==4*Nqx-1), -1,(I+4*i_q==0), def.,gCOND(Lc0(<q>): 1,(I+4*i_q<4*Nqx-1), -1,(I+4*i_q>0), def.,true)) and \
                        gCOND(Lc1(<p>): 1,(J+4*j_q==4*Nqx-1), -1,(J+4*j_q==0), def.,gCOND(Lc1(<q>): 1,(J+4*j_q<4*Nqx-1), -1,(J+4*j_q>0), def.,true)) and \
                        gCOND(Lc2(<p>): 1,(K+4*k_q==4*Nqx-1), -1,(K+4*k_q==0), def.,gCOND(Lc2(<q>): 1,(K+4*k_q<4*Nqx-1), -1,(K+4*k_q>0), def.,true)) ))
                        
                        
                            #
                            # FPC (2D)
                            #
                            OUTIF (nbr_kap_b == -1)
                                REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            END_OUTIF
                            OUTIF (nbr_kap_b == -2)
                                REG-vz cdotu = gNz(Lc0(<q>)*u + Lc1(<q>)*v + Lc2(<q>)*w);
                                REG-vm f_<q> = -f_<q> + (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                            END_OUTIF
                            OUTIF (nbr_kap_b == -3)
                                REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            END_OUTIF
                            OUTIF (nbr_kap_b == -4)
                                REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            END_OUTIF
                            
                            
                            
                            #
                            # LDC (2D)
                            #
                            #OUTIF (nbr_kap_b == -4)
                                #REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                #REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            #END_OUTIF
                            
                            
                            
                            #
                            # FPC (3D)
                            #
                            #OUTIF (nbr_kap_b == -1)
                                #REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                #REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            #END_OUTIF
                            #OUTIF (nbr_kap_b == -2)
                                #REG-vz cdotu = gNz(Lc0(<q>)*u + Lc1(<q>)*v + Lc2(<q>)*w);
                                #REG-vm f_<q> = -f_<q> + (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                            #END_OUTIF
                            #OUTIF (nbr_kap_b == -3)
                                #REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                #REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            #END_OUTIF
                            #OUTIF (nbr_kap_b == -4)
                                #REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                #REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            #END_OUTIF
                            #OUTIF (nbr_kap_b == -5)
                                #REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                #REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            #END_OUTIF
                            #OUTIF (nbr_kap_b == -6)
                                #REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                #REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            #END_OUTIF
                            
                            
                            
                            #
                            # LDC (3D)
                            #
                            #OUTIF (nbr_kap_b == -6)
                                #REG-vm cdotu = (ufloat_t)(gD(Lc0(<q>)*(0.05) + Lc1(<q>)*(0.00) + Lc2(<q>)*(0.00)));
                                #REG-vm f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(gD(Lw(<q>)))*(ufloat_t)(3.0)*cdotu;
                            #END_OUTIF
                            
                            
                            #INFOR N 1   0 IMP_TAB(BC_tab,-1,0)END_IMP 1
                                # Dirichlet (bounce-back).
                                #INIF (IMP_TAB(BC_tab,<N>,1)END_IMP==0)
                                    #INIF FLAGS=t (IMP_TAB(BC_tab,<N>,2)END_IMP!=0.0)and(IMP_TAB(BC_tab,<N>,2)END_IMP!=0.0)
                                        #OUTIF (nbr_kap_b == IMP_TAB(BC_tab,<N>,0)END_IMP)
                                            #REG cdotu = (ufloat_t)( Lc0(<q>)*IMP_TAB(BC_tab,<N>,2)END_IMP + Lc1(<q>)*IMP_TAB(BC_tab,<N>,3)END_IMP + Lc2(<q>)*IMP_TAB(BC_tab,<N>,4)END_IMP );
                                            #REG f_<q> = f_<q> - (ufloat_t)(2.0)*(ufloat_t)(LBM_w(<q>))*(ufloat_t)(3.0)*cdotu;
                                        #END_OUTIF
                                    #END_INIF
                                #END_INIF
                                # Outflow (anti-bounce-back).
                                #INIF (IMP_TAB(BC_tab,<N>,1)END_IMP==1)
                                    #OUTIF (nbr_kap_b == IMP_TAB(BC_tab,<N>,0)END_IMP)
                                        #REG cdotu = CDOTU;
                                        #REG cdotu = Lc0(<q>)*u + Lc1(<q>)*v + Lc2(<q>)*w;
                                        #REG f_<q> = -f_<q> + (ufloat_t)(2.0)*(ufloat_t)(LBM_w(<q>))*((ufloat_t)(1.0) + cdotu*cdotu*(ufloat_t)(4.5) - udotu*(ufloat_t)(1.5));
                                    #END_OUTIF
                                #END_INIF
                            #END_INFOR
                        END_OUTIF
                        
                    END_INIF
                END_INFOR
            END_OUTIF
            <
        END_INFOR
    END_OUTIF

    <
    // Write fi* to global memory.
    INFOR p 1   0 Lsize 1
        REG cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + <p>*n_maxcells] = f_<p>;
    END_INFOR
    REG __syncthreads();
END_OUTFOR 2
INIF Ldim==3
    END_OUTFOR
END_INIF

TEMPLATE NEW_BLOCK
END_TEMPLATE

