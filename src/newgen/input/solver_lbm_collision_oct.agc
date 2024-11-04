# Kernel name and includes.
NAME Collide_LBM_name
NAME_FILE solver_lbm_collide
INCLUDE "solver.h"
INCLUDE "mesh.h"
INCLUDE_GUARD N_Q==LBM_size

# Import BCs from text file.
IMPORT TABLE ../../../input/BC_new.txt BC_tab

# Kernel parameters.
KERNEL_REQUIRE int n_ids_idev_L ROUTINE mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t dx_L ROUTINE dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L ROUTINE tau_vec[L]
KERNEL_REQUIRE ufloat_t tau_ratio ROUTINE tau_ratio_vec_C2F[L]
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
REG int I_kap = threadIdx.x % Nbx;
REG int J_kap = (threadIdx.x / Nbx) % Nbx;
INIF LBM_dim==3
    int K_kap = (threadIdx.x / Nbx) / Nbx;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int i_Q = -1;
REG int nbr_kap_b = -1;
REG int block_on_boundary = -1;
INFOR p 1   0 LBM_size 1
    REG ufloat_t f_<p> = N_Pf(0.0);
END_INFOR
REG ufloat_t rho_kap = N_Pf(0.0);
REG ufloat_t u_kap = N_Pf(0.0);
REG ufloat_t v_kap = N_Pf(0.0);
INIF LBM_dim==3
    REG ufloat_t w_kap = N_Pf(0.0);
END_INIF
REG ufloat_t cdotu = N_Pf(0.0);
REG ufloat_t udotu = N_Pf(0.0);
REG ufloat_t omeg = dx_L / tau_L;
REG ufloat_t omegp = N_Pf(1.0) - omeg;
INFOR p 1   1 LBM_size 1
    REG int nbr_id_<p> = N_Pf(0.0);
END_INFOR


LOOPBLOCKS 2\
    CONDITION ((i_kap_bc<0)||(block_on_boundary==1))\
    REQUIRING 2 i_kap_bc=cblock_ID_nbr_child[i_kap_b] block_on_boundary=cblock_ID_mask[i_kap_b]

# Loop over individual block quadrants.
REG block_on_boundary = cblock_ID_onb[i_kap_b];
OUTIF block_on_boundary == 1
    INFOR p 1   1 LBM_size 1
        REG nbr_id_<p> = cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks];
    END_INFOR
END_OUTIF

INIF LBM_dim==3
    OUTFOR k_q 3   0 Nqx 1
END_INIF
OUTFOR i_qj_q 3   0 Nqx 1   0 Nqx 1
    // Retrieve DDFs and compute macroscopic properties.
    INIF LBM_dim==2
        REG i_Q = i_q + Nqx*j_q;
    INELSE
        REG i_Q = i_q + Nqx*j_q + Nqx*Nqx*k_q;
    END_INIF
    INFOR p 1   0 LBM_size 1
        f_<p> = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells];
        #REG f_<p> = cells_f_F[(size_t)(i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells)];
    END_INFOR
    REG rho_kap = SUM< i 0 LBM_size 1 f_<i> >END_SUM;
    REG u_kap = (SUM< i 0 LBM_size 1 N_Pf(LBM_c0(<i>))*f_<i>>END_SUM) / rho_kap;
    REG v_kap = (SUM< i 0 LBM_size 1 N_Pf(LBM_c1(<i>))*f_<i>>END_SUM) / rho_kap;
    INIF LBM_dim==2
        REG udotu = u_kap*u_kap + v_kap*v_kap;
    INELSE
        REG w_kap = (SUM< i 0 LBM_size 1 N_Pf(LBM_c2(<i>))*f_<i>>END_SUM) / rho_kap;
        REG udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
    END_INIF

    <
    // Eddy viscosity calculation.

    <
    // Collision step.
    INFOR p 1   0 LBM_size 1
        DEFINE CDOTU 
        INIF LBM_c0(<p>)==1
            DEFINE DEF_PUSH + CDOTU u_kap
        END_INIF
        INIF LBM_c0(<p>)==-1
            DEFINE DEF_PUSH + CDOTU (-u_kap)
        END_INIF
        INIF LBM_c1(<p>)==1
            DEFINE DEF_PUSH + CDOTU v_kap
        END_INIF
        INIF LBM_c1(<p>)==-1
            DEFINE DEF_PUSH + CDOTU (-v_kap)
        END_INIF
        INIF LBM_c2(<p>)==1
            DEFINE DEF_PUSH + CDOTU w_kap
        END_INIF
        INIF LBM_c2(<p>)==-1
            DEFINE DEF_PUSH + CDOTU (-w_kap)
        END_INIF
        INIF LBM_c0(<p>)==0 and LBM_c1(<p>)==0 and LBM_c2(<p>)==0
            DEFINE DEF_PUSH + CDOTU N_Pf(0.0)
        END_INIF
        
        REG cdotu = CDOTU;
        #REG cdotu = LBM_c0(<p>)*u_kap + LBM_c1(<p>)*v_kap + LBM_c2(<p>)*w_kap;
        REG f_<p> = f_<p>*omegp + ( N_Pf(LBM_w(<p>))*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
    END_INFOR

    <
    // Impose boundary conditions.
    OUTIF block_on_boundary == 1
        INFOR p 1   1 LBM_size 1
            // nbr <p>
            #REG nbr_kap_b = cblock_ID_nbr[i_kap_b + <p>*n_maxcblocks];
            DEFINE NBR_COND 
            INIF LBM_c0(<p>)==1
                DEFINE DEF_PUSH and NBR_COND (i_q==Nqx-1)
            END_INIF
            INIF LBM_c0(<p>)==-1
                DEFINE DEF_PUSH and NBR_COND (i_q==0)
            END_INIF
            INIF LBM_c1(<p>)==1
                DEFINE DEF_PUSH and NBR_COND (j_q==Nqx-1)
            END_INIF
            INIF LBM_c1(<p>)==-1
                DEFINE DEF_PUSH and NBR_COND (j_q==0)
            END_INIF
            INIF LBM_dim==3 and LBM_c2(<p>)==1
                DEFINE DEF_PUSH and NBR_COND (k_q==Nqx-1)
            END_INIF
            INIF LBM_dim==3 and LBM_c2(<p>)==-1
                DEFINE DEF_PUSH and NBR_COND (k_q==0)
            END_INIF
            
            
            OUTIF NBR_COND
                REG nbr_kap_b = nbr_id_<p>;
                INFOR q 1   1 LBM_size 1
                    INIF (LBM_c0(<p>)==LBM_c0(<q>) or LBM_c0(<p>)==0.0) and (LBM_c1(<p>)==LBM_c1(<q>) or LBM_c1(<p>)==0.0) and (LBM_c2(<p>)==LBM_c2(<q>) or LBM_c2(<p>)==0.0)
                    
                        # Build the cell boundary condition.
                        DEFINE C_COND 
                        INIF LBM_c0(<p>)==1
                            DEFINE DEF_PUSH and C_COND (I_kap+Nbx*i_q==Nbx*Nqx-1)
                        END_INIF
                        INIF LBM_c0(<p>)==-1
                            DEFINE DEF_PUSH and C_COND (I_kap+Nbx*i_q==0)
                        END_INIF
                        INIF LBM_c0(<p>)==0
                            INIF LBM_c0(<q>)==1
                                DEFINE DEF_PUSH and C_COND (I_kap+Nbx*i_q<Nbx*Nqx-1)
                            END_INIF
                            INIF LBM_c0(<q>)==-1
                                DEFINE DEF_PUSH and C_COND (I_kap+Nbx*i_q>0)
                            END_INIF
                        END_INIF
                        INIF LBM_c1(<p>)==1
                            DEFINE DEF_PUSH and C_COND (J_kap+Nbx*j_q==Nbx*Nqx-1)
                        END_INIF
                        INIF LBM_c1(<p>)==-1
                            DEFINE DEF_PUSH and C_COND (J_kap+Nbx*j_q==0)
                        END_INIF
                        INIF LBM_c1(<p>)==0
                            INIF LBM_c1(<q>)==1
                                DEFINE DEF_PUSH and C_COND (J_kap+Nbx*j_q<Nbx*Nqx-1)
                            END_INIF
                            INIF LBM_c1(<q>)==-1
                                DEFINE DEF_PUSH and C_COND (J_kap+Nbx*j_q>0)
                            END_INIF
                        END_INIF
                        INIF LBM_c2(<p>)==1
                            DEFINE DEF_PUSH and C_COND (K_kap+Nbx*k_q==Nbx*Nqx-1)
                        END_INIF
                        INIF LBM_c2(<p>)==-1
                            DEFINE DEF_PUSH and C_COND (K_kap+Nbx*k_q==0)
                        END_INIF
                        INIF LBM_c2(<p>)==0
                            INIF LBM_c2(<q>)==1
                                DEFINE DEF_PUSH and C_COND (K_kap+Nbx*k_q<Nbx*Nqx-1)
                            END_INIF
                            INIF LBM_c2(<q>)==-1
                                DEFINE DEF_PUSH and C_COND (K_kap+Nbx*k_q>0)
                            END_INIF
                        END_INIF
                        
                        # Build cdotu.
                        DEFINE CDOTU 
                        INIF LBM_c0(<q>)==1
                            DEFINE DEF_PUSH + CDOTU u_kap
                        END_INIF
                        INIF LBM_c0(<q>)==-1
                            DEFINE DEF_PUSH + CDOTU (-u_kap)
                        END_INIF
                        INIF LBM_c1(<q>)==1
                            DEFINE DEF_PUSH + CDOTU v_kap
                        END_INIF
                        INIF LBM_c1(<q>)==-1
                            DEFINE DEF_PUSH + CDOTU (-v_kap)
                        END_INIF
                        INIF LBM_c2(<q>)==1
                            DEFINE DEF_PUSH + CDOTU w_kap
                        END_INIF
                        INIF LBM_c2(<q>)==-1
                            DEFINE DEF_PUSH + CDOTU (-w_kap)
                        END_INIF
                        INIF LBM_c0(<q>)==0 and LBM_c1(<p>)==0 and LBM_c2(<p>)==0
                            DEFINE DEF_PUSH + CDOTU N_Pf(0.0)
                        END_INIF
                    
                        # For each boundary edge, construct the conditional and loop over defined boundary conditions.
                        OUTIF C_COND
                            #OUTIF LBM_COND_BC(LBM_c0(<q>),LBM_c0(<p>),  LBM_c1(<q>),LBM_c1(<p>),  LBM_c2(<q>),LBM_c2(<p>))END_LBM_COND_BC
                            INFOR N 1   0 IMP_TAB(BC_tab,-1,0)END_IMP 1
                                # Dirichlet (bounce-back).
                                INIF (IMP_TAB(BC_tab,<N>,1)END_IMP==0)
                                    INIF (IMP_TAB(BC_tab,<N>,2)END_IMP!=0.0)and(IMP_TAB(BC_tab,<N>,2)END_IMP!=0.0)
                                        OUTIF nbr_kap_b == IMP_TAB(BC_tab,<N>,0)END_IMP
                                            REG cdotu = N_Pf(^D< LBM_c0(<q>)*IMP_TAB(BC_tab,<N>,2)END_IMP + LBM_c1(<q>)*IMP_TAB(BC_tab,<N>,3)END_IMP + LBM_c2(<q>)*IMP_TAB(BC_tab,<N>,4)END_IMP >^);
                                            REG f_<q> = f_<q> - N_Pf(2.0)*N_Pf(LBM_w(<q>))*N_Pf(3.0)*cdotu;
                                        END_OUTIF
                                    END_INIF
                                END_INIF
                                # Outflow (anti-bounce-back).
                                INIF (IMP_TAB(BC_tab,<N>,1)END_IMP==1)
                                    OUTIF nbr_kap_b == IMP_TAB(BC_tab,<N>,0)END_IMP
                                        REG cdotu = CDOTU;
                                        #REG cdotu = LBM_c0(<q>)*u_kap + LBM_c1(<q>)*v_kap + LBM_c2(<q>)*w_kap;
                                        REG f_<q> = -f_<q> + N_Pf(2.0)*N_Pf(LBM_w(<q>))*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5));
                                    END_OUTIF
                                END_INIF
                            END_INFOR
                        END_OUTIF
                        
                    END_INIF
                END_INFOR
            END_OUTIF
            <
        END_INFOR
    END_OUTIF

    <
    // Write fi* to global memory.
    INFOR p 1   0 LBM_size 1
        cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + <p>*n_maxcells] = f_<p>;
        #REG cells_f_F[(size_t)(i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells)] = f_<p>;
    END_INFOR
    REG __syncthreads();
END_OUTFOR 2
INIF LBM_dim==3
    END_OUTFOR
END_INIF

END_LOOPBLOCKS

END_FILE

