# Kernel name and includes.
NAME SetInitialConditions_LBM_name
NAME_FILE solver_lbm_set_ic
INCLUDE "solver.h"
INCLUDE "mesh.h"
INCLUDE_GUARD N_Q==LBM_size

# Import IC from text file.
IMPORT VECTOR ../../../input/IC.txt IC_tab

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
KERNEL_REQUIRE ufloat_t rho_t0 ROUTINE N_Pf(IMP_VEC(IC_tab,0)END_IMP)
KERNEL_REQUIRE ufloat_t u_t0 ROUTINE N_Pf(IMP_VEC(IC_tab,1)END_IMP)
KERNEL_REQUIRE ufloat_t v_t0 ROUTINE N_Pf(IMP_VEC(IC_tab,2)END_IMP)
KERNEL_REQUIRE ufloat_t w_t0 ROUTINE N_Pf(IMP_VEC(IC_tab,3)END_IMP)

# Routine parameters.
ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_COND mesh->n_ids[i_dev][L] > 0
ROUTINE_OBJECT Solver_LBM



#
# Kernel definition.
#

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG int i_kap_b = -1;
REG ufloat_t cdotu = N_Pf(0.0);
REG ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0;


LOOPBLOCKS
    OUTFOR i_Q 3   0 N_QUADS 1
    
    // Compute IC.
    INFOR p 1   0 LBM_size 1
	DEFINE CDOTU 
        INIF LBM_c0(<p>)==1
            DEFINE DEF_PUSH + CDOTU u_t0
        END_INIF
        INIF LBM_c0(<p>)==-1
            DEFINE DEF_PUSH + CDOTU (-u_t0)
        END_INIF
        INIF LBM_c1(<p>)==1
            DEFINE DEF_PUSH + CDOTU v_t0
        END_INIF
        INIF LBM_c1(<p>)==-1
            DEFINE DEF_PUSH + CDOTU (-v_t0)
        END_INIF
        INIF LBM_c2(<p>)==1
            DEFINE DEF_PUSH + CDOTU w_t0
        END_INIF
        INIF LBM_c2(<p>)==-1
            DEFINE DEF_PUSH + CDOTU (-w_t0)
        END_INIF
        INIF LBM_c0(<p>)==0 and LBM_c1(<p>)==0 and LBM_c2(<p>)==0
            DEFINE DEF_PUSH + CDOTU N_Pf(0.0)
        END_INIF
        
        REG cdotu = CDOTU;
        #REG cdotu = LBM_c0(<p>)*u_t0 + LBM_c1(<p>)*v_t0 + LBM_c2(<p>)*w_t0;
        REG cells_f_F[threadIdx.x + (size_t)i_Q*M_TBLOCK + (size_t)i_kap_b*M_CBLOCK + (size_t)LBM_pb(<p>)*n_maxcells] = N_Pf(LBM_w(<p>))*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
    END_INFOR

    END_OUTFOR
END_LOOPBLOCKS

END_FILE

