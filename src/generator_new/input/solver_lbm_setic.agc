# /**************************************************************************************/
# /*                                                                                    */
# /*  Author: Khodr Jaber                                                               */
# /*  Affiliation: Turbulence Research Lab, University of Toronto                       */
# /*                                                                                    */
# /**************************************************************************************/

# File metadata and routine parameters.
FILE_NAME solver_lbm_setic
FILE_DIR ../solver_lbm/

ROUTINE_NAME SetInitialConditions
ROUTINE_OBJECT_NAME Solver_LBM
ROUTINE_INCLUDE "custom.h"
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
KERNEL_REQUIRE ufloat_t *cblock_f_X       | mesh->c_cblock_f_X[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask        | mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *cblock_ID_onb         | mesh->c_cblock_ID_onb[i_dev]
KERNEL_REQUIRE ufloat_t dx_L              | mesh->dxf_vec[L]



# Declarations.
REG constexpr int M_TBLOCK = AP->M_TBLOCK;
REG constexpr int M_CBLOCK = AP->M_CBLOCK;
REG constexpr int M_LBLOCK = AP->M_LBLOCK;
REG __shared__ int s_ID_cblock[M_TBLOCK];
REG int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
REG int i_kap_b = -1;
REG int I = threadIdx.x % 4;
REG int J = (threadIdx.x / 4) % 4;
INIF Ldim==3
    int K = (threadIdx.x / 4) / 4;
END_INIF
INFOR p 1   0 Lsize 1
	REG ufloat_t f_<p> = (ufloat_t)(-1.0);
END_INFOR
REG ufloat_t cdotu = (ufloat_t)0.0;
REG ufloat_t udotu = (ufloat_t)0.0;
REG ufloat_t rho = (ufloat_t)1.0;
REG ufloat_t u = (ufloat_t)0.0;
REG ufloat_t v = (ufloat_t)0.0;
REG ufloat_t x = (ufloat_t)0.0;
REG ufloat_t y = (ufloat_t)0.0;
REG ufloat_t w = (ufloat_t)0.0;
REG ufloat_t z = (ufloat_t)0.0;
<



TEMPLATE
TEMPLATE NAME PRIMARY_ORIGINAL

//<Compute macroscopic properties.
INFOR p 1   0 Lsize 1
	REG f_<p> = (ufloat_t)(-1.0);
END_INFOR
OUTIF ( cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] > -1 )
    REG x = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + (ufloat_t)0.5*dx_L;
    REG y = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + (ufloat_t)0.5*dx_L;
    INIF Ldim==2
        REG Cu_ComputeIC<ufloat_t>(rho, u, v, w, x, y, z);
        REG udotu = u*u + v*v;
    INELSE
        REG z = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + (ufloat_t)0.5*dx_L;
        REG Cu_ComputeIC<ufloat_t>(rho, u, v, w, x, y, z);
        REG udotu = u*u + v*v + w*w;
    END_INIF
    INFOR p 1   0 Lsize 1
        REG-vz cdotu = gNz( Lc0(<p>)*u + Lc1(<p>)*v + Lc2(<p>)*w );
        REG-vm f_<p> = rho*(ufloat_t)gD(Lw(<p>))*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
    END_INFOR
END_OUTIF
<

// Write DDFs in proper index.
INFOR p 1   0 Lsize 1
    REG-v cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + Lpb(<p>)*n_maxcells] = f_<p>;
END_INFOR

TEMPLATE NEW_BLOCK
END_TEMPLATE
