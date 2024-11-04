# Kernel name and includes.
NAME Interpolate_Linear_LBM_name
NAME_FILE solver_lbm_interp_linear
INCLUDE "solver.h"
INCLUDE "mesh.h"
INCLUDE_GUARD N_Q==LBM_size

# Kernel parameters.
TEMPLATE int interp_type=0
TEMPLATE_VALS 0
TEMPLATE_VALS 1
KERNEL_REQUIRE int n_ids_idev_L ROUTINE mesh->n_ids[i_dev][L]
KERNEL_REQUIRE long int n_maxcells
KERNEL_REQUIRE int n_maxcblocks
KERNEL_REQUIRE ufloat_t dx_L ROUTINE dxf_vec[L]
KERNEL_REQUIRE ufloat_t tau_L
KERNEL_REQUIRE ufloat_t tau_ratio ROUTINE tau_ratio_L
KERNEL_REQUIRE int *id_set_idev_L ROUTINE &mesh->c_id_set[i_dev][L*n_maxcblocks]
KERNEL_REQUIRE int *cells_ID_mask ROUTINE mesh->c_cells_ID_mask[i_dev] mesh->c_cblock_ID_ref[i_dev]
KERNEL_REQUIRE ufloat_t *cells_f_F ROUTINE mesh->c_cells_f_F[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr ROUTINE mesh->c_cblock_ID_nbr[i_dev]
KERNEL_REQUIRE int *cblock_ID_nbr_child ROUTINE mesh->c_cblock_ID_nbr_child[i_dev]
KERNEL_REQUIRE int *cblock_ID_mask ROUTINE mesh->c_cblock_ID_mask[i_dev]
KERNEL_REQUIRE int *cblock_ID_onb ROUTINE mesh->c_cblock_ID_onb[i_dev]

# Routine parameters.
ROUTINE_REQUIRE int i_dev
ROUTINE_REQUIRE int L
ROUTINE_REQUIRE int var
ROUTINE_REQUIRE ufloat_t tau_L
ROUTINE_REQUIRE ufloat_t tau_ratio_L
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_INTERFACE
ROUTINE_COND mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_ADDED
ROUTINE_OBJECT Solver_LBM



#
# Kernel definition.
#

REG __shared__ int s_ID_cblock[M_TBLOCK];
REG __shared__ ufloat_t s_F[M_TBLOCK];
REG int I_kap = threadIdx.x % Nbx;
REG int J_kap = (threadIdx.x / Nbx) % Nbx;
INIF LBM_dim==3
    REG int K_kap = (threadIdx.x / Nbx) / Nbx;
END_INIF
REG int i_kap_b = -1;
REG int i_kap_bc = -1;
REG int block_on_boundary = -1;
INFOR p 1   0 LBM_size 1
    REG ufloat_t f_<p> = N_Pf(0.0);
END_INFOR
REG ufloat_t tmp_i = N_Pf(0.0);
REG ufloat_t rho_kap = N_Pf(0.0);
REG ufloat_t u_kap = N_Pf(0.0);
REG ufloat_t v_kap = N_Pf(0.0);
REG ufloat_t w_kap = N_Pf(0.0);
REG ufloat_t cdotu = N_Pf(0.0);
REG ufloat_t udotu = N_Pf(0.0);


LOOPBLOCKS 2\
    CONDITION (((interp_type==0)and(block_on_boundary==1))or((interp_type==1)and(cells_ID_mask[i_kap_b]==V_REF_ID_MARK_REFINE)))\
    REQUIRING 2 i_kap_bc=cblock_ID_nbr_child[i_kap_b] block_on_boundary=cblock_ID_mask[i_kap_b]

// Load DDFs and compute macroscopic properties.
INFOR p 1   0 LBM_size 1
    REG f_<p> = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells];
END_INFOR
REG rho_kap = SUM< i 0 LBM_size 1 f_<i> >END_SUM;
REG u_kap = (SUM< i 0 LBM_size 1 LBM_c0(<i>)*f_<i> >END_SUM) / rho_kap;
REG v_kap = (SUM< i 0 LBM_size 1 LBM_c1(<i>)*f_<i> >END_SUM) / rho_kap;
INIF LBM_dim==2
REG udotu = u_kap*u_kap + v_kap*v_kap;
INELSE
REG w_kap = (SUM< i 0 LBM_size 1 N_Pf(LBM_c2(<i>))*f_<i>>END_SUM) / rho_kap;
REG udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
END_INIF

<
// Interpolate rescaled fi to children if applicable.
INFOR p 1   0 LBM_size 1
    //
    // DDF <p>.
    //
    REG cdotu = LBM_c0(<p>)*u_kap + LBM_c1(<p>)*v_kap + LBM_c2(<p>)*w_kap;
    REG tmp_i = N_Pf(LBM_w(<p>))*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
    REG s_F[threadIdx.x] = tmp_i + (f_<p> - tmp_i)*(tau_ratio);
    REG __syncthreads();
    INFOR xc_ixc_jxc_k 4   0 2 1   0 2 1   0 2 1
        INIF (<xc_k>==0 and LBM_dim==2) or (LBM_dim==3)
            //\tChild ^I< <xc_i>+2*<xc_j>+4*<xc_k> >^.
            OUTIF (interp_type == 0 && cells_ID_mask[(i_kap_bc+^I< <xc_i>+2*<xc_j>+4*<xc_k> >^)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1)
            
                DEFINE q ^I<2*<xc_i> + 2*4*<xc_j> + 2*4*4*<xc_k> >^
                
                INIF LBM_dim==2
                    REG cells_f_F[(i_kap_bc+^I< <xc_i>+2*<xc_j>+4*<xc_k> >^)*M_CBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells] = \
                    REG s_F[^I<0+q>^] + \
                    REG (s_F[^I<1+q>^]-s_F[^I<0+q>^])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<4+q>^]-s_F[^I<0+q>^])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<5+q>^]-s_F[^I<4+q>^]-s_F[^I<1+q>^]+s_F[^I<0+q>^])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5));
                INELSE
                    REG cells_f_F[(i_kap_bc+^I< <xc_i>+2*<xc_j>+4*<xc_k> >^)*M_CBLOCK + threadIdx.x + LBM_pb(<p>)*n_maxcells] = \
                    REG s_F[^I<0+q>^] + \
                    REG (s_F[^I<1+q>^] - s_F[^I<0+q>^])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<4+q>^] - s_F[^I<0+q>^])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<16+q>^] - s_F[^I<0+q>^])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<5+q>^] - s_F[^I<1+q>^] - s_F[^I<4+q>^] + s_F[^I<0+q>^])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<17+q>^] - s_F[^I<1+q>^] - s_F[^I<16+q>^] + s_F[^I<0+q>^])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<20+q>^] - s_F[^I<4+q>^] - s_F[^I<16+q>^] + s_F[^I<0+q>^])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) + \
                    REG (s_F[^I<21+q>^] + s_F[^I<1+q>^] + s_F[^I<4+q>^] + s_F[^I<16+q>^] - s_F[^I<5+q>^] - s_F[^I<17+q>^] - s_F[^I<20+q>^] - s_F[^I<0+q>^])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));
                END_INIF
                
            END_OUTIF
        END_INIF
    END_INFOR 
    REG __syncthreads();
    <
END_INFOR

END_LOOPBLOCKS
