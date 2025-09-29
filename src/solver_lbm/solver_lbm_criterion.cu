/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"
#include "solver_lbm.h"





template <typename ufloat_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_ComputeRefCriteria_OLD
(
    const int id_max_curr,
    const int n_maxcells,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    int *__restrict__ cblock_ID_ref,
    const int *__restrict__ cblock_level,
    const int *__restrict__ cblock_ID_nbr,
    const ufloat_t *__restrict__ cells_f_F,
    const int *cells_ID_mask,
    const int S_CRITERION,
    const ufloat_t N_REFINE_START,
    const ufloat_t N_REFINE_INC,
    const ufloat_t N_REFINE_MAX,
    const int MAX_LEVELS_INTERIOR
)
{
    constexpr int VS = LP->VS;
    constexpr int CM = LP->CM;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_QUADS = AP->N_QUADS;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_HBLOCK = AP->M_HBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ ufloat_t s_u[M_HBLOCK];
    __shared__ ufloat_t s_v[M_HBLOCK];
    __shared__ ufloat_t s_w[M_HBLOCK];
    __shared__ ufloat_t s_W[M_TBLOCK];
    __shared__ ufloat_t s_Wmax[M_TBLOCK];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int I_kap = threadIdx.x % 4;
    int J_kap = (threadIdx.x / 4) % 4;
    int K_kap = (threadIdx.x / 4) / 4;
    
    // DDFs and macroscopic properties.
    ufloat_t f_i = (ufloat_t)(0.0);
    ufloat_t rho_kap = (ufloat_t)(0.0);
    ufloat_t u_kap = (ufloat_t)(0.0);
    ufloat_t v_kap = (ufloat_t)(0.0);
    ufloat_t w_kap = (ufloat_t)(0.0);

    // Intermediate vorticity variables.
    ufloat_t uX = (ufloat_t)(0.0);
    ufloat_t uY = (ufloat_t)(0.0);
    ufloat_t uZ = (ufloat_t)(0.0);
    ufloat_t vX = (ufloat_t)(0.0);
    ufloat_t vY = (ufloat_t)(0.0);
    ufloat_t vZ = (ufloat_t)(0.0);
    ufloat_t wX = (ufloat_t)(0.0);
    ufloat_t wY = (ufloat_t)(0.0);
    ufloat_t wZ = (ufloat_t)(0.0);
    ufloat_t tmp __attribute__((unused)) = (ufloat_t)(0.0);
    bool eligible = true;
    
    // Keep in mind that each ID represents a block, not just a cell.
    s_ID_cblock[threadIdx.x] = -1;
    s_Wmax[threadIdx.x] = -1000;
    if (kap < id_max_curr)
    {
        //int i_kap = id_set_idev_L[kap];
        
        s_ID_cblock[threadIdx.x] = kap;
    }
    __syncthreads();
    
    // Now we loop over all cell-blocks and operate on the cells.
    for (int k = 0; k < M_TBLOCK; k++)
    {
        int i_kap_b = s_ID_cblock[k];
        if (i_kap_b > -1 && cblock_ID_ref[i_kap_b] != V_REF_ID_INACTIVE)
        {
            for (int i_Q = 0; i_Q < N_QUADS; i_Q++)
            {
                // Reset variables.
                u_kap = (ufloat_t)(0.0);
                v_kap = (ufloat_t)(0.0);
                if (N_DIM==3)
                    w_kap = (ufloat_t)(0.0);
                
                // Compute local vorticity magnitudes and place in shared memory.
                //
                //
                //
if (VS == VS_D2Q9)
{
                if (CM == CM_BGK)
                {
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; 
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; 
                }
                else // CM == CM_MRT
                {
                    rho_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells];
                    u_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells] / rho_kap;
                    v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells] / rho_kap;
                }
}
if (VS == VS_D3Q19)
{
                if (CM == CM_BGK)
                {
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 10*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 9*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 12*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 11*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 14*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 13*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 16*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 15*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 18*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 17*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
                }
                else // CM == CM_MRT
                {
                    rho_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells];
                    u_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells] / rho_kap;
                    v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells] / rho_kap;
                    v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells] / rho_kap;
                }
}
if (VS == VS_D3Q27)
{
                if (CM == CM_BGK)
                {
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 2*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 1*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 3*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 5*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 7*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 10*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 9*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 12*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 11*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 14*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 13*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 16*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 15*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 18*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 17*n_maxcells]; rho_kap += f_i; u_kap = u_kap; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 20*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 19*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 22*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 21*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 24*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 23*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap -f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 26*n_maxcells]; rho_kap += f_i; u_kap = u_kap -f_i; v_kap = v_kap +f_i; w_kap = w_kap +f_i;
                    f_i = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 25*n_maxcells]; rho_kap += f_i; u_kap = u_kap +f_i; v_kap = v_kap -f_i; w_kap = w_kap -f_i;
                }
                else // CM == CM_MRT
                {
                    rho_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 0*n_maxcells];
                    u_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 4*n_maxcells] / rho_kap;
                    v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 6*n_maxcells] / rho_kap;
                    v_kap = cells_f_F[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + (size_t)threadIdx.x + 8*n_maxcells] / rho_kap;
                }
}
                
                
                
if (N_DIM==2)
{
                s_u[(I_kap+1)+(4+2)*(J_kap+1)] = u_kap;
                s_v[(I_kap+1)+(4+2)*(J_kap+1)] = v_kap;
                __syncthreads();
                if (I_kap==0)
                {
                    s_u[(0)+(4+2)*(J_kap+1)] = (ufloat_t)(4.0)*s_u[(1)+(4+2)*(J_kap+1)] - (ufloat_t)(6.0)*s_u[(2)+(4+2)*(J_kap+1)] + (ufloat_t)(4.0)*s_u[(3)+(4+2)*(J_kap+1)] - (ufloat_t)(1.0)*s_u[(4)+(4+2)*(J_kap+1)];
                    s_u[(5)+(4+2)*(J_kap+1)] = (ufloat_t)(4.0)*s_u[(4)+(4+2)*(J_kap+1)] - (ufloat_t)(6.0)*s_u[(3)+(4+2)*(J_kap+1)] + (ufloat_t)(4.0)*s_u[(2)+(4+2)*(J_kap+1)] - (ufloat_t)(1.0)*s_u[(1)+(4+2)*(J_kap+1)];
                    s_v[(0)+(4+2)*(J_kap+1)] = (ufloat_t)(4.0)*s_v[(1)+(4+2)*(J_kap+1)] - (ufloat_t)(6.0)*s_v[(2)+(4+2)*(J_kap+1)] + (ufloat_t)(4.0)*s_v[(3)+(4+2)*(J_kap+1)] - (ufloat_t)(1.0)*s_v[(4)+(4+2)*(J_kap+1)];
                    s_v[(5)+(4+2)*(J_kap+1)] = (ufloat_t)(4.0)*s_v[(4)+(4+2)*(J_kap+1)] - (ufloat_t)(6.0)*s_v[(3)+(4+2)*(J_kap+1)] + (ufloat_t)(4.0)*s_v[(2)+(4+2)*(J_kap+1)] - (ufloat_t)(1.0)*s_v[(1)+(4+2)*(J_kap+1)];
                }
                if (J_kap==0)
                {
                    s_u[(I_kap+1)+(4+2)*(0)] = (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(1)] - (ufloat_t)(6.0)*s_u[(I_kap+1)+(4+2)*(2)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(3)] - (ufloat_t)(1.0)*s_u[(I_kap+1)+(4+2)*(4)];
                    s_u[(I_kap+1)+(4+2)*(5)] = (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(4)] - (ufloat_t)(6.0)*s_u[(I_kap+1)+(4+2)*(3)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(2)] - (ufloat_t)(1.0)*s_u[(I_kap+1)+(4+2)*(1)];
                    s_v[(I_kap+1)+(4+2)*(0)] = (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(1)] - (ufloat_t)(6.0)*s_v[(I_kap+1)+(4+2)*(2)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(3)] - (ufloat_t)(1.0)*s_v[(I_kap+1)+(4+2)*(4)];
                    s_v[(I_kap+1)+(4+2)*(5)] = (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(4)] - (ufloat_t)(6.0)*s_v[(I_kap+1)+(4+2)*(3)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(2)] - (ufloat_t)(1.0)*s_v[(I_kap+1)+(4+2)*(1)];
                }
                __syncthreads();
                
                uX = (s_u[(I_kap+1 +1)+(4+2)*(J_kap+1)] - s_u[(I_kap+1 -1)+(4+2)*(J_kap+1)])/(2.0*dx_L);
                uY = (s_u[(I_kap+1)+(4+2)*(J_kap+1 +1)] - s_u[(I_kap+1)+(4+2)*(J_kap+1 -1)])/(2.0*dx_L);
                uZ = (ufloat_t)(0.0);
                vX = (s_v[(I_kap+1 +1)+(4+2)*(J_kap+1)] - s_v[(I_kap+1 -1)+(4+2)*(J_kap+1)])/(2.0*dx_L);
                vY = (s_v[(I_kap+1)+(4+2)*(J_kap+1 +1)] - s_v[(I_kap+1)+(4+2)*(J_kap+1 -1)])/(2.0*dx_L);
                vZ = (ufloat_t)(0.0);
                wX = (ufloat_t)(0.0);
                wY = (ufloat_t)(0.0);
                wZ = (ufloat_t)(0.0);
}
else
{
                s_u[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = u_kap;
                s_v[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = v_kap;
                s_w[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = w_kap;
                __syncthreads();
                if (I_kap==0)
                {
                    s_u[(0)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_u[(1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_u[(2)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_u[(3)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_u[(4)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)];
                    s_u[(5)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_u[(4)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_u[(3)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_u[(2)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_u[(1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)];
                    
                    s_v[(0)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_v[(1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_v[(2)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_v[(3)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_v[(4)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)];
                    s_v[(5)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_v[(4)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_v[(3)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_v[(2)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_v[(1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)];
                    s_w[(0)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_w[(1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_w[(2)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_w[(3)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_w[(4)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)];
                    s_w[(5)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_w[(4)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_w[(3)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_w[(2)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_w[(1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)];
                }
                if (J_kap==0)
                {
                    s_u[(I_kap+1)+(4+2)*(0)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_u[(I_kap+1)+(4+2)*(2)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(3)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_u[(I_kap+1)+(4+2)*(4)+(4+2)*(4+2)*(K_kap+1)];
                    s_u[(I_kap+1)+(4+2)*(5)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(4)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_u[(I_kap+1)+(4+2)*(3)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(2)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_u[(I_kap+1)+(4+2)*(1)+(4+2)*(4+2)*(K_kap+1)];
                    s_v[(I_kap+1)+(4+2)*(0)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_v[(I_kap+1)+(4+2)*(2)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(3)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_v[(I_kap+1)+(4+2)*(4)+(4+2)*(4+2)*(K_kap+1)];
                    s_v[(I_kap+1)+(4+2)*(5)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(4)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_v[(I_kap+1)+(4+2)*(3)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(2)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_v[(I_kap+1)+(4+2)*(1)+(4+2)*(4+2)*(K_kap+1)];
                    s_w[(I_kap+1)+(4+2)*(0)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(1)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_w[(I_kap+1)+(4+2)*(2)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(3)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_w[(I_kap+1)+(4+2)*(4)+(4+2)*(4+2)*(K_kap+1)];
                    s_w[(I_kap+1)+(4+2)*(5)+(4+2)*(4+2)*(K_kap+1)] = (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(4)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(6.0)*s_w[(I_kap+1)+(4+2)*(3)+(4+2)*(4+2)*(K_kap+1)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(2)+(4+2)*(4+2)*(K_kap+1)] - (ufloat_t)(1.0)*s_w[(I_kap+1)+(4+2)*(1)+(4+2)*(4+2)*(K_kap+1)];
                }
                if (K_kap==0)
                {
                    s_u[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(0)] = (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(1)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(4)];
                    s_u[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(5)] = (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(4)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_u[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(1)];
                    s_v[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(0)] = (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(1)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(4)];
                    s_v[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(5)] = (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(4)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_v[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(1)];
                    s_w[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(0)] = (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(1)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(4)];
                    s_w[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(5)] = (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(4)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_w[(I_kap+1)+(4+2)*(J_kap)+(4+2)*(4+2)*(1)];
                }
                __syncthreads();
                
                uX = (s_u[(I_kap+1 +1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - s_u[(I_kap+1 -1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)])/(2*dx_L);
                uY = (s_u[(I_kap+1)+(4+2)*(J_kap+1 +1)+(4+2)*(4+2)*(K_kap+1)] - s_u[(I_kap+1)+(4+2)*(J_kap+1 -1)+(4+2)*(4+2)*(K_kap+1)])/(2*dx_L);
                uZ = (s_u[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1 +1)] - s_u[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1 -1)])/(2*dx_L);
                vX = (s_v[(I_kap+1 +1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - s_v[(I_kap+1 -1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)])/(2*dx_L);
                vY = (s_v[(I_kap+1)+(4+2)*(J_kap+1 +1)+(4+2)*(4+2)*(K_kap+1)] - s_v[(I_kap+1)+(4+2)*(J_kap+1 -1)+(4+2)*(4+2)*(K_kap+1)])/(2*dx_L);
                vZ = (s_v[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1 +1)] - s_v[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1 -1)])/(2*dx_L);
                wX = (s_w[(I_kap+1 +1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)] - s_w[(I_kap+1 -1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1)])/(2*dx_L);
                wY = (s_w[(I_kap+1)+(4+2)*(J_kap+1 +1)+(4+2)*(4+2)*(K_kap+1)] - s_w[(I_kap+1)+(4+2)*(J_kap+1 -1)+(4+2)*(4+2)*(K_kap+1)])/(2*dx_L);
                wZ = (s_w[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1 +1)] - s_w[(I_kap+1)+(4+2)*(J_kap+1)+(4+2)*(4+2)*(K_kap+1 -1)])/(2*dx_L);
}
                __syncthreads();
                
                
                
                s_W[threadIdx.x] = 0;
                if (S_CRITERION == 0)
                    s_W[threadIdx.x] = floor(log2( sqrt((wY-vZ)*(wY-vZ) + (uZ-wX)*(uZ-wX) + (vX-uY)*(vX-uY)) ));
                if (S_CRITERION == 1)
                    s_W[threadIdx.x] = floor(log2( abs((uX*vY+vY*wZ+wZ*uX)-(uY*vX+vZ*wY+uZ*wX)) ));
                //
                //
                //
                
                // Set vorticity to zero if the cell is a ghost cell (it has wrong values).
                if (cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] == 2)
                    s_W[threadIdx.x] = (ufloat_t)(0.0);
                __syncthreads();
                
                // Block reduction for maximum.
                for (int s=blockDim.x/2; s>0; s>>=1)
                {
                    if (threadIdx.x < s)
                    {
                        s_W[threadIdx.x] = max( s_W[threadIdx.x],s_W[threadIdx.x + s] );
                    }
                    __syncthreads();
                }
                
                // Store maximum.
                if (threadIdx.x == 0)
                {
                    //s_Wmax[k] = s_W[0];
                    if (s_W[0] > s_Wmax[k])
                        s_Wmax[k] = s_W[0];
                }
                __syncthreads();
            }
        }
    }
    __syncthreads();
    
    // Evaluate criterion.
    if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
    {
        // If vorticity is very large, cap at 1.0 to indicate maximum needed refinement.
        if (s_Wmax[threadIdx.x] > N_REFINE_MAX)
            s_Wmax[threadIdx.x] = N_REFINE_MAX;
        
        int ref_kap = cblock_ID_ref[kap];
        int level_kap = cblock_level[kap];
        int L_desired = MAX_LEVELS_INTERIOR-1;
        for (int p = 1; p <= MAX_LEVELS_INTERIOR-1; p++)
        {
            if (s_Wmax[threadIdx.x] < N_REFINE_START-N_REFINE_INC*p)
                L_desired = (MAX_LEVELS_INTERIOR-1)-p;
        }
        
        // Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
        for (int p = 0; p < N_Q_max; p++)
        {
            if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
                eligible = false;
        }
        
        // If cell-block is unrefined but desired level is higher than current, mark for refinement.
        if (eligible && level_kap != MAX_LEVELS_INTERIOR-1 && ref_kap == V_REF_ID_UNREFINED && L_desired > level_kap)
            cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
        
        // If cell-block is refined (leaf) but desired level is less than current, mark for coarsening.
        if (ref_kap == V_REF_ID_REFINED && L_desired < level_kap+1)
            cblock_ID_ref[kap] = V_REF_ID_MARK_COARSEN;
    }
}

























































































template <typename ufloat_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_ComputeRefCriteria
(
    const int id_max_curr,
    const int n_maxcells,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    int *__restrict__ cblock_ID_ref,
    const int *__restrict__ cblock_level,
    const int *__restrict__ cblock_ID_nbr,
    const ufloat_t *__restrict__ cells_f_F,
    const int *__restrict__ cells_ID_mask,
    const int S_CRITERION,
    const ufloat_t N_REFINE_START,
    const ufloat_t N_REFINE_INC,
    const ufloat_t N_REFINE_MAX,
    const int MAX_LEVELS_INTERIOR
)
{
    constexpr int N_Q = LP->N_Q;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_HBLOCK = AP->M_HBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ ufloat_t s_u[M_HBLOCK];
    __shared__ ufloat_t s_v[M_HBLOCK];
    __shared__ ufloat_t s_w[M_HBLOCK];
    __shared__ ufloat_t s_W[M_TBLOCK];
    
    // Get the cell-block index.
    int i_kap_b = -1;
    if (blockIdx.x < id_max_curr)
        i_kap_b = blockIdx.x;
    
    // Now we loop over all cell-blocks and operate on the cells.
    if (i_kap_b > -1)
    {
        if (cblock_ID_ref[i_kap_b] != V_REF_ID_INACTIVE)
        {
            // Compute cell indices.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            if (N_DIM==3)
                K = (threadIdx.x / 4) / 4;
            
            // Retrieve macroscopic properties.
            //ufloat_t rho = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+0)*n_maxcells];
            ufloat_t u = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells];
            ufloat_t v = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells];
            ufloat_t w = static_cast<ufloat_t>(0.0);
            if (N_DIM==3)
                w = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+3)*n_maxcells];
            
            // Compute velocity gradient.
            ufloat_t uX = static_cast<ufloat_t>(0.0);
            ufloat_t uY = static_cast<ufloat_t>(0.0);
            ufloat_t uZ = static_cast<ufloat_t>(0.0);
            ufloat_t vX = static_cast<ufloat_t>(0.0);
            ufloat_t vY = static_cast<ufloat_t>(0.0);
            ufloat_t vZ = static_cast<ufloat_t>(0.0);
            ufloat_t wX = static_cast<ufloat_t>(0.0);
            ufloat_t wY = static_cast<ufloat_t>(0.0);
            ufloat_t wZ = static_cast<ufloat_t>(0.0);
            if (N_DIM==2)
            {
                s_u[(I+1)+(4+2)*(J+1)] = u;
                s_v[(I+1)+(4+2)*(J+1)] = v;
                __syncthreads();
                
                ExtrapolateToHalo<ufloat_t,N_DIM>(s_u,I,J,K);
                ExtrapolateToHalo<ufloat_t,N_DIM>(s_v,I,J,K);
//                 if (I==0)
//                 {
//                     s_u[(0)+(4+2)*(J+1)] = (ufloat_t)(4.0)*s_u[(1)+(4+2)*(J+1)] - (ufloat_t)(6.0)*s_u[(2)+(4+2)*(J+1)] + (ufloat_t)(4.0)*s_u[(3)+(4+2)*(J+1)] - (ufloat_t)(1.0)*s_u[(4)+(4+2)*(J+1)];
//                     s_u[(5)+(4+2)*(J+1)] = (ufloat_t)(4.0)*s_u[(4)+(4+2)*(J+1)] - (ufloat_t)(6.0)*s_u[(3)+(4+2)*(J+1)] + (ufloat_t)(4.0)*s_u[(2)+(4+2)*(J+1)] - (ufloat_t)(1.0)*s_u[(1)+(4+2)*(J+1)];
//                     s_v[(0)+(4+2)*(J+1)] = (ufloat_t)(4.0)*s_v[(1)+(4+2)*(J+1)] - (ufloat_t)(6.0)*s_v[(2)+(4+2)*(J+1)] + (ufloat_t)(4.0)*s_v[(3)+(4+2)*(J+1)] - (ufloat_t)(1.0)*s_v[(4)+(4+2)*(J+1)];
//                     s_v[(5)+(4+2)*(J+1)] = (ufloat_t)(4.0)*s_v[(4)+(4+2)*(J+1)] - (ufloat_t)(6.0)*s_v[(3)+(4+2)*(J+1)] + (ufloat_t)(4.0)*s_v[(2)+(4+2)*(J+1)] - (ufloat_t)(1.0)*s_v[(1)+(4+2)*(J+1)];
//                 }
//                 if (J==0)
//                 {
//                     s_u[(I+1)+(4+2)*(0)] = (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(1)] - (ufloat_t)(6.0)*s_u[(I+1)+(4+2)*(2)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(3)] - (ufloat_t)(1.0)*s_u[(I+1)+(4+2)*(4)];
//                     s_u[(I+1)+(4+2)*(5)] = (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(4)] - (ufloat_t)(6.0)*s_u[(I+1)+(4+2)*(3)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(2)] - (ufloat_t)(1.0)*s_u[(I+1)+(4+2)*(1)];
//                     s_v[(I+1)+(4+2)*(0)] = (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(1)] - (ufloat_t)(6.0)*s_v[(I+1)+(4+2)*(2)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(3)] - (ufloat_t)(1.0)*s_v[(I+1)+(4+2)*(4)];
//                     s_v[(I+1)+(4+2)*(5)] = (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(4)] - (ufloat_t)(6.0)*s_v[(I+1)+(4+2)*(3)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(2)] - (ufloat_t)(1.0)*s_v[(I+1)+(4+2)*(1)];
//                 }
                __syncthreads();
                
                uX = (s_u[(I+1 +1)+(4+2)*(J+1)] - s_u[(I+1 -1)+(4+2)*(J+1)])/(2.0*dx_L);
                uY = (s_u[(I+1)+(4+2)*(J+1 +1)] - s_u[(I+1)+(4+2)*(J+1 -1)])/(2.0*dx_L);
                vX = (s_v[(I+1 +1)+(4+2)*(J+1)] - s_v[(I+1 -1)+(4+2)*(J+1)])/(2.0*dx_L);
                vY = (s_v[(I+1)+(4+2)*(J+1 +1)] - s_v[(I+1)+(4+2)*(J+1 -1)])/(2.0*dx_L);
            }
            else // N_DIM==3
            {
                s_u[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = u;
                s_v[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = v;
                s_w[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = w;
                __syncthreads();
                
                ExtrapolateToHalo<ufloat_t,N_DIM>(s_u,I,J,K);
                ExtrapolateToHalo<ufloat_t,N_DIM>(s_v,I,J,K);
                ExtrapolateToHalo<ufloat_t,N_DIM>(s_w,I,J,K);
//                 if (I==0)
//                 {
//                     s_u[(0)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_u[(1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_u[(2)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_u[(3)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_u[(4)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)];
//                     s_u[(5)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_u[(4)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_u[(3)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_u[(2)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_u[(1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)];
//                     
//                     s_v[(0)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_v[(1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_v[(2)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_v[(3)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_v[(4)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)];
//                     s_v[(5)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_v[(4)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_v[(3)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_v[(2)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_v[(1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)];
//                     s_w[(0)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_w[(1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_w[(2)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_w[(3)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_w[(4)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)];
//                     s_w[(5)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_w[(4)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_w[(3)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_w[(2)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_w[(1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)];
//                 }
//                 if (J==0)
//                 {
//                     s_u[(I+1)+(4+2)*(0)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_u[(I+1)+(4+2)*(2)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(3)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_u[(I+1)+(4+2)*(4)+(4+2)*(4+2)*(K+1)];
//                     s_u[(I+1)+(4+2)*(5)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(4)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_u[(I+1)+(4+2)*(3)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(2)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_u[(I+1)+(4+2)*(1)+(4+2)*(4+2)*(K+1)];
//                     s_v[(I+1)+(4+2)*(0)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_v[(I+1)+(4+2)*(2)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(3)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_v[(I+1)+(4+2)*(4)+(4+2)*(4+2)*(K+1)];
//                     s_v[(I+1)+(4+2)*(5)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(4)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_v[(I+1)+(4+2)*(3)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(2)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_v[(I+1)+(4+2)*(1)+(4+2)*(4+2)*(K+1)];
//                     s_w[(I+1)+(4+2)*(0)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(1)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_w[(I+1)+(4+2)*(2)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(3)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_w[(I+1)+(4+2)*(4)+(4+2)*(4+2)*(K+1)];
//                     s_w[(I+1)+(4+2)*(5)+(4+2)*(4+2)*(K+1)] = (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(4)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(6.0)*s_w[(I+1)+(4+2)*(3)+(4+2)*(4+2)*(K+1)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(2)+(4+2)*(4+2)*(K+1)] - (ufloat_t)(1.0)*s_w[(I+1)+(4+2)*(1)+(4+2)*(4+2)*(K+1)];
//                 }
//                 if (K==0)
//                 {
//                     s_u[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(0)] = (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(1)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(4)];
//                     s_u[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(5)] = (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(4)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_u[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(1)];
//                     s_v[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(0)] = (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(1)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(4)];
//                     s_v[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(5)] = (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(4)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_v[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(1)];
//                     s_w[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(0)] = (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(1)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(4)];
//                     s_w[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(5)] = (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(4)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(3)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(2)] + (ufloat_t)(4.0)*s_w[(I+1)+(4+2)*(J)+(4+2)*(4+2)*(1)];
//                 }
                __syncthreads();
                
                uX = (s_u[(I+1 +1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - s_u[(I+1 -1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
                uY = (s_u[(I+1)+(4+2)*(J+1 +1)+(4+2)*(4+2)*(K+1)] - s_u[(I+1)+(4+2)*(J+1 -1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
                uZ = (s_u[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 +1)] - s_u[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 -1)])/(2*dx_L);
                vX = (s_v[(I+1 +1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - s_v[(I+1 -1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
                vY = (s_v[(I+1)+(4+2)*(J+1 +1)+(4+2)*(4+2)*(K+1)] - s_v[(I+1)+(4+2)*(J+1 -1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
                vZ = (s_v[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 +1)] - s_v[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 -1)])/(2*dx_L);
                wX = (s_w[(I+1 +1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - s_w[(I+1 -1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
                wY = (s_w[(I+1)+(4+2)*(J+1 +1)+(4+2)*(4+2)*(K+1)] - s_w[(I+1)+(4+2)*(J+1 -1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
                wZ = (s_w[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 +1)] - s_w[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 -1)])/(2*dx_L);
            }
            __syncthreads();
            
            s_W[threadIdx.x] = 0;
            if (S_CRITERION == 0)
                s_W[threadIdx.x] = floor(log2( sqrt((wY-vZ)*(wY-vZ) + (uZ-wX)*(uZ-wX) + (vX-uY)*(vX-uY)) ));
            if (S_CRITERION == 1)
                s_W[threadIdx.x] = floor(log2( abs((uX*vY+vY*wZ+wZ*uX)-(uY*vX+vZ*wY+uZ*wX)) ));
            
            // Set vorticity to zero if the cell is a ghost cell (it has wrong values).
            if (cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] == V_CELLMASK_GHOST)
                s_W[threadIdx.x] = (ufloat_t)(0.0);
            __syncthreads();
            
            // Block reduction for maximum.
            BlockwiseMaximum<ufloat_t>(threadIdx.x,blockDim.x,s_W);
            __syncthreads();
            
            // Evaluate criterion.
            if (threadIdx.x == 0 && cblock_ID_ref[i_kap_b] != V_REF_ID_INACTIVE)
            {
                // If vorticity is very large, cap at 1.0 to indicate maximum needed refinement.
                if (s_W[0] > N_REFINE_MAX)
                    s_W[0] = N_REFINE_MAX;

                int ref_kap = cblock_ID_ref[i_kap_b];
                int level_kap = cblock_level[i_kap_b];
                int L_desired = MAX_LEVELS_INTERIOR-1;
                for (int p = 1; p <= MAX_LEVELS_INTERIOR-1; p++)
                {
                    if (s_W[0] < N_REFINE_START-N_REFINE_INC*p)
                        L_desired = (MAX_LEVELS_INTERIOR-1)-p;
                }

                // Don't refine near invalid fine-grid boundaries. Only in the interior for quality purposes.
                bool eligible = true;
                for (int p = 0; p < N_Q_max; p++)
                {
                    if (cblock_ID_nbr[i_kap_b + p*n_maxcblocks] == N_SKIPID)
                        eligible = false;
                }

                // If cell-block is unrefined but desired level is higher than current, mark for refinement.
                if (eligible && level_kap != MAX_LEVELS_INTERIOR-1 && ref_kap == V_REF_ID_UNREFINED && L_desired > level_kap)
                    cblock_ID_ref[i_kap_b] = V_REF_ID_MARK_REFINE;

                // If cell-block is refined (leaf) but desired level is less than current, mark for coarsening.
                if (ref_kap == V_REF_ID_REFINED && L_desired < level_kap+1)
                    cblock_ID_ref[i_kap_b] = V_REF_ID_MARK_COARSEN;
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeRefCriteria(int i_dev, int L, int var)
{
    //for (int L = 0; L < MAX_LEVELS; L++)
    //    S_RefreshVariables(0,L);
    
    // Solution-based criterion. Only one type implemented for current solver.
    {
        //Cu_ComputeRefCriteria_OLD<ufloat_t,AP,LP> <<<(M_TBLOCK+mesh->id_max[i_dev][MAX_LEVELS]-1)/M_TBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(
        //Cu_ComputeRefCriteria_SEMI<ufloat_t,AP,LP> <<<(M_TBLOCK+mesh->id_max[i_dev][MAX_LEVELS]-1)/M_TBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(
        Cu_ComputeRefCriteria<ufloat_t,AP,LP> <<<mesh->id_max[i_dev][MAX_LEVELS],M_TBLOCK,0,mesh->streams[i_dev]>>>(
            mesh->id_max[i_dev][MAX_LEVELS], mesh->n_maxcells, mesh->n_maxcblocks, mesh->dxf_vec[L],
            mesh->c_cblock_ID_ref[i_dev], mesh->c_cblock_level[i_dev], mesh->c_cblock_ID_nbr[i_dev],
            mesh->c_cells_f_F[i_dev], mesh->c_cells_ID_mask[i_dev],
            S_CRITERION, mesh->N_REFINE_START, mesh->N_REFINE_INC, mesh->N_REFINE_MAX, mesh->MAX_LEVELS_INTERIOR
        );
    }
    
    return 0;
}
