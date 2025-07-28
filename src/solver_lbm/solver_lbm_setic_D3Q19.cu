/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "custom.h"
#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_SetInitialConditions_D3Q19
(
	const int n_ids_idev_L,
	const long int n_maxcells,
	const int n_maxcblocks,
	const int *__restrict__ id_set_idev_L,
	const int *__restrict__ cells_ID_mask,
	ufloat_t *__restrict__ cells_f_F,
	const ufloat_t *__restrict__ cblock_f_X,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb,
	const ufloat_t dx_L
)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q = LP->VS;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = (threadIdx.x / 4) / 4;

    s_ID_cblock[threadIdx.x] = -1;
    if ((threadIdx.x<M_LBLOCK)and(kap<n_ids_idev_L))
    {
        s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
    }
    __syncthreads();
    
    // Loop over block Ids.
    for (int k = 0; k < M_LBLOCK; k += 1)
    {
        int i_kap_b = s_ID_cblock[k];
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1))
        {
            // Compute cell coordinates. Obtain the initial conditions from the custom routine.
            ufloat_t x __attribute__((unused)) = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + (ufloat_t)0.5*dx_L;
            ufloat_t y __attribute__((unused)) = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + (ufloat_t)0.5*dx_L;
            ufloat_t z __attribute__((unused)) = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + (ufloat_t)0.5*dx_L;
            ufloat_t rho;
            ufloat_t u;
            ufloat_t v;
            ufloat_t w;
            Cu_ComputeIC<ufloat_t>(rho, u, v, w, x, y, z);
            ufloat_t udotu = u*u + v*v + w*w;
            
            // Compute equilibrium distributions from initial macroscopic conditions.
            #pragma unroll
            for (int p = 0; p < N_Q; p++)
            {
                ufloat_t cdotu = (ufloat_t)V_CONN_ID[0+p*27]*u + (ufloat_t)V_CONN_ID[1+p*27]*v + (ufloat_t)V_CONN_ID[2+p*27]*w;
                ufloat_t f_p = rho*(ufloat_t)LBMw[p]*( (ufloat_t)1.0 + (ufloat_t)3.0*cdotu + (ufloat_t)4.5*cdotu*cdotu - (ufloat_t)1.5*udotu );
                if ( cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] != V_CELLMASK_SOLID )
                    cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = f_p;
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_SetInitialConditions_D3Q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_SetInitialConditions_D3Q19<ufloat_t,ufloat_g_t,AP,LP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->dxf_vec[L]);
	}

	return 0;
}

