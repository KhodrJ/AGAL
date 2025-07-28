/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:19 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_ImposeBC_D3Q19
(
	const int n_ids_idev_L,
	const long int n_maxcells,
	const int n_maxcblocks,
	const int n_maxcells_b,
	const int n_maxblocks_b,
	const ufloat_t dx_L,
	const ufloat_t dx_L_g,
	const ufloat_t tau_L,
	const int *__restrict__ id_set_idev_L,
	const int *__restrict__ cells_ID_mask,
	ufloat_t *__restrict__ cells_f_F,
	const ufloat_g_t *__restrict__ cells_f_X_b,
	const ufloat_t *__restrict__ cblock_f_X,
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb,
	const int *__restrict__ cblock_ID_onb_solid,
	const bool geometry_init,
	const int force_type,
	const int bc_type
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int N_Q = LP->VS;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_nbr[N_Q_max];
    //__shared__ ufloat_t s_u[3*M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = 0;
    if (N_DIM==3)
        K = (threadIdx.x / 4) / 4;
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
        int valid_block = -1;
        
        // Load data for conditions on cell-blocks.
        if (i_kap_b>-1)
        {
            valid_block=cblock_ID_onb[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((valid_block==1)))
        {
            // Compute cell coordinates and retrieve macroscopic properties.
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            int block_mask = -1;
            if (geometry_init)
                block_mask = cblock_ID_onb_solid[i_kap_b];
            ufloat_t x __attribute__((unused)) = cblock_f_X[i_kap_b + 0*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + I);
            ufloat_t y __attribute__((unused)) = cblock_f_X[i_kap_b + 1*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + J);
            ufloat_t z __attribute__((unused)) = cblock_f_X[i_kap_b + 2*n_maxcblocks] + dx_L*((ufloat_t)(0.5) + K);
            
            // Get macroscopic properties and place them in shared memory.
            ufloat_t rho = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+0)*n_maxcells];
            ufloat_t u = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+1)*n_maxcells];
            ufloat_t v = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+2)*n_maxcells];
            ufloat_t w;
            if (N_DIM==3)
                w = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + (N_Q+3)*n_maxcells];
            
            // Retrieve indices of neighboring blocks.
            if (threadIdx.x==0)
            {
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads();
            
            // Loop over pairs of DDFs and impose BC.
            #pragma unroll
            for (int p = 1; p < N_Q; p++)
            {
                if ( (N_DIM==2 && (p==1||(p+1)%3==0))   ||   (N_DIM==3 && (p==26||((p-1)%2==0&&p<25))) )
                {
                    // Retrieve the DDF. Use correct order of access this time.
                    ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
                    ufloat_t f_q = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells];
                    
                    //
                    // BC of p.
                    //
                    
                    {
                        // Compute incremented local indices. Seeking the index of the cell in opposite direction of p.
                        int Ip = I + V_CONN_ID[LBMpb[p] + 0*27];
                        int Jp = J + V_CONN_ID[LBMpb[p] + 1*27];
                        int Kp = 0;
                        if (N_DIM==3)
                                Kp = K + V_CONN_ID[LBMpb[p] + 2*27];
                        
                        // Identify the correct neighboring block and cell.
                        int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                        
                        // Impose default BC.
                        if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                            f_p = Cu_ImposeBC(nbr_kap_b, f_p, rho, u, v, w, x, y, z, (ufloat_t)LBMw[p], (ufloat_t)V_CONN_ID[p+0*27], (ufloat_t)V_CONN_ID[p+1*27], (ufloat_t)V_CONN_ID[p+2*27]);
                    }

                    // Do interpolated bounce-back, if applicable.
                    if (valid_mask == V_CELLMASK_BOUNDARY)
                    {
                        // Get face-cell link.
                        ufloat_g_t dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + p*n_maxcells_b] / dx_L_g;
                        
                        // Re-compute incremented local indices.
                        int Ip = I + V_CONN_ID[LBMpb[p] + 0*27];
                        int Jp = J + V_CONN_ID[LBMpb[p] + 1*27];
                        int Kp = 0;
                        if (N_DIM==3)
                                Kp = K + V_CONN_ID[LBMpb[p] + 2*27];
                        
                        // Identify the correct neighboring block and cell.
                        int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                        Ip = (4 + (Ip % 4)) % 4;
                        Jp = (4 + (Jp % 4)) % 4;
                        if (N_DIM==3)
                                Kp = (4 + (Kp % 4)) % 4;
                        int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                        
                        // Retrieve DDF from 'behind.'
                        ufloat_t f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + p*n_maxcells];
                        
                        // ULI.
                        if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                        {
                            f_p = (ufloat_t)(2.0)*dQ*f_p + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                        }
                        // DLI.
                        if (dQ >= (ufloat_g_t)(0.5))
                        {
                            f_p = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_p + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_q;
                        }
                    }
                    
                    //
                    // BC of pb.
                    //
                    
                    {
                        // Compute incremented local indices.  Seeking the index of the cell in opposite direction of pb.
                        int Ip = I + V_CONN_ID[p + 0*27];
                        int Jp = J + V_CONN_ID[p + 1*27];
                        int Kp = 0;
                        if (N_DIM==3)
                                Kp = K + V_CONN_ID[p + 2*27];
                        
                        // Identify the correct neighboring block and cell.
                        int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                        
                        // Impose default BC.
                        if (nbr_kap_b < 0 && nbr_kap_b != N_SKIPID)
                            f_q = Cu_ImposeBC(nbr_kap_b, f_q, rho, u, v, w, x, y, z, (ufloat_t)LBMw[LBMpb[p]], (ufloat_t)V_CONN_ID[LBMpb[p]+0*27], (ufloat_t)V_CONN_ID[LBMpb[p]+1*27], (ufloat_t)V_CONN_ID[LBMpb[p]+2*27]);
                    }
                    
                    // Do interpolated bounce-back, if applicable.
                    if (valid_mask == V_CELLMASK_BOUNDARY)
                    {
                        // Get face-cell link.
                        ufloat_g_t dQ = cells_f_X_b[block_mask*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells_b] / dx_L_g;
                        
                        // Re-compute incremented local indices.
                        int Ip = I + V_CONN_ID[p + 0*27];
                        int Jp = J + V_CONN_ID[p + 1*27];
                        int Kp = 0;
                        if (N_DIM==3)
                                Kp = K + V_CONN_ID[p + 2*27];
                        
                        // Identify the correct neighboring block and cell.
                        int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                        Ip = (4 + (Ip % 4)) % 4;
                        Jp = (4 + (Jp % 4)) % 4;
                        if (N_DIM==3)
                                Kp = (4 + (Kp % 4)) % 4;
                        int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                        
                        // Retrieve DDF from 'behind.'
                        ufloat_t f_m = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + LBMpb[p]*n_maxcells];
                        
                        // ULI.
                        if (dQ > 0 && dQ < (ufloat_g_t)(0.5))
                        {
                            f_q = (ufloat_t)(2.0)*dQ*f_q + ((ufloat_t)(1.0) - (ufloat_t)(2.0)*dQ)*f_m;
                        }
                        
                        // DLI.
                        if (dQ >= (ufloat_g_t)(0.5))
                        {
                            f_q = ((ufloat_t)(1.0)/((ufloat_t)(2.0)*dQ))*f_q + (((ufloat_t)(2.0)*dQ - (ufloat_t)(1.0))/((ufloat_t)(2.0)*dQ))*f_p;
                        }
                    }
                    
                    // Write fi* to global memory.
                    if (valid_mask != -1)
                    {
                        cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = f_p;
                        cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + LBMpb[p]*n_maxcells] = f_q;
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ImposeBC_D3Q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_ImposeBC_D3Q19<ufloat_t,ufloat_g_t,AP,LP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, mesh->n_solidb, dxf_vec[L], (ufloat_g_t)dxf_vec[L], tau_vec[L], &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cblock_f_X[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->geometry_init, S_FORCE_TYPE, S_BC_TYPE);
	}

	return 0;
}
