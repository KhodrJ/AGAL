/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*  Last Updated: Sun Jul 27 18:28:18 2025                                            */
/*                                                                                    */
/**************************************************************************************/

#include "solver_lbm.h"
#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
__global__
void Cu_Stream_Original_D3Q19
(
	const int n_ids_idev_L,
	const long int n_maxcells,
	const int n_maxcblocks,
	const int n_maxcells_b,
	const int *__restrict__ id_set_idev_L,
	const int *__restrict__ cells_ID_mask,
	ufloat_t *__restrict__ cells_f_F,
	const ufloat_g_t *__restrict__ cells_f_X_b,
	const int *__restrict__ cblock_ID_nbr,
	const int *__restrict__ cblock_ID_nbr_child,
	const int *__restrict__ cblock_ID_mask,
	const int *__restrict__ cblock_ID_onb_solid,
	const bool geometry_init
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int N_Q = LP->VS;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_nbr[27];
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
        int i_kap_bc = -1;
        int valid_block = -1;
        
        // Load data for conditions on cell-blocks.
        if (i_kap_b>-1)
        {
            i_kap_bc=cblock_ID_nbr_child[i_kap_b];
            valid_block=cblock_ID_mask[i_kap_b];
        }
        
        // Latter condition is added only if n>0.
        if ((i_kap_b>-1)&&((i_kap_bc<0)||(valid_block>-3)))
        {
            // Get masks for possible boundary nodes.
            int valid_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            if (geometry_init && valid_mask == -2)
                valid_block = cblock_ID_onb_solid[i_kap_b];
         
            // Load neighbor-block indices into shared memory.
            if (threadIdx.x==0)
            {
                for (int p = 0; p < N_Q_max; p++)
                {
                    if (V_CONN_ID[p+0*27] >= 0 && V_CONN_ID[p+1*27] >= 0 && V_CONN_ID[p+2*27] >= 0)
                        s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
                }
            }
            __syncthreads();
            
            //
            #pragma unroll
            for (int p = 1; p < N_Q; p++)
            {
                if ( (N_DIM==2 && (p==1||(p+1)%3==0))   ||   (N_DIM==3 && (p==26||((p-1)%2==0&&p<25))) )
                {
                    // Load current DDF. Get the face-cell link, if applicable.
                    ufloat_t f_p = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells];
                    ufloat_g_t dQ = (ufloat_g_t)(-1.0);
                    if (valid_mask == V_CELLMASK_BOUNDARY)
                        dQ = cells_f_X_b[valid_block*M_CBLOCK + threadIdx.x + p*n_maxcells_b];
                    
                    // 
                    int Ip = I + V_CONN_ID[p + 0*27];
                    int Jp = J + V_CONN_ID[p + 1*27];
                    int Kp = 0;
                    if (N_DIM==3)
                            Kp = K + V_CONN_ID[p + 2*27];
                    
                    // Assign the correct neighbor cell-block ID.
                    int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                    Ip = (4 + (Ip % 4)) % 4;
                    Jp = (4 + (Jp % 4)) % 4;
                    if (N_DIM==3)
                            Kp = (4 + (Kp % 4)) % 4;
                    int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                    
                    // Retrieve neighboring DDFs, if applicable.
                    ufloat_t f_pb = (ufloat_t)(-1.0);
                    if ( nbr_kap_b>=0 )
                        f_pb = cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + 3*n_maxcells];
                    
                    // Exchange, if applicable.
                    if ( valid_mask != -1 && f_pb>=0 && dQ < 0)
                    {
                        cells_f_F[nbr_kap_b*M_CBLOCK + nbr_kap_c + LBMpb[p]*n_maxcells] = f_p;
                        cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + p*n_maxcells] = f_pb;
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Stream_Original_D3Q19(int i_dev, int L)
{
	if (mesh->n_ids[i_dev][L] > 0)
	{
		Cu_Stream_Original_D3Q19<ufloat_t,ufloat_g_t,AP,LP><<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(mesh->n_ids[i_dev][L], n_maxcells, n_maxcblocks, mesh->n_maxcells_b, &mesh->c_id_set[i_dev][L*n_maxcblocks], mesh->c_cells_ID_mask[i_dev], mesh->c_cells_f_F[i_dev], mesh->c_cells_f_X_b[i_dev], mesh->c_cblock_ID_nbr[i_dev], mesh->c_cblock_ID_nbr_child[i_dev], mesh->c_cblock_ID_mask[i_dev], mesh->c_cblock_ID_onb_solid[i_dev], mesh->geometry_init);
	}

	return 0;
}

