/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

/*
                d8888                   d8b 888 d8b                                    
               d88888                   Y8P 888 Y8P                                    
              d88P888                       888                                        
             d88P 888 888  888 888  888 888 888 888  8888b.  888d888 888  888          
            d88P  888 888  888 `Y8bd8P' 888 888 888     "88b 888P"   888  888          
           d88P   888 888  888   X88K   888 888 888 .d888888 888     888  888          
          d8888888888 Y88b 888 .d8""8b. 888 888 888 888  888 888     Y88b 888          
88888888 d88P     888  "Y88888 888  888 888 888 888 "Y888888 888      "Y88888 88888888 
                                                                          888          
                                                                     Y8b d88P          
                                                                      "Y88P"           
*/



template <class T>
__global__
void Cu_Concat(int N, T *arr, int N2, T *arr2)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < N2)
    {
        arr[N+kap] = arr2[kap];
    }
}

template <class T>
__global__
void Cu_ConcatReverse(int N, T *arr, int N2, T *arr2)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < N2)
    {
        arr[N+kap] = arr2[N2-1-kap];
    }
}

template <class T>
__global__
void Cu_Debug_ModVals(T *arr, int start, int N, T *arr2)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < N)
    {
        arr[start+kap] = arr2[kap];
    }
}

// Used purely for debugging.
template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_CudaModVals(int var)
{
    // Set IDs 1, 9, 14 to coarsen to check coarsening.
    if (var == 0)
    {
        int i_dev = 0;
        for (int xc = 0; xc < N_CHILDREN/2; xc++)
            cblock_ID_ref[i_dev][64 + xc] = V_REF_ID_MARK_REFINE;
        cblock_ID_ref[i_dev][9] = V_REF_ID_MARK_COARSEN;
        gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], n_maxcblocks*sizeof(int), cudaMemcpyHostToDevice) );
    }
    
    if (var == 1)
    {
        int i_dev = 0;
        cblock_ID_ref[i_dev][9] = V_REF_ID_MARK_REFINE;
        gpuErrchk( cudaMemcpy(c_cblock_ID_ref[i_dev], cblock_ID_ref[i_dev], n_maxcblocks*sizeof(int), cudaMemcpyHostToDevice) );
    }

    return 0;    
}



/*
         .d8888b.                                               888    d8b          d8b 888                      
        d88P  Y88b                                              888    Y8P          Y8P 888                      
        888    888                                              888                     888                      
        888         .d88b.  88888b.  88888b.   .d88b.   .d8888b 888888 888 888  888 888 888888 888  888          
        888        d88""88b 888 "88b 888 "88b d8P  Y8b d88P"    888    888 888  888 888 888    888  888          
        888    888 888  888 888  888 888  888 88888888 888      888    888 Y88  88P 888 888    888  888          
        Y88b  d88P Y88..88P 888  888 888  888 Y8b.     Y88b.    Y88b.  888  Y8bd8P  888 Y88b.  Y88b 888          
88888888 "Y8888P"   "Y88P"  888  888 888  888  "Y8888   "Y8888P  "Y888 888   Y88P   888  "Y888  "Y88888 88888888 
                                                                                                    888          
                                                                                               Y8b d88P          
                                                                                                "Y88P"           
*/



template <const ArgsPack *AP>
__global__
void Cu_UpdateBoundaries
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_nbr, int *cblock_ID_nbr_child, int *cblock_ID_mask, int *cblock_ID_onb
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        int nbr_kap_p = 0;
        int block_on_boundary = 0;
        int block_mask = cblock_ID_mask[kap];
        //int i_kap_bc = cblock_ID_nbr_child[kap];
        
        for (int p = 1; p < N_Q_max; p++)
        {
            nbr_kap_p = cblock_ID_nbr[kap + p*n_maxcblocks];
            if (nbr_kap_p < 0)
            {
                cblock_ID_nbr_child[kap + p*n_maxcblocks] = nbr_kap_p;
                if (nbr_kap_p != N_SKIPID)
                    block_on_boundary = 1;
            }
        }
        if (block_mask == V_BLOCKMASK_SOLIDA)
            block_on_boundary = 1;
        
        cblock_ID_onb[kap] = block_on_boundary;
    }
}

template <const ArgsPack *AP>
__global__
void Cu_UpdateConnectivity
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child,
    int *scattered_map
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_CHILDREN = AP->N_CHILDREN;
    constexpr int M_BLOCK = AP->M_BLOCK;
    __shared__ int s_ID_child[M_BLOCK];
    __shared__ int s_ID_child_nbrs[M_BLOCK*N_CHILDREN];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int t_id_incrementor = threadIdx.x % N_CHILDREN;
    int t_id_builder = threadIdx.x / N_CHILDREN;
    int id_nbr_0_child, id_nbr_1_child, id_nbr_2_child, id_nbr_3_child, id_nbr_4_child, id_nbr_5_child,  id_nbr_6_child, id_nbr_7_child, id_nbr_8_child;
    int id_nbr_9_child, id_nbr_10_child, id_nbr_11_child, id_nbr_12_child, id_nbr_13_child, id_nbr_14_child, id_nbr_15_child, id_nbr_16_child, id_nbr_17_child, id_nbr_18_child, id_nbr_19_child, id_nbr_20_child, id_nbr_21_child, id_nbr_22_child, id_nbr_23_child, id_nbr_24_child, id_nbr_25_child, id_nbr_26_child;
    
    // Initialize shared memory.
    s_ID_child[threadIdx.x] = -1;
    s_ID_child_nbrs[threadIdx.x] = -1;
    
    if (kap < id_max_curr)
    {
        if (scattered_map[kap] > -1 && cblock_ID_nbr_child[kap + 0*n_maxcblocks] > -1)
        {
            s_ID_child[threadIdx.x] = cblock_ID_nbr_child[kap + 0*n_maxcblocks]; //scattered_map[kap];
            id_nbr_0_child = cblock_ID_nbr_child[kap + 0*n_maxcblocks];
            id_nbr_1_child = cblock_ID_nbr_child[kap + 1*n_maxcblocks];
            id_nbr_2_child = cblock_ID_nbr_child[kap + 2*n_maxcblocks];
            id_nbr_3_child = cblock_ID_nbr_child[kap + 3*n_maxcblocks];
            id_nbr_4_child = cblock_ID_nbr_child[kap + 4*n_maxcblocks];
            id_nbr_5_child = cblock_ID_nbr_child[kap + 5*n_maxcblocks];
            id_nbr_6_child = cblock_ID_nbr_child[kap + 6*n_maxcblocks];
            id_nbr_7_child = cblock_ID_nbr_child[kap + 7*n_maxcblocks];
            id_nbr_8_child = cblock_ID_nbr_child[kap + 8*n_maxcblocks];
if (N_DIM==3)
{
            id_nbr_9_child = cblock_ID_nbr_child[kap + 9*n_maxcblocks];
            id_nbr_10_child = cblock_ID_nbr_child[kap + 10*n_maxcblocks];
            id_nbr_11_child = cblock_ID_nbr_child[kap + 11*n_maxcblocks];
            id_nbr_12_child = cblock_ID_nbr_child[kap + 12*n_maxcblocks];
            id_nbr_13_child = cblock_ID_nbr_child[kap + 13*n_maxcblocks];
            id_nbr_14_child = cblock_ID_nbr_child[kap + 14*n_maxcblocks];
            id_nbr_15_child = cblock_ID_nbr_child[kap + 15*n_maxcblocks];
            id_nbr_16_child = cblock_ID_nbr_child[kap + 16*n_maxcblocks];
            id_nbr_17_child = cblock_ID_nbr_child[kap + 17*n_maxcblocks];
            id_nbr_18_child = cblock_ID_nbr_child[kap + 18*n_maxcblocks];
            id_nbr_19_child = cblock_ID_nbr_child[kap + 19*n_maxcblocks];
            id_nbr_20_child = cblock_ID_nbr_child[kap + 20*n_maxcblocks];
            id_nbr_21_child = cblock_ID_nbr_child[kap + 21*n_maxcblocks];
            id_nbr_22_child = cblock_ID_nbr_child[kap + 22*n_maxcblocks];
            id_nbr_23_child = cblock_ID_nbr_child[kap + 23*n_maxcblocks];
            id_nbr_24_child = cblock_ID_nbr_child[kap + 24*n_maxcblocks];
            id_nbr_25_child = cblock_ID_nbr_child[kap + 25*n_maxcblocks];
            id_nbr_26_child = cblock_ID_nbr_child[kap + 26*n_maxcblocks];            
}
        }
    }
    __syncthreads();

if (N_DIM==2)
{
    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 0*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 1*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 0;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 2*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 3;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 3*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 4*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 5*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 6*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 7*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 8*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();
}
else
{
    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 0*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 1*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 2*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 3*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 4*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 5*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 6*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 0;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 4;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 7*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 7;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 8*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 0;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 2;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 9*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 5;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 7;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 10*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 0;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 1;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 11*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 6;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 7;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 12*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 6;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 13*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 1;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 5;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 14*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 4;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 6;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 15*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 1;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 3;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 16*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 4;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 5;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 17*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 2;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 3;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 18*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_19_child<0 ? id_nbr_19_child:id_nbr_19_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 19*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_20_child<0 ? id_nbr_20_child:id_nbr_20_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 20*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_21_child<0 ? id_nbr_21_child:id_nbr_21_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_7_child<0 ? id_nbr_7_child:id_nbr_7_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 21*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_8_child<0 ? id_nbr_8_child:id_nbr_8_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_22_child<0 ? id_nbr_22_child:id_nbr_22_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 22*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_18_child<0 ? id_nbr_18_child:id_nbr_18_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_23_child<0 ? id_nbr_23_child:id_nbr_23_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_9_child<0 ? id_nbr_9_child:id_nbr_9_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 23*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_10_child<0 ? id_nbr_10_child:id_nbr_10_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_24_child<0 ? id_nbr_24_child:id_nbr_24_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_17_child<0 ? id_nbr_17_child:id_nbr_17_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 24*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_2_child<0 ? id_nbr_2_child:id_nbr_2_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_14_child<0 ? id_nbr_14_child:id_nbr_14_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_3_child<0 ? id_nbr_3_child:id_nbr_3_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_16_child<0 ? id_nbr_16_child:id_nbr_16_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_5_child<0 ? id_nbr_5_child:id_nbr_5_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_25_child<0 ? id_nbr_25_child:id_nbr_25_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_11_child<0 ? id_nbr_11_child:id_nbr_11_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 25*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();

    if (kap < id_max_curr)
    {    if (s_ID_child[threadIdx.x] > -1)
        {
            s_ID_child_nbrs[0 + threadIdx.x*N_CHILDREN] = id_nbr_12_child<0 ? id_nbr_12_child:id_nbr_12_child + 7;
            s_ID_child_nbrs[1 + threadIdx.x*N_CHILDREN] = id_nbr_26_child<0 ? id_nbr_26_child:id_nbr_26_child + 6;
            s_ID_child_nbrs[2 + threadIdx.x*N_CHILDREN] = id_nbr_6_child<0 ? id_nbr_6_child:id_nbr_6_child + 5;
            s_ID_child_nbrs[3 + threadIdx.x*N_CHILDREN] = id_nbr_15_child<0 ? id_nbr_15_child:id_nbr_15_child + 4;
            s_ID_child_nbrs[4 + threadIdx.x*N_CHILDREN] = id_nbr_4_child<0 ? id_nbr_4_child:id_nbr_4_child + 3;
            s_ID_child_nbrs[5 + threadIdx.x*N_CHILDREN] = id_nbr_13_child<0 ? id_nbr_13_child:id_nbr_13_child + 2;
            s_ID_child_nbrs[6 + threadIdx.x*N_CHILDREN] = id_nbr_0_child<0 ? id_nbr_0_child:id_nbr_0_child + 1;
            s_ID_child_nbrs[7 + threadIdx.x*N_CHILDREN] = id_nbr_1_child<0 ? id_nbr_1_child:id_nbr_1_child + 0;
        }
    }
    __syncthreads();
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        if (ID_child_q > -1) cblock_ID_nbr[ID_child_q + t_id_incrementor + 26*n_maxcblocks] = s_ID_child_nbrs[threadIdx.x + q*M_BLOCK];
    }
    __syncthreads();
}
}

template <const ArgsPack *AP>
__global__
void Cu_UpdateMasks_1
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_mask, int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int mask_ID = V_BLOCKMASK_REGULAR;
    
    if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
    {
        // Masking is only necessary for blocks with children.
        int i_c = cblock_ID_nbr_child[kap];
        if (i_c > -1)
        {
            // Loop over neighbor-children. A value of N_SKPID indicates that at least one child is gonna be masked.
            for (int p = 1; p < N_Q_max; p++)
            {
                // Focus only on N_SKIPID, other negative values represent domain boundaries which don't require masks.
                if (cblock_ID_nbr_child[kap + p*n_maxcblocks] == N_SKIPID)
                    mask_ID = V_BLOCKMASK_INTERFACE;
            }
        }
        
        if (cblock_ID_mask[kap] > -1)
            cblock_ID_mask[kap] = mask_ID;
    }
}

template <const ArgsPack *AP>
__global__
void Cu_UpdateMasks_2
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_nbr, int *cblock_ID_ref, int *cells_ID_mask
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int Nqx = AP->Nqx;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_ID_mask[M_TBLOCK];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int I_kap = threadIdx.x % 4;
    int J_kap = (threadIdx.x / 4) % 4;
    int K_kap = (threadIdx.x / 4) / 4;
    
    // Keep in mind that each ID represents a block, not just a cell.
    s_ID_cblock[threadIdx.x] = -1;
    if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
        s_ID_cblock[threadIdx.x] = kap;
    __syncthreads();

    for (int k = 0; k < M_TBLOCK; k++)
    {
        int i_kap_b = s_ID_cblock[k];
        
        if (i_kap_b > -1)
        {
if (N_DIM==2)
{
            for (int j_q = 0; j_q < Nqx; j_q += 1)
            {
                for (int i_q = 0; i_q < Nqx; i_q += 1)
                {
                    int i_Q = i_q + Nqx*j_q;
                    s_ID_mask[threadIdx.x] = cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x];
                    if (s_ID_mask[threadIdx.x] >= 0)
                    {
                        s_ID_mask[threadIdx.x] = 0;
                        if ((cblock_ID_nbr[i_kap_b + 1*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 2*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 3*n_maxcblocks]==N_SKIPID)and((i_q==0)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 4*n_maxcblocks]==N_SKIPID)and((j_q==0)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 5*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 6*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 7*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 8*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)))
                            s_ID_mask[threadIdx.x] = 1;
                        if ((cblock_ID_nbr[i_kap_b + 1*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)))
                        {
                            if ((I_kap >= 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        if ((cblock_ID_nbr[i_kap_b + 2*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)))
                        {
                            if ((J_kap >= 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        if ((cblock_ID_nbr[i_kap_b + 3*n_maxcblocks]==N_SKIPID)and((i_q==0)))
                        {
                            if ((I_kap < 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        if ((cblock_ID_nbr[i_kap_b + 4*n_maxcblocks]==N_SKIPID)and((j_q==0)))
                        {
                            if ((J_kap < 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        if ((cblock_ID_nbr[i_kap_b + 5*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)))
                        {
                            if ((I_kap >= 2)and(J_kap >= 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        if ((cblock_ID_nbr[i_kap_b + 6*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)))
                        {
                            if ((I_kap < 2)and(J_kap >= 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        if ((cblock_ID_nbr[i_kap_b + 7*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)))
                        {
                            if ((I_kap < 2)and(J_kap < 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        if ((cblock_ID_nbr[i_kap_b + 8*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)))
                        {
                            if ((I_kap >= 2)and(J_kap < 2))
                                s_ID_mask[threadIdx.x] = 2;
                        }
                        cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] = s_ID_mask[threadIdx.x];
                    }
                    __syncthreads();
                }
            }
}
else
{
            for (int k_q = 0; k_q < Nqx; k_q += 1)
            {
                for (int j_q = 0; j_q < Nqx; j_q += 1)
                {
                    for (int i_q = 0; i_q < Nqx; i_q += 1)
                    {
                        int i_Q = i_q+Nqx*j_q+Nqx*Nqx*k_q;
                        s_ID_mask[threadIdx.x] = cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x];
                        if (s_ID_mask[threadIdx.x] >= 0)
                        {
                            s_ID_mask[threadIdx.x] = 0;
                            if ((cblock_ID_nbr[i_kap_b + 1*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 2*n_maxcblocks]==N_SKIPID)and((i_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 3*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 4*n_maxcblocks]==N_SKIPID)and((j_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 5*n_maxcblocks]==N_SKIPID)and((k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 6*n_maxcblocks]==N_SKIPID)and((k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 7*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 8*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 9*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 10*n_maxcblocks]==N_SKIPID)and((i_q==0)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 11*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 12*n_maxcblocks]==N_SKIPID)and((j_q==0)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 13*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 14*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 15*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 16*n_maxcblocks]==N_SKIPID)and((i_q==0)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 17*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 18*n_maxcblocks]==N_SKIPID)and((j_q==0)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 19*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 20*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 21*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 22*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 23*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 24*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 25*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)and(k_q==Nqx-1)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 26*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)and(k_q==0)))
                                s_ID_mask[threadIdx.x] = 1;
                            if ((cblock_ID_nbr[i_kap_b + 1*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)))
                            {
                                if ((I_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 2*n_maxcblocks]==N_SKIPID)and((i_q==0)))
                            {
                                if ((I_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 3*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)))
                            {
                                if ((J_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 4*n_maxcblocks]==N_SKIPID)and((j_q==0)))
                            {
                                if ((J_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 5*n_maxcblocks]==N_SKIPID)and((k_q==Nqx-1)))
                            {
                                if ((K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 6*n_maxcblocks]==N_SKIPID)and((k_q==0)))
                            {
                                if ((K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 7*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)))
                            {
                                if ((I_kap >= 2)and(J_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 8*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)))
                            {
                                if ((I_kap < 2)and(J_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 9*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(k_q==Nqx-1)))
                            {
                                if ((I_kap >= 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 10*n_maxcblocks]==N_SKIPID)and((i_q==0)and(k_q==0)))
                            {
                                if ((I_kap < 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 11*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)and(k_q==Nqx-1)))
                            {
                                if ((J_kap >= 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 12*n_maxcblocks]==N_SKIPID)and((j_q==0)and(k_q==0)))
                            {
                                if ((J_kap < 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 13*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)))
                            {
                                if ((I_kap >= 2)and(J_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 14*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)))
                            {
                                if ((I_kap < 2)and(J_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 15*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(k_q==0)))
                            {
                                if ((I_kap >= 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 16*n_maxcblocks]==N_SKIPID)and((i_q==0)and(k_q==Nqx-1)))
                            {
                                if ((I_kap < 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 17*n_maxcblocks]==N_SKIPID)and((j_q==Nqx-1)and(k_q==0)))
                            {
                                if ((J_kap >= 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 18*n_maxcblocks]==N_SKIPID)and((j_q==0)and(k_q==Nqx-1)))
                            {
                                if ((J_kap < 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 19*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)and(k_q==Nqx-1)))
                            {
                                if ((I_kap >= 2)and(J_kap >= 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 20*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)and(k_q==0)))
                            {
                                if ((I_kap < 2)and(J_kap < 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 21*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==Nqx-1)and(k_q==0)))
                            {
                                if ((I_kap >= 2)and(J_kap >= 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 22*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==0)and(k_q==Nqx-1)))
                            {
                                if ((I_kap < 2)and(J_kap < 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 23*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)and(k_q==Nqx-1)))
                            {
                                if ((I_kap >= 2)and(J_kap < 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 24*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)and(k_q==0)))
                            {
                                if ((I_kap < 2)and(J_kap >= 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 25*n_maxcblocks]==N_SKIPID)and((i_q==0)and(j_q==Nqx-1)and(k_q==Nqx-1)))
                            {
                                if ((I_kap < 2)and(J_kap >= 2)and(K_kap >= 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            if ((cblock_ID_nbr[i_kap_b + 26*n_maxcblocks]==N_SKIPID)and((i_q==Nqx-1)and(j_q==0)and(k_q==0)))
                            {
                                if ((I_kap >= 2)and(J_kap < 2)and(K_kap < 2))
                                    s_ID_mask[threadIdx.x] = 2;
                            }
                            cells_ID_mask[i_kap_b*M_CBLOCK + i_Q*M_TBLOCK + threadIdx.x] = s_ID_mask[threadIdx.x];
                        }
                        __syncthreads();
                    }
                }
            }
}
        }
    }
}



/*
                d8888 888                           d8b 888    888                             
               d88888 888                           Y8P 888    888                             
              d88P888 888                               888    888                             
             d88P 888 888  .d88b.   .d88b.  888d888 888 888888 88888b.  88888b.d88b.           
            d88P  888 888 d88P"88b d88""88b 888P"   888 888    888 "88b 888 "888 "88b          
           d88P   888 888 888  888 888  888 888     888 888    888  888 888  888  888          
          d8888888888 888 Y88b 888 Y88..88P 888     888 Y88b.  888  888 888  888  888          
88888888 d88P     888 888  "Y88888  "Y88P"  888     888  "Y888 888  888 888  888  888 88888888 
                               888                                                             
                          Y8b d88P                                                             
                           "Y88P"                                                              
*/





template <const ArgsPack *AP>
__global__
void Cu_RefineCells_Prep
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref, int *cblock_ID_nbr,
    int *efficient_map
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        int ref_kap = cblock_ID_ref[kap];
        if (ref_kap == V_REF_ID_MARK_REFINE || ref_kap == V_REF_ID_MARK_COARSEN)
        {
            for (int p = 0; p < N_Q_max; p++)
                efficient_map[kap + p*n_maxcblocks] = cblock_ID_nbr[kap + p*n_maxcblocks];
        }
    }
}

// NOTE: I'm trying to mark unrefined cell-blocks violating the quality criterion. After that, I'll go back to cell-blocks marked for coarsening and check the children - if they have at least one violating the criterion, unmark them.
template <const ArgsPack *AP>
__global__
void Cu_RefineCells_Q1_1
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
    {
        bool has_refined_nbr = false;
        for (int p = 1; p < N_Q_max; p++)
        {
            if (cblock_ID_nbr_child[kap + p*n_maxcblocks] >= 0)
                has_refined_nbr = true;
        }
        
        if (has_refined_nbr)
            cblock_ID_ref[kap] = V_REF_ID_UNREFINED_VIO;
    }
}

// NOTE: The coarsening corrector. Basically, if at least ONE child is near a refined neighbor AND the current block is near a boundary, do not proceed with coarsening as this creates an interface with refinement scale of 4 rather than 2.
template <const ArgsPack *AP>
__global__
void Cu_RefineCells_Q1_2
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
    constexpr int M_BLOCK = AP->M_BLOCK;
    constexpr int N_CHILDREN = AP->N_CHILDREN;
    __shared__ int s_ID_child[M_BLOCK*N_CHILDREN];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int ref_kap = -1;
    
    // Initialize shared memory array.
    for (int xc = 0; xc < N_CHILDREN; xc++)
        s_ID_child[xc + threadIdx.x*N_CHILDREN] = -1;
    __syncthreads();
    
    // Fill shared memory array with child IDs of refined cell-blocks.
    if (kap < id_max_curr)
    {
        ref_kap = cblock_ID_ref[kap];
        
        if (ref_kap == V_REF_ID_MARK_COARSEN)
        {
            int i_c = cblock_ID_nbr_child[kap];
            
            for (int xc = 0; xc < N_CHILDREN; xc++)
                s_ID_child[xc + threadIdx.x*N_CHILDREN] = i_c + xc;
        }
    }
    __syncthreads();
    
    // Go through recorded children and replace IDs with refinement IDs.
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int i_q = s_ID_child[threadIdx.x + q*M_BLOCK];
        if (i_q >= 0)
            s_ID_child[threadIdx.x + q*M_BLOCK] = cblock_ID_ref[i_q];
    }
    __syncthreads();
    
    // Now, evaluate the quality criterion.
    if (kap < id_max_curr && ref_kap == V_REF_ID_MARK_COARSEN)
    {
        //bool near_boundary = false; [DEPRECATED]
        bool has_violating_child = false;
        //int nbr_id_p; [DEPRECATED]
        
        for (int xc = 0; xc < N_CHILDREN; xc++)
        {
            if (s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_UNREFINED_VIO || s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_MARK_REFINE)
                has_violating_child = true;
        }
        
        // [DEPRECATED]
//         for (int p = 1; p < N_Q_max; p++)
//         {
//             nbr_id_p = cblock_ID_nbr[kap + p*n_maxcblocks];
//             if (nbr_id_p < 0)
//                 near_boundary = true;
//         }
        
        // If both violating conditions are satisifed, revert to a regular refined cell.
        if (has_violating_child)
        //if (near_boundary || has_violating_child) [DEPRECATED]
            cblock_ID_ref[kap] = V_REF_ID_REFINED;
    }
}



// Formerly: Cu_RefineCells_S1.
template <typename ufloat_t, const ArgsPack *AP>
__global__
void Cu_AddRemoveBlocks
(
    int id_max_curr, int n_maxcblocks, ufloat_t dx,
    int *cblock_ID_nbr_child, int *cblock_ID_ref, int *cblock_level, ufloat_t *cblock_f_X,
    int *scattered_map
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_BLOCK = AP->M_BLOCK;
    constexpr int N_CHILDREN = AP->N_CHILDREN;
    
    // Child constructor parameters.
    unsigned int t_id_incrementor = threadIdx.x%N_CHILDREN; //threadIdx.x&(N_CHILDREN-1);
    unsigned int t_id_builder = threadIdx.x/N_CHILDREN; //threadIdx.x>>log2(N_CHILDREN);
    unsigned int t_xi_builder = threadIdx.x%2; //threadIdx.x&(1);
    unsigned int t_xj_builder = (threadIdx.x/2)%2; //(threadIdx.x>>log2(2))&(1);
    unsigned int t_xk_builder = (threadIdx.x/4)%2; //(threadIdx.x>>log2(4))&(1);
    
    __shared__ int s_ID_child[M_BLOCK];
    __shared__ int s_ref[M_BLOCK];
    __shared__ int s_level[M_BLOCK];
    __shared__ ufloat_t s_x[M_BLOCK];
    __shared__ ufloat_t s_y[M_BLOCK];
    __shared__ ufloat_t s_z[M_BLOCK];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    // Initialize shared memory.
    s_ID_child[threadIdx.x] = -1;
    s_ref[threadIdx.x] = -1;
    s_level[threadIdx.x] = -1;
    s_x[threadIdx.x] = -1;
    s_y[threadIdx.x] = -1;
if (N_DIM==3)
    s_z[threadIdx.x] = -1;

    __syncthreads();
    
    if (kap < id_max_curr)
    {
        // If this particular ID on the map is a valid ID, we need to transcribe the spatial location and level corresponding to the parent.
        // I've reset s_ID_child to -1 to know which values should be skipped.
        //if (scattered_map[kap] > -1)
        if (cblock_ID_ref[kap] == V_REF_ID_MARK_REFINE)
        {
            s_ID_child[threadIdx.x] = scattered_map[kap];
            if (scattered_map[kap] < 0)
                printf("Uh oh...(S1, scatter)\n");
            s_ref[threadIdx.x] = V_REF_ID_NEW;
            s_level[threadIdx.x] = cblock_level[kap];
            s_x[threadIdx.x] = cblock_f_X[kap + 0*n_maxcblocks];
            s_y[threadIdx.x] = cblock_f_X[kap + 1*n_maxcblocks];
if (N_DIM==3)
            s_z[threadIdx.x] = cblock_f_X[kap + 2*n_maxcblocks];

            
            cblock_ID_nbr_child[kap] = s_ID_child[threadIdx.x];
            //for (int xc = 0; xc < N_CHILDREN; xc++)
            //    cblock_ID_child[kap + xc*n_maxcblocks] = s_ID_child[threadIdx.x]+xc;
        }
        if (cblock_ID_ref[kap] == V_REF_ID_MARK_COARSEN)
        {
            s_ID_child[threadIdx.x] = cblock_ID_nbr_child[kap];
            s_ref[threadIdx.x] = V_REF_ID_REMOVE;
            
            if (s_ID_child[threadIdx.x] >= id_max_curr)
                printf("Uh oh...(S1, ID max. violated)\n");
            
            cblock_ID_nbr_child[kap] = N_SKIPID;
            //for (int xc = 0; xc < N_CHILDREN; xc++)
            //    cblock_ID_child[kap + xc*n_maxcblocks] = N_SKIPID;
        }
    }
    __syncthreads();
    
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int ID_child_q = s_ID_child[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        int ref_q = s_ref[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ];
        
        // Only write if there actually is a child.
        //if (ID_child_q > -1)
        if (ref_q == V_REF_ID_NEW)
        {
            if (ID_child_q < 0)
                printf("Uh oh...(S1, new)\n");
            int level_q = s_level[ t_id_builder + q*(M_BLOCK/N_CHILDREN) ] + 1;
            cblock_level[ID_child_q + t_id_incrementor] = level_q;
            cblock_ID_ref[ID_child_q + t_id_incrementor] = V_REF_ID_NEW;
            cblock_f_X[ID_child_q + t_id_incrementor + 0*n_maxcblocks] = s_x[t_id_builder + q*(M_BLOCK/N_CHILDREN)] + t_xi_builder*(dx/(1<<level_q));
            cblock_f_X[ID_child_q + t_id_incrementor + 1*n_maxcblocks] = s_y[t_id_builder + q*(M_BLOCK/N_CHILDREN)] + t_xj_builder*(dx/(1<<level_q));
if (N_DIM==3)
{
            cblock_f_X[ID_child_q + t_id_incrementor + 2*n_maxcblocks] = s_z[t_id_builder + q*(M_BLOCK/N_CHILDREN)] + t_xk_builder*(dx/(1<<level_q));
}
            cblock_ID_nbr_child[ID_child_q + t_id_incrementor] = N_SKIPID;
        }
        if (ref_q == V_REF_ID_REMOVE)
        {
            if (ID_child_q < 0)
                printf("Uh oh...(S1, remove)\n");
            cblock_ID_ref[ID_child_q + t_id_incrementor] = V_REF_ID_REMOVE;
        }
    }
}

// Here, update the child_nbr IDs according to refinement only.
template <const ArgsPack *AP>
__global__
void Cu_RefineCells_S2_1
(
    int id_max_curr, int n_maxcblocks, 
    int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child,
    int *efficient_map, int *scattered_map
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_BLOCK = AP->M_BLOCK;
    __shared__ int s_ID_nbr[M_BLOCK*9];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
    
// #if (N_DIM==3)
    for (int k = 0; k < (N_DIM==2?1:3); k++)
    {
// #else
//     int k = 0;
// #endif
    
    // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
    if (kap < id_max_curr && efficient_map[kap] >= 0)
    {
        for (int p = 0; p < 9; p++)
            s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
    }
    __syncthreads();
    
    // Replace neighbor Ids with their respective marks.
    for (int p = 0; p < 9; p++)
    {
        int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
        if (i_p > -1)
        {
            s_ID_nbr[threadIdx.x + p*M_BLOCK] = cblock_ID_ref[i_p];
            if (s_ID_nbr[threadIdx.x + p*M_BLOCK] == V_REF_ID_MARK_REFINE)
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = scattered_map[i_p];
            else
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
        }
    }
    __syncthreads();
    
    // Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
    if (kap < id_max_curr && efficient_map[kap] >= 0)
    {
        for (int p = 0; p < 9; p++)
        {
            if (s_ID_nbr[p + threadIdx.x*9] > -1)
                cblock_ID_nbr_child[kap + (k*9+p)*n_maxcblocks] = s_ID_nbr[p + threadIdx.x*9];
        }
    }
    
// #if (N_DIM==3)
if (N_DIM==3)
{
    __syncthreads();
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
}
    }
// #endif
}

// Update due to coarsening.
template <const ArgsPack *AP>
__global__
void Cu_RefineCells_S2_2
(
    int id_max_curr, int n_maxcblocks, 
    int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child,
    int *efficient_map
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_BLOCK = AP->M_BLOCK;
    __shared__ int s_ID_nbr[M_BLOCK*9];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
    
// #if (N_DIM==3)
    for (int k = 0; k < (N_DIM==2?1:3); k++)
    {
// #else
//     int k = 0;
// #endif
    
    // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
    if (kap < id_max_curr && efficient_map[kap] >= 0)
    {
        for (int p = 0; p < 9; p++)
            s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
    }
    __syncthreads();
    
    // Replace neighbor Ids with their respective marks.
    for (int p = 0; p < 9; p++)
    {
        int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
        if (i_p > -1)
            s_ID_nbr[threadIdx.x + p*M_BLOCK] = cblock_ID_ref[i_p];
    }
    __syncthreads();
    
    // Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
    if (kap < id_max_curr && efficient_map[kap] >= 0)
    {
        for (int p = 0; p < 9; p++)
        {
            if ((k*9+p) > 0 && s_ID_nbr[p + threadIdx.x*9] == V_REF_ID_MARK_COARSEN)
                cblock_ID_nbr_child[kap + (k*9+p)*n_maxcblocks] = N_SKIPID;
        }
    }
    
// #if (N_DIM==3)
if (N_DIM==3)
{
    __syncthreads();
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
}
    }
// #endif
}

__global__
void Cu_RefineCells_S3
(
    int id_max_curr_wnew,
    int *cblock_ID_ref
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr_wnew)
    {
        int ref_kap = cblock_ID_ref[kap];
        if (ref_kap == V_REF_ID_MARK_REFINE)
            cblock_ID_ref[kap] = V_REF_ID_REFINED;
        if (ref_kap == V_REF_ID_NEW || ref_kap == V_REF_ID_MARK_COARSEN)
            cblock_ID_ref[kap] = V_REF_ID_UNREFINED;
        if (ref_kap == V_REF_ID_REMOVE)
            cblock_ID_ref[kap] = V_REF_ID_INACTIVE;
        
        // If at this point this wasn't overwritten with 'V_REF_ID_REMOVE', it can be safely reverted to 'V_REF_ID_UNREFINED' instead.
        if (ref_kap == V_REF_ID_UNREFINED_VIO)
            cblock_ID_ref[kap] = V_REF_ID_UNREFINED;
    }
}

__global__
void Cu_RefineCells_Cancel
(
    int id_max_curr_wnew,
    int *cblock_ID_ref
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr_wnew)
    {
        int ref_kap = cblock_ID_ref[kap];
        if (ref_kap == V_REF_ID_MARK_REFINE)
            cblock_ID_ref[kap] = V_REF_ID_UNREFINED;
    }
}

template <const ArgsPack *AP>
__global__
void Cu_RefineCells_S4
(
    int id_max_curr_wnew,
    int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
    constexpr int M_BLOCK = AP->M_BLOCK;
    constexpr int N_CHILDREN = AP->N_CHILDREN;
    //__shared__ int s_ID_ref_parent[M_BLOCK];
    __shared__ int s_ID_child[M_BLOCK*N_CHILDREN];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int ref_kap = -1;
    int new_ref_id = -1;
    
    // Initialize shared memory array.
    //s_ID_ref_parent[M_BLOCK] = V_REF_ID_REFINED;
    for (int xc = 0; xc < N_CHILDREN; xc++)
        s_ID_child[xc + threadIdx.x*N_CHILDREN] = -1;
    __syncthreads();
    
    // Fill shared memory array with child IDs of refined cell-blocks.
    if (kap < id_max_curr_wnew)
    {
        ref_kap = cblock_ID_ref[kap];
        
        // If a cell-block is refined, loop over its children and check whether any of them have been refined too. If so, this cell-block is a branch.
        // A refined cell-block always has children with ID >= 0, no need for conditional.
        if (ref_kap == V_REF_ID_REFINED || ref_kap == V_REF_ID_REFINED_WCHILD)
        {
            int i_c = cblock_ID_nbr_child[kap];
            new_ref_id = V_REF_ID_REFINED;
            
            for (int xc = 0; xc < N_CHILDREN; xc++)
                s_ID_child[xc + threadIdx.x*N_CHILDREN] = i_c + xc;
        }
    }
    __syncthreads();
    
    // Loop over shared memory array and retrieve refinement IDs. Replace the child IDs in shared memory.
    // Doing it this way guarantees a degree of coalescence since children are all in a row in memory.
    for (int q = 0; q < N_CHILDREN; q++)
    {
        int i_q = s_ID_child[threadIdx.x + q*M_BLOCK];
        if (i_q >= 0)
        {
            s_ID_child[threadIdx.x + q*M_BLOCK] = cblock_ID_ref[i_q];
            
            //int ref_q = cblock_ID_ref[i_q];
            //if (ref_q == V_REF_ID_REFINED)
            //    s_ID_ref_parent[(threadIdx.x + q*M_BLOCK)/N_CHILDREN] = V_REF_ID_REFINED_WCHILD;
        }
    }
    __syncthreads();
    
    // Loop over children at parent level and check refinement IDs. If at least one child is refined, mark the parent as a branch instead of leaf.
    if (kap < id_max_curr_wnew)
    {
        for (int xc = 0; xc < N_CHILDREN; xc++)
        {
            if (s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_REFINED || s_ID_child[xc + threadIdx.x*N_CHILDREN] == V_REF_ID_REFINED_WCHILD)
                new_ref_id = V_REF_ID_REFINED_WCHILD;
        }
        
        if (new_ref_id > -1)
            cblock_ID_ref[kap] = new_ref_id;
    }
}

__global__
void Cu_CoarsenCells_S1
(
    int n_ids_idev_L, int n_ids_marked_removal, int *ids_marked,
    int *id_set_idev_L
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < n_ids_idev_L)
    {
        // Loop over IDs.
        for (int k = 0; k < n_ids_marked_removal; k++)
        {
            if (id_set_idev_L[kap] == ids_marked[k])
            {
                id_set_idev_L[kap] = N_SKIPID;
            }
        }
    }
}

template <const ArgsPack *AP>
__global__
void Cu_CoarsenCells_S2
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        if (cblock_ID_ref[kap] == V_REF_ID_REMOVE)
        {
            for (int p = 0; p < N_Q_max; p++)
            {
                if (p > 0)
                    cblock_ID_nbr[kap + p*n_maxcblocks] = N_SKIPID;
                cblock_ID_nbr_child[kap + p*n_maxcblocks] = N_SKIPID;
            }
        }
    }
}

__global__
void Cu_FreezeRefined
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        if (cblock_ID_ref[kap] == V_REF_ID_REFINED)
            cblock_ID_ref[kap] = V_REF_ID_REFINED_PERM;
        if (cblock_ID_ref[kap] == V_REF_ID_REFINED_WCHILD)
            cblock_ID_ref[kap] = V_REF_ID_REFINED_WCHILD_PERM;
    }
}

__global__
void Cu_UnfreezeRefined
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        if (cblock_ID_ref[kap] == V_REF_ID_REFINED_PERM)
            cblock_ID_ref[kap] = V_REF_ID_REFINED;
        if (cblock_ID_ref[kap] == V_REF_ID_REFINED_WCHILD_PERM)
            cblock_ID_ref[kap] = V_REF_ID_REFINED_WCHILD;
    }
}

__global__
void Cu_RefinementValidator_1
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        int nbr_child_0 = cblock_ID_nbr_child[kap];
        if (cblock_ID_ref[kap] == V_REF_ID_REFINED && nbr_child_0 < 0)
        {
            printf("Uh oh (refined block violated condition)...\n");
        }
        if (cblock_ID_ref[kap] == V_REF_ID_UNREFINED && nbr_child_0 > -1)
        {
            printf("Uh oh (unrefined block violated condition)...\n");
        }
    }
}

template <const ArgsPack *AP>
__global__
void Cu_RefinementValidator_2
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_ref, int *cblock_level, int *cblock_ID_nbr, int *cblock_ID_nbr_child
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        for (int p = 0; p < N_Q_max; p++)
        {
            int nbr_id_p = cblock_ID_nbr[kap + p*n_maxcblocks];
            if (nbr_id_p > -1)
            {
                int ref_kap_nbr_p = cblock_ID_ref[nbr_id_p];
                if ( (ref_kap_nbr_p == V_REF_ID_REFINED || ref_kap_nbr_p == V_REF_ID_REFINED_WCHILD || ref_kap_nbr_p == V_REF_ID_REFINED_PERM || ref_kap_nbr_p == V_REF_ID_REFINED_WCHILD_PERM || ref_kap_nbr_p == V_REF_ID_MARK_REFINE) && cblock_ID_nbr_child[kap + p*n_maxcblocks] == N_SKIPID)
                    printf("Uh oh (bad connectivity data for block %i on level %i, direction %i)...\n", kap, cblock_level[kap], p);
            }
        }
    }
}

template <const ArgsPack *AP>
__global__
void Cu_RefinementValidator_3
(
    int id_max_curr, int n_maxcblocks,
    int *cells_ID_mask, int *cblock_ID_mask, int *cblock_ID_ref, int *cblock_ID_nbr_child
)
{
    constexpr int N_QUADS = AP->N_QUADS;
    constexpr int N_CHILDREN = AP->N_CHILDREN;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    // Keep in mind that each ID represents a block, not just a cell.
    s_ID_cblock[threadIdx.x] = -1;
    if (kap < id_max_curr && cblock_ID_ref[kap] != V_REF_ID_INACTIVE)
        s_ID_cblock[threadIdx.x] = kap;
    __syncthreads();

    for (int k = 0; k < M_TBLOCK; k++)
    {
        int i_kap_b = s_ID_cblock[k];
        
        if (i_kap_b > -1)
        {
            int mask_block_ID = cblock_ID_mask[i_kap_b];
            int child_zero_ID = cblock_ID_nbr_child[i_kap_b];
            
            if (child_zero_ID > -1)
            {
                for (int xc = 0; xc < N_CHILDREN; xc++)
                {
                    for (int Q = 0; Q < N_QUADS; Q++)
                    {
                        int mask_cell_ID = cells_ID_mask[(child_zero_ID+xc)*M_TBLOCK + Q*M_TBLOCK + threadIdx.x];
                        
                        if (mask_cell_ID > 0 && mask_block_ID != 1)
                            printf("Uh oh...block has a ghost layer in children but is not marked with right mask...\n");
                    }
                }
            }
        }
    }
}

__global__
void Cu_RefinementValidator_4
(
    int id_max_curr, int n_maxcblocks,
    int *cblock_ID_nbr_child
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < n_maxcblocks)
    {
        int i_c = cblock_ID_nbr_child[kap];
        
        if (i_c >= id_max_curr)
            printf("Hmmm...ID max [%i]. violated at prep. stage (%i, i_c=%i)\n", id_max_curr, kap, i_c);
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_FreezeRefinedCells(int var)
{
    for (int i_dev = 0; i_dev < N_DEV; i_dev++)
    {
        // Freeze.
        if (var == 0)
        {
            Cu_FreezeRefined<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max[i_dev][MAX_LEVELS], n_maxcblocks,
                c_cblock_ID_ref[i_dev]
            );
        }
        
        // Unfreeze.
        if (var == 1)
        {
            Cu_UnfreezeRefined<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max[i_dev][MAX_LEVELS], n_maxcblocks,
                c_cblock_ID_ref[i_dev]
            );
        }
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_RefineAndCoarsenBlocks(int var)
{
    for (int i_dev = 0; i_dev < N_DEV; i_dev++)
    {
#if (P_SHOW_REFINE==1)
        double tot_time = 0.0;
        double tmp_time = 0.0;
        tic_simple("[Pre]");
#endif
        // - c_tmp_1 is used to store the indices of blocks marked for refinement and those for marked for coarsening afterwards.
        // - c_tmp_2 is used to store the map.
        // - c_tmp_3 is used to store spliced gaps in reverse order.
        // - c_tmp_4 is used to store levels of the blocks marked for refinement.
        // - c_tmp_5 is used to store a copy of modified ID of the ID set before they are filtered of blocks marked for removal.
        // - Using n_ids_marked_refine, the gap set is spliced at the end and reversed in order before scattering and using in later routines.
        
        // First, collect an array of cblocks marked for refinement along with their count via thrust routines 'copy_if' and 'count_if', respectively.
        int id_max_curr = id_max[i_dev][MAX_LEVELS];
        int n_ids_marked_refine = thrust::count_if(
            thrust::device, c_cblock_ID_ref_dptr[i_dev], c_cblock_ID_ref_dptr[i_dev] + id_max_curr, is_marked_for_refinement()
        );
        int n_ids_marked_coarsen = thrust::count_if(
            thrust::device, c_cblock_ID_ref_dptr[i_dev], c_cblock_ID_ref_dptr[i_dev] + id_max_curr, is_marked_for_coarsening()
        );
        int id_max_curr_wnew = id_max_curr + N_CHILDREN*n_ids_marked_refine;
        
        // First, collect an array of cblocks marked for refinement along with their count via thrust routines 'copy_if' and 'count_if', respectively.
        // Make sure that this step is skipped if there are not enough gaps left in the gap set.
        bool proceed_refinement = n_ids_marked_refine > 0 && N_CHILDREN*n_ids_marked_refine <= n_gaps[i_dev];
        bool proceed_coarsening = n_ids_marked_coarsen > 0;
        
        // TODO
        std::cout << "Numbers (no. ref.): " << n_ids_marked_refine << std::endl;
        std::cout << "Numbers (no. coarse[1].): " << n_ids_marked_coarsen << std::endl;
        
        if (!proceed_refinement && n_ids_marked_refine > 0)
        {
            std::cout << "Canceling, not enough space to refine further (n_gaps=" << n_gaps[i_dev] << ", requested=" << N_CHILDREN*n_ids_marked_refine << ")..." << std::endl;
            n_ids_marked_refine = 0;
            id_max_curr_wnew = id_max_curr;
            
            Cu_RefineCells_Cancel<<<(M_BLOCK+id_max_curr_wnew-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr_wnew, c_cblock_ID_ref[i_dev]
            );
        }
        
        // Reset temporary arrays if we are proceeding with refinement/coarsening.
        if (proceed_refinement || proceed_coarsening)
        {
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
                n_maxcblocks, c_tmp_1[i_dev], -1
            );
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
                n_maxcblocks, c_tmp_2[i_dev], -1
            );
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
                n_maxcblocks, c_tmp_3[i_dev], -1
            );
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
                n_maxcblocks, c_tmp_4[i_dev], -1
            );
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
                n_maxcblocks, c_tmp_5[i_dev], -1
            );
            Cu_ResetToValue<<<(M_BLOCK+(n_maxcblocks*N_Q_max)-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                n_maxcblocks*N_Q_max, c_tmp_6[i_dev], -1
            );
            Cu_ResetToValue<<<(M_BLOCK+(n_maxcblocks*N_Q_max)-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                n_maxcblocks*N_Q_max, c_tmp_7[i_dev], -1
            );
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
                n_maxcblocks, c_tmp_8[i_dev], -1
            );
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[Pre]",T_US);
        tot_time += tmp_time;
        ref_printer << var << " " << n_ids_marked_refine << " " << n_ids_marked_coarsen << " " << tmp_time << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S1]");
#endif        
        // Now we need to set up the scatter map for efficiently establishing connectivity. The plan is to 1) loop over cblocks and copy their neighbors down if marked for refinement/coarsening, 2) sort them and perform a unique copy in case of repitions (cblocks near each other may call on the same neighbors), 3) scatter them and use the scattered map to update connectivity only of cblocks in the vincinity of marked cblocks.
        if (proceed_refinement || proceed_coarsening)
        {
            // Loop over cblocks and copy down the neigbors IDs of marked cblocks in efficient_map. The efficient_map must be reset before this step.
            Cu_RefineCells_Prep<AP> <<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr, n_maxcblocks,
                c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev],
                c_tmp_6[i_dev]
            );

            // Count the number of recorded non-negative IDs.
            int n_nonnegative_prev = thrust::count_if(
                thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_maxcblocks*N_Q_max, is_nonnegative()
            );
            
            // Regular-copy non-negative IDs in efficient_map.
            thrust::copy_if(
                thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_maxcblocks*N_Q_max, c_tmp_7_dptr[i_dev], is_nonnegative()
            );

            // Sort the copied IDs to prepare for the unique-copy.
            thrust::sort(
                thrust::device, c_tmp_7_dptr[i_dev], c_tmp_7_dptr[i_dev] + n_nonnegative_prev
            );
            
            // Reset in preparation for unique copy.
            Cu_ResetToValue<<<(M_BLOCK+(n_maxcblocks*N_Q_max)-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                n_maxcblocks*N_Q_max, c_tmp_6[i_dev], -1
            );
            
            // Perform the unique-copy. At this stage, c_tmp_6 shouldn't exceed n_maxcblocks in value so it is safe to scatter to a new c_tmp.
            thrust::unique_copy(
                thrust::device, c_tmp_7_dptr[i_dev], c_tmp_7_dptr[i_dev] + n_nonnegative_prev, c_tmp_6_dptr[i_dev]
            );
            
            // Re-count the number of recorded non-negative IDs.
            n_nonnegative_prev = thrust::count_if(
                thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_nonnegative_prev, is_nonnegative()
            );
            
            // Scatter the IDs.
            thrust::scatter(
                thrust::device, c_tmp_6_dptr[i_dev], c_tmp_6_dptr[i_dev] + n_nonnegative_prev, c_tmp_6_dptr[i_dev], c_tmp_8_dptr[i_dev]
            );
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S1]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S2]");
#endif
        // Copy marked block indices into temporary array so that they're contiguous in memory, capture gaps required for proceeding steps.
        if (proceed_refinement)
        {
            // Copy indices of blocks marked for refinement.
            thrust::copy_if(
                //thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(id_max_curr), c_cblock_ID_ref_dptr[i_dev], c_tmp_1_dptr[i_dev], is_marked_for_refinement()
                thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr, c_cblock_ID_ref_dptr[i_dev], c_tmp_1_dptr[i_dev], is_marked_for_refinement()
            );
            
            // Splice the gap set starting at (n_gaps-1) - n_ids_marked_refine, reverse its order with thrust.
            thrust::reverse_copy(
                thrust::device,  c_gap_set_dptr[i_dev] + (n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine), c_gap_set_dptr[i_dev] + (n_gaps[i_dev]), c_tmp_3_dptr[i_dev]
                //thrust::device,  &c_gap_set_dptr[i_dev][n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine], &c_gap_set_dptr[i_dev][n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine] + N_CHILDREN*n_ids_marked_refine, c_tmp_3_dptr[i_dev]
            );
            
            // Update the ID max. since new children IDs might exceed current one after previous shuffling of gaps.
            id_max_curr_wnew = std::max(id_max_curr_wnew, *thrust::max_element(thrust::device, c_tmp_3_dptr[i_dev], c_tmp_3_dptr[i_dev] + N_CHILDREN*n_ids_marked_refine) + 1);
            
            // Contract the child indices (using reversed data in temporary array, append to the end of c_tmp_3.
            Cu_ContractByFrac<int,AP> <<<(M_BLOCK+N_CHILDREN*n_ids_marked_refine-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(
                //N_CHILDREN*n_ids_marked_refine, c_tmp_3[i_dev], N_CHILDREN, &c_gap_set[i_dev][n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine]
                
                N_CHILDREN*n_ids_marked_refine, c_tmp_3[i_dev], N_CHILDREN, &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine]
            );
            
            // Scatter gaps to indices of marked cells for the write process, then retrieve newly-generated blocks by level for insertion in ID sets.
            thrust::scatter(
                thrust::device, &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine], &c_tmp_3[i_dev][N_CHILDREN*n_ids_marked_refine] + n_ids_marked_refine, c_tmp_1_dptr[i_dev], c_tmp_2_dptr[i_dev]
            );
        
            // Call S2 routine where new child IDs are inserted in the cblock_ID_nbr_child array.
            Cu_RefineCells_S2_1<AP> <<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr, n_maxcblocks,
                c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
                c_tmp_8[i_dev], c_tmp_2[i_dev]
            );
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S2]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S3]");
#endif
        // Now correct the marks for coarsening.
        if (proceed_coarsening)
        {
            // Call the first part of quality control on coarsening.
            Cu_RefineCells_Q1_1<AP> <<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr, n_maxcblocks,
                c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
            );
            gpuErrchk( cudaPeekAtLastError() );
            
            // Call the second part of quality control on coarsening.
            Cu_RefineCells_Q1_2<AP> <<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr, n_maxcblocks,
                c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
            );
            gpuErrchk( cudaPeekAtLastError() );
            
            // Re-evaluate whether to coarsen or not.
            n_ids_marked_coarsen = thrust::count_if(
            thrust::device, c_cblock_ID_ref_dptr[i_dev], c_cblock_ID_ref_dptr[i_dev] + id_max_curr, is_marked_for_coarsening()
            );
            proceed_coarsening = n_ids_marked_coarsen > 0;
            
            // TODO
            std::cout << "Numbers (no. coarse[2].): " << n_ids_marked_coarsen << std::endl;
            
            // If we are still proceeding with coarsening...
            if (proceed_coarsening)
            {
                // Retrieve indices of blocks marked for coarsening, append to c_tmp_1. Needed for Cu_RefineCells_S2.
                thrust::copy_if(
                    thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr, c_cblock_ID_ref_dptr[i_dev],     &c_tmp_1_dptr[i_dev][n_ids_marked_refine], is_marked_for_coarsening()
                );
                
                // Call S2 routine where new child IDs are inserted in the cblock_ID_nbr_child array.
                Cu_RefineCells_S2_2<AP> <<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                    id_max_curr, n_maxcblocks,
                    c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
                    c_tmp_8[i_dev]
                );
                gpuErrchk( cudaPeekAtLastError() );
            }
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S3]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " " << n_ids_marked_coarsen << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S4]");
#endif
        // Scatter gaps to indices of marked cells for the write process, then retrieve newly-generated blocks by level for insertion in ID sets.
        if (proceed_refinement || proceed_coarsening)
        {
            Cu_AddRemoveBlocks<ufloat_t,AP> <<<(M_BLOCK+id_max_curr-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr, n_maxcblocks, dx_cblock,
                c_cblock_ID_nbr_child[i_dev], c_cblock_ID_ref[i_dev], c_cblock_level[i_dev], c_cblock_f_X[i_dev],
                c_tmp_2[i_dev]
            );
            gpuErrchk( cudaPeekAtLastError() );
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S4]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S5]");
#endif
        // Retrieve blocks marked for removal. Run through ID sets and filter out marked blocks. Placed after Cu_AddRemoveBlocks so that children of  blocks marked for coarsening are now marked for removal.
        if (proceed_coarsening)
        {    
            // Retrieve the blocks to be removed, store in c_tmp_5.
            thrust::copy_if(
                thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_5_dptr[i_dev], is_marked_for_removal()
            );
            
            // Copy the oldly-set levels of children-to-be-removed. This allows us to remove them from the correct ID set array.
            thrust::copy_if(
                thrust::device, c_cblock_level_dptr[i_dev], c_cblock_level_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_4_dptr[i_dev], is_marked_for_removal()
            );
            
            // Sort the levels (keys, stored in c_tmp_4) and the associated IDs for removal (values, stored in c_tmp_5). Reduce by key to get the number of gaps to insert into the ID set.
            thrust::stable_sort_by_key(
                thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_coarsen, c_tmp_5_dptr[i_dev]
            );
                    
            int n_rem_levels_L_prev = 0;
            int n_rem_levels_L = 0;
            for (int L = 1; L < MAX_LEVELS; L++)
            {
                // Count the number of IDs to be removed on level L.
                n_rem_levels_L = thrust::count_if(
                    thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_coarsen, is_equal_to(L)
                );
                
                if (n_rem_levels_L > 0)
                {
                    // Copy id set to temporary array and change any IDs marked for removal to N_SKIPID.
                    Cu_CoarsenCells_S1<<<(M_BLOCK+n_ids[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                        n_ids[i_dev][L], n_rem_levels_L, &c_tmp_5[i_dev][n_rem_levels_L_prev],
                        &c_id_set[i_dev][L*n_maxcblocks]
                    );
                    gpuErrchk( cudaPeekAtLastError() );
                    
                    // Use remove_if to store only those IDs not marked for removal.
                    thrust::remove_if(
                        thrust::device, &c_id_set_dptr[i_dev][L*n_maxcblocks], &c_id_set_dptr[i_dev][L*n_maxcblocks] + n_ids[i_dev][L], is_removed()
                    );
                    
                    n_ids[i_dev][L] -= n_rem_levels_L;
                    n_rem_levels_L_prev += n_rem_levels_L;
                    std::cout << "Removed " << n_rem_levels_L << " cblocks on Level " << L << "..." << std::endl;
                }
            }
            
            // Insert removed IDs into the gap set. These are the ones stored in c_tmp_5.
            // Concatenation happens at n_gaps - Nc*n_marked_refine since Nc*n_marked_refine gaps will be cut off at the end of the routine. Gaps to be used for refinement have already been processed, so this shouldn't be problematic.
            Cu_ConcatReverse<<<(M_BLOCK+n_rem_levels_L_prev-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                n_gaps[i_dev] - N_CHILDREN*n_ids_marked_refine, c_gap_set[i_dev], n_rem_levels_L_prev, c_tmp_5[i_dev]
            );
            //std::cout << "Ngaps updated from " << n_gaps[i_dev] << " to ";
            n_gaps[i_dev] += n_rem_levels_L_prev;
            //std::cout << n_gaps[i_dev] << " after removals." << std::endl;
            
            gpuErrchk( cudaPeekAtLastError() );
            Cu_CoarsenCells_S2<AP> <<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr, n_maxcblocks, c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
            );
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S5]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S6]");
#endif
        if (proceed_refinement)
        {
            // Copy the newly-set levels of gaps spliced from the gap set. This allows us to insert them into the correct ID set array.
            thrust::copy_if(
                thrust::device, c_cblock_level_dptr[i_dev], c_cblock_level_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_4_dptr[i_dev], is_newly_generated()
            );
            
            thrust::copy_if(
                thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max_curr_wnew, c_cblock_ID_ref_dptr[i_dev], c_tmp_3_dptr[i_dev], is_newly_generated()
            );
            
            // Sort the levels (keys, stored in c_tmp_4) and the associated gaps (values, stored in c_tmp_3). Reduce by key to get the number of gaps to insert into the ID set.
            thrust::stable_sort_by_key(
                thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_refine, c_tmp_3_dptr[i_dev]
            );
            
            // Loop over the reduced level-keys and 1) retrieve total number to be added, 2) concatenate ID sets using total number per level, 3) update the n_ids array host-side.
            int n_new_levels_L_prev = 0;
            int n_new_levels_L = 0;
            for (int L = 1; L < MAX_LEVELS; L++)
            {
                n_new_levels_L = thrust::count_if(
                    thrust::device, c_tmp_4_dptr[i_dev], c_tmp_4_dptr[i_dev] + N_CHILDREN*n_ids_marked_refine, is_equal_to(L)
                );
                
                if (n_new_levels_L > 0)
                {
                    Cu_Concat<<<(M_BLOCK+n_new_levels_L-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                        n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_new_levels_L, &c_tmp_3[i_dev][n_new_levels_L_prev]
                    );
                    gpuErrchk( cudaPeekAtLastError() );
                    n_ids[i_dev][L] += n_new_levels_L;
                    
                    thrust::sort(
                        thrust::device, &c_id_set_dptr[i_dev][L*n_maxcblocks], &c_id_set_dptr[i_dev][L*n_maxcblocks] + n_ids[i_dev][L]
                    );
                    
                    std::cout << "Inserted " << n_new_levels_L << " cblocks on Level " << L << "..." << std::endl;
                }
                n_new_levels_L_prev += n_new_levels_L;
            }
            //std::cout << "Ngaps updated from " << n_gaps[i_dev] << " to ";
            n_gaps[i_dev] -= N_CHILDREN*n_ids_marked_refine;
            //std::cout << n_gaps[i_dev] << " after insertions." << std::endl;
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S6]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S7]");
#endif
        // Update the ID max. Loop over refinement indices and remove marks. Update branch / leaf IDs.
        if (proceed_refinement || proceed_coarsening)
        {
            // Update the ID max.
            for (int L = 1; L < MAX_LEVELS; L++)
                id_max[i_dev][L] = *thrust::max_element(thrust::device, &c_id_set_dptr[i_dev][L*n_maxcblocks], &c_id_set_dptr[i_dev][L*n_maxcblocks] + n_ids[i_dev][L]) + 1;
            id_max[i_dev][MAX_LEVELS] = *std::max_element(id_max[i_dev], id_max[i_dev] + MAX_LEVELS);
            
            // Update cell-block masks.
            Cu_UpdateMasks_1<AP> <<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max[i_dev][MAX_LEVELS], n_maxcblocks,
                c_cblock_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
            );
            gpuErrchk( cudaPeekAtLastError() );
            
            // Interpolate data to newly-added cell-blocks.
            for (int L = 0; L < MAX_LEVELS-1; L++)
                M_Interpolate(i_dev, L, V_INTERP_ADDED);
            
            // Call S3 routine where refinement IDs are reset for marked and newly-generated blocks.
            Cu_RefineCells_S3<<<(M_BLOCK+id_max_curr_wnew-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr_wnew, c_cblock_ID_ref[i_dev]
            );
            
            // Call S4 routine where branch and leaf cell-block identifications are updated.
            Cu_RefineCells_S4<AP> <<<(M_BLOCK+id_max_curr_wnew-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max_curr_wnew, c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
            );
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S7]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " ";
#endif
        
        
        
#if (P_SHOW_REFINE==1)
        tic_simple("[S8]");
#endif
        // Update connectivity.
        if (proceed_refinement || proceed_coarsening)
        {    
            // Update cell-block connectivity.
            Cu_UpdateConnectivity<AP> <<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max[i_dev][MAX_LEVELS], n_maxcblocks,
                c_cblock_ID_ref[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev],
                c_tmp_8[i_dev]
            );
            gpuErrchk( cudaPeekAtLastError() );
            
            // Update boundary information.
            Cu_UpdateBoundaries<AP> <<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max[i_dev][MAX_LEVELS], n_maxcblocks,
                c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_onb[i_dev]
            );
            gpuErrchk( cudaPeekAtLastError() );
            
            // Update cell masks.
            Cu_UpdateMasks_2<AP> <<<(M_TBLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_TBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
                id_max[i_dev][MAX_LEVELS], n_maxcblocks,
                c_cblock_ID_nbr[i_dev], c_cblock_ID_ref[i_dev], c_cells_ID_mask[i_dev]
            );
            gpuErrchk( cudaPeekAtLastError() );
        }
#if (P_SHOW_REFINE==1)
        cudaDeviceSynchronize();
        tmp_time = toc_simple("[S8]",T_US);
        tot_time += tmp_time;
        ref_printer << tmp_time << " " << tot_time << "\n";
#endif
        
        
        
        //Cu_RefinementValidator_1<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
        //    n_maxcblocks, n_maxcblocks,
        //    c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
        //);
         //Cu_RefinementValidator_2<AP> <<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
         //    n_maxcblocks, n_maxcblocks,
         //    c_cblock_ID_ref[i_dev], c_cblock_level[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev]
         //);
        //Cu_RefinementValidator_3<AP> <<<(M_TBLOCK+n_maxcblocks-1)/M_TBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
         //    n_maxcblocks, n_maxcblocks,
        //    c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_ref[i_dev], c_cblock_ID_nbr_child[i_dev]
         //);
        
        // TEST
        //cudaMemcpy(tmp_1[i_dev], &c_cblock_ID_nbr_child[i_dev][1*n_maxcblocks], id_max_curr_wnew*sizeof(int), cudaMemcpyDeviceToHost);
        //std::cout << "DEBUG\n";
        //for (int k = 0; k < id_max_curr_wnew; k++)
        //    std::cout << tmp_1[i_dev][k] << " ";
        //std::cout << std::endl;
    }

    return 0;
}
