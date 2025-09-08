/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_CheckMasks ]=================================================  */
/*                                                                                    */
/*  This kernel identifies the boundary cells (contained within the interior of the   */
/*  domain) that are adjacent to solid cells. The cell masks of each block are        */
/*  placed in shared memory along with a one-cell surrounding halo. Since cells       */
/*  will consider all possible directions, placement in shared memory prevents        */
/*  searching through the same data over and over again from global memory.           */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_CheckMasks
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    int *__restrict__ cells_ID_mask,
    const int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_nbr_child,
    int *__restrict__ cblock_ID_onb,
    int *tmp_1
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int M_HBLOCK = AP->M_HBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_ID_nbr[27];
    __shared__ int s_ID_mask[M_HBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    int i_kap_b = -1;
    int I = threadIdx.x % 4;
    int J = (threadIdx.x / 4) % 4;
    int K = 0;
    if (N_DIM==3)
        K = (threadIdx.x / 4) / 4;
    bool near_a_solid_cell = false;
    
    s_ID_cblock[threadIdx.x] = -1;
    if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
    {
        s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
    }
    for (int k = 0; k < M_HBLOCK/M_TBLOCK+1; k++)
    {
        if (k*M_TBLOCK + threadIdx.x < M_HBLOCK)
            s_ID_mask[k*M_TBLOCK + threadIdx.x] = N_SKIPID;
    }
    __syncthreads();
    
    // Loop over block Ids.
    for (int k = 0; k < M_LBLOCK; k += 1)
    {
        i_kap_b = s_ID_cblock[k];

        // Latter condition is added only if n>0.
        if (i_kap_b > -1 && cblock_ID_nbr_child[i_kap_b] < 0 && (cblock_ID_mask[i_kap_b]<0 && cblock_ID_mask[i_kap_b] != V_BLOCKMASK_SOLID))
        {
            // Load neighbor-block indices into shared memory.
            if (threadIdx.x==0)
            {
                //#pragma unroll
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads();
            
            // Retrieve cell masks from the current block and from one cell-layer around it from neighboring blocks.
            for (int p = 1; p < N_Q_max; p++)
            {
                // nbr_kap_b is the index of the neighboring block w.r.t the current cell.
                // nbr_kap_c is the index of the cell in that neighboring block.
                // nbr_kap_h is the index of the halo to store that value.
                
                // First, increment indices along pth direction. Store the resulting halo index.
                int Ip = I + V_CONN_ID[p + 0*27];
                int Jp = J + V_CONN_ID[p + 1*27];
                int Kp = 0;
                if (N_DIM==3)
                    Kp = K + V_CONN_ID[p + 2*27];
                int nbr_kap_h = (Ip+1) + 6*(Jp+1);
                if (N_DIM==3)
                    nbr_kap_h += 36*(Kp+1);
                
                // Then, identify the appropriate neighbor block to store the retrieved cell masks.
                int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                Ip = (4 + (Ip % 4)) % 4;
                Jp = (4 + (Jp % 4)) % 4;
                if (N_DIM==3)
                    Kp = (4 + (Kp % 4)) % 4;
                int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                
                // Write cell mask to the halo.
                bool changed = (Ip != I+V_CONN_ID[p + 0*27] || V_CONN_ID[p + 0*27]==0) && (Jp != J+V_CONN_ID[p + 1*27] || V_CONN_ID[p + 1*27]==0) && (Kp != K+V_CONN_ID[p + 2*27] || V_CONN_ID[p + 2*27]==0);
                if (changed && nbr_kap_b > -1)
                    s_ID_mask[nbr_kap_h] = cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c];
            }
            int curr_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            s_ID_mask[(I+1)+6*(J+1)+(N_DIM-2)*36*(K+1)] = curr_mask;
            __syncthreads();
            
            // Now go through the shared memory array and check if the current cells are adjacent to any solid cells.
            for (int p = 0; p < N_Q_max; p++)
            {
                // First, increment indices along pth direction. Store the resulting halo index.
                int Ip = I + V_CONN_ID[p + 0*27];
                int Jp = J + V_CONN_ID[p + 1*27];
                int Kp = 0;
                if (N_DIM==3)
                    Kp = K + V_CONN_ID[p + 2*27];
                int nbr_kap_h = (Ip+1) + 6*(Jp+1);
                if (N_DIM==3)
                    nbr_kap_h += 36*(Kp+1);
                
                // Now, check the neighboring cell mask for all cells using values stored in shared memory.
                if (s_ID_mask[nbr_kap_h] == V_CELLMASK_SOLID)
                    near_a_solid_cell = true;
            }
            
            
            
            // [DEPRECATED]
            // Each cell checks the mask of its neighboring cell, if it exists.
            // If at least one is a solid cell, then mark this as 'on the boundary'.
            /*
            int curr_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            for (int p = 0; p < N_Q_max; p++)
            {
                int Ip = I + V_CONN_ID[p + 0*27];
                int Jp = J + V_CONN_ID[p + 1*27];
                int Kp = 0;
                if (N_DIM==3)
                    Kp = K + V_CONN_ID[p + 2*27];
                
                int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                Ip = (4 + (Ip % 4)) % 4;
                Jp = (4 + (Jp % 4)) % 4;
                if (N_DIM==3)
                    Kp = (4 + (Kp % 4)) % 4;
                int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                
                if (nbr_kap_b >= 0 && cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c] == V_CELLMASK_SOLID)
                    near_a_solid_cell = true;
            }
            */
            
            
            
            s_D[threadIdx.x] = 0;
            if (near_a_solid_cell && curr_mask != V_CELLMASK_SOLID)
            {
                cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = V_CELLMASK_BOUNDARY;
                s_D[threadIdx.x] = 1;
            }
            __syncthreads();
            
            // Block reduction for sum.
            for (int s=blockDim.x/2; s>0; s>>=1)
            {
                if (threadIdx.x < s)
                {
                    s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
                }
                __syncthreads();
            }
            
            if (threadIdx.x == 0 && s_D[threadIdx.x] > 0)
            {
                tmp_1[i_kap_b] = s_D[threadIdx.x];
                cblock_ID_onb[i_kap_b] = 1;
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_UpdateMasks_WARP
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask
)
{
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int kap_b = kap / 32;
    int tid = threadIdx.x % 32;
    int i_kap_b = -1;
    
    if (kap_b < n_ids_idev_L)
        i_kap_b = id_set_idev_L[kap_b];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Load cell mask values.
        int cellmask = cells_ID_mask[M_CBLOCK*i_kap_b + tid];
        int cellmask2 = cells_ID_mask[M_CBLOCK*i_kap_b + tid+32];
        int s_D = 0;
        if (cellmask == V_CELLMASK_SOLID || cellmask2 == V_CELLMASK_SOLID)
            s_D = 1;
        
        // Warp reduction for sum.
        for (int offset = 16; offset > 0; offset /= 2)
            s_D += __shfl_down_sync(0xFFFFFFFF, s_D, offset);
        
        // If at least one cell is solid, mark this block.
        if (tid==0 && __shfl_sync(0xFFFFFFFF, s_D, 0)>0)
            cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_UpdateMasks
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask
)
{
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_D[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Load cell mask values.
        int cellmask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
        s_D[threadIdx.x] = 0;
        if (cellmask == V_CELLMASK_SOLID)
            s_D[threadIdx.x] = 1;
        __syncthreads();
        
        // Block reduction for sum.
        for (int s=blockDim.x/2; s>0; s>>=1)
        {
            if (threadIdx.x < s)
            {
                s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        // If at least one cell is solid, mark this block.
        if (threadIdx.x==0 && s_D[0]>0)
            cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_UpdateMasks_Vis
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    int *__restrict__ cells_ID_mask,
    const int *__restrict__ cblock_ID_nbr_child
)
{
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Load cell mask values.
        int cellmask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
        bool has_child = cblock_ID_nbr_child[i_kap_b] < 0;
        
        if (has_child && cellmask == V_CELLMASK_SOLID)
            cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = V_CELLMASK_SOLID_VIS;
    }
}

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_MarkInterior_V2
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    int *__restrict__ cells_ID_mask,
    const int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_nbr_child,
    int *__restrict__ cblock_ID_onb,
    int *tmp_1
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    constexpr int M_HBLOCK = AP->M_HBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_ID_nbr[27];
    __shared__ int s_ID_mask[M_HBLOCK];
    
    // Reset halo array.
    for (int k = 0; k < M_HBLOCK/M_TBLOCK+1; k++)
    {
        if (k*M_TBLOCK + threadIdx.x < M_HBLOCK)
            s_ID_mask[k*M_TBLOCK + threadIdx.x] = N_SKIPID;
    }
    __syncthreads();
    
    if (kap_b < n_ids_idev_L)
        i_kap_b = id_set_idev_L[kap_b];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        int i_kap_bc = cblock_ID_nbr_child[i_kap_b];
        int valid_mask = cblock_ID_mask[i_kap_b];
        
        // Latter condition is added only if n>0.
        if (i_kap_bc < 0 && (valid_mask < 0 && valid_mask != V_BLOCKMASK_SOLID))
        {
            // Calculate cell indices.
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            if (N_DIM==3)
                K = (threadIdx.x / 4) / 4;
            
            // Load neighbor-block indices into shared memory.
            if (threadIdx.x==0)
            {
                //#pragma unroll
                for (int p = 0; p < N_Q_max; p++)
                    s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
            }
            __syncthreads();
            
            // Retrieve cell masks from the current block and from one cell-layer around it from neighboring blocks.
            for (int p = 1; p < N_Q_max; p++)
            {
                // nbr_kap_b is the index of the neighboring block w.r.t the current cell.
                // nbr_kap_c is the index of the cell in that neighboring block.
                // nbr_kap_h is the index of the halo to store that value.
                
                // First, increment indices along pth direction. Store the resulting halo index.
                int Ip = I + V_CONN_ID[p + 0*27];
                int Jp = J + V_CONN_ID[p + 1*27];
                int Kp = 0;
                if (N_DIM==3)
                    Kp = K + V_CONN_ID[p + 2*27];
                int nbr_kap_h = (Ip+1) + 6*(Jp+1);
                if (N_DIM==3)
                    nbr_kap_h += 36*(Kp+1);
                
                // Then, identify the appropriate neighbor block to store the retrieved cell masks.
                int nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
                Ip = (4 + (Ip % 4)) % 4;
                Jp = (4 + (Jp % 4)) % 4;
                if (N_DIM==3)
                    Kp = (4 + (Kp % 4)) % 4;
                int nbr_kap_c = Ip + 4*Jp + 16*Kp;
                
                // Write cell mask to the halo.
                bool changed = (Ip != I+V_CONN_ID[p + 0*27] || V_CONN_ID[p + 0*27]==0) && (Jp != J+V_CONN_ID[p + 1*27] || V_CONN_ID[p + 1*27]==0) && (Kp != K+V_CONN_ID[p + 2*27] || V_CONN_ID[p + 2*27]==0);
                if (changed && nbr_kap_b > -1)
                    s_ID_mask[nbr_kap_h] = cells_ID_mask[nbr_kap_b*M_CBLOCK + nbr_kap_c];
            }
            int curr_mask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
            s_ID_mask[(I+1)+6*(J+1)+(N_DIM-2)*36*(K+1)] = curr_mask;
            __syncthreads();
            
            // Now go through the shared memory array and check if the current cells are adjacent to any solid cells.
            bool near_a_solid_cell = false;
            for (int p = 0; p < N_Q_max; p++)
            {
                // First, increment indices along pth direction. Store the resulting halo index.
                int Ip = I + V_CONN_ID[p + 0*27];
                int Jp = J + V_CONN_ID[p + 1*27];
                int Kp = 0;
                if (N_DIM==3)
                    Kp = K + V_CONN_ID[p + 2*27];
                int nbr_kap_h = (Ip+1) + 6*(Jp+1);
                if (N_DIM==3)
                    nbr_kap_h += 36*(Kp+1);
                
                // Now, check the neighboring cell mask for all cells using values stored in shared memory.
                if (s_ID_mask[nbr_kap_h] == V_CELLMASK_SOLID)
                    near_a_solid_cell = true;
            }
            
            
            s_D[threadIdx.x] = 0;
            if (near_a_solid_cell && curr_mask != V_CELLMASK_SOLID)
            {
                cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = V_CELLMASK_BOUNDARY;
                s_D[threadIdx.x] = 1;
            }
            __syncthreads();
            
            // Block reduction for sum.
            for (int s=blockDim.x/2; s>0; s>>=1)
            {
                if (threadIdx.x < s)
                {
                    s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
                }
                __syncthreads();
            }
            
            if (threadIdx.x == 0 && s_D[threadIdx.x] > 0)
            {
                tmp_1[i_kap_b] = s_D[threadIdx.x];
                cblock_ID_onb[i_kap_b] = 1;
            }
        }
    }
}
