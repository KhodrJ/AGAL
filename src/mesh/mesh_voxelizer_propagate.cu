/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_Voxelize_Propagate ]====================================================  */
/*                                                                                    */
/*  Propagates the solid cell mask values that were first assigned adjacent to the    */
/*  solid surface. This is the first step in the process, and only cell mask values   */
/*  are changed. In the next step, block masks are updated to account for modified    */
/*  cell-blocks. This accelerates the algorithm in that only cell-blocks with the     */
/*  appropriate value are considered for propagation.                                 */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_Propagate_Right_WARP
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_ID_mask[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // Get the left neighbor.
    int nbr_start_left;
    if (N_DIM==2) nbr_start_left = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
    else          nbr_start_left = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1 && nbr_start_left < 0)
    {
        // Compute cell coordinates.
        //int I = threadIdx.x % 4;
        int J = (threadIdx.x / 4) % 4;
        int K = 0;
        if (N_DIM==3)
            K = (threadIdx.x / 4) / 4;
        
        // First read of cell masks.
        s_ID_mask[threadIdx.x] = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
        int status = s_ID_mask[3 + 4*J + 16*K];
        
        // Traverse along +x and update masks.
        int nbr_right = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
        int k = 0;
        while (nbr_right > -1 && k < 1000)
        {
            // Read current cell-block data. Switch the cells to solid if those at the -x edge were solid (i.e., based on status values).
            int cellmask = cells_ID_mask[nbr_right*M_CBLOCK + threadIdx.x];
            if (status == V_CELLMASK_SOLID && cellmask == V_CELLMASK_INTERIOR)
                cellmask = V_CELLMASK_SOLID;
            
            // Update shared memory array and status. Then, switch guard cells back to interior cells before writing.
            s_ID_mask[threadIdx.x] = cellmask;
            status = s_ID_mask[3 + 4*J + 16*K];
            if (cellmask == V_CELLMASK_DUMMY_I)
                cellmask = V_CELLMASK_INTERIOR;
            
            // Write data to current cell-block.
            cells_ID_mask[nbr_right*M_CBLOCK + threadIdx.x] = cellmask;
            
            // Get next neighbor block.
            nbr_right = cblock_ID_nbr[nbr_right + 1*n_maxcblocks];
            k++;
        }
        
        if (k == 100)
            printf("MAX REACHED DURING PROPAGATION...\n");
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_Propagate_Right
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const bool is_dense
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_ID_mask[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // Get the left neighbor.
    int i_nbr_start_left;
    if (N_DIM==2) i_nbr_start_left = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
    else          i_nbr_start_left = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1 && i_nbr_start_left < 0)
    {
        // Compute cell coordinates.
        //int I = threadIdx.x % 4;
        int J = (threadIdx.x / 4) % 4;
        int K = 0;
        if (N_DIM==3)
            K = (threadIdx.x / 4) / 4;
        
        // First read of cell masks.
        s_ID_mask[threadIdx.x] = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
        int status = s_ID_mask[3 + 4*J + 16*K];
        if (i_nbr_start_left == N_SPECID)
            status = V_CELLMASK_SOLID;
        
        // Get first neighbor block in the other direction now.
        int i_nbr_right = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
        
        // Traverse along +x/-x and update masks.
        int k = 0;
        while (i_nbr_right > -1 && k < 1000)
        {
            // Read current cell-block data. Switch the cells to solid if those at the -x edge were solid (i.e., based on status values).
            int cellmask = cells_ID_mask[i_nbr_right*M_CBLOCK + threadIdx.x];
            if (status == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                cellmask = V_CELLMASK_SOLID;
            
            // Update shared memory array and status. Then, switch guard cells back to interior cells before writing.
            s_ID_mask[threadIdx.x] = cellmask;
            status = s_ID_mask[3 + 4*J + 16*K];
            if (is_dense && cellmask == V_CELLMASK_DUMMY_I)
                cellmask = V_CELLMASK_INTERIOR;
            
            // Write data to current cell-block.
            cells_ID_mask[i_nbr_right*M_CBLOCK + threadIdx.x] = cellmask;
            
            // Get next neighbor block.
            i_nbr_right = cblock_ID_nbr[i_nbr_right + 1*n_maxcblocks];
            k++;
        }
        
        if (k == 1000)
            printf("MAX REACHED DURING PROPAGATION...\n");
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_Propagate_Left
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_ID_mask[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
    // Get the right neighbor.
    int i_nbr_start_right = cblock_ID_nbr[i_kap_b + 1*n_maxcblocks];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1 && i_nbr_start_right < 0)
    {
        // Compute cell coordinates.
        //int I = threadIdx.x % 4;
        int J = (threadIdx.x / 4) % 4;
        int K = 0;
        if (N_DIM==3)
            K = (threadIdx.x / 4) / 4;
        
        // First read of cell masks.
        s_ID_mask[threadIdx.x] = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
        int status = s_ID_mask[0 + 4*J + 16*K];
        if (i_nbr_start_right == N_SPECID)
            status = V_CELLMASK_SOLID;
        
        // Get first neighbor block in the other direction now.
        int i_nbr_left = -1;
        if (N_DIM == 2) i_nbr_left = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
        else            i_nbr_left = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
        
        // Traverse along +x/-x and update masks.
        int k = 0;
        while (i_nbr_left > -1 && k < 1000)
        {
            // Read current cell-block data. Switch the cells to solid if those at the -x edge were solid (i.e., based on status values).
            int cellmask = cells_ID_mask[i_nbr_left*M_CBLOCK + threadIdx.x];
            if (status == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                cellmask = V_CELLMASK_SOLID;
            
            // Update shared memory array and status. Then, switch guard cells back to interior cells before writing.
            s_ID_mask[threadIdx.x] = cellmask;
            status = s_ID_mask[0 + 4*J + 16*K];
            if (cellmask == V_CELLMASK_DUMMY_I)
                cellmask = V_CELLMASK_INTERIOR;
            
            // Write data to current cell-block.
            cells_ID_mask[i_nbr_left*M_CBLOCK + threadIdx.x] = cellmask;
            
            // Get next neighbor block.
            if (N_DIM == 2) i_nbr_left = cblock_ID_nbr[i_nbr_left + 3*n_maxcblocks];
            else            i_nbr_left = cblock_ID_nbr[i_nbr_left + 2*n_maxcblocks];
            k++;
        }
        
        if (k == 1000)
            printf("MAX REACHED DURING PROPAGATION...\n");
    }
}
