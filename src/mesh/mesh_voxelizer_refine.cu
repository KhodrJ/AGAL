/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_MarkBoundary ]===============================================  */
/*                                                                                    */
/*  This kernel traverses the cell-blocks according to the 'secondary' mode of        */
/*  access, where threads are assigned to individual cell-blocks and access data      */
/*  from arrays arranged according to the Structure of Arrays format, in order to     */
/*  determine which cell-blocks are adjacent to entirely solid-blocks. These          */
/*  indicate a sort of boundary around the geometry at the block level (the Ids       */
/*  updated in cells_ID_mask indicates this boundary at the cell level according to   */
/*  Cu_FillBins).                                                                     */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_MarkBoundary
(
    const int id_max_curr,
    const int n_maxcblocks,
    const bool hit_max,
    const int L,
    int *__restrict__ cblock_ID_ref,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_level
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_BLOCK = AP->M_BLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    __shared__ int s_ID_nbr[M_BLOCK*9];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int mask_kap;
    bool mark_solid_boundary = false;
    bool eligible = true;
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
    
    for (int k = 0; k < (N_DIM==2?1:3); k++)
    {
        // Load mask.
        mask_kap = cblock_ID_mask[kap];
        
        // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
        if (kap < id_max_curr && cblock_level[kap] == L && mask_kap == V_BLOCKMASK_SOLID)
        {
            for (int p = 0; p < 9; p++)
                s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
        }
        __syncthreads();
        
        // Replace neighbor Ids with their respective marks.
        for (int p = 0; p < 9; p++)
        {
            int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
            if (i_p > -1 && cblock_ID_mask[i_p] == V_BLOCKMASK_REGULAR)
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
            else
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
        }
        __syncthreads();
        
        // Run again and check if any of the masks indicated adjacency to regular blocks.
        if (kap < id_max_curr)
        {
            for (int p = 0; p < 9; p++)
            {
                if (s_ID_nbr[p + threadIdx.x*9] == 1)
                    mark_solid_boundary = true;
            }
        }
        
        if (N_DIM==3)
        {
            __syncthreads();
            
            for (int p = 0; p < 9; p++)
                s_ID_nbr[p + threadIdx.x*9] = -1;
            __syncthreads();
        }
    }
    
    // If near at least one regular block, this block is on the boundary of the solid.
    if (kap < id_max_curr && mark_solid_boundary && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
    {
        cblock_ID_mask[kap] = V_BLOCKMASK_SOLIDB;
        
        // Only refine if not on the finest grid level.
        if (!hit_max)
        {
            for (int p = 0; p < N_Q_max; p++)
            {
                if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
                    eligible = false;
            }
            
            if (eligible)
                cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
        }
    }
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_MarkExterior ]===============================================  */
/*                                                                                    */
/*  Cell-blocks adjacent to those that lie on the boundary of the voxelized solid     */
/*  are marked with this kernel. These cell-blocks contain at least one boundary      */
/*  fluid cell that may require interaction with the solid.                           */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_MarkExterior
(
    const int id_max_curr,
    const int n_maxcblocks,
    const bool hit_max,
    const int L,
    int *__restrict__ cblock_ID_ref,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_level
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_BLOCK = AP->M_BLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    __shared__ int s_ID_nbr[M_BLOCK*9];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    bool mark_for_refinement = false;
    bool eligible = true;
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
    
    for (int k = 0; k < (N_DIM==2?1:3); k++)
    {
        // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
        if (kap < id_max_curr && cblock_level[kap]==L) // && cblock_ID_mask[kap]==V_BLOCKMASK_REGULAR)
        {
            for (int p = 0; p < 9; p++)
                s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
        }
        __syncthreads();
        
        // Replace neighbor Ids with their respective marks.
        for (int p = 0; p < 9; p++)
        {
            int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
            if (i_p > -1 && cblock_ID_mask[i_p] == V_BLOCKMASK_SOLIDB)
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
            else
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
        }
        __syncthreads();
        
        // Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
        if (kap < id_max_curr)
        {
            for (int p = 0; p < 9; p++)
            {
                if (s_ID_nbr[p + threadIdx.x*9] == 1)
                    mark_for_refinement = true;
            }
        }
        
        if (N_DIM==3)
        {
            __syncthreads();
            
            for (int p = 0; p < 9; p++)
                s_ID_nbr[p + threadIdx.x*9] = -1;
            __syncthreads();
        }
    }
    
    // If at least one neighbor was a boundary-interface block, then mark intermediate.
    // Make sure to refine only eligible blocks (should be currently unrefined and then 2:1 balanced afterwards).
    if (kap < id_max_curr && mark_for_refinement && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
    {
        if (!hit_max)
        {
            for (int p = 0; p < N_Q_max; p++)
            {
                if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
                    eligible = false;
            }
            
            if (eligible) // || cblock_ID_mask[kap] == V_BLOCKMASK_SOLID)
                cblock_ID_ref[kap] = V_REF_ID_INDETERMINATE_E;
        }
        
        if (cblock_ID_mask[kap] == V_BLOCKMASK_REGULAR)
            cblock_ID_mask[kap] = V_BLOCKMASK_SOLIDA;
    }
}

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_UpdateSolidChildren
(
    const int id_max_curr,
    const int n_maxcblocks,
    const int L,
    const int *__restrict__ cblock_ID_mask,
    int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_ID_nbr_child,
    const int *__restrict__ cblock_level
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int N_CHILDREN = AP->N_CHILDREN;
    constexpr int M_BLOCK = AP->M_BLOCK;
    constexpr int N_LEFT = N_DIM==2 ? 3 : 2;
    constexpr int N_RIGHT = 1;
    __shared__ int s_ID_child[M_BLOCK*N_CHILDREN];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    for (int p = 0; p < N_CHILDREN; p++)
        s_ID_child[p + threadIdx.x*N_CHILDREN] = -1;
    __syncthreads();
    
    // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
    if (kap < id_max_curr && cblock_level[kap]==L && cblock_ID_mask[kap]==V_BLOCKMASK_SOLID)
    {
        for (int p = 0; p < N_CHILDREN; p++)
            s_ID_child[p + threadIdx.x*N_CHILDREN] = cblock_ID_nbr_child[kap] + p;
    }
    __syncthreads();
    
    // Replace neighbor Ids with their respective marks.
    for (int p = 0; p < N_CHILDREN; p++)
    {
        int i_p = s_ID_child[threadIdx.x + p*M_BLOCK];
        if (i_p > -1 && cblock_ID_nbr[i_p + N_LEFT*n_maxcblocks] == N_SKIPID)
            cblock_ID_nbr[i_p + N_LEFT*n_maxcblocks] = N_SPECID;
        if (i_p > -1 && cblock_ID_nbr[i_p + N_RIGHT*n_maxcblocks] == N_SKIPID)
            cblock_ID_nbr[i_p + N_RIGHT*n_maxcblocks] = N_SPECID;
    }
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_Propagate ]==================================================  */
/*                                                                                    */
/*  Propagates the intermediate marks for refinement. Each unmarked cell-block that   */
/*  is adjacent to at least one cell-block marked with an intermediate flag will      */
/*  also be marked as such. This is performed a number of times Nprop that is         */
/*  calculated using the specified near-wall distance refinement criterion and the    */
/*  total length along one axis of the cell-blocks on the current grid level.         */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_Propagate
(
    const int id_max_curr,
    const int n_maxcblocks,
    const int L,
    int *__restrict__ cblock_ID_ref,
    const int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_level,
    int *__restrict__ tmp_1,
    const int jprop
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_BLOCK = AP->M_BLOCK;
    constexpr int N_Q_max = AP->N_Q_max;
    __shared__ int s_ID_nbr[M_BLOCK*9];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    bool mark_for_refinement = false;
    bool eligible = true;
    
    // To prevent a race condition, alternative between even and odd intermediate states.
    bool check_even = true;
    if (jprop%2 == 1)
        check_even = false;
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
    
    for (int k = 0; k < (N_DIM==2?1:3); k++)
    {
        // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
        if (kap < id_max_curr && cblock_ID_mask[kap] > -1 && cblock_level[kap] == L)
        {
            for (int p = 0; p < 9; p++)
                s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
        }
        __syncthreads();
        
        // Replace neighbor Ids with their respective marks.
        for (int p = 0; p < 9; p++)
        {
            int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
            if (check_even)
            {
                if (i_p > -1 && cblock_ID_ref[i_p] == V_REF_ID_INDETERMINATE_E)
                    s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
                else
                    s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
            }
            else
            {
                if (i_p > -1 && tmp_1[i_p] == V_REF_ID_INDETERMINATE_O)
                    s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
                else
                    s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
            }
        }
        __syncthreads();
        
        // Run again and check if any of the marks indicated refinement. If so, replace child-nbr accordingly.
        if (kap < id_max_curr)
        {
            for (int p = 0; p < 9; p++)
            {
                if (s_ID_nbr[p + threadIdx.x*9] == 1)
                    mark_for_refinement = true;
            }
        }
        
        if (N_DIM==3)
        {
            __syncthreads();
            
            for (int p = 0; p < 9; p++)
                s_ID_nbr[p + threadIdx.x*9] = -1;
            __syncthreads();
        }
    }
    
    // If at least one neighbor was a boundary-interface block, then mark intermediate.
    // Make sure to refine only eligible blocks (should be currently unrefined, 2:1 balanced afterwards).
    if (kap < id_max_curr && mark_for_refinement && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
    {
        for (int p = 0; p < N_Q_max; p++)
        {
            if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
                eligible = false;
        }
        
        if (eligible)
        {
            if (check_even)
                tmp_1[kap] = V_REF_ID_INDETERMINATE_O;
            else
                cblock_ID_ref[kap] = V_REF_ID_INDETERMINATE_E;
        }
    }
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_Finalize ]===================================================  */
/*                                                                                    */
/*  Finalize the previous intermediate marks for refinement by adjusting the flag.    */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_Finalize
(
    const int id_max_curr,
    int *__restrict__ cblock_ID_ref,
    int *__restrict__ tmp_1
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    // Note: ref Id will be corrected after this. tmp_1 is reset before Cu_MarkBlocks_GetMasks, so if that is changed then
    // make sure tmp_1 has been reset properly before calling propagation.
    if (kap < id_max_curr && (cblock_ID_ref[kap] == V_REF_ID_INDETERMINATE_E || tmp_1[kap] == V_REF_ID_INDETERMINATE_O))
    {
        cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
        tmp_1[kap] = -1;
    }
    
}
