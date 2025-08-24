/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

// template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
// __global__
// void Cu_Voxelize_S1
// (
//     const int L,
//     const ufloat_t dx_L,
//     const int n_maxcblocks,
//     const int id_max_curr_L,
//     const ufloat_t *__restrict__ cblock_f_X,
//     int *__restrict__ cblock_ID_mask,
//     const int *__restrict__ cblock_level,
//     const int *__restrict__ binned_face_ids_n_3D,
//     const int G_BIN_DENSITY
// )
// {
//     constexpr int N_DIM = AP->N_DIM;
//     int kap = blockIdx.x*blockDim.x + threadIdx.x;
//     
//     if (kap < id_max_curr_L && cblock_level[kap]==L)
//     {
//         // Load cell-block coordinates (increment by dx_L so that it's less likely to be on the boundary of a bin).
//         ufloat_t vxp = cblock_f_X[kap + 0*n_maxcblocks] + dx_L;
//         ufloat_t vyp = cblock_f_X[kap + 1*n_maxcblocks] + dx_L;
//         ufloat_t vzp = cblock_f_X[kap + 2*n_maxcblocks] + dx_L;
//         
//         // Compute the global bin index for this cell-block.
//         int bin_id_x = (int)(vxp*G_BIN_DENSITY);
//         int bin_id_y = (int)(vyp*G_BIN_DENSITY);
//         int bin_id_z = 0;
//         if (N_DIM==3)
//             bin_id_z = (int)(vzp*G_BIN_DENSITY);
//         int global_bin_id = bin_id_x + G_BIN_DENSITY*bin_id_y + G_BIN_DENSITY*G_BIN_DENSITY*bin_id_z;
//         
//         // If the associated 3D bin has a positive size, mark this cell-block for consideration in S2.
//         if (binned_face_ids_n_3D[global_bin_id] > 0)
//             cblock_ID_mask[kap] = V_BLOCKMASK_DUMMY_I;
//     }
// }

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_S1
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int L,
    const ufloat_t dx_L,
    const int n_maxcblocks,
    const ufloat_t *__restrict__ cblock_f_X,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_level,
    const int *__restrict__ binned_face_ids_n_3D,
    const int G_BIN_DENSITY
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_D[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x*M_LBLOCK < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x*M_LBLOCK];

    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Compute cell coordinates.
        int I = threadIdx.x % 4;
        int J = (threadIdx.x / 4) % 4;
        int K = 0;
        if (N_DIM==3)
            K = (threadIdx.x / 4) / 4;
        
        // Compute cell coordinates.
        ufloat_g_t vxp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L;
        ufloat_g_t vyp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L;
        ufloat_g_t vzp = (ufloat_g_t)0.0;
        if (N_DIM==3)
            vzp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L;
        
        // For each face, check if the current cell is within the appropriate bounds. If at least one condition is 
        // satisfied, exit the loop and make a note.
        int bin_id_x = (int)(vxp*G_BIN_DENSITY);
        int bin_id_y = (int)(vyp*G_BIN_DENSITY);
        int bin_id_z = 0;
        if (N_DIM==3)
            bin_id_z = (int)(vzp*G_BIN_DENSITY);
        
        // Now find the total number of intersections a ray makes in the direction with the smallest number of bins.
        int global_bin_id = bin_id_x + G_BIN_DENSITY*bin_id_y + G_BIN_DENSITY*G_BIN_DENSITY*bin_id_z;
        int n_f = binned_face_ids_n_3D[global_bin_id];
        s_D[threadIdx.x] = n_f;
        __syncthreads();
        
        // Block reduction for s_D. If at least one cell was in a valid bin, mark this cell-block.
//         for (int s=blockDim.x/2; s>0; s>>=1)
//         {
//             if (threadIdx.x < s)
//             {
//                 s_D[threadIdx.x] = max(s_D[threadIdx.x],s_D[threadIdx.x + s]);
//             }
//             __syncthreads();
//         }
//         if (threadIdx.x==0 && s_D[0] > 0)
        if (threadIdx.x==0 && n_f>0)
            cblock_ID_mask[i_kap_b] = V_BLOCKMASK_DUMMY_I;
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_S2
(
    const int L,
    const int n_maxcblocks,
    const int id_max_curr_L,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_level
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_BLOCK = AP->M_BLOCK;
    __shared__ int s_ID_nbr[M_BLOCK*9];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    for (int p = 0; p < 9; p++)
        s_ID_nbr[p + threadIdx.x*9] = -1;
    __syncthreads();
    
    // Load level, initialize boolean.
    int level_kap = cblock_level[kap];
    bool mark_adjacent_solid_surface = false;
    
    for (int k = 0; k < (N_DIM==2?1:3); k++)
    {
        // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
        if (kap < id_max_curr_L && level_kap == L)
        {
            for (int p = 0; p < 9; p++)
                s_ID_nbr[p + threadIdx.x*9] = cblock_ID_nbr[kap + (k*9+p)*n_maxcblocks];
        }
        __syncthreads();
        
        // Replace neighbor Ids with their respective marks.
        for (int p = 0; p < 9; p++)
        {
            int i_p = s_ID_nbr[threadIdx.x + p*M_BLOCK];
            if (i_p > -1 && cblock_ID_mask[i_p] == V_BLOCKMASK_DUMMY_I)
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = 1;
            else
                s_ID_nbr[threadIdx.x + p*M_BLOCK] = -1;
        }
        __syncthreads();
        
        // Run again and check if any of the masks indicated adjacency to regular blocks.
        if (kap < id_max_curr_L && level_kap == L)
        {
            for (int p = 0; p < 9; p++)
            {
                if (s_ID_nbr[p + threadIdx.x*9] == 1)
                    mark_adjacent_solid_surface = true;
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
    if (kap < id_max_curr_L && mark_adjacent_solid_surface)
        cblock_ID_mask[kap + 1*n_maxcblocks] = V_BLOCKMASK_DUMMY_II;
}


template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_S3_V1
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    const bool hit_max,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int n_faces,
    const int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    const ufloat_g_t *__restrict__ geom_f_face_Xt,
    const int *__restrict__ binned_face_ids_n_2D,
    const int *__restrict__ binned_face_ids_N_2D,
    const int *__restrict__ binned_face_ids_2D,
    const int G_BIN_DENSITY
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_ID_mask[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x*M_LBLOCK < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x*M_LBLOCK];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1)
    {
        // Compute cell coordinates.
        int I = threadIdx.x % 4;
        int J = (threadIdx.x / 4) % 4;
        int K = 0;
        if (N_DIM==3)
            K = (threadIdx.x / 4) / 4;
        
        int intersect_counter = 0;
        if (cblock_ID_mask[i_kap_b + 1*n_maxcblocks] == V_BLOCKMASK_DUMMY_II)
        {
            ufloat_g_t vxp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L;
            ufloat_g_t vyp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L;
            ufloat_g_t vzp = (ufloat_g_t)0.0;
            if (N_DIM==3)
                vzp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L;
            
            int bin_id_y = (int)(vyp*G_BIN_DENSITY);
            int bin_id_z = 0;
            if (N_DIM==3)
                bin_id_z = (int)(vzp*G_BIN_DENSITY);
            int global_bin_id = bin_id_y + G_BIN_DENSITY*bin_id_z;
            
            int n_f = binned_face_ids_n_2D[global_bin_id];
            int N_f = binned_face_ids_N_2D[global_bin_id];
            for (int p = 0; p < n_f; p++)
            {
                int f_p = binned_face_ids_2D[N_f+p];
                ufloat_g_t vx1 = geom_f_face_X[f_p + 0*n_faces_a];
                ufloat_g_t vy1 = geom_f_face_X[f_p + 1*n_faces_a];
                ufloat_g_t vx2 = geom_f_face_X[f_p + 3*n_faces_a];
                ufloat_g_t vy2 = geom_f_face_X[f_p + 4*n_faces_a];
                
                if (N_DIM==2)
                {
                    ufloat_g_t nx = vy2-vy1;
                    ufloat_g_t ny = vx1-vx2;
                    
                    // Find the distance along a ray with direction [1,0].
                    ufloat_g_t tmp = (vx1-vxp) + (vy1-vyp)*(ny/nx);
                    if (tmp > 0)
                    {
                        ufloat_g_t tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
                        bool C = CheckPointInLine(tmp2, vyp, vx1, vy1, vx2, vy2);
                        if (C)
                            intersect_counter++;
                    }
                }
                else
                {
                    ufloat_g_t vz1 = geom_f_face_X[f_p + 2*n_faces_a];
                    ufloat_g_t vz2 = geom_f_face_X[f_p + 5*n_faces_a];
                    ufloat_g_t vx3 = geom_f_face_X[f_p + 6*n_faces_a];
                    ufloat_g_t vy3 = geom_f_face_X[f_p + 7*n_faces_a];
                    ufloat_g_t vz3 = geom_f_face_X[f_p + 8*n_faces_a];
                    
                    ufloat_g_t ex1 = vx2-vx1;
                    ufloat_g_t ey1 = vy2-vy1;
                    ufloat_g_t ez1 = vz2-vz1;
                    ufloat_g_t ex2 = vx3-vx1;
                    ufloat_g_t ey2 = vy3-vy1;
                    ufloat_g_t ez2 = vz3-vz1;
                    ufloat_g_t nx = (ufloat_g_t)0.0;
                    ufloat_g_t ny = (ufloat_g_t)0.0;
                    ufloat_g_t nz = (ufloat_g_t)0.0;
                    Cross(ex1, ey1, ez1, ex2, ey2, ez2, nx, ny, nz);
                    ufloat_g_t tmp = Tsqrt(nx*nx + ny*ny + nz*nz);
                    nx /= tmp;
                    ny /= tmp;
                    nz /= tmp;
                    
                    // Find the distance along a ray with direction [1,0,0].
                    tmp = (vx1-vxp) + (vy1-vyp)*(ny/nx) + (vz1-vzp)*(nz/nx);
                    if (tmp > 0)
                    {
                        ufloat_g_t tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
                        bool C = CheckPointInTriangle(tmp2, vyp, vzp, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, nx, ny, nz, ex1, ey1, ez1, ex2, ey2, ez2);
                        if (C)
                            intersect_counter++;
                    }
                }
            }
            
            // Now, if there are an even number of intersections, the current cell is in the solid.
            // Otherwise, it is a fluid cell.
            s_D[threadIdx.x] = 0;
            int cellmask = V_CELLMASK_INTERIOR;
            if (intersect_counter%2 == 0)
                cellmask = V_CELLMASK_DUMMY_I;
            if (intersect_counter%2 == 1)
            {
                cellmask = V_CELLMASK_SOLID;
                
                // In either case of solid or surface-of-solid, we want to write this mask value.
                //s_D[threadIdx.x] = 1;
            }
            s_ID_mask[threadIdx.x] = cellmask;
            __syncthreads();
            
            // Block reduction for sum.
            //for (int s=blockDim.x/2; s>0; s>>=1)
            //{
            //    if (threadIdx.x < s)
            //    {
            //        s_D[threadIdx.x] = s_D[threadIdx.x] + s_D[threadIdx.x + s];
            //    }
            //    __syncthreads();
            //}
            
            //if (s_D[0]>0)
            {
                // If at least one cell is solid, update the block mask.
                //if (threadIdx.x == 0)
                //    cblock_ID_mask[i_kap_b] = V_BLOCKMASK_DUMMY_I;
                
                /*
                // Internally propagate the cell mask values to the x-edges of the cell-block.
                for (int l = 0; l < 9; l++)
                {
                    if (I > 0)
                    {
                        int nbr_mask = s_ID_mask[(I-1) + 4*J + 16*K];
                        if (nbr_mask == V_CELLMASK_DUMMY_II && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_II;
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask != V_CELLMASK_DUMMY_II)
                            cellmask = V_CELLMASK_DUMMY_I;
                    }
                    if (I < 3)
                    {
                        int nbr_mask = s_ID_mask[(I+1) + 4*J + 16*K];
                        if (nbr_mask == V_CELLMASK_DUMMY_II && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_II;
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask != V_CELLMASK_DUMMY_II)
                            cellmask = V_CELLMASK_DUMMY_I;
                    }
                    if (J > 0)
                    {
                        int nbr_mask = s_ID_mask[I + 4*(J-1) + 16*K];
                        if (nbr_mask == V_CELLMASK_DUMMY_II && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_II;
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask != V_CELLMASK_DUMMY_II)
                            cellmask = V_CELLMASK_DUMMY_I;
                    }
                    if (J < 3)
                    {
                        int nbr_mask = s_ID_mask[I + 4*(J+1) + 16*K];
                        if (nbr_mask == V_CELLMASK_DUMMY_II && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_II;
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask != V_CELLMASK_DUMMY_II)
                            cellmask = V_CELLMASK_DUMMY_I;
                    }
                    if (N_DIM==3 && K > 0)
                    {
                        int nbr_mask = s_ID_mask[I + 4*J + 16*(K-1)];
                        if (nbr_mask == V_CELLMASK_DUMMY_II && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_II;
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask != V_CELLMASK_DUMMY_II)
                            cellmask = V_CELLMASK_DUMMY_I;
                    }
                    if (N_DIM==3 && K < 3)
                    {
                        int nbr_mask = s_ID_mask[I + 4*J + 16*(K+1)];
                        if (nbr_mask == V_CELLMASK_DUMMY_II && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_II;
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask != V_CELLMASK_DUMMY_II)
                            cellmask = V_CELLMASK_DUMMY_I;
                    }
                    s_ID_mask[threadIdx.x] = cellmask;
                    __syncthreads();
                }
                */
                
                // If there are solid masks in this block, place guard in the masks for the propagation.
                //if (cellmask == V_CELLMASK_INTERIOR)
                //    cellmask = V_CELLMASK_SOLIDS;
                cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = cellmask;
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_V2
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    const bool hit_max,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_ref,
    const int *__restrict__ cblock_ID_nbr,
    const int n_faces,
    const int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    const ufloat_g_t *__restrict__ geom_f_face_Xt,
    const int *__restrict__ binned_face_ids_n,
    const int *__restrict__ binned_face_ids_N,
    const int *__restrict__ binned_face_ids,
    const int G_BIN_DENSITY
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_fI[M_TBLOCK];
    __shared__ ufloat_g_t s_fD[16];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    
    s_ID_cblock[threadIdx.x] = -1;
    s_fI[threadIdx.x] = -1;
    if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
    {
        s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
    }
    __syncthreads();
    
    // Loop over block Ids.
    for (int k = 0; k < M_LBLOCK; k += 1)
    {
        int i_kap_b = s_ID_cblock[k];
        
        // Latter condition is added only if n>0.
        if (i_kap_b > -1)
        {
            // Compute cell-center coordinates.
            ufloat_g_t vxp = cblock_f_X[i_kap_b + 0*n_maxcblocks] + (threadIdx.x % 4)*dx_L + 0.5*dx_L;
            ufloat_g_t vyp = cblock_f_X[i_kap_b + 1*n_maxcblocks] + ((threadIdx.x / 4) % 4)*dx_L + 0.5*dx_L;
            ufloat_g_t vzp = (ufloat_g_t)0.0;
            if (N_DIM==3)
                vzp = cblock_f_X[i_kap_b + 2*n_maxcblocks] + ((threadIdx.x / 4) / 4)*dx_L + 0.5*dx_L;
            
            // For each face, check if the current cell is within the appropriate bounds. If at least
            // one condition is satisfied, exit the loop and make a note.
            int global_bin_id = ((int)(vyp*G_BIN_DENSITY)) + G_BIN_DENSITY*((int)(vzp*G_BIN_DENSITY));
            
            // Now find the total number of intersections a ray makes in the direction with the smallest number of bins.
            //s_fI[threadIdx.x] = binned_face_ids[global_bin_id+threadIdx.x];
            int n_f = binned_face_ids_n[global_bin_id];
            int N_f = 0;
            int pmin = -1;
            ufloat_g_t tmpmin = (ufloat_t)1.0;
            ufloat_g_t dotmin = (ufloat_t)0.0;
            if (n_f > 0)
            {
                N_f = binned_face_ids_N[global_bin_id];
                for (int j = 0; j < n_f/M_TBLOCK+1; j++)
                {
                    // Read the next M_TBLOCK faces.
                    int plim = M_TBLOCK;
                    if ((j+1)*M_TBLOCK >= n_f)
                        plim = M_TBLOCK - ((j+1)*M_TBLOCK - n_f);
                    if (threadIdx.x < plim)
                        s_fI[threadIdx.x] = binned_face_ids[N_f + j*M_TBLOCK + threadIdx.x];
                    __syncthreads();
                    
                    for (int p = 0; p < plim; p++)
                    {
                        //int f_p = binned_face_ids[N_f+p];
                        int f_p = s_fI[p];
                        
                        // Load face data.
                        if (threadIdx.x < 16)
                            s_fD[threadIdx.x] = geom_f_face_Xt[threadIdx.x + f_p*16];
                        __syncthreads();
                        
                        if (N_DIM==2)
                        {
                            // Load normal.
                            ufloat_g_t nx = s_fD[9];
                            ufloat_g_t ny = s_fD[10];
                            
                            // Load vertices 1 and 2.
                            ufloat_g_t vx = s_fD[0];
                            ufloat_g_t vy = s_fD[1];
                            ufloat_g_t vx2 = s_fD[3];
                            ufloat_g_t vy2 = s_fD[4];
                            
                            // Find the distance along a ray with direction [1,0].
                            ufloat_g_t tmp = (vx-vxp) + (vy-vyp)*(ny/nx);
                            if (tmp > 0)
                            {
                                ufloat_g_t tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
                                
                                // First check if point is inside the line.
                                bool C = -( ((vx-tmp2)*(vx2-vx))+((vy-vyp)*(vy2-vy)) ) > 0;
                                
                                // Second check if point is inside the line.
                                C = C && ( ((vx2-tmp2)*(vx2-vx))+((vy2-vyp)*(vy2-vy)) ) > 0;
                                
                                if (C)
                                {
                                    pmin = p;
                                    tmpmin = tmp;
                                    dotmin = nx*(tmp2 - vxp);
                                }
                            }
                        }
                        else
                        {
                            // Load normal.
                            ufloat_g_t nx = s_fD[9];
                            ufloat_g_t ny = s_fD[10];
                            ufloat_g_t nz = s_fD[11];
                            
                            // Load vertex 1.
                            ufloat_g_t vx = s_fD[0];
                            ufloat_g_t vy = s_fD[1];
                            ufloat_g_t vz = s_fD[2];
                            
                            // Find the distance along a ray with direction [1,0,0].
                            ufloat_g_t tmp = (vx-vxp) + (vy-vyp)*(ny/nx) + (vz-vzp)*(nz/nx);
                            if (tmp > 0)
                            {
                                ufloat_g_t tmp2 = vxp + tmp; // Stores the x-component of the intersection point.
                                
                                // First check that point is inside the triangle.
                                ufloat_g_t vx2 = s_fD[3] - vx; // vx2 stores x-comp. of edge 1: vx2-vx1.
                                ufloat_g_t vy2 = s_fD[4] - vy; // I don't need real vx2 individually.
                                ufloat_g_t vz2 = s_fD[5] - vz;
                                ufloat_g_t sx = vy2*nz - vz2*ny;
                                ufloat_g_t sy = vz2*nx - vx2*nz;
                                ufloat_g_t sz = vx2*ny - vy2*nx;
                                ufloat_g_t C = (vx-tmp2)*sx + (vy-vyp)*sy + (vz-vzp)*sz > 0;
                                
                                // Second check that point is inside the triangle.
                                vx += vx2; // Recover vertex 2.
                                vy += vy2;
                                vz += vz2;
                                vx2 = s_fD[6] - vx; // vx2 stores x-comp. of edge 2: vx3-vx2.
                                vy2 = s_fD[7] - vy; // I don't need real vx2 individually.
                                vz2 = s_fD[8] - vz;
                                sx = vy2*nz - vz2*ny;
                                sy = vz2*nx - vx2*nz;
                                sz = vx2*ny - vy2*nx;
                                C = C && (vx-tmp2)*sx + (vy-vyp)*sy + (vz-vzp)*sz > 0;
                                
                                // Second check that point is inside the triangle.
                                vx += vx2; // Recover vertex 3.
                                vy += vy2;
                                vz += vz2;
                                vx2 = s_fD[0] - vx; // vx2 stores x-comp. of edge 3: vx1-vx3.
                                vy2 = s_fD[1] - vy; // I don't need real vx2 individually.
                                vz2 = s_fD[2] - vz;
                                sx = vy2*nz - vz2*ny;
                                sy = vz2*nx - vx2*nz;
                                sz = vx2*ny - vy2*nx;
                                C = C && (vx-tmp2)*sx + (vy-vyp)*sy + (vz-vzp)*sz > 0;
                                
                                if (C)
                                {
                                    pmin = p;
                                    tmpmin = tmp;
                                    dotmin = nx*(tmp2 - vxp);
                                }
                            }
                        }
                        
                        __syncthreads();
                    }
                }
            }
            
            // Now, if there are an even number of intersections, the current cell is in the solid.
            // Otherwise, it is a fluid cell.
            s_D[threadIdx.x] = 0;
            if (pmin != -1 && dotmin >= (ufloat_g_t)0.0)
            {
                cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = V_CELLMASK_SOLID;
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
            
            // If at least one cell is solid, update the block mask.
            if (threadIdx.x == 0 && s_D[threadIdx.x]>0)
                cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
        }
    }
}

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
void Cu_Voxelize_Propagate_S1
(
    const int j_parity,
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    const ufloat_t dxb_L,
    int *__restrict__ cells_ID_mask,
    const ufloat_t *__restrict__ cblock_f_X,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_ID_cblock[M_TBLOCK];
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_ID_mask[M_TBLOCK];
    __shared__ int s_ID_mask_nbr_R[M_TBLOCK];
    int kap = blockIdx.x*M_LBLOCK + threadIdx.x;
    
    s_ID_cblock[threadIdx.x] = -1;
    s_ID_mask_nbr_R[threadIdx.x] = N_SKIPID;
    if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
    {
        s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
    }
    __syncthreads();
    
    // Loop over block Ids.
    for (int k = 0; k < M_LBLOCK; k += 1)
    {
        int i_kap_b = s_ID_cblock[k];
        int valid_block = -1;
        bool parity_match = false;
        
        if (i_kap_b > -1)
        {
            valid_block = cblock_ID_mask[i_kap_b];
            parity_match = (int)((cblock_f_X[i_kap_b]+(ufloat_t)1e-5)/(dxb_L))%2   ==   j_parity;
        }
        
        // Latter condition is added only if n>0.
        if (i_kap_b > -1 && valid_block == V_BLOCKMASK_DUMMY_I && parity_match)
        {
            int I = threadIdx.x % 4;
            int J = (threadIdx.x / 4) % 4;
            int K = 0;
            if (N_DIM==3)
                K = (threadIdx.x / 4) / 4;
            
            // Get left and right neighbor blocks.
            int nbr_kap_R = i_kap_b;
            for (int l = 0; l < 1; l++)
            {
                i_kap_b = nbr_kap_R;
                nbr_kap_R = cblock_ID_nbr[nbr_kap_R + 1*n_maxcblocks];
                
                // Read neighbor data, if valid.
                // It's safe to have __syncthreads() since all threads have the same nbr_kap_R value.
                if (nbr_kap_R > -1)// && cblock_ID_mask[nbr_kap_R] != V_BLOCKMASK_DUMMY_I)
                {
                    // Get the current cell masks.
                    int cellmask = cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
                    s_ID_mask[threadIdx.x] = cellmask; //cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x];
                    __syncthreads();
                    cellmask = cells_ID_mask[nbr_kap_R*M_CBLOCK + threadIdx.x];
                    if (I == 0)
                    {
                        int cellmask_nbr = s_ID_mask[3 + 4*J + 16*K];
                        bool changeable = cellmask == V_CELLMASK_INTERIOR || cellmask == V_CELLMASK_SOLIDS;
                        if (cellmask_nbr == V_CELLMASK_DUMMY_I && changeable)
                            cellmask = V_CELLMASK_DUMMY_I;
                    }
                    s_ID_mask_nbr_R[threadIdx.x] = cellmask;
                    __syncthreads();
                    
                    // Internal propagation to the right.
                    for (int j = 0; j < 3; j++)
                    {
                        if (I == j+1)
                        {
                            int cellmask_nbr = s_ID_mask_nbr_R[(I-1) + 4*J + 16*K];
                            bool changeable = cellmask == V_CELLMASK_INTERIOR || cellmask == V_CELLMASK_SOLIDS;
                            if (cellmask_nbr == V_CELLMASK_DUMMY_I && changeable)
                                cellmask = V_CELLMASK_DUMMY_I;
                        }
                        //if (I == 2-j)
                        //{
                        //    int cellmask_nbr = s_ID_mask_nbr_R[(I+1) + 4*J + 16*K];
                        //    bool changeable = cellmask == V_CELLMASK_INTERIOR || cellmask == V_CELLMASK_SOLIDS;
                        //    if (cellmask_nbr == V_CELLMASK_DUMMY_I && changeable)
                        //        cellmask = V_CELLMASK_DUMMY_I;
                        //}
                        s_ID_mask_nbr_R[threadIdx.x] = cellmask;
                        __syncthreads();
                    }
                    
                    // Update the cell-block mask based on current cell mask values.
                    s_D[threadIdx.x] = 0;
                    if (cellmask != V_CELLMASK_INTERIOR)
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
                    
                    // If at least one cell is solid, update the block mask.
                    if (s_D[0]>0)
                    {
                        // If at least one cell is solid, update the block mask.
                        if (threadIdx.x == 0)
                            cblock_ID_mask[nbr_kap_R + 1*n_maxcblocks] = V_BLOCKMASK_DUMMY_II;
                        cells_ID_mask[nbr_kap_R*M_CBLOCK + threadIdx.x] = cellmask;
                    }
                }
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_Propagate_S2
(
    const int id_max_curr,
    const int L,
    const int n_maxcblocks,
    int *__restrict__ cblock_ID_mask,
    const int *__restrict__ cblock_ID_nbr,
    const int *__restrict__ cblock_level
)
{
    constexpr int M_BLOCK = AP->M_BLOCK;
    __shared__ int s_ID_nbr[M_BLOCK];
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    s_ID_nbr[threadIdx.x] = N_SKIPID;
    __syncthreads();
    
    // Load mask and level.
    int level_kap = cblock_level[kap];
    int mask_kap_0 = cblock_ID_mask[kap + 0*n_maxcblocks];
    int mask_kap_1 = cblock_ID_mask[kap + 1*n_maxcblocks];
    
    // First, read neighbor Ids and place in shared memory. Arrange for contiguity.
    if (kap < id_max_curr && level_kap == L && mask_kap_1 == V_BLOCKMASK_DUMMY_II)
    {
        // Note: x+ neighbor has same value in 2D and 3D.
        s_ID_nbr[threadIdx.x] = cblock_ID_nbr[kap + 1*n_maxcblocks];
    }
    __syncthreads();
    
    // Replace +x neighbor Ids with their respective marks.
    int i_p = s_ID_nbr[threadIdx.x];
    if (i_p > -1)
        i_p = cblock_ID_mask[i_p + 1*n_maxcblocks];
    if (i_p != V_BLOCKMASK_DUMMY_II)
        s_ID_nbr[threadIdx.x] = 1;
    else
        s_ID_nbr[threadIdx.x] = -1;
    __syncthreads();
    
    // Run again and check if any of the masks indicated adjacency to regular blocks.
    if (kap < id_max_curr && level_kap == L && mask_kap_1 == V_BLOCKMASK_DUMMY_II)
    {
        // If neighbor is not marked as a dummy, then propagation proceeds from this block. Convert in next step.
        if (s_ID_nbr[threadIdx.x] == 1)
            mask_kap_0 = V_BLOCKMASK_DUMMY_II;
        
        // If neighbor is explicily marked as dummy, then convert it into solid and finalize.
        if (s_ID_nbr[threadIdx.x] == -1)
            mask_kap_0 = V_BLOCKMASK_SOLID;
        
        cblock_ID_mask[kap] = mask_kap_0;
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_Propagate_S3
(
    const int id_max_curr,
    const int L,
    const int n_maxcblocks,
    int *__restrict__ cblock_ID_mask
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < id_max_curr)
    {
        int mask_kap_0 = cblock_ID_mask[kap + 0*n_maxcblocks];
        int mask_kap_1 = cblock_ID_mask[kap + 1*n_maxcblocks];
        
        // If this block was used for propagation, turn it into a solid block to finalize.
        if (mask_kap_0 == V_BLOCKMASK_DUMMY_I)
            mask_kap_0 = V_BLOCKMASK_SOLID;
        
        // If this block needs to be used for propagation
        if (mask_kap_1 == V_BLOCKMASK_DUMMY_II)
        {
            mask_kap_0 = V_BLOCKMASK_DUMMY_I;
            mask_kap_1 = V_BLOCKMASK_REGULAR;
        }
        
        cblock_ID_mask[kap + 0*n_maxcblocks] = mask_kap_0;
        cblock_ID_mask[kap + 1*n_maxcblocks] = mask_kap_1;
    }
}

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

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_MarkBlocks_MarkInterior ]===============================================  */
/*                                                                                    */
/*  Cell-blocks adjacent to those that lie on the boundary of the voxelized solid     */
/*  are marked with this kernel. These cell-blocks contain at least one boundary      */
/*  fluid cell that may require interaction with the solid.                           */
/*                                                                                    */
/**************************************************************************************/

template <const ArgsPack *AP>
__global__
void Cu_MarkBlocks_MarkInterior
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
        if (kap < id_max_curr && cblock_level[kap] == L)
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
    // Make sure to refine only eligible blocks (should be currently unrefined, 2:1 balanced afterwards).
    if (kap < id_max_curr && mark_for_refinement && cblock_ID_ref[kap] == V_REF_ID_UNREFINED)
    {
        if (!hit_max)
        {
            for (int p = 0; p < N_Q_max; p++)
            {
                if (cblock_ID_nbr[kap + p*n_maxcblocks] == N_SKIPID)
                    eligible = false;
            }
            
            if (eligible)
                cblock_ID_ref[kap] = V_REF_ID_INDETERMINATE_E;
        }
        
        if (cblock_ID_mask[kap] == V_BLOCKMASK_REGULAR)
            cblock_ID_mask[kap] = V_BLOCKMASK_SOLIDA;
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
        cblock_ID_ref[kap] = V_REF_ID_MARK_REFINE;
    
}





template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_FillBinned_S1(int i_dev, int L)
{
    if (n_ids[i_dev][L] > 0)
    {
        // Calculate constants.
        ufloat_g_t R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(ufloat_g_t)L);
        int Nprop_i = (int)(R/(ufloat_g_t)(4.0*sqrt(2.0)*dxf_vec[L])) + 1;   // For filling-in the interior.
        int Nprop_d = (int)(R/(ufloat_g_t)(4.0*sqrt(2.0)*dxf_vec[L])) + 1;   // For satisfying the near-wall distance criterion.
        bool hit_max = L==MAX_LEVELS-1;
        
        // Voxelize the solid, filling in all solid cells.
        cudaDeviceSynchronize();
        tic_simple("");
        Cu_Voxelize_S1<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], L, (ufloat_t)dxf_vec[L], n_maxcblocks,
            c_cblock_f_X[i_dev], c_cblock_ID_mask[i_dev], c_cblock_level[i_dev],
            geometry->c_binned_face_ids_n_3D[i_dev], geometry->G_BIN_DENSITY
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", Voxelize1"; toc_simple("",T_US,1);
        tic_simple("");
        Cu_Voxelize_S2<ufloat_t,ufloat_g_t,AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
            L, n_maxcblocks, id_max[i_dev][L],
            c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", Voxelize2"; toc_simple("",T_US,1);
        tic_simple("");
        Cu_Voxelize_S3_V1<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L], hit_max,
            c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
            geometry->n_faces[i_dev], geometry->n_faces_a[i_dev], geometry->c_geom_f_face_X[i_dev], geometry->c_geom_f_face_Xt[i_dev],
            geometry->c_binned_face_ids_n_2D[i_dev], geometry->c_binned_face_ids_N_2D[i_dev], geometry->c_binned_face_ids_2D[i_dev], geometry->G_BIN_DENSITY
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", Voxelize3"; toc_simple("",T_US,1);
        
        // Propagate preliminary solid masks within the interior from both sides.
        for (int j = 0; j < 0; j++)
        {
            cudaDeviceSynchronize();
            tic_simple("");
            for (int l = 0; l < 2; l++)
            {
                Cu_Voxelize_Propagate_S1<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
                    l%2, n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, (ufloat_t)4.0*dxf_vec[L],
                    c_cells_ID_mask[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev]
                );
                Cu_Voxelize_Propagate_S2<ufloat_t,ufloat_g_t,AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                    id_max[i_dev][L], L, n_maxcblocks,
                    c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
                );
            }
            Cu_Voxelize_Propagate_S3<ufloat_t,ufloat_g_t,AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK,0,streams[i_dev]>>>(
                id_max[i_dev][L], L, n_maxcblocks,
                c_cblock_ID_mask[i_dev]
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", VoxelizePropagate (" << j << ")"; toc_simple("",T_US,1);
        }
        
        /*
        // Identify the cell-blocks on the boundary of the solid. Mark them for refinement, if eligible.
        tic_simple("");
        Cu_MarkBlocks_MarkBoundary<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
            id_max[i_dev][L], n_maxcblocks, hit_max, L,
            c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", MarkBoundary"; toc_simple("",T_US,1);
        
        // Now, mark blocks adjacent to these solid-boundary cell-blocks for refinement if eligible as well.
        tic_simple("");
        Cu_MarkBlocks_MarkInterior<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
            id_max[i_dev][L], n_maxcblocks, hit_max, L,
            c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", MarkInterior"; toc_simple("",T_US,1);
        
        // Propagate these latter marks until the specified near-wall refinement criterion is approximately reached.
        for (int j = 0; j < Nprop_d; j++)
        {
            tic_simple("");
            Cu_MarkBlocks_Propagate<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
                id_max[i_dev][L], n_maxcblocks, L,
                c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev], c_tmp_1[i_dev], j
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", Propagate (" << j << ")"; toc_simple("",T_US,1);
        }
        
        // Finalize the intermediate marks for refinement.
        tic_simple("");
        Cu_MarkBlocks_Finalize<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
            id_max[i_dev][L], c_cblock_ID_ref[i_dev], c_tmp_1[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", Finalize"; toc_simple("",T_US,1);
        */
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_FillBinned_S2(int i_dev, int L)
{
    if (n_ids[i_dev][L] > 0)
    {
        // Reset one of the intermediate arrays in preparation for copying.
        if (L == 0)
        {
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_tmp_1[i_dev], 0);
            Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_tmp_2[i_dev], 0);
        }
        
        // Update solid-adjacent cell masks and indicate adjacency of blocks to the geometry boundary.
        cudaDeviceSynchronize();
        tic_simple("");
        Cu_MarkBlocks_CheckMasks<AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
            c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev], c_cblock_ID_onb[i_dev],
            c_tmp_1[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_UPDATEMASKS | L=" << L << ", UpdateMasks"; toc_simple("",T_US,1);
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_FillBinned_S2A(int i_dev)
{
    // Declare an 'old' number of solid cell-blocks prior to adjustment via padding.
    int n_solidb_old = 0;
    
    // Compute the number of solid-adjacent cells, and the number of blocks these cells occupy.
    n_solida = thrust::reduce(thrust::device, c_tmp_1_dptr[i_dev], c_tmp_1_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], 0);
    n_solidb = thrust::count_if(thrust::device, c_tmp_1_dptr[i_dev], c_tmp_1_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], is_positive());
    n_solidb_old = n_solidb;
    n_solidb = ((n_solidb + 128) / 128) * 128;
    n_maxcells_b = n_solidb*M_CBLOCK;
    std::cout << "Counted " << n_solida << " cells adjacent to the solid boundary (" << (double)n_solida / (double)n_maxcells << ", in " << n_solidb << " blocks or " << n_maxcells_b << " cells)..." << std::endl;
    
    // If there were solid-adjacent cells, then allocate memory for face-cell linkages.
    if (n_solidb > 0)
    {
        // Allocate memory for the solid cell linkage data.
        cells_ID_mask_b[i_dev] = new int[n_maxcells_b*N_Q_max];
        cells_f_X_b[i_dev] = new ufloat_g_t[n_maxcells_b*N_Q_max];
        cblock_ID_onb_solid[i_dev] = new int[n_maxcblocks];
        gpuErrchk( cudaMalloc((void **)&c_cells_ID_mask_b[i_dev], n_maxcells_b*N_Q_max*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_cells_f_X_b[i_dev], n_maxcells_b*N_Q_max*sizeof(ufloat_g_t)) );
        gpuErrchk( cudaMalloc((void **)&c_cblock_ID_onb_solid[i_dev], n_maxcblocks*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_cblock_ID_face[i_dev], n_solidb*N_Q_max*sizeof(int)) );
        
        // Reset some arrays. Make a device pointer to the new cblock_ID_onb_solid array.
        Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_cblock_ID_onb_solid[i_dev], -1);
        Cu_ResetToValue<<<(M_BLOCK+(n_solidb*N_Q_max)-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_solidb*N_Q_max, c_cblock_ID_face[i_dev], -1);
        thrust::device_ptr<int> *c_cblock_ID_onb_solid_dptr = new thrust::device_ptr<int>[N_DEV];
        c_cblock_ID_onb_solid_dptr[i_dev] = thrust::device_pointer_cast(c_cblock_ID_onb_solid[i_dev]);
        
        // Now create the map from block Ids in their usual order to the correct region in the linkage data arrays.
        // Note: Make sure c_tmp_1 still has the number of solid-adjacent cells for each block.
        //
        // Copy the indices of cell-blocks with a positive number of solid-adjacent cells so that they are contiguous.
        thrust::copy_if(thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], c_tmp_1_dptr[i_dev], c_tmp_2_dptr[i_dev], is_positive());
        //
        // Now scatter the addresses of these copied Ids so that cell-blocks know where to find the data of their solid-adjacent cells
        // in the new arrays.
        thrust::scatter(thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + n_solidb_old, c_tmp_2_dptr[i_dev], c_cblock_ID_onb_solid_dptr[i_dev] );
        
        // Report available memory after these new allocations.
        cudaMemGetInfo(&free_t, &total_t);
        std::cout << "[-] After allocations:\n";
        std::cout << "    Free: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_IdentifyFaces(int i_dev, int L)
{
    // Use the solver-defined face-cell link calculation procedure (might move this back to the Mesh since it doesn't seem to depend on the solver).
    if (n_solidb > 0)
    {
        cudaDeviceSynchronize();
        tic_simple("");
        solver->S_IdentifyFaces(0,L);
        cudaDeviceSynchronize();
        std::cout << "MESH_IDENTIFYFACES | L=" << L << ", IdentifyFaces"; toc_simple("",T_US,1);
    }
    
    return 0;
}
