/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/


template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_V1_WARP
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int n_faces,
    const int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    const ufloat_g_t *__restrict__ geom_f_face_Xt,
    const int *__restrict__ binned_face_ids_n_3D,
    const int *__restrict__ binned_face_ids_N_3D,
    const int *__restrict__ binned_face_ids_3D,
    const int G_BIN_DENSITY
)
{
    constexpr int N_Q_max = AP->N_Q_max;
    constexpr int N_DIM = AP->N_DIM;
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
        // Compute cell coordinates.
        int I = tid % 4;
        int J = (tid / 4) % 4;
        int K = 0;
        if (N_DIM==3)
            K = (tid / 4) / 4;
        
        // Compute cell coordinates.
        vec3<ufloat_g_t> vp
        (
            cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L,
            cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L,
            (N_DIM==2) ? (ufloat_g_t)0.0 : cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L
        );
        
        // Note: For warp-level operation, I assume that the second set of cells are in the same bin.
        // Compute global bin index for this thread.
        int Ibx = (int)(vp.x*G_BIN_DENSITY);
        int Iby = (int)(vp.y*G_BIN_DENSITY);
        int Ibz = 0;
        if (N_DIM==3)
            Ibz = (int)(vp.z*G_BIN_DENSITY);
        int global_bin_id = Ibx + G_BIN_DENSITY*Iby + G_BIN_DENSITY*G_BIN_DENSITY*Ibz;
        
        // Initialize trackers for minimal face-distance.
        int pmin = -1;
        ufloat_g_t dmin = (ufloat_g_t)1.0;
        ufloat_g_t dotmin = (ufloat_g_t)1.0;
        int pmin2 = -1;
        ufloat_g_t dmin2 = (ufloat_g_t)1.0;
        ufloat_g_t dotmin2 = (ufloat_g_t)1.0;
        int n_f = binned_face_ids_n_3D[global_bin_id];
        
        // If bin is nonempty, traverse the faces and get the signed distance to the closest face.
        // Only consider faces within a distance of dx (these would be adjacent to the surface).
        if (n_f > 0)
        {
            int N_f = binned_face_ids_N_3D[global_bin_id];
            for (int p = 0; p < n_f; p++)
            {
                int f_p = binned_face_ids_3D[N_f+p];
                vec3<ufloat_g_t> v1
                (
                    geom_f_face_X[f_p + 0*n_faces_a],
                    geom_f_face_X[f_p + 1*n_faces_a],
                    geom_f_face_X[f_p + 2*n_faces_a]
                );
                vec3<ufloat_g_t> v2
                (
                    geom_f_face_X[f_p + 3*n_faces_a],
                    geom_f_face_X[f_p + 4*n_faces_a],
                    geom_f_face_X[f_p + 5*n_faces_a]
                );
                vec3<ufloat_g_t> v3
                (
                    geom_f_face_X[f_p + 6*n_faces_a],
                    geom_f_face_X[f_p + 7*n_faces_a],
                    geom_f_face_X[f_p + 8*n_faces_a]
                );
                vec3<ufloat_g_t> n = FaceNormalUnit<ufloat_g_t,N_DIM>(v1,v2,v3);
                
                // Account for all directions within the cell-neighbor halo.
                vec3<ufloat_g_t> ray
                (
                    static_cast<ufloat_g_t>(1.0),
                    static_cast<ufloat_g_t>(0.0),
                    static_cast<ufloat_g_t>(0.0)
                );
                for (int w = 0; w < 2; w++)
                {
                    if (w == 1)
                        vp.z += static_cast<ufloat_g_t>(2.0)*dx_L;
                    
                    ufloat_g_t d = DotV(v1-vp,n) / DotV(ray,n);
                    vec3<ufloat_g_t> vi = vp + ray*d;
                    {
                        d = Tabs(d);
                        if (w == 0)
                        {
                            if (d < dx_L && (d < dmin || pmin == -1) && CheckPointInTriangleA(vi,v1,v2,v3,n))
                            {
                                pmin = p;
                                dmin = d;
                                dotmin = DotV(vi-vp,n);
                            }
                        }
                        if (w == 1)
                        {
                            if (d < dx_L && (d < dmin2 || pmin2 == -1) && CheckPointInTriangleA(vi,v1,v2,v3,n))
                            {
                                pmin2 = p;
                                dmin2 = d;
                                dotmin2 = DotV(vi-vp,n);
                            }
                        }
                    }
                    
                    if (w == 1)
                        vp.z -= static_cast<ufloat_g_t>(2.0)*dx_L;
                }
            }
        }
        
        // Now, if there are an even number of intersections, the current cell is in the solid.
        // Otherwise, it is a fluid cell.
        int s_D = 0;
        int cellmask = V_CELLMASK_INTERIOR;
        int cellmask2 = V_CELLMASK_INTERIOR;
        if (pmin != -1)
        {
            if (dotmin >= 0)
                cellmask = V_CELLMASK_SOLID;
            else
                cellmask = V_CELLMASK_DUMMY_I;
            
            s_D = 1;
        }
        if (pmin2 != -1)
        {
            if (dotmin2 >= 0)
                cellmask2 = V_CELLMASK_SOLID;
            else
                cellmask2 = V_CELLMASK_DUMMY_I;
            
            s_D = 1;
        }
        
        // Warp reduction for sum.
        for (int offset = 16; offset > 0; offset /= 2)
            s_D += __shfl_down_sync(0xFFFFFFFF, s_D, offset);
        
        // If at least one cell is solid, update the block mask.
        if (__shfl_sync(0xFFFFFFFF, s_D, 0)>0)
        {
            for (int w = 0; w < 2; w++)
            {
                //if (tid == 0 && w == 0)
                //    cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
                
                if (w == 1)
                    cellmask = cellmask2;
                int s_ID_mask = cellmask;
                
                // Internally propagate the cell mask values to the x-edges of the cell-block.
                for (int l = 0; l < 4; l++)
                {
                    int nbr_mask = __shfl_up_sync(0xFFFFFFFF, s_ID_mask, 1);
                    if (I > 0)
                    {
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_I;
                        if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                            cellmask = V_CELLMASK_SOLID;
                    }
                    nbr_mask = __shfl_down_sync(0xFFFFFFFF, s_ID_mask, 1);
                    if (I < 3)
                    {
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_I;
                        if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                            cellmask = V_CELLMASK_SOLID;
                    }
                    s_ID_mask = cellmask;
                }
                
                // If there are solid masks in this block, place guard in the masks for the propagation.
                cells_ID_mask[i_kap_b*M_CBLOCK + tid + 32*w] = cellmask;
            }
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_V1
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int n_faces,
    const int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    const ufloat_g_t *__restrict__ geom_f_face_Xt,
    const int *__restrict__ binned_face_ids_n_3D,
    const int *__restrict__ binned_face_ids_N_3D,
    const int *__restrict__ binned_face_ids_3D,
    const int G_BIN_DENSITY
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_ID_mask[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x];
    
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
        vec3<ufloat_g_t> vp
        (
            cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L,
            cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L,
            (N_DIM==2) ? (ufloat_g_t)0.0 : cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L
        );
        
        // Compute global bin index for this thread.
        int Ibx = (int)(vp.x*G_BIN_DENSITY);
        int Iby = (int)(vp.y*G_BIN_DENSITY);
        int Ibz = 0;
        if (N_DIM==3)
            Ibz = (int)(vp.z*G_BIN_DENSITY);
        int global_bin_id = Ibx + G_BIN_DENSITY*Iby + G_BIN_DENSITY*G_BIN_DENSITY*Ibz;
        
        // Initialize trackers for minimal face-distance.
        int pmin = -1;
        ufloat_g_t dmin = (ufloat_g_t)1.0;
        ufloat_g_t dotmin = (ufloat_g_t)1.0;
        int n_f = binned_face_ids_n_3D[global_bin_id];
        
        // If bin is nonempty, traverse the faces and get the signed distance to the closest face.
        // Only consider faces within a distance of dx (these would be adjacent to the surface).
        if (n_f > 0)
        {
            int N_f = binned_face_ids_N_3D[global_bin_id];
            for (int p = 0; p < n_f; p++)
            {
                int f_p = binned_face_ids_3D[N_f+p];
                vec3<ufloat_g_t> v1
                (
                    geom_f_face_X[f_p + 0*n_faces_a],
                    geom_f_face_X[f_p + 1*n_faces_a],
                    geom_f_face_X[f_p + 2*n_faces_a]
                );
                vec3<ufloat_g_t> v2
                (
                    geom_f_face_X[f_p + 3*n_faces_a],
                    geom_f_face_X[f_p + 4*n_faces_a],
                    geom_f_face_X[f_p + 5*n_faces_a]
                );
                vec3<ufloat_g_t> v3
                (
                    geom_f_face_X[f_p + 6*n_faces_a],
                    geom_f_face_X[f_p + 7*n_faces_a],
                    geom_f_face_X[f_p + 8*n_faces_a]
                );
                vec3<ufloat_g_t> n = FaceNormalUnit<ufloat_g_t,N_DIM>(v1,v2,v3);
                
                
                
                
                // Voxelize the face into the current cell-block with a triangle-bin overlap test.
                vec3<ufloat_g_t> ray
                (
                    static_cast<ufloat_g_t>(1.0),
                    static_cast<ufloat_g_t>(0.0),
                    static_cast<ufloat_g_t>(0.0)
                );
                ufloat_g_t d = DotV(v1-vp,n) / DotV(ray,n);
                vec3<ufloat_g_t> vi = vp + ray*d;
                {
                    d = Tabs(d);
                    if (d < dx_L && (d < dmin || pmin == -1) && CheckPointInTriangleA(vi,v1,v2,v3,n))
                    {
                        pmin = p;
                        dmin = d;
                        dotmin = DotV(vi-vp,n);
                    }
                }
                
                // If this triangle is the closest to this point, consider the relative orientation if a snapped ray also intersects it.
                /*
                vec3<ufloat_g_t> vi = PointFaceIntersection<ufloat_g_t,N_DIM>(vp,v1,v2,v3,n);
                {
                    vec3<ufloat_g_t> vd = vi-vp;
                    ufloat_g_t d = NormV(vd);
                    if (d < static_cast<ufloat_g_t>(2.0)*dx_L && (d < static_cast<ufloat_g_t>(2.0)*dmin || pmin == -1))
                    {
                        // Convert vd into a snapped ray.
                        vd.Normalize();
                        vd.Snap();
                        
                        // Perform ray cast with the snapped ray. Update intersection point.
                        d = DotV(vd,n);
                        if (Tabs(d) > EPS<ufloat_g_t>())
                        {
                            d = DotV(v1-vp,n) / d;
                            vi = vp + vd*d;
                            d = Tabs(d);
                        
                            // Now only store the result of this snapped ray instead of the nearest-distance ray.
                            if (d < dx_L && (d < dmin || pmin == -1) && CheckPointInTriangleA(vi,v1,v2,v3,n))
                            {
                                pmin = p;
                                dmin = d;
                                dotmin = DotV(vi-vp,n);
                            }
                        }
                    }
                }
                */
            }
        }
        
        // Now, if there are an even number of intersections, the current cell is in the solid.
        // Otherwise, it is a fluid cell.
        s_D[threadIdx.x] = 0;
        int cellmask = V_CELLMASK_INTERIOR;
        if (pmin != -1)
        {
            if (dotmin >= 0)
                cellmask = V_CELLMASK_SOLID;
            else
                cellmask = V_CELLMASK_DUMMY_I;
            
            s_D[threadIdx.x] = 1;
        }
        s_ID_mask[threadIdx.x] = cellmask;
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
        
        if (s_D[0]>0)
        {
            // If at least one cell is solid, update the block mask.
            //if (threadIdx.x == 0)
            //    cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
            
            // Internally propagate the cell mask values to the x-edges of the cell-block.
            for (int l = 0; l < 9; l++)
            {
                if (I > 0)
                {
                    int nbr_mask = s_ID_mask[(I-1) + 4*J + 16*K];
                    if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                        cellmask = V_CELLMASK_DUMMY_I;
                    if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                        cellmask = V_CELLMASK_SOLID;
                }
                if (I < 3)
                {
                    int nbr_mask = s_ID_mask[(I+1) + 4*J + 16*K];
                    if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                        cellmask = V_CELLMASK_DUMMY_I;
                    if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                        cellmask = V_CELLMASK_SOLID;
                }
                s_ID_mask[threadIdx.x] = cellmask;
                __syncthreads();
            }
            
            // If there are solid masks in this block, place guard in the masks for the propagation.
            cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = cellmask;
        }
    }
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
__global__
void Cu_Voxelize_V2_WARP
(
    const int n_ids_idev_L,
    const int *__restrict__ id_set_idev_L,
    const int n_maxcblocks,
    const ufloat_t dx_L,
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int n_faces,
    const int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    const ufloat_g_t *__restrict__ geom_f_face_Xt,
    const int *__restrict__ binned_face_ids_n_3D,
    const int *__restrict__ binned_face_ids_N_3D,
    const int *__restrict__ binned_face_ids_3D,
    const int G_BIN_DENSITY
)
{
    constexpr int M_WARP = 32;
    constexpr int N_DIM = AP->N_DIM;
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
        // Compute cell coordinates.
        int I = tid % 4;
        int J = (tid / 4) % 4;
        int K = 0;
        if (N_DIM==3)
            K = (tid / 4) / 4;
        
        // Compute cell coordinates.
        vec3<ufloat_g_t> vp
        (
            cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L,
            cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L,
            (N_DIM==2) ? (ufloat_g_t)0.0 : cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L
        );
        
        // For each face, check if the current cell is within the appropriate bounds. If at least
        // one condition is satisfied, exit the loop and make a note.
        int global_bin_id = (int)(vp.x*G_BIN_DENSITY) + G_BIN_DENSITY*((int)(vp.y*G_BIN_DENSITY)) + G_BIN_DENSITY*G_BIN_DENSITY*((int)(vp.z*G_BIN_DENSITY));
        
        // Now find the total number of intersections a ray makes in the direction with the smallest number of bins.
        int n_f = binned_face_ids_n_3D[global_bin_id];
        int pmin = -1;
        ufloat_g_t dmin = (ufloat_t)1.0;
        ufloat_g_t dotmin = (ufloat_t)0.0;
        int pmin2 = -1;
        ufloat_g_t dmin2 = (ufloat_t)1.0;
        ufloat_g_t dotmin2 = (ufloat_t)0.0;
        if (n_f > 0)
        {
            int N_f = binned_face_ids_N_3D[global_bin_id];
            for (int j = 0; j < n_f/M_WARP+1; j++)
            {
                // Read the next M_TBLOCK faces.
                int plim = M_WARP;
                int s_fI = -1;
                if ((j+1)*M_WARP >= n_f)
                    plim = M_WARP - ((j+1)*M_WARP - n_f);
                if (tid < plim)
                    s_fI = binned_face_ids_3D[N_f + j*M_WARP + tid];
                __syncthreads();
                
                for (int p = 0; p < plim; p++)
                {
                    int f_p = __shfl_sync(0xFFFFFFFF, s_fI, p);
                    
                    // Load face data.
                    ufloat_g_t s_fD = -1;
                    if (tid < 16)
                        s_fD = geom_f_face_Xt[tid + f_p*16];
                    
                    // Load face data.
                    vec3<ufloat_g_t> v1(__shfl_sync(0xFFFFFFFF, s_fD, 0), __shfl_sync(0xFFFFFFFF, s_fD, 1), __shfl_sync(0xFFFFFFFF, s_fD, 2));
                    vec3<ufloat_g_t> v2(__shfl_sync(0xFFFFFFFF, s_fD, 3), __shfl_sync(0xFFFFFFFF, s_fD, 4), __shfl_sync(0xFFFFFFFF, s_fD, 5));
                    vec3<ufloat_g_t> v3(__shfl_sync(0xFFFFFFFF, s_fD, 6), __shfl_sync(0xFFFFFFFF, s_fD, 7), __shfl_sync(0xFFFFFFFF, s_fD, 8));
                    vec3<ufloat_g_t> n(__shfl_sync(0xFFFFFFFF, s_fD, 9), __shfl_sync(0xFFFFFFFF, s_fD, 10), __shfl_sync(0xFFFFFFFF, s_fD, 11));
                    
                    // Find the distance along a ray with direction [1,0,0].
                    {
                        vec3<ufloat_g_t> ray
                        (
                            static_cast<ufloat_g_t>(1.0),
                            static_cast<ufloat_g_t>(0.0),
                            static_cast<ufloat_g_t>(0.0)
                        );
                        
                        for (int w = 0; w < 2; w++)
                        {
                            if (w == 1)
                                vp.z += static_cast<ufloat_g_t>(2.0)*dx_L;
                            
                            ufloat_g_t d = DotV(v1-vp,n) / DotV(ray,n);
                            vec3<ufloat_g_t> vi = vp + ray*d;
                            d = Tabs(d);
                            if (w == 0)
                            {
                                if (d < dx_L && (d < dmin || pmin == -1) && CheckPointInTriangleA(vi,v1,v2,v3,n))
                                {
                                    pmin = p;
                                    dmin = d;
                                    dotmin = DotV(vi-vp,n);
                                }
                            }
                             if (w == 1)
                            {
                                if (d < dx_L && (d < dmin2 || pmin2 == -1) && CheckPointInTriangleA(vi,v1,v2,v3,n))
                                {
                                    pmin2 = p;
                                    dmin2 = d;
                                    dotmin2 = DotV(vi-vp,n);
                                }
                            }
                            
                            if (w == 1)
                                vp.z -= static_cast<ufloat_g_t>(2.0)*dx_L;
                        }
                    }
                }
            }
        }
        
        // Now, if there are an even number of intersections, the current cell is in the solid.
        // Otherwise, it is a fluid cell.
        int s_D = 0;
        int cellmask = V_CELLMASK_INTERIOR;
        int cellmask2 = V_CELLMASK_INTERIOR;
        if (pmin != -1)
        {
            if (dotmin >= 0)
                cellmask = V_CELLMASK_SOLID;
            else
                cellmask = V_CELLMASK_DUMMY_I;
            
            s_D = 1;
        }
        if (pmin2 != -1)
        {
            if (dotmin2 >= 0)
                cellmask2 = V_CELLMASK_SOLID;
            else
                cellmask2 = V_CELLMASK_DUMMY_I;
            
            s_D = 1;
        }
        
        // Warp reduction for sum.
        for (int offset = 16; offset > 0; offset /= 2)
            s_D += __shfl_down_sync(0xFFFFFFFF, s_D, offset);
        
        // If at least one cell is solid, update the block mask.
        if (__shfl_sync(0xFFFFFFFF, s_D, 0)>0)
        {
            for (int w = 0; w < 2; w++)
            {
                //if (tid == 0 && w == 0)
                //    cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
                
                if (w == 1)
                    cellmask = cellmask2;
                int s_ID_mask = cellmask;
                
                // Internally propagate the cell mask values to the x-edges of the cell-block.
                for (int l = 0; l < 4; l++)
                {
                    int nbr_mask = __shfl_up_sync(0xFFFFFFFF, s_ID_mask, 1);
                    if (I > 0)
                    {
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_I;
                        if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                            cellmask = V_CELLMASK_SOLID;
                    }
                    nbr_mask = __shfl_down_sync(0xFFFFFFFF, s_ID_mask, 1);
                    if (I < 3)
                    {
                        if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                            cellmask = V_CELLMASK_DUMMY_I;
                        if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                            cellmask = V_CELLMASK_SOLID;
                    }
                    s_ID_mask = cellmask;
                }
                
                // If there are solid masks in this block, place guard in the masks for the propagation.
                cells_ID_mask[i_kap_b*M_CBLOCK + tid + 32*w] = cellmask;
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
    int *__restrict__ cells_ID_mask,
    int *__restrict__ cblock_ID_mask,
    const ufloat_t *__restrict__ cblock_f_X,
    const int *__restrict__ cblock_ID_nbr,
    const int n_faces,
    const int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    const ufloat_g_t *__restrict__ geom_f_face_Xt,
    const int *__restrict__ binned_face_ids_n_3D,
    const int *__restrict__ binned_face_ids_N_3D,
    const int *__restrict__ binned_face_ids_3D,
    const int G_BIN_DENSITY
)
{
    constexpr int N_DIM = AP->N_DIM;
    constexpr int M_TBLOCK = AP->M_TBLOCK;
    constexpr int M_CBLOCK = AP->M_CBLOCK;
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_fI[M_TBLOCK];
    __shared__ ufloat_g_t s_fD[16];
    __shared__ int s_ID_mask[M_TBLOCK];
    
    s_fI[threadIdx.x] = -1;    
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
        vec3<ufloat_g_t> vp
        (
            cblock_f_X[i_kap_b + 0*n_maxcblocks] + I*dx_L + 0.5*dx_L,
            cblock_f_X[i_kap_b + 1*n_maxcblocks] + J*dx_L + 0.5*dx_L,
            (N_DIM==2) ? (ufloat_g_t)0.0 : cblock_f_X[i_kap_b + 2*n_maxcblocks] + K*dx_L + 0.5*dx_L
        );
        
        // For each face, check if the current cell is within the appropriate bounds. If at least
        // one condition is satisfied, exit the loop and make a note.
        int global_bin_id = (int)(vp.x*G_BIN_DENSITY) + G_BIN_DENSITY*((int)(vp.y*G_BIN_DENSITY)) + G_BIN_DENSITY*G_BIN_DENSITY*((int)(vp.z*G_BIN_DENSITY));
        
        // Now find the total number of intersections a ray makes in the direction with the smallest number of bins.
        //s_fI[threadIdx.x] = binned_face_ids[global_bin_id+threadIdx.x];
        int n_f = binned_face_ids_n_3D[global_bin_id];
        int pmin = -1;
        ufloat_g_t dmin = (ufloat_t)1.0;
        ufloat_g_t dotmin = (ufloat_t)0.0;
        if (n_f > 0)
        {
            int N_f = binned_face_ids_N_3D[global_bin_id];
            for (int j = 0; j < n_f/M_TBLOCK+1; j++)
            {
                // Read the next M_TBLOCK faces.
                int plim = M_TBLOCK;
                if ((j+1)*M_TBLOCK >= n_f)
                    plim = M_TBLOCK - ((j+1)*M_TBLOCK - n_f);
                if (threadIdx.x < plim)
                    s_fI[threadIdx.x] = binned_face_ids_3D[N_f + j*M_TBLOCK + threadIdx.x];
                __syncthreads();
                
                for (int p = 0; p < plim; p++)
                {
                    //int f_p = binned_face_ids[N_f+p];
                    int f_p = s_fI[p];
                    
                    // Load face data.
                    if (threadIdx.x < 16)
                        s_fD[threadIdx.x] = geom_f_face_Xt[threadIdx.x + f_p*16];
                    __syncthreads();
                    
                    // Load face data.
                    vec3<ufloat_g_t> v1(s_fD[0], s_fD[1], s_fD[2]);
                    vec3<ufloat_g_t> v2(s_fD[3], s_fD[4], s_fD[5]);
                    vec3<ufloat_g_t> v3(s_fD[6], s_fD[7], s_fD[8]);
                    vec3<ufloat_g_t> n (s_fD[9], s_fD[10], s_fD[11]);
                    
                    // Find the distance along a ray with direction [1,0,0].
                    {
                        vec3<ufloat_g_t> ray
                        (
                            static_cast<ufloat_g_t>(1.0),
                            static_cast<ufloat_g_t>(0.0),
                            static_cast<ufloat_g_t>(0.0)
                        );
                        ufloat_g_t d = DotV(v1-vp,n) / DotV(ray,n);
                        vec3<ufloat_g_t> vi = vp + ray*d;
                        {
                            d = Tabs(d);
                            if (d < dx_L && (d < dmin || pmin == -1) && CheckPointInTriangleA(vi,v1,v2,v3,n))
                            {
                                pmin = p;
                                dmin = d;
                                dotmin = DotV(vi-vp,n);
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
        int cellmask = V_CELLMASK_INTERIOR;
        if (pmin != -1)
        {
            if (dotmin >= 0)
                cellmask = V_CELLMASK_SOLID;
            else
                cellmask = V_CELLMASK_DUMMY_I;
            
            s_D[threadIdx.x] = 1;
        }
        s_ID_mask[threadIdx.x] = cellmask;
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
        
        if (s_D[0]>0)
        {
            // If at least one cell is solid, update the block mask.
            //if (threadIdx.x == 0)
            //    cblock_ID_mask[i_kap_b] = V_BLOCKMASK_SOLID;
            
            // Internally propagate the cell mask values to the x-edges of the cell-block.
            for (int l = 0; l < 9; l++)
            {
                if (I > 0)
                {
                    int nbr_mask = s_ID_mask[(I-1) + 4*J + 16*K];
                    if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                        cellmask = V_CELLMASK_DUMMY_I;
                    if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                        cellmask = V_CELLMASK_SOLID;
                }
                if (I < 3)
                {
                    int nbr_mask = s_ID_mask[(I+1) + 4*J + 16*K];
                    if (nbr_mask == V_CELLMASK_DUMMY_I && cellmask == V_CELLMASK_INTERIOR)
                        cellmask = V_CELLMASK_DUMMY_I;
                    if (nbr_mask == V_CELLMASK_SOLID && cellmask != V_CELLMASK_DUMMY_I)
                        cellmask = V_CELLMASK_SOLID;
                }
                s_ID_mask[threadIdx.x] = cellmask;
                __syncthreads();
            }
            
            // If there are solid masks in this block, place guard in the masks for the propagation.
            cells_ID_mask[i_kap_b*M_CBLOCK + threadIdx.x] = cellmask;
        }
    }
}
