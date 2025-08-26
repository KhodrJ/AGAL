/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

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
                if (tid == 0 && w == 0)
                    cblock_ID_mask[i_kap_b] = global_bin_id;
                
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
    constexpr int N_Q_max = AP->N_Q_max;
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
            if (threadIdx.x == 0)
                cblock_ID_mask[i_kap_b] = global_bin_id;
            
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
                if (tid == 0 && w == 0)
                    cblock_ID_mask[i_kap_b] = global_bin_id;
                
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
            if (threadIdx.x == 0)
                cblock_ID_mask[i_kap_b] = global_bin_id;
            
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
void Cu_Voxelize_Propagate_WARP
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
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_ID_mask[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x*M_LBLOCK < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x*M_LBLOCK];
    
    // Get the left neighbor.
    int nbr_start_left;
    if (N_DIM==2) nbr_start_left = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
    else          nbr_start_left = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1 && nbr_start_left < 0)
    {
        // Compute cell coordinates.
        int I = threadIdx.x % 4;
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
        while (nbr_right > -1 && k < 100)
        {
            int cellmask = cells_ID_mask[nbr_right*M_CBLOCK + threadIdx.x];
            if (status == V_CELLMASK_SOLID && cellmask == V_CELLMASK_INTERIOR)
                cellmask = V_CELLMASK_SOLID;
            s_ID_mask[threadIdx.x] = cellmask;
            cells_ID_mask[nbr_right*M_CBLOCK + threadIdx.x] = cellmask;
            status = s_ID_mask[3 + 4*J + 16*K];
            
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
void Cu_Voxelize_Propagate
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
    constexpr int M_LBLOCK = AP->M_LBLOCK;
    __shared__ int s_D[M_TBLOCK];
    __shared__ int s_ID_mask[M_TBLOCK];
    
    int i_kap_b = -1;
    if (blockIdx.x*M_LBLOCK < n_ids_idev_L)
        i_kap_b = id_set_idev_L[blockIdx.x*M_LBLOCK];
    
    // Get the left neighbor.
    int nbr_start_left;
    if (N_DIM==2) nbr_start_left = cblock_ID_nbr[i_kap_b + 3*n_maxcblocks];
    else          nbr_start_left = cblock_ID_nbr[i_kap_b + 2*n_maxcblocks];
    
    // If cell-block Id is valid.
    if (i_kap_b > -1 && nbr_start_left < 0)
    {
        // Compute cell coordinates.
        int I = threadIdx.x % 4;
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
        while (nbr_right > -1 && k < 100)
        {
            int cellmask = cells_ID_mask[nbr_right*M_CBLOCK + threadIdx.x];
            if (status == V_CELLMASK_SOLID && cellmask == V_CELLMASK_INTERIOR)
                cellmask = V_CELLMASK_SOLID;
            s_ID_mask[threadIdx.x] = cellmask;
            cells_ID_mask[nbr_right*M_CBLOCK + threadIdx.x] = cellmask;
            status = s_ID_mask[3 + 4*J + 16*K];
            
            // Get next neighbor block.
            nbr_right = cblock_ID_nbr[nbr_right + 1*n_maxcblocks];
            k++;
        }
        
        if (k == 100)
            printf("MAX REACHED DURING PROPAGATION...\n");
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
        int L_bin = std::min(L, geometry->bins->n_levels-1);
        
        // Voxelize the solid, filling in all solid cells.
        tic_simple("");
        Cu_Voxelize_V2_WARP<ufloat_t,ufloat_g_t,AP> <<<(32+(32*n_ids[i_dev][L])-1)/32,32,0,streams[i_dev]>>>(
        //Cu_Voxelize_V1<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L],
            c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
            geometry->n_faces, geometry->n_faces_a, geometry->c_geom_f_face_X, geometry->c_geom_f_face_Xt,
            geometry->bins->c_binned_face_ids_n_3D[0], geometry->bins->c_binned_face_ids_N_3D[0], geometry->bins->c_binned_face_ids_3D[0], geometry->bins->n_bin_density[0]
        );
        cudaDeviceSynchronize();
        thrust::device_ptr<int> mask_ptr = thrust::device_pointer_cast(c_cells_ID_mask[i_dev]);
        std::cout << "Counted " << thrust::count_if(thrust::device, mask_ptr, mask_ptr + id_max[i_dev][L]*M_CBLOCK, is_equal_to(V_CELLMASK_SOLID)) << " solid cells..." << std::endl;
        std::cout << "MESH_VOXELIZE | L=" << L << ", Voxelize"; toc_simple("",T_US,1);
        
        // Propagate preliminary solid masks within the interior from both sides.
        tic_simple("");
        Cu_Voxelize_Propagate<ufloat_t,ufloat_g_t,AP> <<<(M_LBLOCK+n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
            c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", VoxelizePropagate"; toc_simple("",T_US,1);
        
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
