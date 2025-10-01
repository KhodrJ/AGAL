/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry_bins.h"

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_ComputeVoxelRayIndicators_MD ]==========================================  */
/*                                                                                    */
/*  This version of the "ray-indicator" kernel loops over cell-centers defined by     */
/*  the bounding box of each face and casts rays in all surrounding directions to     */
/*  see if the faces will need to be considered during link-length calculation. A     */
/*  triangle for which no ray casts produce intersections will never be seen by       */
/*  voxels, so they can be filtered out safely.                                       */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, int N_DIM>
__global__
void Cu_ComputeVoxelRayIndicators_MD
(
    const ufloat_g_t dx_L,
    const ufloat_g_t dx_Lo2,
    const ufloat_g_t Nx,
    const ufloat_g_t Ny,
    const ufloat_g_t Nz,
    const long int n_faces,
    const long int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    int *__restrict__ ray_indicators,
    const int NVDP=16
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < n_faces)
    {
        // Load face vertices.
        vec3<ufloat_g_t> v1, v2, v3;
        LoadFaceData<ufloat_g_t,FaceArrangement::SoA>(kap, geom_f_face_X, NVDP, n_faces_a, v1, v2, v3);
        vec3<ufloat_g_t> n = FaceNormalUnit<ufloat_g_t,N_DIM>(v1,v2,v3);
        
        // 
        bool found = false;
        if (N_DIM==2)
        {
            constexpr int N_Q_max = 9;
            
            // Compute bounding box limits.
            int ixmin = static_cast<int>( Tmin(v1.x,v2.x) * Nx);
            int iymin = static_cast<int>( Tmin(v1.y,v2.y) * Ny);
            int ixmax = static_cast<int>( Tmax(v1.x,v2.x) * Nx);
            int iymax = static_cast<int>( Tmax(v1.y,v2.y) * Ny);
            
            // Loop over all possible cells in the bounding box.
            for (int p = 1; p < N_Q_max; p++)
            {
                // Consider rays in half the possible directions. Both senses will be accounted for.
                if (p==1||(p+1)%3==0)
                {
                    for (int iy = iymin; iy <= iymax; iy++)
                    for (int ix = ixmin; ix <= ixmax; ix++)
                    {
                        vec3<ufloat_g_t> vp
                        (
                            dx_Lo2 + dx_L*static_cast<ufloat_g_t>(ix),
                            dx_Lo2 + dx_L*static_cast<ufloat_g_t>(iy),
                            static_cast<ufloat_g_t>(0.0)
                        );
                        
                        vec3<ufloat_g_t> ray
                        (
                            static_cast<ufloat_g_t>(V_CONN_ID[p+0*27]),
                            static_cast<ufloat_g_t>(V_CONN_ID[p+1*27]),
                            static_cast<ufloat_g_t>(0.0)
                        );
                        ufloat_g_t d = DotV(ray,n);
                        if (d != static_cast<ufloat_g_t>(0.0))
                        {
                            d = DotV(v1-vp,n) / d;
                            vec3<ufloat_g_t> vi = vp + ray*d;
                            
                            if (CheckPointInLineExtended(vi,v1,v2))
                            {
                                found = true;
                                ixmax = ixmin-1;
                                iymax = iymin-1;
                            }
                        }
                    }
                }
                
                if (found)
                    break;
            }
        }
        else // N_DIM==3
        {
            constexpr int N_Q_max = 27;
            
            // Compute bounding box limits.
            int ixmin = static_cast<int>( (Tmin(Tmin(v1.x,v2.x),v3.x)) * Nx);
            int iymin = static_cast<int>( (Tmin(Tmin(v1.y,v2.y),v3.y)) * Ny);
            int izmin = static_cast<int>( (Tmin(Tmin(v1.z,v2.z),v3.z)) * Nz);
            int ixmax = static_cast<int>( (Tmax(Tmax(v1.x,v2.x),v3.x)) * Nx);
            int iymax = static_cast<int>( (Tmax(Tmax(v1.y,v2.y),v3.y)) * Ny);
            int izmax = static_cast<int>( (Tmax(Tmax(v1.z,v2.z),v3.z)) * Nz);
            
            // Loop over all possible cells in the bounding box.
            for (int p = 1; p < N_Q_max; p++)
            {
                // Consider rays in half the possible directions. Both senses will be accounted for.
                if (p==26||((p-1)%2==0&&p<25))
                {
                    for (int iz = izmin; iz <= izmax; iz++)
                    for (int iy = iymin; iy <= iymax; iy++)
                    for (int ix = ixmin; ix <= ixmax; ix++)
                    {
                        vec3<ufloat_g_t> vp
                        (
                            dx_Lo2 + dx_L*static_cast<ufloat_g_t>(ix),
                            dx_Lo2 + dx_L*static_cast<ufloat_g_t>(iy),
                            dx_Lo2 + dx_L*static_cast<ufloat_g_t>(iz)
                        );
                        
                        vec3<ufloat_g_t> ray
                        (
                            static_cast<ufloat_g_t>(V_CONN_ID[p+0*27]),
                            static_cast<ufloat_g_t>(V_CONN_ID[p+1*27]),
                            static_cast<ufloat_g_t>(V_CONN_ID[p+2*27])
                        );
                        ufloat_g_t d = DotV(v1-vp,n) / DotV(ray,n);
                        vec3<ufloat_g_t> vi = vp + ray*d;
                        
                        if (CheckPointInTriangleExtended(vi,v1,v2,v3,n))
                        {
                            found = true;
                            ixmax = ixmin-1;
                            iymax = iymin-1;
                            izmax = izmin-1;
                        }
                    }
                }
                
                if (found)
                    break;
            }
        }
        
        // If there was an intersection with one of the cells in the bounding box, consider this face during binning.
        if (found)
            ray_indicators[kap] = kap;
    }
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_ComputeVoxelRayIndicators_1D ]==========================================  */
/*                                                                                    */
/*  This version of the "ray-indicator" kernel loops over cell-centers defined by     */
/*  the yz-plane bounding box of each face and casts a single ray along +x to see     */
/*  if the faces will need to be considered during voxelization. A triangle for       */
/*  which no ray casts produce intersections will never be seen by voxels, so they    */
/*  can be filtered out safely.                                                       */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, int N_DIM>
__global__
void Cu_ComputeVoxelRayIndicators_1D
(
    const ufloat_g_t dx_L,
    const ufloat_g_t dx_Lo2,
    const ufloat_g_t Nx,
    const ufloat_g_t Ny,
    const ufloat_g_t Nz,
    const long int n_faces,
    const long int n_faces_a,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    int *__restrict__ ray_indicators,
    const int NVDP=16
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (kap < n_faces)
    {
        // Load face vertices.
        vec3<ufloat_g_t> v1, v2, v3;
        LoadFaceData<ufloat_g_t,FaceArrangement::SoA>(kap, geom_f_face_X, NVDP, n_faces_a, v1, v2, v3);
        vec3<ufloat_g_t> n = FaceNormalUnit<ufloat_g_t,N_DIM>(v1,v2,v3);
        
        // 
        bool found = false;
        if (N_DIM==2)
        {
            // Compute bounding box limits.
            //int iymin = static_cast<int>( Tround((Tmin(v1.y,v2.y) - dx_Lo2)/dx_L) );
            //int iymax = static_cast<int>( Tround((Tmax(v1.y,v2.y) - dx_Lo2)/dx_L) );
            int iymin = static_cast<int>( Tmin(v1.y,v2.y) * Ny);
            int iymax = static_cast<int>( Tmax(v1.y,v2.y) * Ny);
            
            // Consider rays in half the possible directions. Both senses will be accounted for.
            for (int iy = iymin; iy <= iymax; iy++)
            {
                vec3<ufloat_g_t> vp
                (
                    static_cast<ufloat_g_t>(0.0),
                    dx_Lo2 + dx_L*static_cast<ufloat_g_t>(iy),
                    static_cast<ufloat_g_t>(0.0)
                );
                
                ufloat_g_t d = (v1.x-vp.x) + (v1.y-vp.y)*(n.y/n.x);
                vec3<ufloat_g_t> vi = vp;
                vi.x += d;
                
                if (CheckPointInFaceAABB<ufloat_g_t,2>(vi,v1,v2,v3))
                {
                    found = true;
                    iymax = iymin-1;
                }
            }
        }
        else // N_DIM==3
        {
            // Compute bounding box limits.
            int iymin = static_cast<int>( (Tmin(Tmin(v1.y,v2.y),v3.y)) * Ny);
            int izmin = static_cast<int>( (Tmin(Tmin(v1.z,v2.z),v3.z)) * Nz);
            int iymax = static_cast<int>( (Tmax(Tmax(v1.y,v2.y),v3.y)) * Ny);
            int izmax = static_cast<int>( (Tmax(Tmax(v1.z,v2.z),v3.z)) * Nz);
            
            // Consider a single ray in +x.
            for (int iz = izmin; iz <= izmax; iz++)
            for (int iy = iymin; iy <= iymax; iy++)
            {
                vec3<ufloat_g_t> vp
                (
                    static_cast<ufloat_g_t>(0.0),
                    dx_Lo2 + dx_L*static_cast<ufloat_g_t>(iy),
                    dx_Lo2 + dx_L*static_cast<ufloat_g_t>(iz)
                );
                
                ufloat_g_t d = (v1.x-vp.x) + (v1.y-vp.y)*(n.y/n.x) + (v1.z-vp.z)*(n.z/n.x);
                vec3<ufloat_g_t> vi = vp;
                vi.x += d;
                
                // If a small AABB intersects the triangle, consider it.
                if (CheckPointInFaceAABB<ufloat_g_t,3>(vi,v1,v2,v3))
                {
                    found = true;
                    iymax = iymin-1;
                    izmax = izmin-1;
                }
            }
        }
        
        // If there was an intersection with one of the cells in the bounding box, consider this face during binning.
        if (found)
            ray_indicators[kap] = kap;
    }
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_ComputeBoundingBoxLimits2D ]============================================  */
/*                                                                                    */
/*  Faces are traversed, and the limits of their bounding boxes are computed and      */
/*  truncated to produce indices that map to the uniformly-sized bins. Each face      */
/*  may cross up to 4^(D-1) bins depending on their size and orientation. A           */
/*  duplicate of the current face index is stored in bounding_box_index_limits for    */
/*  each overlapped bin index stored in bounding_box_limits, so that the result       */
/*  after sorting by key is a set of pairs (face,bin) that accounts for bin overlap   */
/*  properly. The 2D version of this kernel does not requiring modifying the bin      */
/*  volume.                                                                           */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, int N_DIM>
__global__
void Cu_ComputeBoundingBoxLimits2D
(
    const long int n_faces,
    const long int n_faces_a,
    const int n_bin_spec,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    int *__restrict__ bounding_box_limits,
    int *__restrict__ bounding_box_index_limits,
    const ufloat_g_t dx,
    const ufloat_g_t Lx,
    const ufloat_g_t Ly,
    const ufloat_g_t Lz,
    const int G_BIN_DENSITY,
    const int *__restrict__ ray_indicators,
    const bool use_ray,
    const bool use_map,
    const int n_filtered,
    const int NVDP=16
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int map_val = kap;
    
    // If using rays, get the current indicator value.
    if (use_ray && kap < n_faces)
        map_val = ray_indicators[kap];
    
    if (kap < n_faces && map_val > -1)
    {
        // Load face data.
        vec3<ufloat_g_t> v1, v2, v3;
        LoadFaceData<ufloat_g_t,FaceArrangement::SoA>(kap, geom_f_face_X, NVDP, n_faces_a, v1, v2, v3);
        
        if (N_DIM==2)
        {
            ufloat_g_t vBx_m = fmin(v1.x, v2.x);
            ufloat_g_t vBx_M = fmax(v1.x, v2.x);
            ufloat_g_t vBy_m = fmin(v1.y, v2.y);
            ufloat_g_t vBy_M = fmax(v1.y, v2.y);
            
            // C is used to determine if a face is completely outside of the bounding box.
            bool C = true;
            if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
                C = false;
            if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
                C = false;
            
            int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY);
            int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY);
            
            // Note: this assumes that the faces intersect, at most, eight bins.
            int counter = 0;
            for (int J = bin_id_yl; J < bin_id_yL+1; J++)
            {
                if (C && counter < n_bin_spec)
                {
                    int global_id = J;
                    if (use_map)
                    {
                        bounding_box_limits[map_val + counter*n_filtered] = global_id;
                        bounding_box_index_limits[map_val + counter*n_filtered] = kap;
                    }
                    else
                    {
                        bounding_box_limits[kap + counter*n_faces] = global_id;
                        bounding_box_index_limits[kap + counter*n_faces] = kap;
                    }
                    counter++;
                }
            }
        }
        else // N_DIM==3
        {
            ufloat_g_t vBx_m = fmin(fmin(v1.x, v2.x), v3.x);
            ufloat_g_t vBx_M = fmax(fmax(v1.x, v2.x), v3.x);
            ufloat_g_t vBy_m = fmin(fmin(v1.y, v2.y), v3.y);
            ufloat_g_t vBy_M = fmax(fmax(v1.y, v2.y), v3.y);
            ufloat_g_t vBz_m = fmin(fmin(v1.z, v2.z), v3.z);
            ufloat_g_t vBz_M = fmax(fmax(v1.z, v2.z), v3.z);
            
            // C is used to determine if a face is completely outside of the bounding box.
            bool C = true;
            if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
                C = false;
            if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
                C = false;
            if (vBz_m<-dx&&vBz_M<-dx || vBz_m>Lz+dx&&vBz_M>Lz+dx)
                C = false;
            
            int bin_id_yl = (int)(vBy_m*G_BIN_DENSITY);
            int bin_id_zl = (int)(vBz_m*G_BIN_DENSITY);
            int bin_id_yL = (int)(vBy_M*G_BIN_DENSITY);
            int bin_id_zL = (int)(vBz_M*G_BIN_DENSITY);
            
            // Note: this assumes that the faces intersect, at most, eight bins.
            int counter = 0;
            for (int K = bin_id_zl; K < bin_id_zL+1; K++)
            {
                for (int J = bin_id_yl; J < bin_id_yL+1; J++)
                {
                    if (C && counter < n_bin_spec)
                    {
                        int global_id = J + G_BIN_DENSITY*K;
                        if (use_map)
                        {
                            bounding_box_limits[map_val + counter*n_filtered] = global_id;
                            bounding_box_index_limits[map_val + counter*n_filtered] = kap;
                        }
                        else
                        {
                            bounding_box_limits[kap + counter*n_faces] = global_id;
                            bounding_box_index_limits[kap + counter*n_faces] = kap;
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ Cu_ComputeBoundingBoxLimits3D ]============================================  */
/*                                                                                    */
/*  Faces are traversed, and the limits of their bounding boxes are computed and      */
/*  truncated to produce indices that map to the uniformly-sized bins. Each face      */
/*  may cross up to 4^D bins depending on their size and orientation. A duplicate     */
/*  of the current face index is stored in bounding_box_index_limits for each         */
/*  overlapped bin index stored in bounding_box_limits, so that the result after      */
/*  sorting by key is a set of pairs (face,bin) that accounts for bin overlap         */
/*  properly. The 3D version of this kernel requires extending the bin volume by an   */
/*  amount dx in each direction to account for cut-links that may cross bins for      */
/*  cells lying directly on a bin boundary.                                           */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_g_t, int N_DIM>
__global__
void Cu_ComputeBoundingBoxLimits3D
(
    const long int n_faces,
    const long int n_faces_a,
    const int n_bin_spec,
    const ufloat_g_t *__restrict__ geom_f_face_X,
    int *__restrict__ bounding_box_limits,
    int *__restrict__ bounding_box_index_limits,
    const ufloat_g_t dx,
    const ufloat_g_t Lx,
    const ufloat_g_t Ly,
    const ufloat_g_t Lz,
    const ufloat_g_t Lx0g,
    const ufloat_g_t Ly0g,
    const ufloat_g_t Lz0g,
    const int G_BIN_DENSITY,
    const int *__restrict__ ray_indicators,
    const bool use_ray,
    const bool use_map,
    const int n_filtered,
    const int NVDP=16
)
{
    int kap = blockIdx.x*blockDim.x + threadIdx.x;
    int map_val = kap;
    
    // If using rays, get the current indicator value.
    if (use_ray && kap < n_faces)
        map_val = ray_indicators[kap];
    
    if (kap < n_faces && map_val > -1)
    {
        // Load face data,
        vec3<ufloat_g_t> v1, v2, v3;
        LoadFaceData<ufloat_g_t,FaceArrangement::SoA>(kap, geom_f_face_X, NVDP, n_faces_a, v1, v2, v3);
        
        if (N_DIM==2)
        {
            vec3<ufloat_g_t> vBm (Tmin(v1.x, v2.x), Tmin(v1.y, v2.y));
            vec3<ufloat_g_t> vBM (Tmax(v1.x, v2.x), Tmax(v1.y, v2.y));
            
            // C is used to determine if a face is completely outside of the bounding box.
            bool C = true;
            if ((vBm.x<-dx && vBM.x<-dx) || (vBm.x>Lx+dx && vBM.x>Lx+dx))
                C = false;
            if ((vBm.y<-dx && vBM.y<-dx) || (vBm.y>Ly+dx && vBM.y>Ly+dx))
                C = false;
            
            if (C)
            {
                int bin_id_xl = (int)((vBm.x-0*dx)*G_BIN_DENSITY)-1;
                int bin_id_yl = (int)((vBm.y-0*dx)*G_BIN_DENSITY)-1;
                int bin_id_xL = (int)((vBM.x+0*dx)*G_BIN_DENSITY)+1;
                int bin_id_yL = (int)((vBM.y+0*dx)*G_BIN_DENSITY)+1;
                
                // Note: this assumes that the faces intersect, at most, eight bins.
                int counter = 0;
                for (int J = bin_id_yl; J < bin_id_yL+1; J++)
                {
                    for (int I = bin_id_xl; I < bin_id_xL+1; I++)
                    {
                        vec3<ufloat_g_t> vm(I*Lx0g-dx,J*Ly0g-dx);
                        vec3<ufloat_g_t> vM((I+1)*Lx0g+dx,(J+1)*Ly0g+dx);
                        C = LineBinOverlap2D(vm,vM,v1,v2);
                        
                        if (C && counter < n_bin_spec)
                        {
                            int global_id = I + G_BIN_DENSITY*J;
                            if (use_map)
                            {
                                bounding_box_limits[map_val + counter*n_filtered] = global_id;
                                bounding_box_index_limits[map_val + counter*n_filtered] = kap;
                            }
                            else
                            {
                                bounding_box_limits[kap + counter*n_faces] = global_id;
                                bounding_box_index_limits[kap + counter*n_faces] = kap;
                            }
                            counter++;
                        }
                    }
                }
            }
        }
        else // N_DIM==3
        {
            vec3<ufloat_g_t> vBm (Tmin(Tmin(v1.x, v2.x), v3.x), Tmin(Tmin(v1.y, v2.y), v3.y), Tmin(Tmin(v1.z, v2.z), v3.z));
            vec3<ufloat_g_t> vBM (Tmax(Tmax(v1.x, v2.x), v3.x), Tmax(Tmax(v1.y, v2.y), v3.y), Tmax(Tmax(v1.z, v2.z), v3.z));
            
            // C is used to determine if a face is completely outside of the bounding box.
            bool C = true;
            if ((vBm.x<-dx&&vBM.x<-dx) || (vBm.x>Lx+dx&&vBM.x>Lx+dx))
                C = false;
            if ((vBm.y<-dx&&vBM.y<-dx) || (vBm.y>Ly+dx&&vBM.y>Ly+dx))
                C = false;
            if ((vBm.z<-dx&&vBM.z<-dx) || (vBm.z>Lz+dx&&vBM.z>Lz+dx))
                C = false;
            
            if (C)
            {
                int bin_id_xl = (int)((vBm.x)*G_BIN_DENSITY)-1;
                int bin_id_yl = (int)((vBm.y)*G_BIN_DENSITY)-1;
                int bin_id_zl = (int)((vBm.z)*G_BIN_DENSITY)-1;
                int bin_id_xL = (int)((vBM.x)*G_BIN_DENSITY)+1;
                int bin_id_yL = (int)((vBM.y)*G_BIN_DENSITY)+1;
                int bin_id_zL = (int)((vBM.z)*G_BIN_DENSITY)+1;
                
                // Note: this assumes that the faces intersect, at most, eight bins.
                int counter = 0;
                for (int K = bin_id_zl; K < bin_id_zL+1; K++)
                {
                    for (int J = bin_id_yl; J < bin_id_yL+1; J++)
                    {
                        for (int I = bin_id_xl; I < bin_id_xL+1; I++)
                        {
                            vec3<ufloat_g_t> vm(I*Lx0g-dx,J*Ly0g-dx,K*Lz0g-dx);
                            vec3<ufloat_g_t> vM((I+1)*Lx0g+dx,(J+1)*Ly0g+dx,(K+1)*Lz0g+dx);
                            C = TriangleBinOverlap3D(vm,vM,v1,v2,v3);
                            
                            if (C && counter < n_bin_spec)
                            {
                                int global_id = I + G_BIN_DENSITY*J + G_BIN_DENSITY*G_BIN_DENSITY*K;
                                if (use_map)
                                {
                                    bounding_box_limits[map_val + counter*n_filtered] = global_id;
                                    bounding_box_index_limits[map_val + counter*n_filtered] = kap;
                                }
                                else
                                {
                                    bounding_box_limits[kap + counter*n_faces] = global_id;
                                    bounding_box_index_limits[kap + counter*n_faces] = kap;
                                }
                                counter++;
                            }
                        }
                    }
                }
            }
        }
    }
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_MakeBinsGPU ]============================================================  */
/*                                                                                    */
/*  Performs a uniform spatial binning of geometry faces inside of the domain in      */
/*  parallel on the GPU. Faces outside of the domain are filtered out. The result     */
/*  is the allocation of memory for and filling of three sets of arrays: 1)           */
/*  c_binned_ids_v/b, a set of contiguous binned faces such that the first batch      */
/*  correspond to the faces of bin 0, the second batch corresponds to bin 1 and so    */
/*  on, 2) c_binned_ids_n_v/b, the sizes of the n_bins_2D/b bins, and 3)              */
/*  c_binned_ids_N_v/b, the starting indices for the faces of each bin in             */
/*  c_binned_ids_v/b. The set of arrays with '_v' corresponds to a 2D binning which   */
/*  enables a raycast algorithm for solid-cell identification. The one with '_b'      */
/*  corresponds to the 3D binning, where the bins are extended in volume by an        */
/*  amount dx specified by the mesh resolution and which is used to restrict the      */
/*  search-space when cells are computing the lengths of cut-links across the         */
/*  geometry.                                                                         */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
template <bool make_2D>
int Geometry<ufloat_t,ufloat_g_t,AP>::Bins::G_MakeBinsGPU(int L)
{
    std::cout << "CONSTRUCTION OF 1D BINS ON LEVEL " << L << std::endl;
    
    // Some constants.
    ufloat_g_t *c_geom_f_face_X = geometry->c_geom_f_face_X;
    long int n_faces = geometry->n_faces;
    long int n_faces_a = geometry->n_faces_a;
    ufloat_g_t Lx0g = Lx0g_vec[L + 0*n_bin_levels];
    ufloat_g_t Ly0g = Lx0g_vec[L + 1*n_bin_levels];
    ufloat_g_t Lz0g = Lx0g_vec[L + 2*n_bin_levels];
    int use_debug = false;   // Indicates to print out binned Ids for debugging.
    int use_ray = true;      // Indicates to use the ray indicators.
    int use_map = true;      // Indicates to use the indicator map array.
    int use_zip = true;      // Indicates to perform compaction before sorting by key.
    
    // Correct optimizations, if necessary.
    if (L == n_bin_levels-1 && n_bin_levels != n_max_levels_wall)
    {
        use_ray = false;
        use_map = false;
        std::cout << "G_BIN_LEVELS < MAX_LEVELS_WALL, not using ray indicator optimization..." << std::endl;
    }
    
    // Proceed only if there are actual faces loaded in the current object.
    if (n_faces > 0)
    {
        // o====================================================================================
        // | Preparation.
        // o====================================================================================
        
        // Compute constants.
        long int n_limits_2D = 1;
        long int n_limits_3D = 1;
        for (int d = 0; d < N_DIM-1; d++) n_limits_2D *= (2+n_bin_spec);
        for (int d = 0; d < N_DIM-0; d++) n_limits_3D *= (2+n_bin_spec);
        long int n_filtered = 0;
        std::cout << "Using a bin radius of " << n_limits_3D << " (" << n_limits_2D << ")..." << std::endl;
        
        // Declare and allocate std::vector<int> bin arrays, which will be updated during traversal.
        n_bins_2D[L] = 1; for (int d = 0; d < N_DIM-1; d++) { n_bins_2D[L] *= n_bin_density[L]; }
        n_bins_3D[L] = 1; for (int d = 0; d < N_DIM-0; d++) { n_bins_3D[L] *= n_bin_density[L]; }
        long int n_lim_size_2D = n_limits_2D*n_faces;
        long int n_lim_size_3D = n_limits_3D*n_faces;
        
        // Some array declarations.
        int *c_ray_indicators;               // Used to filter out faces that never encounter a ray.
        int *c_indicator_map;                // Stored mapped values of ray indicators.
        int *c_bounding_box_limits;          // Stores bin values for the faces.
        int *c_bounding_box_index_limits;    // Stores copies of face values for each bin.
        int *c_tmp_b_i;                      // Used to store unique bin Ids.
        int *c_tmp_b_ii;                     // Used to gather starting-location indices.
        
        // Declare and allocate memory for the c_bounding_box_limits.
        tic_simple("");
        cudaDeviceSynchronize();
        if (use_ray)
            gpuErrchk( cudaMalloc((void **)&c_ray_indicators, n_faces*sizeof(int)) );
        if (use_ray && use_map)
            gpuErrchk( cudaMalloc((void **)&c_indicator_map, n_faces*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_tmp_b_i, n_bins_3D[L]*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_tmp_b_ii, n_bins_3D[L]*sizeof(int)) );
        //
        // Allocate memory for bins sizes and locations.
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_3D[L], n_bins_3D[L]*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_3D[L], n_bins_3D[L]*sizeof(int)) );
        //
        // Reset values.
        if (use_ray)
            Cu_ResetToValue<<<(M_BLOCK+n_faces-1)/M_BLOCK, M_BLOCK>>>(n_faces, c_ray_indicators, -1);
        if (use_ray && use_map)
            Cu_ResetToValue<<<(M_BLOCK+n_faces-1)/M_BLOCK, M_BLOCK>>>(n_faces, c_indicator_map, -1);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_3D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_3D[L], c_binned_face_ids_n_3D[L], 0);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_3D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_3D[L], c_binned_face_ids_N_3D[L], 0);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_3D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_3D[L], c_tmp_b_i, -1);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_3D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_3D[L], c_tmp_b_ii, -1);
        
        // Wrap raw pointers with thrust device_ptr.
        thrust::device_ptr<int> ray_ptr = thrust::device_pointer_cast(c_ray_indicators);
        thrust::device_ptr<int> map_ptr = thrust::device_pointer_cast(c_indicator_map);
        thrust::device_ptr<int> bnb_ptr = thrust::device_pointer_cast(c_binned_face_ids_n_3D[L]);
        thrust::device_ptr<int> bNb_ptr = thrust::device_pointer_cast(c_binned_face_ids_N_3D[L]);
        thrust::device_ptr<int> c_tmp_b_i_ptr = thrust::device_pointer_cast(c_tmp_b_i);
        thrust::device_ptr<int> c_tmp_b_ii_ptr = thrust::device_pointer_cast(c_tmp_b_ii);
        cudaDeviceSynchronize();
        std::cout << "Memory allocation (1): "; toc_simple("",T_US,1);
        
        // Load connectivity arrays in constant GPU memory if not already done.
        if (!init_conn)
            InitConnectivity<AP->N_DIM>();
        
        
        
        // o====================================================================================
        // | Ray-indicator computation.
        // o====================================================================================
        
        if (use_ray)
        {
            std::cout << "RAY COMPUTATION:" << std::endl;
            // STEP 1A: Traverse faces and identify the bins they should go in.
            cudaDeviceSynchronize();
            tic_simple("");
            Cu_ComputeVoxelRayIndicators_1D<ufloat_g_t,AP->N_DIM><<<(M_BLOCK+n_faces-1)/M_BLOCK,M_BLOCK>>>(
                dxf_vec[L], static_cast<ufloat_g_t>(0.5)*dxf_vec[L],
                static_cast<ufloat_g_t>(Nxi_L[L]), static_cast<ufloat_g_t>(Nxi_L[L + 1*n_bin_levels]), static_cast<ufloat_g_t>(Nxi_L[L + 2*n_bin_levels]),
                n_faces, n_faces_a,
                c_geom_f_face_X, c_ray_indicators
            );
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS PRE | L=" << L << ", Computing ray indicators"; toc_simple("",T_US,1);
            
            if (use_map)
            {
                // Step 1B: Collect ray indicators.
                tic_simple("");
                auto filtered_result = thrust::copy_if(thrust::device, ray_ptr, ray_ptr + n_faces, map_ptr, is_nonnegative());
                n_filtered = filtered_result - map_ptr;
                n_lim_size_2D = n_limits_2D*n_filtered;
                n_lim_size_3D = n_limits_3D*n_filtered;
                cudaDeviceSynchronize();
                std::cout << "MAKE_BINS PRE | L=" << L << ", Gathered ray indicators (filtered=" << n_filtered << "/" << n_faces << " [ " << 100.00*(double)n_filtered/(double)n_faces << "%])"; toc_simple("",T_US,1);
                
                // Step 1C: Scatter new ray indicators.
                tic_simple("");
                auto counting_iter = thrust::counting_iterator<int>(0);
                thrust::scatter(
                    thrust::device, counting_iter, counting_iter + n_filtered,
                    map_ptr,
                    ray_ptr
                );
                cudaDeviceSynchronize();
                std::cout << "MAKE_BINS PRE | L=" << L << ", Scattered new ray indicators"; toc_simple("",T_US,1);
            }
        }
        
        // Other memory allocations.
        tic_simple("");
        gpuErrchk( cudaMalloc((void **)&c_bounding_box_limits, n_lim_size_3D*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_bounding_box_index_limits, n_lim_size_3D*sizeof(int)) );
        Cu_ResetToValue<<<(M_BLOCK+n_lim_size_3D-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_3D, c_bounding_box_limits, n_bins_3D[L]);
        Cu_ResetToValue<<<(M_BLOCK+n_lim_size_3D-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_3D, c_bounding_box_index_limits, -1);
        thrust::device_ptr<int> bbi_ptr = thrust::device_pointer_cast(c_bounding_box_index_limits);
        thrust::device_ptr<int> bb_ptr = thrust::device_pointer_cast(c_bounding_box_limits);
        cudaDeviceSynchronize();
        std::cout << "Memory allocation (2): "; toc_simple("",T_US,1);
        
        
        
        // o====================================================================================
        // | 3D binning.
        // o====================================================================================
        
        // STEP 2: Traverse faces and identify the bins they should go in.
        tic_simple("");
        cudaDeviceSynchronize();
        Cu_ComputeBoundingBoxLimits3D<ufloat_g_t,AP->N_DIM><<<(M_BLOCK+n_faces-1)/M_BLOCK,M_BLOCK>>>(
            n_faces, n_faces_a, n_limits_3D,
            c_geom_f_face_X, c_bounding_box_limits, c_bounding_box_index_limits,
            (ufloat_g_t)dxf_vec[L]+EPS<ufloat_g_t>(), (ufloat_g_t)Lx, (ufloat_g_t)Ly, (ufloat_g_t)Lz,
            Lx0g, Ly0g, Lz0g, n_bin_density[L],
            c_ray_indicators,
            use_ray, use_map, n_filtered
        );
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Computing bounding box limits"; toc_simple("",T_US,1);
        
        // STEP 3: If selected, make a zip iterator out of the bounding box limits and index arrays, and then remove all invalid bins.
        // This might speed up the sort-by-key that follows.
        if (use_zip)
        {
            tic_simple("");
            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(bb_ptr, bbi_ptr));
            auto zipped_end = thrust::remove_if(thrust::device, zipped, zipped + n_lim_size_3D, is_equal_to_zip(n_bins_3D[L]));
            n_lim_size_3D = zipped_end - zipped;
            //n_lim_size_3D = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_3D, is_nonnegative_and_less_than(n_bins_3D[L]));
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 3D | L=" << L << ", Compaction"; toc_simple("",T_US,1);
        }
        
        // STEP 4: Sort by key.
        tic_simple("");
        thrust::sort_by_key(thrust::device, bb_ptr, bb_ptr + n_lim_size_3D, bbi_ptr);
        int n_binned_faces_3D = n_lim_size_3D;
        if (!use_zip)
            n_binned_faces_3D = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_3D, is_nonnegative_and_less_than(n_bins_3D[L]));
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Sort by key (n_binned_faces=" << n_binned_faces_3D << ")"; toc_simple("",T_US,1);
        
        // STEP 5: Reduce the keys to get the number of faces in each bin. Scatter them to c_binned_face_ids_n_2D.
        tic_simple("");
        auto result = thrust::reduce_by_key(
            thrust::device, bb_ptr, bb_ptr + n_lim_size_3D,
            thrust::make_constant_iterator(1),
            c_tmp_b_i_ptr,    // stores unique keys
            c_tmp_b_ii_ptr    // stores reduction
        );
        int n_unique_bins_b = result.first - c_tmp_b_i_ptr;
        if (!use_zip)
            n_unique_bins_b = thrust::count_if(thrust::device, c_tmp_b_i_ptr, c_tmp_b_i_ptr + n_bins_3D[L], is_nonnegative_and_less_than(n_bins_3D[L]));
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Reduction (nbins=" << n_unique_bins_b << ") by key"; toc_simple("",T_US,1);
        
        // STEP 6: Scatter the bin sizes.
        tic_simple("");
        thrust::scatter(
            thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_b,
            c_tmp_b_i_ptr,
            bnb_ptr
        );
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Scatter (1)"; toc_simple("",T_US,1);
        
        // STEP 7: Get the difference in the bounding box limits to identify the starting location for the Ids of each individual bin.
        tic_simple("");
        thrust::adjacent_difference(thrust::device, bb_ptr, bb_ptr + n_lim_size_3D, bb_ptr);
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Adjacent difference"; toc_simple("",T_US,1);
        
        // STEP 8: Gather the indices of the starting locations.
        tic_simple("");
        auto counting_iter = thrust::counting_iterator<int>(1);
        thrust::transform(
            thrust::device, counting_iter, counting_iter + n_lim_size_3D,
            bb_ptr, bb_ptr,
            replace_diff_with_indexM1()
        );
        thrust::copy_if(thrust::device, &bb_ptr[1], &bb_ptr[1] + (n_lim_size_3D-1), &c_tmp_b_ii_ptr[1], is_positive());
        int fZ = 0;
        cudaMemcpy(c_tmp_b_ii, &fZ, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Copy-if"; toc_simple("",T_US,1);
        
        // STEP 9: Now scatter the starting-location indices.
        tic_simple("");
        thrust::scatter(
            thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_b,
            c_tmp_b_i_ptr,
            bNb_ptr
        );
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Scatter (2)"; toc_simple("",T_US,1);
        
        // Copy the indices of the binned faces.
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_3D[L], n_binned_faces_3D*sizeof(int)) );
        init_bins_3D = 1;
        cudaMemcpy(c_binned_face_ids_3D[L], c_bounding_box_index_limits, n_binned_faces_3D*sizeof(int), cudaMemcpyDeviceToDevice);
        //
        // Copy the GPU data to the CPU for drawing.
        binned_face_ids_n_3D[L] = new int[n_bins_3D[L]];
        binned_face_ids_N_3D[L] = new int[n_bins_3D[L]];
        binned_face_ids_3D[L] = new int[n_binned_faces_3D];
        cudaMemcpy(binned_face_ids_n_3D[L], c_binned_face_ids_n_3D[L], n_bins_3D[L]*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(binned_face_ids_N_3D[L], c_binned_face_ids_N_3D[L], n_bins_3D[L]*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(binned_face_ids_3D[L], c_binned_face_ids_3D[L], n_binned_faces_3D*sizeof(int), cudaMemcpyDeviceToHost);
        
        // DEBUG
        if (use_debug)
        {
            for (int i = 0; i < n_bins_3D[L]; i++)
            {
                int nbins_i = binned_face_ids_n_3D[L][i];
                int Nbins_i = binned_face_ids_N_3D[L][i];
                if (nbins_i > 0)
                {
                    std::cout << "Bin " << i << ": (nbins=" << nbins_i << ",Nbins=" << Nbins_i << ")" << std::endl;
                    for (int j = 0; j < nbins_i; j++)
                        std::cout << binned_face_ids_3D[L][Nbins_i+j] << " ";
                    std::cout << std::endl;
                }
            }
        }
        
        
        
        // o====================================================================================
        // | 2D binning.
        // o====================================================================================
        
        if (make_2D)
        {
            // Preparation of 2D binning.
            gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_2D[L], n_bins_2D[L]*sizeof(int)) );
            gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_2D[L], n_bins_2D[L]*sizeof(int)) );
            thrust::device_ptr<int> bnv_ptr = thrust::device_pointer_cast(c_binned_face_ids_n_2D[L]);
            thrust::device_ptr<int> bNv_ptr = thrust::device_pointer_cast(c_binned_face_ids_N_2D[L]);
            
            // Reset values for the 2D binning part. I can reuse the temporary allocations since they are always larger than needed.
            tic_simple("");
            cudaDeviceSynchronize();
            Cu_ResetToValue<<<(M_BLOCK+n_bins_2D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_2D[L], c_binned_face_ids_n_2D[L], 0);
            Cu_ResetToValue<<<(M_BLOCK+n_bins_2D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_2D[L], c_binned_face_ids_N_2D[L], 0);
            Cu_ResetToValue<<<(M_BLOCK+n_lim_size_2D-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_2D, c_bounding_box_limits, n_bins_2D[L]);
            Cu_ResetToValue<<<(M_BLOCK+n_lim_size_2D-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_2D, c_bounding_box_index_limits, -1);
            Cu_ResetToValue<<<(M_BLOCK+n_bins_2D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_2D[L], c_tmp_b_i, -1);
            Cu_ResetToValue<<<(M_BLOCK+n_bins_2D[L]-1)/M_BLOCK, M_BLOCK>>>(n_bins_2D[L], c_tmp_b_ii, -1);
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Memory allocation"; toc_simple("",T_US,1);
            
            // STEP 2: Traverse faces and identify the bins they should go in.
            tic_simple("");
            cudaDeviceSynchronize();
            Cu_ComputeBoundingBoxLimits2D<ufloat_g_t,AP->N_DIM><<<(M_BLOCK+n_faces-1)/M_BLOCK,M_BLOCK>>>(
                n_faces, n_faces_a, n_limits_2D,
                c_geom_f_face_X, c_bounding_box_limits, c_bounding_box_index_limits,
                (ufloat_g_t)dxf_vec[L]+EPS<ufloat_g_t>(), (ufloat_g_t)Lx, (ufloat_g_t)Ly, (ufloat_g_t)Lz, n_bin_density[L],
                c_ray_indicators,
                use_ray, use_map, n_filtered
            );
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Computing bounding box limits"; toc_simple("",T_US,1);
            
            // STEP 3: If selected, make a zip iterator out of the bounding box limits and index arrays, and then remove all invalid bins.
            // This might speed up the sort-by-key that follows.
            if (use_zip)
            {
                tic_simple("");
                auto zipped = thrust::make_zip_iterator(thrust::make_tuple(bb_ptr, bbi_ptr));
                auto zipped_end = thrust::remove_if(thrust::device, zipped, zipped + n_lim_size_2D, is_equal_to_zip(n_bins_2D[L]));
                n_lim_size_2D = zipped_end - zipped;
                //n_lim_size_2D = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_2D, is_nonnegative_and_less_than(n_bins_2D[L]));
                cudaDeviceSynchronize();
                std::cout << "MAKE_BINS 2D | L=" << L << ", Compaction"; toc_simple("",T_US,1);
            }
            
            // STEP 4: Sort by key.
            tic_simple("");
            thrust::sort_by_key(thrust::device, bb_ptr, bb_ptr + n_lim_size_2D, bbi_ptr);
            int n_binned_faces_2D = n_lim_size_2D;
            if (!use_zip)
                n_binned_faces_2D = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_2D, is_nonnegative_and_less_than(n_bins_2D[L]));
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Sort by key (n_binned_faces=" << n_binned_faces_2D << ")"; toc_simple("",T_US,1);
            
            // STEP 5: Reduce the keys to get the number of faces in each bin. Scatter them to c_binned_face_ids_n_2D.
            tic_simple("");
            result = thrust::reduce_by_key(
                thrust::device, bb_ptr, bb_ptr + n_lim_size_2D,
                thrust::make_constant_iterator(1),
                c_tmp_b_i_ptr,    // stores unique keys
                c_tmp_b_ii_ptr    // stores reduction
            );
            int n_unique_bins_v = result.first - c_tmp_b_i_ptr;
            if (!use_zip)
                n_unique_bins_v = thrust::count_if(thrust::device, c_tmp_b_i_ptr, c_tmp_b_i_ptr + n_bins_2D[L], is_nonnegative_and_less_than(n_bins_2D[L]));
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Reduction (nbins=" << n_unique_bins_v << ") by key "; toc_simple("",T_US,1);
            
            // STEP 6: Scatter the bin sizes.
            tic_simple("");
            thrust::scatter(
                thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_v,
                c_tmp_b_i_ptr,
                bnv_ptr
            );
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Scatter (1)"; toc_simple("",T_US,1);
            
            // STEP 7: Get the difference in the bounding box limits to identify the starting location for the Ids of each individual bin.
            tic_simple("");
            thrust::adjacent_difference(thrust::device, bb_ptr, bb_ptr + n_lim_size_2D, bb_ptr);
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Adjacent difference"; toc_simple("",T_US,1);
            
            // STEP 8: Gather the indices of the starting locations.
            tic_simple("");
            counting_iter = thrust::counting_iterator<int>(1);
            thrust::transform(
                thrust::device, counting_iter, counting_iter + n_lim_size_2D,
                bb_ptr, bb_ptr,
                replace_diff_with_indexM1()
            );
            thrust::copy_if(thrust::device, &bb_ptr[1], &bb_ptr[1] + (n_lim_size_2D-1), &c_tmp_b_ii_ptr[1], is_positive());
            fZ = 0;
            cudaMemcpy(c_tmp_b_ii, &fZ, sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Copy-if"; toc_simple("",T_US,1);
            
            // STEP 9: Now scatter the starting-location indices.
            tic_simple("");
            thrust::scatter(
                thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_v,
                c_tmp_b_i_ptr,
                bNv_ptr
            );
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS 2D | L=" << L << ", Scatter (2)"; toc_simple("",T_US,1);
            
            // Copy the indices of the binned faces.
            gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_2D[L], n_binned_faces_2D*sizeof(int)) );
            init_bins_2D = 1;
            cudaMemcpy(c_binned_face_ids_2D[L], c_bounding_box_index_limits, n_binned_faces_2D*sizeof(int), cudaMemcpyDeviceToDevice);
            //
            // Copy the GPU data to the CPU for drawing.
            binned_face_ids_n_2D[L] = new int[n_bins_2D[L]];
            binned_face_ids_N_2D[L] = new int[n_bins_2D[L]];
            binned_face_ids_2D[L] = new int[n_binned_faces_2D];
            cudaMemcpy(binned_face_ids_n_2D[L], c_binned_face_ids_n_2D[L], n_bins_2D[L]*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(binned_face_ids_N_2D[L], c_binned_face_ids_N_2D[L], n_bins_2D[L]*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(binned_face_ids_2D[L], c_binned_face_ids_2D[L], n_binned_faces_2D*sizeof(int), cudaMemcpyDeviceToHost);
            
            // DEBUG
            if (use_debug)
            {
                for (int i = 0; i < n_bins_2D[L]; i++)
                {
                    int nbins_i = binned_face_ids_n_2D[L][i];
                    int Nbins_i = binned_face_ids_N_2D[L][i];
                    if (nbins_i > 0)
                    {
                        std::cout << "Bin " << i << ": (nbins=" << nbins_i << ",Nbins=" << Nbins_i << ")" << std::endl;
                        for (int j = 0; j < nbins_i; j++)
                            std::cout << binned_face_ids_2D[L][Nbins_i+j] << " ";
                        std::cout << std::endl;
                    }
                }
            }
        }
        
        
        // o====================================================================================
        // | Finish.
        // o====================================================================================
        
        // Free temporary arrays.
        std::cout << std::endl;
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
        if (use_ray)
            cudaFree(c_ray_indicators);
        if (use_ray && use_map)
            cudaFree(c_indicator_map);
        cudaFree(c_bounding_box_limits);
        cudaFree(c_bounding_box_index_limits);
        cudaFree(c_tmp_b_i);
        cudaFree(c_tmp_b_ii);
    }
    
    return 0;
}

/**************************************************************************************/
/*                                                                                    */
/*  ===[ G_MakeBinsGPU_MD ]=========================================================  */
/*                                                                                    */
/*  This variant of G_MakeBinsGPU does the same thing but strictly for a set of 3D    */
/*  bins on a single grid level and with multi-directional ray casting to include     */
/*  faces that may require consideration when computing the lengths of links of       */
/*  boundary cells that cut through the geometry surface.                             */
/*                                                                                    */
/**************************************************************************************/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::Bins::G_MakeBinsGPU_MD(int L)
{
    std::cout << "CONSTRUCTION OF MD BINS ON LEVEL " << L << std::endl;
    
    // Some constants.
    ufloat_g_t *c_geom_f_face_X = geometry->c_geom_f_face_X;
    long int n_faces = geometry->n_faces;
    long int n_faces_a = geometry->n_faces_a;
    ufloat_g_t Lx0g = Lx0g_vec[L + 0*n_bin_levels];
    ufloat_g_t Ly0g = Lx0g_vec[L + 1*n_bin_levels];
    ufloat_g_t Lz0g = Lx0g_vec[L + 2*n_bin_levels];
    int use_debug = false;   // Indicates to print out binned Ids for debugging.
    int use_ray = true;      // Indicates to use the ray indicators.
    int use_map = true;      // Indicates to use the indicator map array.
    int use_zip = true;      // Indicates to perform compaction before sorting by key.
    
    // Correct optimizations, if necessary.
    if (n_max_levels_wall != n_bin_level_MD+1)
    {
        use_ray = false;
        use_map = false;
        std::cout << "G_BIN_LEVELS < MAX_LEVELS_WALL, not using ray indicator optimization..." << std::endl;
    }
    
    // Proceed only if there are actual faces loaded in the current object.
    if (n_faces > 0)
    {
        // o====================================================================================
        // | Preparation.
        // o====================================================================================
        
        // Compute constants.
        long int n_limits_MD = 1;
        for (int d = 0; d < N_DIM; d++) n_limits_MD *= (2+n_bin_spec);
        long int n_filtered = 0;
        std::cout << "Using a bin radius of " << n_limits_MD << std::endl;
        
        // Declare and allocate std::vector<int> bin arrays, which will be updated during traversal.
        n_bins_MD = 1; for (int d = 0; d < N_DIM-0; d++) { n_bins_MD *= n_bin_density[L]; }
        long int n_lim_size_MD = n_limits_MD*n_faces;
        
        // Some array declarations.
        int *c_ray_indicators;               // Used to filter out faces that never encounter a ray.
        int *c_indicator_map;                // Stored mapped values of ray indicators.
        int *c_bounding_box_limits;          // Stores bin values for the faces.
        int *c_bounding_box_index_limits;    // Stores copies of face values for each bin.
        int *c_tmp_b_i;                      // Used to store unique bin Ids.
        int *c_tmp_b_ii;                     // Used to gather starting-location indices.
        
        // Declare and allocate memory for the c_bounding_box_limits.
        cudaDeviceSynchronize();
        tic_simple("");
        if (use_ray)
            gpuErrchk( cudaMalloc((void **)&c_ray_indicators, n_faces*sizeof(int)) );
        if (use_ray && use_map)
            gpuErrchk( cudaMalloc((void **)&c_indicator_map, n_faces*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_tmp_b_i, n_bins_MD*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_tmp_b_ii, n_bins_MD*sizeof(int)) );
        //
        // Allocate memory for bins sizes and locations.
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_MD, n_bins_MD*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_MD, n_bins_MD*sizeof(int)) );
        //
        // Reset values.
        if (use_ray)
            Cu_ResetToValue<<<(M_BLOCK+n_faces-1)/M_BLOCK, M_BLOCK>>>(n_faces, c_ray_indicators, -1);
        if (use_ray && use_map)
            Cu_ResetToValue<<<(M_BLOCK+n_faces-1)/M_BLOCK, M_BLOCK>>>(n_faces, c_indicator_map, -1);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_MD-1)/M_BLOCK, M_BLOCK>>>(n_bins_MD, c_binned_face_ids_n_MD, 0);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_MD-1)/M_BLOCK, M_BLOCK>>>(n_bins_MD, c_binned_face_ids_N_MD, 0);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_MD-1)/M_BLOCK, M_BLOCK>>>(n_bins_MD, c_tmp_b_i, -1);
        Cu_ResetToValue<<<(M_BLOCK+n_bins_MD-1)/M_BLOCK, M_BLOCK>>>(n_bins_MD, c_tmp_b_ii, -1);
        
        // Wrap raw pointers with thrust device_ptr.
        thrust::device_ptr<int> ray_ptr = thrust::device_pointer_cast(c_ray_indicators);
        thrust::device_ptr<int> map_ptr = thrust::device_pointer_cast(c_indicator_map);
        thrust::device_ptr<int> bnb_ptr = thrust::device_pointer_cast(c_binned_face_ids_n_MD);
        thrust::device_ptr<int> bNb_ptr = thrust::device_pointer_cast(c_binned_face_ids_N_MD);
        thrust::device_ptr<int> c_tmp_b_i_ptr = thrust::device_pointer_cast(c_tmp_b_i);
        thrust::device_ptr<int> c_tmp_b_ii_ptr = thrust::device_pointer_cast(c_tmp_b_ii);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
        std::cout << "Memory allocation (1): "; toc_simple("",T_US,1);
        
        // Load connectivity arrays in constant GPU memory if not already done.
        if (!init_conn)
            InitConnectivity<AP->N_DIM>();
        
        
        
        // o====================================================================================
        // | Ray-indicator computation.
        // o====================================================================================
        
        if (use_ray)
        {
            std::cout << "RAY COMPUTATION:" << std::endl;
            // STEP 1A: Traverse faces and identify the bins they should go in.
            tic_simple("");
            cudaDeviceSynchronize();
            Cu_ComputeVoxelRayIndicators_MD<ufloat_g_t,AP->N_DIM><<<(M_BLOCK+n_faces-1)/M_BLOCK,M_BLOCK>>>(
                dxf_vec[L], static_cast<ufloat_g_t>(0.5)*dxf_vec[L],
                static_cast<ufloat_g_t>(Nxi_L[L]), static_cast<ufloat_g_t>(Nxi_L[L + 1*n_bin_levels]), static_cast<ufloat_g_t>(Nxi_L[L + 2*n_bin_levels]),
                n_faces, n_faces_a,
                c_geom_f_face_X, c_ray_indicators
            );
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS PRE | L=" << L << ", Computing ray indicators"; toc_simple("",T_US,1);
            
            if (use_map)
            {
                // Step 1B: Collect ray indicators.
                tic_simple("");
                auto filtered_result = thrust::copy_if(thrust::device, ray_ptr, ray_ptr + n_faces, map_ptr, is_nonnegative());
                n_filtered = filtered_result - map_ptr;
                n_lim_size_MD = n_limits_MD*n_filtered;
                cudaDeviceSynchronize();
                std::cout << "MAKE_BINS PRE | L=" << L << ", Gathered ray indicators (filtered=" << n_filtered << "/" << n_faces << " [ " << 100.00*(double)n_filtered/(double)n_faces << "%])"; toc_simple("",T_US,1);
                
                // Step 1C: Scatter new ray indicators.
                tic_simple("");
                auto counting_iter = thrust::counting_iterator<int>(0);
                thrust::scatter(
                    thrust::device, counting_iter, counting_iter + n_filtered,
                    map_ptr,
                    ray_ptr
                );
                cudaDeviceSynchronize();
                std::cout << "MAKE_BINS PRE | L=" << L << ", Scattered new ray indicators"; toc_simple("",T_US,1);
            }
        }
        
        // Other memory allocations.
        tic_simple("");
        gpuErrchk( cudaMalloc((void **)&c_bounding_box_limits, n_lim_size_MD*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_bounding_box_index_limits, n_lim_size_MD*sizeof(int)) );
        Cu_ResetToValue<<<(M_BLOCK+n_lim_size_MD-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_MD, c_bounding_box_limits, n_bins_MD);
        Cu_ResetToValue<<<(M_BLOCK+n_lim_size_MD-1)/M_BLOCK, M_BLOCK>>>(n_lim_size_MD, c_bounding_box_index_limits, -1);
        thrust::device_ptr<int> bbi_ptr = thrust::device_pointer_cast(c_bounding_box_index_limits);
        thrust::device_ptr<int> bb_ptr = thrust::device_pointer_cast(c_bounding_box_limits);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
        std::cout << "Memory allocation (2): "; toc_simple("",T_US,1);
        
        
        
        // o====================================================================================
        // | 3D binning.
        // o====================================================================================
        
        // STEP 2: Traverse faces and identify the bins they should go in.
        tic_simple("");
        cudaDeviceSynchronize();
        Cu_ComputeBoundingBoxLimits3D<ufloat_g_t,AP->N_DIM><<<(M_BLOCK+n_faces-1)/M_BLOCK,M_BLOCK>>>(
            n_faces, n_faces_a, n_limits_MD,
            c_geom_f_face_X, c_bounding_box_limits, c_bounding_box_index_limits,
            (ufloat_g_t)dxf_vec[L]+EPS<ufloat_g_t>(), (ufloat_g_t)Lx, (ufloat_g_t)Ly, (ufloat_g_t)Lz,
            Lx0g, Ly0g, Lz0g, n_bin_density[L],
            c_ray_indicators,
            use_ray, use_map, n_filtered
        );
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS MD | L=" << L << ", Computing bounding box limits"; toc_simple("",T_US,1);
        
        // STEP 3: If selected, make a zip iterator out of the bounding box limits and index arrays, and then remove all invalid bins.
        // This might speed up the sort-by-key that follows.
        if (use_zip)
        {
            tic_simple("");
            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(bb_ptr, bbi_ptr));
            auto zipped_end = thrust::remove_if(thrust::device, zipped, zipped + n_lim_size_MD, is_equal_to_zip(n_bins_MD));
            n_lim_size_MD = zipped_end - zipped;
            //n_lim_size_MD = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_MD, is_nonnegative_and_less_than(n_bins_MD));
            cudaDeviceSynchronize();
            std::cout << "MAKE_BINS MD | L=" << L << ", Compaction"; toc_simple("",T_US,1);
        }
        
        // STEP 4: Sort by key.
        tic_simple("");
        thrust::sort_by_key(thrust::device, bb_ptr, bb_ptr + n_lim_size_MD, bbi_ptr);
        int n_binned_faces_MD = n_lim_size_MD;
        if (!use_zip)
            n_binned_faces_MD = thrust::count_if(thrust::device, bb_ptr, bb_ptr + n_lim_size_MD, is_nonnegative_and_less_than(n_bins_MD));
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS MD | L=" << L << ", Sort by key (n_binned_faces=" << n_binned_faces_MD << ")"; toc_simple("",T_US,1);
        
        // STEP 5: Reduce the keys to get the number of faces in each bin. Scatter them to c_binned_face_ids_n_2D.
        tic_simple("");
        auto result = thrust::reduce_by_key(
            thrust::device, bb_ptr, bb_ptr + n_lim_size_MD,
            thrust::make_constant_iterator(1),
            c_tmp_b_i_ptr,    // stores unique keys
            c_tmp_b_ii_ptr    // stores reduction
        );
        int n_unique_bins_b = result.first - c_tmp_b_i_ptr;
        if (!use_zip)
            n_unique_bins_b = thrust::count_if(thrust::device, c_tmp_b_i_ptr, c_tmp_b_i_ptr + n_bins_MD, is_nonnegative_and_less_than(n_bins_MD));
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS 3D | L=" << L << ", Reduction (nbins=" << n_unique_bins_b << ") by key"; toc_simple("",T_US,1);
        
        // STEP 6: Scatter the bin sizes.
        tic_simple("");
        thrust::scatter(
            thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_b,
            c_tmp_b_i_ptr,
            bnb_ptr
        );
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS MD | L=" << L << ", Scatter (1)"; toc_simple("",T_US,1);
        
        // STEP 7: Get the difference in the bounding box limits to identify the starting location for the Ids of each individual bin.
        tic_simple("");
        thrust::adjacent_difference(thrust::device, bb_ptr, bb_ptr + n_lim_size_MD, bb_ptr);
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS MD | L=" << L << ", Adjacent difference"; toc_simple("",T_US,1);
        
        // STEP 8: Gather the indices of the starting locations.
        tic_simple("");
        auto counting_iter = thrust::counting_iterator<int>(1);
        thrust::transform(
            thrust::device, counting_iter, counting_iter + n_lim_size_MD,
            bb_ptr, bb_ptr,
            replace_diff_with_indexM1()
        );
        thrust::copy_if(thrust::device, &bb_ptr[1], &bb_ptr[1] + (n_lim_size_MD-1), &c_tmp_b_ii_ptr[1], is_positive());
        int fZ = 0;
        cudaMemcpy(c_tmp_b_ii, &fZ, sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS MD | L=" << L << ", Copy-if"; toc_simple("",T_US,1);
        
        // STEP 9: Now scatter the starting-location indices.
        tic_simple("");
        thrust::scatter(
            thrust::device, c_tmp_b_ii_ptr, c_tmp_b_ii_ptr + n_unique_bins_b,
            c_tmp_b_i_ptr,
            bNb_ptr
        );
        cudaDeviceSynchronize();
        std::cout << "MAKE_BINS MD | L=" << L << ", Scatter (2)"; toc_simple("",T_US,1);
        
        // Copy the indices of the binned faces.
        gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_MD, n_binned_faces_MD*sizeof(int)) );
        cudaMemcpy(c_binned_face_ids_MD, c_bounding_box_index_limits, n_binned_faces_MD*sizeof(int), cudaMemcpyDeviceToDevice);
        init_bins_MD = 1;
        //
        // Copy the GPU data to the CPU for drawing.
        binned_face_ids_n_MD = new int[n_bins_MD];
        binned_face_ids_N_MD = new int[n_bins_MD];
        binned_face_ids_MD = new int[n_binned_faces_MD];
        cudaMemcpy(binned_face_ids_n_MD, c_binned_face_ids_n_MD, n_bins_MD*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(binned_face_ids_N_MD, c_binned_face_ids_N_MD, n_bins_MD*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(binned_face_ids_MD, c_binned_face_ids_MD, n_binned_faces_MD*sizeof(int), cudaMemcpyDeviceToHost);
        
        // DEBUG
        if (use_debug)
        {
            for (int i = 0; i < n_bins_MD; i++)
            {
                int nbins_i = binned_face_ids_n_MD[i];
                int Nbins_i = binned_face_ids_N_MD[i];
                if (nbins_i > 0)
                {
                    std::cout << "Bin " << i << ": (nbins=" << nbins_i << ",Nbins=" << Nbins_i << ")" << std::endl;
                    for (int j = 0; j < nbins_i; j++)
                        std::cout << binned_face_ids_MD[Nbins_i+j] << " ";
                    std::cout << std::endl;
                }
            }
        }
        
        
        // o====================================================================================
        // | Finish.
        // o====================================================================================
        
        // Free temporary arrays.
        std::cout << std::endl;
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
        if (use_ray)
            cudaFree(c_ray_indicators);
        if (use_ray && use_map)
            cudaFree(c_indicator_map);
        cudaFree(c_bounding_box_limits);
        cudaFree(c_bounding_box_index_limits);
        cudaFree(c_tmp_b_i);
        cudaFree(c_tmp_b_ii);
    }
    
    return 0;
}
