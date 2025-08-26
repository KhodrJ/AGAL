/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef UTIL_TRIBIN_OLD_H
#define UTIL_TRIBIN_OLD_H

// o====================================================================================
// | TO BE REMOVED [TODO].
// o====================================================================================

template <typename T>
__device__
void Cross(T &ax, T &ay, T &az, T &bx, T &by, T &bz, T &sx, T &sy, T &sz)
{
    sx = ay*bz - az*by;
    sy = az*bx - ax*bz;
    sz = ax*by - ay*bx;
}

template <typename T>
__device__
bool CheckPointInLine(T &vxp, T &vyp, T &vx1, T &vy1, T &vx2, T &vy2)
{
    bool C = true;
    
    // First point.
    C = C &&   -( (vx1-vxp)*(vx2-vx1) + (vy1-vyp)*(vy2-vy1) ) > 0;
    C = C &&   ( (vx2-vxp)*(vx2-vx1) + (vy2-vyp)*(vy2-vy1) ) > 0;
    
    return C;
}

template <typename T>
__device__
bool CheckPointInLine3D(T &vxp, T &vyp, T &vzp, T &vx1, T &vy1, T &vz1, T &vx2, T &vy2, T &vz2)
{
    bool C = true;
    
    // First point.
    C = C &&   -( (vx1-vxp)*(vx2-vx1) + (vy1-vyp)*(vy2-vy1) + (vz1-vzp)*(vz2-vz1) ) > 0;
    C = C &&   ( (vx2-vxp)*(vx2-vx1) + (vy2-vyp)*(vy2-vy1) + (vz2-vzp)*(vz2-vz1) ) > 0;
    
    return C;
}

template <typename T>
__device__
bool CheckPointInTriangle(T &vxp, T &vyp, T &vzp, T &vx1, T &vy1, T &vz1, T &vx2, T &vy2, T &vz2, T &vx3, T &vy3, T &vz3, T &nx, T &ny, T &nz, T &sx, T &sy, T &sz, T &ex, T &ey, T &ez)
{
    bool C = true;
    
    // First edge.
    ex = vx2-vx1;
    ey = vy2-vy1;
    ez = vz2-vz1;
    Cross(ex,ey,ez,nx,ny,nz,sx,sy,sz);
    C = C &&   (vx1-vxp)*sx + (vy1-vyp)*sy + (vz1-vzp)*sz > 0;
    
    // Second edge.
    ex = vx3-vx2;
    ey = vy3-vy2;
    ez = vz3-vz2;
    Cross(ex,ey,ez,nx,ny,nz,sx,sy,sz);
    C = C &&   (vx2-vxp)*sx + (vy2-vyp)*sy + (vz2-vzp)*sz > 0;
    
    // Third edge.
    ex = vx1-vx3;
    ey = vy1-vy3;
    ez = vz1-vz3;
    Cross(ex,ey,ez,nx,ny,nz,sx,sy,sz);
    C = C &&   (vx3-vxp)*sx + (vy3-vyp)*sy + (vz3-vzp)*sz > 0;
    
    return C;
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckInLine(T d, T vxp, T vx0, T vx1)
{
    return (d > (T)0.0 && d < (T)1.0 && vxp > vx0 && vxp < vx1);
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckInRect(T d, T vxp, T vyp, T vx0, T vy0, T vx1, T vy1)
{
    return (d > (T)0.0 && d < (T)1.0 && vxp > vx0 && vxp < vx1 && vyp > vy0 && vyp < vy1);
}

// Using in:
// - Control volume force calculations [solver_lbm_compute_forces.cu].
template <typename T>
__device__
bool CheckPointInRegion2D(T vxp, T vyp, T vxm, T vxM, T vym, T vyM)
{
    return (vxp > vxm && vxp < vxM && vyp > vym && vyp < vyM);
}

// Using in:
// - Control volume force calculations [solver_lbm_compute_forces.cu].
template <typename T>
__device__
bool CheckPointInRegion3D(T vxp, T vyp, T vzp, T vxm, T vxM, T vym, T vyM, T vzm, T vzM)
{
    return (vxp > vxm && vxp < vxM && vyp > vym && vyp < vyM && vzp > vzm && vzp < vzM);
}













template <typename ufloat_g_t, int N_DIM>
__host__ __device__ __forceinline__
bool IncludeInBin
(
    const vec3<ufloat_g_t> &vm,
    const vec3<ufloat_g_t> &vM,
    const vec3<ufloat_g_t> &vBm,
    const vec3<ufloat_g_t> &vBM,
    const vec3<ufloat_g_t> &v1,
    const vec3<ufloat_g_t> &v2,
    const vec3<ufloat_g_t> &v3
)
{
    // vm/M are the bin dimensions.
    // vBm/M is the face bounding box.
    // v1/2/3 are the face vertices.
    
    ufloat_g_t tmp = (ufloat_g_t)(0.0);
    ufloat_g_t ex1 = (ufloat_g_t)(0.0);
    ufloat_g_t ey1 = (ufloat_g_t)(0.0);
    
if (N_DIM==2)
{
    // Only consider this calculation if the bounding box intersects the bin.
    if ( !( (vBm.x < vm.x && vBM.x < vm.x) || (vBm.x > vM.x && vBM.x > vM.x) || (vBm.y < vm.y && vBM.y < vm.y) || (vBm.y > vM.y && vBM.y > vM.y) ) )
    {
        // Check if bounding box is entirely inside current bin.
        if (vBm.x > vm.x && vBM.x < vM.x && vBm.y > vm.y && vBM.y < vM.y) { return true; }
        
        // Check if at least one of the vertices is inside the bin.
        if (v1.x > vm.x && v1.x < vM.x) { return true; }
        if (v2.x > vm.x && v2.x < vM.x) { return true; }
        if (v1.y > vm.y && v1.y < vM.y) { return true; }
        if (v2.y > vm.y && v2.y < vM.y) { return true; }
        
        // Check the bottom edge of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vm.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ex1,vm.x,vM.x)) { return true; }
        }
        
        // Check the top edge of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vM.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ex1,vm.x,vM.x)) { return true; }
        }
        
        // Check the left edge of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vm.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ey1,vm.y,vM.y)) { return true; }
        }
        
        // Check the right edge of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vM.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            if (CheckInLine(tmp,ey1,vm.y,vM.y)) { return true; }
        }
    }
}
else
{
    ufloat_g_t ez1 = (ufloat_g_t)(0.0);
    
    if ( !( (vBm.x < vm.x && vBM.x < vm.x) || (vBm.x > vM.x && vBM.x > vM.x) || (vBm.y < vm.y && vBM.y < vm.y) || (vBm.y > vM.y && vBM.y > vM.y) || (vBm.z < vm.z && vBM.z < vm.z) || (vBm.z > vM.z && vBM.z > vM.z) ) )
    {
        // Check if bounding box is entirely inside current bin.
        if (vBm.x > vm.x && vBM.x < vM.x && vBm.y > vm.y && vBM.y < vM.y && vBm.z > vm.z && vBM.z < vM.z) { return true; }
        
        // Check if bounding box completely surrounds the bin.
        if (vBm.x < vm.x && vBM.x > vM.x && vBm.y < vm.y && vBM.y > vM.y) { return true; }
        if (vBm.y < vm.y && vBM.y > vM.y && vBm.z < vm.z && vBM.z > vM.z) { return true; }
        if (vBm.z < vm.z && vBM.z > vM.z && vBm.x < vm.x && vBM.x > vM.x) { return true; }
        
        // Check if at least one of the vertices is inside the bin.
        if (v1.x > vm.x && v1.x < vM.x && v1.y > vm.y && v1.y < vM.y && v1.z > vm.z && v1.z < vM.z) { return true; }
        if (v2.x > vm.x && v2.x < vM.x && v2.y > vm.y && v2.y < vM.y && v2.z > vm.z && v2.z < vM.z) { return true; }
        if (v3.x > vm.x && v3.x < vM.x && v3.y > vm.y && v3.y < vM.y && v3.z > vm.z && v3.z < vM.z) { return true; }
        
        // Check the bottom face of the bin.
        {
            ez1 = v2.z-v1.z;
            tmp = (vm.z-v1.z)/(ez1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v3.z-v2.z;
            tmp = (vm.z-v2.z)/(ez1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v1.z-v3.z;
            tmp = (vm.z-v3.z)/(ez1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        
        // Check the top face of the bin.
        {
            ez1 = v2.z-v1.z;
            tmp = (vM.z-v1.z)/(ez1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v3.z-v2.z;
            tmp = (vM.z-v2.z)/(ez1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        {
            ez1 = v1.z-v3.z;
            tmp = (vM.z-v3.z)/(ez1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ey1,vm.x,vm.y,vM.x,vM.y)) { return true; }
        }
        
        // Check the back face of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vm.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v3.y-v2.y;
            tmp = (vm.y-v2.y)/(ey1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v1.y-v3.y;
            tmp = (vm.y-v3.y)/(ey1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        
        // Check the front face of the bin.
        {
            ey1 = v2.y-v1.y;
            tmp = (vM.y-v1.y)/(ey1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v3.y-v2.y;
            tmp = (vM.y-v2.y)/(ey1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        {
            ey1 = v1.y-v3.y;
            tmp = (vM.y-v3.y)/(ey1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ex1,ez1,vm.x,vm.z,vM.x,vM.z)) { return true; }
        }
        
        // Check the left face of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vm.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v3.x-v2.x;
            tmp = (vm.x-v2.x)/(ex1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v1.x-v3.x;
            tmp = (vm.x-v3.x)/(ex1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        
        // Check the right face of the bin.
        {
            ex1 = v2.x-v1.x;
            tmp = (vM.x-v1.x)/(ex1);
            ex1 = v1.x + tmp*(v2.x-v1.x);
            ey1 = v1.y + tmp*(v2.y-v1.y);
            ez1 = v1.z + tmp*(v2.z-v1.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v3.x-v2.x;
            tmp = (vM.x-v2.x)/(ex1);
            ex1 = v2.x + tmp*(v3.x-v2.x);
            ey1 = v2.y + tmp*(v3.y-v2.y);
            ez1 = v2.z + tmp*(v3.z-v2.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
        {
            ex1 = v1.x-v3.x;
            tmp = (vM.x-v3.x)/(ex1);
            ex1 = v3.x + tmp*(v1.x-v3.x);
            ey1 = v3.y + tmp*(v1.y-v3.y);
            ez1 = v3.z + tmp*(v1.z-v3.z);
            if (CheckInRect(tmp,ey1,ez1,vm.y,vm.z,vM.y,vM.z)) { return true; }
        }
    }
}
    
    return false;
}

#endif
