/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef UTIL_TRIBIN
#define UTIL_TRIBIN

template <typename T, int axis=0>
__host__ __device__ __forceinline__ bool TriangleBinOverlap1D_AxisGap
(
    const T &v1x,
    const T &v1y,
    const T &v2x,
    const T &v2y,
    const T &rmx,
    const T &rmy,
    const T &rMx,
    const T &rMy
)
{
    bool b = false;
    
    T ex;
    T ey;
    if (axis == 0) {ex = static_cast<T>(1.0); ey = static_cast<T>(0.0);}
    if (axis == 1) {ex = static_cast<T>(0.0); ey = static_cast<T>(1.0);}
    if (axis == 2) {ex = v2y-v1y; ey = v1x-v2x;}
    
    T t = Tmax(v1x*ex+v1y*ey, v2x*ex+v2y*ey);
    T r = Tmin(Tmin(Tmin(rmx*ex+rmy*ey, rMx*ex+rmy*ey), rmx*ex+rMy*ey), rMx*ex+rMy*ey);
    b = b || (t<r);
    
    t = Tmin(v1x*ex+v1y*ey, v2x*ex+v2y*ey);
    r = Tmax(Tmax(Tmax(rmx*ex+rmy*ey, rMx*ex+rmy*ey), rmx*ex+rMy*ey), rMx*ex+rMy*ey);
    b = b || (r<t);
    
    return b;
}

template <typename T>
__host__ __device__ __forceinline__ bool LineBinOverlap2D(const vec2<T> &vm, const vec2<T> &vM, const vec2<T> &v1, const vec2<T> &v2)
{
    // Tests.
    if (TriangleBinOverlap1D_AxisGap<T,0>(v1.x,v1.y,v2.x,v2.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap1D_AxisGap<T,1>(v1.x,v1.y,v2.x,v2.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap1D_AxisGap<T,2>(v1.x,v1.y,v2.x,v2.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    
    return true;
}
template <typename T>
__host__ __device__ __forceinline__ bool LineBinOverlap2D(const vec3<T> &vm, const vec3<T> &vM, const vec3<T> &v1, const vec3<T> &v2)
{
    // Tests.
    if (TriangleBinOverlap1D_AxisGap<T,0>(v1.x,v1.y,v2.x,v2.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap1D_AxisGap<T,1>(v1.x,v1.y,v2.x,v2.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap1D_AxisGap<T,2>(v1.x,v1.y,v2.x,v2.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    
    return true;
}

















template <typename T>
__host__ __device__ __forceinline__ bool TrianglePlaneCutsBox(const vec3<T> &vm, const vec3<T> &vM, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3)
{
    vec3<T> n = FaceNormal<T,3>(v1,v2,v3);
    vec3<T> c(
        n.x > 0 ? (vM.x-vm.x) : static_cast<T>(0.0),
        n.y > 0 ? (vM.y-vm.y) : static_cast<T>(0.0),
        n.z > 0 ? (vM.z-vm.z) : static_cast<T>(0.0)
    );
    T d = DotV(n,vm);
    T d1 = DotV(n,c-v1);
    T d2 = DotV(n,(vM-vm-c)-v1);
    
    return (d+d1)*(d+d2) <= 0;
}

template <typename T, int axis=0>
__host__ __device__ __forceinline__ bool TriangleBinOverlap2D_AxisGap
(
    const T &v1x,
    const T &v1y,
    const T &v2x,
    const T &v2y,
    const T &v3x,
    const T &v3y,
    const T &rmx,
    const T &rmy,
    const T &rMx,
    const T &rMy
)
{
    bool b = false;
    
    T ex;
    T ey;
    if (axis == 0) {ex = static_cast<T>(1.0); ey = static_cast<T>(0.0);}
    if (axis == 1) {ex = static_cast<T>(0.0); ey = static_cast<T>(1.0);}
    if (axis == 2) {ex = v2y-v1y; ey = v1x-v2x;}
    if (axis == 3) {ex = v3y-v2y; ey = v2x-v3x;}
    if (axis == 4) {ex = v1y-v3y; ey = v3x-v1x;}
    
    T t = Tmax(Tmax(v1x*ex+v1y*ey, v2x*ex+v2y*ey), v3x*ex+v3y*ey);
    T r = Tmin(Tmin(Tmin(rmx*ex+rmy*ey, rMx*ex+rmy*ey), rmx*ex+rMy*ey), rMx*ex+rMy*ey);
    b = b || (t<r);
    
    t = Tmin(Tmin(v1x*ex+v1y*ey, v2x*ex+v2y*ey), v3x*ex+v3y*ey);
    r = Tmax(Tmax(Tmax(rmx*ex+rmy*ey, rMx*ex+rmy*ey), rmx*ex+rMy*ey), rMx*ex+rMy*ey);
    b = b || (r<t);
    
    return b;
}

template <typename T>
__host__ __device__ __forceinline__ bool TriangleBinOverlap3D(const vec3<T> &vm, const vec3<T> &vM, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3)
{
    // Plane-cut test.
    if (!TrianglePlaneCutsBox(vm,vM,v1,v2,v3)) {return false;}
    
    // XY tests.
    if (TriangleBinOverlap2D_AxisGap<T,0>(v1.x,v1.y,v2.x,v2.y,v3.x,v3.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,1>(v1.x,v1.y,v2.x,v2.y,v3.x,v3.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,2>(v1.x,v1.y,v2.x,v2.y,v3.x,v3.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,3>(v1.x,v1.y,v2.x,v2.y,v3.x,v3.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,4>(v1.x,v1.y,v2.x,v2.y,v3.x,v3.y,vm.x,vm.y,vM.x,vM.y)) {return false;}
    
    // YZ tests.
    if (TriangleBinOverlap2D_AxisGap<T,0>(v1.y,v1.z,v2.y,v2.z,v3.y,v3.z,vm.y,vm.z,vM.y,vM.z)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,1>(v1.y,v1.z,v2.y,v2.z,v3.y,v3.z,vm.y,vm.z,vM.y,vM.z)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,2>(v1.y,v1.z,v2.y,v2.z,v3.y,v3.z,vm.y,vm.z,vM.y,vM.z)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,3>(v1.y,v1.z,v2.y,v2.z,v3.y,v3.z,vm.y,vm.z,vM.y,vM.z)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,4>(v1.y,v1.z,v2.y,v2.z,v3.y,v3.z,vm.y,vm.z,vM.y,vM.z)) {return false;}
    
    // ZX tests.
    if (TriangleBinOverlap2D_AxisGap<T,0>(v1.z,v1.x,v2.z,v2.x,v3.z,v3.x,vm.z,vm.x,vM.z,vM.x)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,1>(v1.z,v1.x,v2.z,v2.x,v3.z,v3.x,vm.z,vm.x,vM.z,vM.x)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,2>(v1.z,v1.x,v2.z,v2.x,v3.z,v3.x,vm.z,vm.x,vM.z,vM.x)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,3>(v1.z,v1.x,v2.z,v2.x,v3.z,v3.x,vm.z,vm.x,vM.z,vM.x)) {return false;}
    if (TriangleBinOverlap2D_AxisGap<T,4>(v1.z,v1.x,v2.z,v2.x,v3.z,v3.x,vm.z,vm.x,vM.z,vM.x)) {return false;}
    
    return true;
}



















template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInTriangleAABB(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, T eps=static_cast<T>(EPS<float>()*2.0F))
{
//     EPS<float>()*static_cast<T>(2.0)
    vec3<T> vm = vp - eps;
    vec3<T> vM = vp + eps;
    
    return TriangleBinOverlap3D(vm,vM,v1,v2,v3);
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInTriangleSphere(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
    vec3<T> vi = PointTriangleIntersection(vp,v1,v2,v3,n);
    
    return PointInSphere(vi,vp,EPS<T>()*static_cast<T>(100.0),static_cast<T>(0.0));
}

#endif
