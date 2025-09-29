/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef UTIL_H
#define UTIL_H

// o====================================================================================
// | Templated math operations.
// o====================================================================================

template <class T> __host__ __device__ __forceinline__ T EPS();
template <> __host__ __device__ __forceinline__ float EPS() { return FLT_EPSILON; }
template <> __host__ __device__ __forceinline__ double EPS() { return DBL_EPSILON; }
//
__host__ __device__ __forceinline__ int imin(int a, int b) { return (a < b) ? a : b; }
__host__ __device__ __forceinline__ int imax(int a, int b) { return (a > b) ? a : b; }
//
template <class T> __host__ __device__ __forceinline__ T Tmin(T a, T b);
template <> __host__ __device__ __forceinline__ int Tmin(int a, int b) { return imin(a,b); }
template <> __host__ __device__ __forceinline__  float Tmin(float a, float b) { return fminf(a,b); }
template <> __host__ __device__ __forceinline__  double Tmin(double a, double b) { return fmin(a,b); }
//
template <class T> __host__ __device__ __forceinline__ T Tmax(T a, T b);
template <> __host__ __device__ __forceinline__ int Tmax(int a, int b) { return imax(a,b); }
template <> __host__ __device__ __forceinline__  float Tmax(float a, float b) { return fmaxf(a,b); }
template <> __host__ __device__ __forceinline__  double Tmax(double a, double b) { return fmax(a,b); }
//
template <class T> __host__ __device__ __forceinline__ T Tabs(T a);
template <> __host__ __device__ __forceinline__ int Tabs(int a) { return abs(a); }
template <> __host__ __device__ __forceinline__  float Tabs(float a) { return fabsf(a); }
template <> __host__ __device__ __forceinline__  double Tabs(double a) { return fabs(a); }
//
template <class T> __host__ __device__ __forceinline__ T Tpow(T a, T b);
template <> __host__ __device__ __forceinline__  float Tpow(float a, float b) { return powf(a,b); }
template <> __host__ __device__ __forceinline__  double Tpow(double a, double b) { return pow(a,b); }
//
template <class T> __host__ __device__ __forceinline__ T Tsqrt(T a);
template <> __host__ __device__ __forceinline__  float Tsqrt(float a) { return sqrtf(a); }
template <> __host__ __device__ __forceinline__  double Tsqrt(double a) { return sqrt(a); }
//
template <class T> __host__ __device__ __forceinline__ T Tacos(T a);
template <> __host__ __device__ __forceinline__  float Tacos(float a) { return acosf(a); }
template <> __host__ __device__ __forceinline__  double Tacos(double a) { return acos(a); }
//
template <class T> __host__ __device__ __forceinline__ T Tround(T a);
template <> __host__ __device__ __forceinline__  float Tround(float a) { return roundf(a); }
template <> __host__ __device__ __forceinline__  double Tround(double a) { return round(a); }
//
template <class T> __host__ __device__ __forceinline__ T Tceil(T a);
template <> __host__ __device__ __forceinline__  float Tceil(float a) { return ceilf(a); }
template <> __host__ __device__ __forceinline__  double Tceil(double a) { return ceil(a); }
//
template <class T> __host__ __device__ __forceinline__ T Tfloor(T a);
template <> __host__ __device__ __forceinline__  float Tfloor(float a) { return floorf(a); }
template <> __host__ __device__ __forceinline__  double Tfloor(double a) { return floor(a); }
//
template <class T> __host__ __device__ __forceinline__ bool LTZ_E(T a) { return a < -EPS<T>(); } // Less-than-zero exclusive.
template <class T> __host__ __device__ __forceinline__ bool LTZ_I(T a) { return a <  EPS<T>(); } // Less-than-zero inclusive.
template <class T> __host__ __device__ __forceinline__ bool GTZ_E(T a) { return a >  EPS<T>(); } // Greater-than-zero exclusive.
template <class T> __host__ __device__ __forceinline__ bool GTZ_I(T a) { return a > -EPS<T>(); } // Greater-than-zero inclusive.
//
template <class T> __host__ __device__ __forceinline__ int sign(T a)
{
    if (a > static_cast<T>(0)) return 1;
    if (a < static_cast<T>(0)) return -1;
    return 0;
}
template <class T> __host__ __device__ __forceinline__ int Tsign(T a, T tol=EPS<T>())
{
    if (a > tol) return 1;
    if (a < -tol) return -1;
    return 0;
}

// o====================================================================================
// | Connectivity arrays.
// o====================================================================================

bool init_conn = false;
__constant__ int V_CONN_ID[81];
__constant__ int V_CONN_ID_PB[27];
__constant__ int V_CONN_MAP[27];
int V_CONN_ID_H[81];
int V_CONN_ID_PB_H[27];
int V_CONN_MAP_H[27];

template <int N_DIM>
inline int InitConnectivity()
{
    int V_CONN_ID_2D[81] = {0, 1, 0, -1, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int V_CONN_ID_3D[81] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};
    
    int V_CONN_ID_PB_2D[27] = {0, 3, 4, 1, 2, 7, 8, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    int V_CONN_ID_PB_3D[27] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25};
    
    int V_CONN_MAP_2D[27] = {7, 4, 8, 3, 0, 1, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int V_CONN_MAP_3D[27] = {20, 12, 26, 10, 6, 15, 24, 17, 21, 8, 4, 13, 2, 0, 1, 14, 3, 7, 22, 18, 23, 16, 5, 9, 25, 11, 19};
    
    if (N_DIM==2)
    {
        cudaMemcpyToSymbol(V_CONN_ID, V_CONN_ID_2D, sizeof(int)*81);
        cudaMemcpyToSymbol(V_CONN_ID_PB, V_CONN_ID_PB_2D, sizeof(int)*27);
        cudaMemcpyToSymbol(V_CONN_MAP, V_CONN_MAP_2D, sizeof(int)*27);
        for (int p = 0; p < 81; p++) V_CONN_ID_H[p] = V_CONN_ID_2D[p];
        for (int p = 0; p < 27; p++) V_CONN_ID_PB_H[p] = V_CONN_ID_PB_2D[p];
        for (int p = 0; p < 27; p++) V_CONN_MAP_H[p] = V_CONN_MAP_2D[p];
    }
    if (N_DIM==3)
    {
        cudaMemcpyToSymbol(V_CONN_ID, V_CONN_ID_3D, sizeof(int)*81);
        cudaMemcpyToSymbol(V_CONN_ID_PB, V_CONN_ID_PB_3D, sizeof(int)*27);
        cudaMemcpyToSymbol(V_CONN_MAP, V_CONN_MAP_3D, sizeof(int)*27);
        for (int p = 0; p < 81; p++) V_CONN_ID_H[p] = V_CONN_ID_3D[p];
        for (int p = 0; p < 27; p++) V_CONN_ID_PB_H[p] = V_CONN_ID_PB_3D[p];
        for (int p = 0; p < 27; p++) V_CONN_MAP_H[p] = V_CONN_MAP_3D[p];
    }
    init_conn = true;
    
    return 0;
}

// o====================================================================================
// | Vec2.
// o====================================================================================

template <typename T>
struct vec2
{
    T x;
    T y;
    
    // Constructors.
    __host__ __device__ vec2() : x(0), y(0) {}
    __host__ __device__ vec2(T x_) : x(x_), y(0) {}
    __host__ __device__ vec2(T x_, T y_) : x(x_), y(y_) {}
    
    // Useful operation overloads.
    __host__ __device__ vec2<T> operator+(const vec2<T> &w) const
    {
        return vec2<T>(x + w.x, y + w.y);
    }
    __host__ __device__ vec2<T> operator+(const T &a) const
    {
        return vec2<T>(x + a, y + a);
    }
    __host__ __device__ vec2<T> operator-(const vec2<T> &w) const
    {
        return vec2<T>(x - w.x, y - w.y);
    }
    __host__ __device__ vec2<T> operator-(const T &a) const
    {
        return vec2<T>(x - a, y - a);
    }
    __host__ __device__ vec2<T> operator*(T a) const
    {
        return vec2<T>(x*a, y*a);
    }
    __host__ __device__ bool operator==(const vec2<T> &w) const
    {
        return x==w.x && y==w.y;
    }
    __host__ __device__ vec2<T> &operator+=(const vec2<T> & w)
    {
        x += w.x;
        y += w.y;
        return *this;
    }
    __host__ __device__ vec2<T> &operator*=(const vec2<T> &w)
    {
        x *= w.x;
        y *= w.y;
        return *this;
    }
    
    // Useful routines.
    __host__ __device__ T Min() const
    {
        return Tmin(x,y);
    }
    __host__ __device__ T Max() const
    {
        return Tmax(x,y);
    }
    __host__ __device__ bool InRegion_Cube(const vec2<T> &vm, const vec2<T> &vM) const
    {
        return (x >= vm.x && x <= vM.x && y >= vm.y && y <= vM.y);
    }
    __host__ __device__ int Set(T x_, T y_)
    {
        x = x_;
        y = y_;
        return 0;
    }
    __host__ __device__ int Normalize()
    {
        if (!(x == static_cast<T>(0.0) && y == static_cast<T>(0.0)))
        {
            T onorm = static_cast<T>(1.0)/Tsqrt(x*x + y*y);
            x *= onorm;
            y *= onorm;
        }
        return 0;
    }
    __host__ __device__ int Snap()
    {
        x = round(x);
        y = round(y);
        return 0;
    }
    template <int N>
    __host__ __device__ int AddToComp(const T &a)
    {
        if (N == 0) x += a;
        if (N == 1) y += a;
        return 0;
    }
};

template <typename T>
__host__ __device__ __forceinline__ T DotV(const vec2<T> &a, const vec2<T> &b)
{
    return (a.x*b.x + a.y*b.y);
}

template <typename T>
__host__ __device__ __forceinline__ T NormV(const vec2<T> &a)
{
    return Tsqrt(a.x*a.x + a.y*a.y);
}

template <typename T>
__host__ __device__ __forceinline__ T NormSqV(const vec2<T> &a)
{
    return a.x*a.x + a.y*a.y;
}

template <typename T>
__host__ __device__ __forceinline__ vec2<T> UnitV(const vec2<T> &a)
{
    T onorm = static_cast<T>(1.0)/NormV(a);
    return a*onorm;
}

template <typename T>
__host__ __device__ __forceinline__ T DistV(const vec2<T> &a, const vec2<T> &b)
{
    return NormV(a-b);
}

template <typename T>
__host__ __device__ __forceinline__ vec2<T> ReducedCrossWUnitV(const vec2<T> &a)
{
    // Returns the cross product of a vector a=(ax,ay,0), b=(0,0,1).
    return vec2<T>(a.y,-a.x);
}

// o====================================================================================
// | Vec3.
// o====================================================================================

template <typename T>
struct vec3
{
    T x;
    T y;
    T z;
    
    // Constructors.
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(T x_) : x(x_), y(0), z(0) {}
    __host__ __device__ vec3(T x_, T y_) : x(x_), y(y_), z(0) {}
    __host__ __device__ vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    
    // Useful operation overloads.
    __host__ __device__ vec3<T> operator+(const vec3<T> &w) const
    {
        return vec3<T>(x + w.x, y + w.y, z + w.z);
    }
    __host__ __device__ vec3<T> operator+(const T &a) const
    {
        return vec3<T>(x + a, y + a, z + a);
    }
    __host__ __device__ vec3<T> operator-(const vec3<T> &w) const
    {
        return vec3<T>(x - w.x, y - w.y, z - w.z);
    }
    __host__ __device__ vec3<T> operator-(const T &a) const
    {
        return vec3<T>(x - a, y - a, z - a);
    }
    __host__ __device__ vec3<T> operator*(T a) const
    {
        return vec3<T>(x*a, y*a, z*a);
    }
    __host__ __device__ bool operator==(const vec3<T> &w) const
    {
        return x==w.x && y==w.y && z==w.z;
    }
    __host__ __device__ vec3<T> &operator+=(const vec3<T> & w)
    {
        x += w.x;
        y += w.y;
        z += w.z;
        return *this;
    }
    __host__ __device__ vec3<T> &operator*=(const vec3<T> &w)
    {
        x *= w.x;
        y *= w.y;
        z *= w.z;
        return *this;
    }
    
    // Useful routines.
    __host__ __device__ T Min() const
    {
        return Tmin(Tmin(x,y),z);
    }
    __host__ __device__ T Max() const
    {
        return Tmax(Tmax(x,y),z);
    }
    __host__ __device__ bool InRegion_Cube(const vec3<T> &vm, const vec3<T> &vM) const
    {
        return (x >= vm.x && x <= vM.x && y >= vm.y && y <= vM.y && z >= vm.z && z <= vM.z);
    }
    __host__ __device__ int Set(T x_, T y_, T z_)
    {
        x = x_;
        y = y_;
        z = z_;
        return 0;
    }
    __host__ __device__ int Normalize()
    {
        if (!(x == static_cast<T>(0.0) && y == static_cast<T>(0.0) && z == static_cast<T>(0.0)))
        {
            T onorm = static_cast<T>(1.0)/Tsqrt(x*x + y*y + z*z);
            x *= onorm;
            y *= onorm;
            z *= onorm;
        }
        return 0;
    }
    __host__ __device__ int Snap()
    {
        x = round(x);
        y = round(y);
        z = round(z);
        return 0;
    }
    template <int N>
    __host__ __device__ int AddToComp(const T &a)
    {
        if (N == 0) x += a;
        if (N == 1) y += a;
        if (N == 2) z += a;
        return 0;
    }
};

template <typename T>
__host__ __device__ vec3<T> __forceinline__ CrossV(const vec3<T> &a, const vec3<T> &b)
{
    return vec3<T>
    (
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

template <typename T>
__host__ __device__ __forceinline__ T DotV2D(const vec3<T> &a, const vec3<T> &b)
{
    return (a.x*b.x + a.y*b.y);
}

template <typename T>
__host__ __device__ __forceinline__ T DotV(const vec3<T> &a, const vec3<T> &b)
{
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}

template <typename T>
__host__ __device__ __forceinline__ T NormV(const vec3<T> &a)
{
    return Tsqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

template <typename T>
__host__ __device__ __forceinline__ T NormSqV(const vec3<T> &a)
{
    return a.x*a.x + a.y*a.y + a.z*a.z;
}

template <typename T>
__host__ __device__ __forceinline__ vec3<T> UnitV(const vec3<T> &a)
{
    T onorm = static_cast<T>(1.0)/NormV(a);
    return a*onorm;
}

template <typename T>
__host__ __device__ __forceinline__ T DistV(const vec3<T> &a, const vec3<T> &b)
{
    return NormV(a-b);
}

// o====================================================================================
// | Geometry face operations.
// o====================================================================================

template <typename T, int N_DIM>
__host__ __device__ __forceinline__ vec3<T> FaceNormal(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3)
{
    if (N_DIM==2)
        return vec3<T>(v2.y-v1.y,v1.x-v2.x);
    else
        return CrossV(v2-v1,v3-v1);
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__ vec3<T> FaceNormalUnit(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3)
{
    vec3<T> n = FaceNormal<T,N_DIM>(v1,v2,v3);
    return UnitV(n);
}

template <typename T>
__host__ __device__ __forceinline__ vec3<T> InwardNormal(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &n)
{
    return CrossV(v2-v1,n);
}

template <typename T>
__host__ __device__ __forceinline__ vec3<T> InwardNormalUnit(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &n)
{
    vec3<T> e = CrossV(v2-v1,n);
    return UnitV(e);
}

template <typename T>
__host__ __device__ __forceinline__
vec3<T> PointLineIntersection(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2)
{
    vec3<T> e = v2-v1;
    vec3<T> w = vp-v1;
    T t = DotV(e,w)/DotV(e,e);
    
    return v1 + e*t;
}

template <typename T>
__host__ __device__ __forceinline__
vec3<T> PointLineIntersectionClamped(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2)
{
    vec3<T> e = v2-v1;
    vec3<T> w = vp-v1;
    T t = DotV(e,w)/DotV(e,e);
    t = Tmax(static_cast<T>(0.0),Tmin(static_cast<T>(1.0),t));
    
    return v1 + e*t;
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInLine(const vec2<T> &vp, const vec2<T> &v1, const vec2<T> &v2)
{
    // Assumes that the point is on an infinite line defined by the segment.
    
    // First vertex.
    if (DotV(vp-v1,v2-v1) <= -EPS<T>())
        return false;
    
    // Second vertex.
    if (-DotV(vp-v2,v2-v1) <= -EPS<T>())
        return false;
    
    return true;
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInLine(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2)
{
    // Assumes that the point is on an infinite line defined by the segment.
    
    // First vertex.
    if (DotV2D(vp-v1,v2-v1) <= -EPS<T>())
        return false;
    
    // Second vertex.
    if (-DotV2D(vp-v2,v2-v1) <= -EPS<T>())
        return false;
    
    return true;
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInLineExtended(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, T eps=EPS<T>())
{
    // Assumes that the point is on an infinite line defined by the segment.
    
    // First vertex.
    if (DotV2D(vp-v1,v2-v1) <= -eps)
        return false;
    
    // Second vertex.
    if (-DotV2D(vp-v2,v2-v1) <= -eps)
        return false;
    
    return true;
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInTriangleExtended(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n, T eps=EPS<T>())
{
    // First edge.
    vec3<T> ven = InwardNormalUnit(v1,v2,n);
    T s = DotV(v1-vp,ven);
    if (s <= -eps)
        return false;
    
    // Second edge.
    ven = InwardNormalUnit(v2,v3,n);
    s = DotV(v2-vp,ven);
    if (s <= -eps)
        return false;
    
    // Third edge.
    ven = InwardNormalUnit(v3,v1,n);
    s = DotV(v3-vp,ven);
    if (s <= -eps)
        return false;
    
    return true;
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInTriangle(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
    // First edge.
    vec3<T> ven = InwardNormalUnit(v1,v2,n);
    T s = DotV(v1-vp,ven);
    if ( LTZ_I(s) )
        return false;
    
    // Second edge.
    ven = InwardNormalUnit(v2,v3,n);
    s = DotV(v2-vp,ven);
    if ( LTZ_I(s) )
        return false;
    
    // Third edge.
    ven = InwardNormalUnit(v3,v1,n);
    s = DotV(v3-vp,ven);
    if ( LTZ_I(s) )
        return false;
    
    return true;
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
bool CheckPointInFace(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
    if (N_DIM==2)
        return CheckPointInLine(vp,v1,v2);
    else
        return CheckPointInTriangle(vp,v1,v2,v3,n);
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
bool CheckPointInFaceExtended(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n, T eps=EPS<T>())
{
    if (N_DIM==2)
        return CheckPointInLineExtended(vp,v1,v2,eps);
    else
        return CheckPointInTriangleExtended(vp,v1,v2,v3,n,eps);
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInTriangleShifted(T &d, vec3<T> vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n, const T shift=EPS<T>())
{
    // Shift the point vp towards the center of the triangle.
    vec3<T> vc = (v1+v2+v3)*static_cast<T>(1.0/3.0);
    vc = UnitV(vc - vp)*shift;
    d += vc.x;
    vp = vp + vc;
    
    return CheckPointInTriangleI(vp,v1,v2,v3,n);
}

template <typename T>
__host__ __device__ __forceinline__
bool CheckPointInTriangleProjected(const T &vpx, const T &vpy, const T &v1x, const T &v1y, const T &v2x, const T &v2y, const T &v3x, const T &v3y)
{
    // First edge.
    vec2<T> ven = UnitV(vec2<T>(v2y-v1y, v1x-v2x));
    T s = DotV(vec2<T>(v1x-vpx,v1y-vpy),ven);
    if ( GTZ_I(s) )
        return false;
    
    // Second edge.
    ven = UnitV(vec2<T>(v3y-v2y, v2x-v3x));
    s = DotV(vec2<T>(v2x-vpx,v2y-vpy),ven);
    if ( GTZ_I(s) )
        return false;
    
    // Third edge.
    ven = UnitV(vec2<T>(v1y-v3y, v3x-v1x));
    s = DotV(vec2<T>(v3x-vpx,v3y-vpy),ven);
    if ( GTZ_I(s) )
        return false;
    
    return true;
}

template <typename T>
__host__ __device__ __forceinline__
vec3<T> PointTriangleIntersection(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
    T dT = DotV(v1-vp,n) / DotV(n,n);
    vec3<T> vi = vp + n*dT;
    bool in_T = CheckPointInTriangleI(vi, v1, v2, v3, n);
    
    if (!in_T)
    {
        // Line 1.
        vi = PointLineIntersectionClamped(vp,v1,v2);
        dT = NormV(vp-vi);
        
        // Line 2.
        vec3<T> vic = PointLineIntersectionClamped(vp,v2,v3);
        T dL = NormV(vp-vic);
        if (dL < dT)
        {
            vi = vic;
            dT = dL;
        }
        
        // Line 3.
        vic = PointLineIntersectionClamped(vp,v3,v1);
        dL = NormV(vp-vic);
        if (dL < dT)
        {
            vi = vic;
            dT = dL;
        }
    }
    
    return vi;
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
vec3<T> PointFaceIntersection(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
    if (N_DIM==2)
        return PointLineIntersection(vp, v1, v2);
    return PointTriangleIntersection(vp, v1, v2, v3, n);
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
bool RayFaceIntersection_OneSided(const T &max_length, const vec3<T> &vp, const vec3<T> &ray, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
    T d = DotV(v1-vp,n) / DotV(ray,n);
    vec3<T> vi = vp + ray*d;
    
    return ((d > 0 && d < max_length) && CheckPointInFace<T,N_DIM>(vi,v1,v2,v3,n));
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
bool RayFaceIntersection_TwoSided(const T &max_length, const vec3<T> &vp, const vec3<T> &ray, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
    T d = DotV(v1-vp,n) / DotV(ray,n);
    vec3<T> vi = vp + ray*d;
    
    return (Tabs(d) < max_length && CheckPointInFace<T,N_DIM>(vi,v1,v2,v3,n));
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
bool PointInIndexedBin
(
    const vec3<T> &vp,
    const int &Id,
    const int &Nx,
    const int &Ny,
    const T &Lx,
    const T &Ly,
    const T &Lz,
    const T &dx_x,
    const T &dx_y,
    const T &dx_z
)
{
    bool b = true;
    
    // Along x.
    {
        int Id_x = Id % Nx;
        b = b && (vp.x >= static_cast<T>(Id_x)*Lx-dx_x && vp.x <= static_cast<T>(Id_x+1)*Lx+dx_x);
    }
    
    // Along y.
    {
        int Id_y = (Id/Nx) % Ny;
        b = b && (vp.y >= static_cast<T>(Id_y)*Ly-dx_y && vp.y <= static_cast<T>(Id_y+1)*Ly+dx_y);
    }
    
    // Along z.
    if (N_DIM==3)
    {
        int Id_z = (Id/Nx) / Ny;
        b = b && (vp.z >= static_cast<T>(Id_z)*Lz-dx_z && vp.z <= static_cast<T>(Id_z+1)*Lz+dx_z);
    }
    
    return b;
}

template <typename T>
__host__ __device__ __forceinline__
bool PointInCircle(const vec2<T> &vi, const vec2<T> &vc, const T &R, const T eps=EPS<T>())
{
    return NormSqV(vi-vc) < (R+eps)*(R+eps);
}

template <typename T>
__host__ __device__ __forceinline__
bool PointInCircle(const vec3<T> &vi, const vec3<T> &vc, const T &R, const T eps=EPS<T>())
{
    return NormSqV(vi-vc) < (R+eps)*(R+eps);
}

template <typename T>
__host__ __device__ __forceinline__
bool PointInSphere(const vec3<T> &vi, const vec3<T> &vc, const T &R, const T eps=EPS<T>())
{
    return NormSqV(vi-vc) < (R+eps)*(R+eps);
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
bool PointInSphereV(const vec3<T> &vi, const vec3<T> &vc, const T &R, const T eps=EPS<T>())
{
    if (N_DIM==2)
        return PointInCircle(vi,vc,R,eps);
    else
        return PointInSphere(vi,vc,R,eps);
}

// o====================================================================================
// | Auxilliary.
// o====================================================================================

template <typename T>
__device__ __forceinline__
int BlockwiseReduction(int t, int B, T *s_data)
{
    for (int s=B/2; s>0; s>>=1)
    {
        if (t < s)
        {
            s_data[t] = s_data[t] + s_data[t+s];
        }
        __syncthreads();
    }
    
    return 0;
}

template <typename T>
__device__ __forceinline__
int BlockwiseMaximum(int t, int B, T *s_data)
{
    for (int s=B/2; s>0; s>>=1)
    {
        if (t < s)
        {
            s_data[t] = Tmax(s_data[t], s_data[t+s]);
        }
        __syncthreads();
    }
    
    return 0;
}

#include "util_index.h"
#include "util_tribin.h"         // Triangle-AABB bin overlap.
#include "util_tribin_old.h"     // My own triangle-AABB bin overlap test. [REMOVE]
#include "util_math.h"           // Useful mathematics routines.
#include "debug_matlab.h"


#endif
