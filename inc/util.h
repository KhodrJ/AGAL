#ifndef UTIL_H
#define UTIL_H

template <class T> __host__ __device__ __forceinline__ T EPS(T a);
template <> __host__ __device__ __forceinline__ float EPS(float a) { return 1e-7F; }
template <> __host__ __device__ __forceinline__ double EPS(double a) { return 1e-15; }

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
	__host__ __device__ vec2<T> operator-(const vec2<T> &w) const
	{
		return vec2<T>(x - w.x, y - w.y);
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
	__host__ __device__ T min() const
	{
		return Tmin(x,y);
	}
	__host__ __device__ T max() const
	{
		return Tmax(x,y);
	}
	__host__ __device__ bool InRegion_Cube(const vec2<T> &vm, const vec2<T> &vM) const
	{
		return (x >= vm.x && x <= vM.x && y >= vm.y && y <= vM.y);
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
	__host__ __device__ vec3<T> operator-(const vec3<T> &w) const
	{
		return vec3<T>(x - w.x, y - w.y, z - w.z);
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
	__host__ __device__ T min() const
	{
		return Tmin(Tmin(x,y),z);
	}
	__host__ __device__ T max() const
	{
		return Tmax(Tmax(x,y),z);
	}
	__host__ __device__ bool InRegion_Cube(const vec3<T> &vm, const vec3<T> &vM) const
	{
		return (x >= vm.x && x <= vM.x && y >= vm.y && y <= vM.y && z >= vm.z && z <= vM.z);
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
__device__
void GetShortestLinkVertex2D(T &vxp, T &vyp, T &vx1, T &vy1, T &vx2, T &vy2, T &sx, T &sy, T &nx, T &ny, T &dmin, T &tmin)
{
	// First link.
	sx = vx1 + (T)(1e-4)*(vx2-vx1) - vxp;
	sy = vy1 + (T)(1e-4)*(vy2-vy1) - vyp;
	dmin = Tsqrt(sx*sx + sy*sy);
	tmin = sx*nx + sy*ny;
	tmin = Tacos(tmin);
	
	// Second link.
	sx = vx2 - (T)(1e-4)*(vx2-vx1) - vxp;
	sy = vy2 - (T)(1e-4)*(vy2-vy1) - vyp;
	if (Tsqrt(sx*sx + sy*sy) < dmin)
	{
		dmin = Tsqrt(sx*sx + sy*sy);
		tmin = sx*nx + sy*ny;
		tmin = Tacos(tmin);
	}
}

/*
template <typename T>
__device__
void GetShortestLinkEdge3D(T &vxp, T &vyp, T &vzp, T &vx1, T &vy1, T &vz1, T &vx2, T &vy2, T &vz2, T &sx, T &sy, T &sz, T &nx, T &ny, T &nz, T &dmin, T &tmin, T &tmp, bool &C)
{
	// ith link.
	tmp = (vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1);
	tmp = ( (vx1-vxp)*(vx2-vx1) + (vy1-vyp)*(vy2-vy1) + (vz1-vzp)*(vz2-vz1) ) / tmp; // tmp now stores dot(v1-vx,v2-v1)/norm(v2-v1)^2;
	sx = vx1 - (vx2-vx1)*tmp;
	sy = vy1 - (vy2-vy1)*tmp;
	sz = vz1 - (vz2-vz1)*tmp;
	C = CheckPointInLine3D(sx, sy, sz, vx1, vy1, vz1, vx2, vy2, vz2);
	dmin = Tsqrt((vxp-sx)*(vxp-sx) + (vyp-sy)*(vyp-sy) + (vzp-sz)*(vzp-sz)); // tmp now stores the length of the link
	tmin = (vxp-sx)*nx + (vyp-sy)*ny + (vzp-sz)*nz;
	//tmin = Tacos(tmin);
}
template <typename T>
__device__
void GetShortedLinkVertex3D(T &vxp, T &vyp, T &vzp, T &vx1, T &vy1, T &vz1, T &vx2, T &vy2, T &vz2, T &vx3, T &vy3, T &vz3, T &nx, T &ny, T &nz, T &dmin, T &tmin, T &tmp)
{
	// First vertex.
	dmin = Tsqrt((vxp-vx1)*(vxp-vx1) + (vyp-vy1)*(vyp-vy1) + (vzp-vz1)*(vzp-vz1));
	tmin = (vxp-vx1)*nx + (vyp-vy1)*ny + (vzp-vz1)*nz;
	
	// Second vertex.
	tmp = Tsqrt((vxp-vx2)*(vxp-vx2) + (vyp-vy2)*(vyp-vy2) + (vzp-vz2)*(vzp-vz2));
	if (tmp < dmin)
	{
		dmin = tmp;
		tmin = (vxp-vx2)*nx + (vyp-vy2)*ny + (vzp-vz2)*nz;
	}
	
	// Third vertex.
	tmp = Tsqrt((vxp-vx3)*(vxp-vx3) + (vyp-vy3)*(vyp-vy3) + (vzp-vz3)*(vzp-vz3));
	if (tmp < dmin)
	{
		dmin = tmp;
		tmin = (vxp-vx3)*nx + (vyp-vy3)*ny + (vzp-vz3)*nz;
	}
}
*/

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



















template <typename T, int N_DIM>
__device__ __forceinline__ vec3<T> FaceNormal(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3)
{
	if (N_DIM==2)
		return vec3<T>(v2.y-v1.y,v1.x-v2.x);
	return CrossV(v2-v1,v3-v1);
}

template <typename T, int N_DIM>
__device__ __forceinline__ vec3<T> FaceNormalUnit(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3)
{
	vec3<T> n = FaceNormal<T,N_DIM>(v1,v2,v3);
	return UnitV(n);
}

template <typename T>
__device__ __forceinline__ vec3<T> InwardNormal(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &n)
{
	return CrossV(v2-v1,n);
}

template <typename T>
__device__ __forceinline__ vec3<T> InwardNormalUnit(const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &n)
{
	vec3<T> e = CrossV(v2-v1,n);
	return UnitV(e);
}

template <typename T>
__device__ __forceinline__
vec3<T> PointLineIntersection(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2)
{
	vec3<T> e = v2-v1;
	vec3<T> w = vp-v1;
	T t = DotV(e,w)/DotV(e,e);
	t = Tmax(static_cast<T>(0.0),Tmin(static_cast<T>(1.0),t));
	
	return v1 + e*t;
}

template <typename T>
__device__ __forceinline__
// bool CheckPointInTriangleA(T vxp, T vyp, T vzp, T vx1, T vy1, T vz1, T vx2, T vy2, T vz2, T vx3, T vy3, T vz3, T nx, T ny, T nz)
bool CheckPointInTriangleA(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
	// First edge.
	vec3<T> ven = InwardNormalUnit(v1,v2,n);
	T s = DotV(v1-vp,ven);
	if (s <= 0)
		return false;
	
	// Second edge.
	ven = InwardNormalUnit(v2,v3,n);
	s = DotV(v2-vp,ven);
	if (s <= 0)
		return false;
	
	// Third edge.
	ven = InwardNormalUnit(v3,v1,n);
	s = DotV(v3-vp,ven);
	if (s <= 0)
		return false;
	
	return true;
}

template <typename T>
__device__ __forceinline__
vec3<T> PointTriangleIntersection(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
	T dT = DotV(v1-vp,n);
	vec3<T> vi = vp + n*dT;
	bool in_T = CheckPointInTriangleA(vi, v1, v2, v3, n);
	
	if (!in_T)
	{
		// Line 1.
		vi = PointLineIntersection(vp,v1,v2);
		dT = NormV(vp-vi);
		
		// Line 2.
		vec3<T> vic = PointLineIntersection(vp,v2,v3);
		T dL = NormV(vp-vic);
		if (dL < dT)
		{
			vi = vic;
			dT = dL;
		}
		
		// Line 3.
		vic = PointLineIntersection(vp,v3,v1);
		dL = NormV(vp-vic);
		if (dL < dT)
		{
			vi = vic;
			dT = dL;
		}
	}
	
	return vi;
	//return (vix-vxp)*nx + (vix-vxp)*ny + (vix-vxp)*nz;
}

template <typename T, int N_DIM>
__device__ __forceinline__
vec3<T> PointFaceIntersection(const vec3<T> &vp, const vec3<T> &v1, const vec3<T> &v2, const vec3<T> &v3, const vec3<T> &n)
{
	if (N_DIM==2)
		return PointLineIntersection(vp, v1, v2);
	return PointTriangleIntersection(vp, v1, v2, v3, n);
}

template <typename T, int N_DIM>
__device__ __forceinline__
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



#include "util_tribin.h"


#endif
