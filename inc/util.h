#ifndef UTIL_H
#define UTIL_H

template <class T> __device__ __forceinline__ T EPS(T a);
template <> __device__ __forceinline__ float EPS(float a) { return 1e-7F; }
template <> __device__ __forceinline__ double EPS(double a) { return 1e-15; }

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

template <typename T>
__device__
bool CheckInLine(T &d, T &vxp, T &vx0, T &vx1)
{
	return (d > (T)0.0 && d < (T)1.0 && vxp > vx0 && vxp < vx1);
}

template <typename T>
__device__
bool CheckInRect(T &d, T &vxp, T &vyp, T &vx0, T &vy0, T &vx1, T &vy1)
{
	return (d > (T)0.0 && d < (T)1.0 && vxp > vx0 && vxp < vx1 && vyp > vy0 && vyp < vy1);
}

#endif
