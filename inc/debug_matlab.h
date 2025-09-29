/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef DEBUG_MATLAB
#define DEBUG_MATLAB

// o====================================================================================
// | Standard output helpers.
// o====================================================================================

inline int DebugDrawSquareInMATLAB(std::ofstream &out, double x0, double x1, double y0, double y1, double c0, double c1, double c2)
{
    out << "plot([" << x0 << " " << x1 << " " << x1 << " " << x0 << " " << x0 << "],[" << y0 << " " << y0 << " " << y1 << " " << y1 << " " << y0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    
    return 0;
}

inline int DebugDrawCubeInMATLAB(std::ofstream &out, double x0, double x1, double y0, double y1, double z0, double z1, double c0, double c1, double c2)
{
    out << "plot3([" << x0 << " " << x1 << "],[" << y0 << " " << y0 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x0 << " " << x0 << "],[" << y0 << " " << y1 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x0 << " " << x1 << "],[" << y1 << " " << y1 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x1 << " " << x1 << "],[" << y0 << " " << y1 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    
    out << "plot3([" << x0 << " " << x1 << "],[" << y0 << " " << y0 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x0 << " " << x0 << "],[" << y0 << " " << y1 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x0 << " " << x1 << "],[" << y1 << " " << y1 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x1 << " " << x1 << "],[" << y0 << " " << y1 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    
    out << "plot3([" << x0 << " " << x0 << "],[" << y0 << " " << y0 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x0 << " " << x0 << "],[" << y1 << " " << y1 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x1 << " " << x1 << "],[" << y0 << " " << y0 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    out << "plot3([" << x1 << " " << x1 << "],[" << y1 << " " << y1 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
    
    return 0;
}

inline int DebugDrawLineInMATLAB(std::ofstream &out, double vx1, double vy1, double vx2, double vy2, double c0, double c1, double c2)
{
    out << "plot([" << vx1 << " " << vx2 << "],[" << vy1 << " " << vy2 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "],'Marker','o');\n";
    
    return 0;
}

inline int DebugDrawTriangleInMATLAB(std::ofstream &out, double vx1, double vy1, double vz1, double vx2, double vy2, double vz2, double vx3, double vy3, double vz3, double c0, double c1, double c2)
{
    out << "fill3([" << vx1 << " " << vx2 << " " << vx3 << " " << vx1 << "],[" << vy1 << " " << vy2 << " " << vy3 << " " << vy1 << "],[" << vz1 << " " << vz2 << " " << vz3 << " " << vz1 << "],[" << c0 << " " << c1 << " " << c2 << "]);\n";
    
    return 0;
}

// o====================================================================================
// | Device debug helpers.
// o====================================================================================

template <typename T>
__host__ __device__ __forceinline__
int DebugDraw2DSquareInMATLAB_DEV(vec3<T> vm, vec3<T> vM, char c1=' ', char c2=' ')
{
    printf("plot([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'%c%c');\n", vm.x, vM.x, vM.x, vm.x, vm.x, vm.y, vm.y, vM.y, vM.y, vm.y, c1, c2);
    
    return 0;
}

template <typename T, int DIM, int SIGN>
__host__ __device__ __forceinline__
int DebugDraw3DSquareInMATLAB_DEV(vec3<T> vm, vec3<T> vM, char c1=' ', char c2=' ')
{
    // DIM is the dimension of the plane which the square lies in.
    // SIGN indicates if the plane is aligned with the lower or upper bound of the given bounding box.
    
    if (DIM==2 && SIGN==0) printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'%c%c');\n", vm.x, vM.x, vM.x, vm.x, vm.x, vm.y, vm.y, vM.y, vM.y, vm.y, vm.z, vm.z, vm.z, vm.z, vm.z, c1, c2);
    if (DIM==2 && SIGN==1) printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'%c%c');\n", vm.x, vM.x, vM.x, vm.x, vm.x, vm.y, vm.y, vM.y, vM.y, vm.y, vM.z, vM.z, vM.z, vM.z, vM.z, c1, c2);
    
    if (DIM==0 && SIGN==0) printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'%c%c');\n", vm.x, vm.x, vm.x, vm.x, vm.x, vm.y, vM.y, vM.y, vm.y, vm.y, vm.z, vm.z, vM.z, vM.z, vm.z, c1, c2);
    if (DIM==0 && SIGN==1) printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'%c%c');\n", vM.x, vM.x, vM.x, vM.x, vM.x, vm.y, vM.y, vM.y, vm.y, vm.y, vm.z, vm.z, vM.z, vM.z, vm.z, c1, c2);
    
    if (DIM==1 && SIGN==0) printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'%c%c');\n", vm.x, vM.x, vM.x, vm.x, vm.x, vm.y, vm.y, vm.y, vm.y, vm.y, vm.z, vm.z, vM.z, vM.z, vm.z, c1, c2);
    if (DIM==1 && SIGN==1) printf("plot3([%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],[%17.15f %17.15f %17.15f %17.15f %17.15f],'%c%c');\n", vm.x, vM.x, vM.x, vm.x, vm.x, vM.y, vM.y, vM.y, vM.y, vM.y, vm.z, vm.z, vM.z, vM.z, vm.z, c1, c2);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDrawSquareInMATLAB_DEV(vec3<T> vm, vec3<T> vM, char c1=' ', char c2=' ')
{
    DebugDraw3DSquareInMATLAB_DEV<T,2,0>(vm,vM);
    DebugDraw3DSquareInMATLAB_DEV<T,2,1>(vm,vM);
    DebugDraw3DSquareInMATLAB_DEV<T,1,0>(vm,vM);
    DebugDraw3DSquareInMATLAB_DEV<T,1,1>(vm,vM);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDraw2DPointInMATLAB_DEV(vec2<T> vp, char c1='k', char c2=' ')
{
    printf("plot(%17.15f,%17.15f,'%c%c');\n", vp.x, vp.y, c1, c2);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDraw2DPointInMATLAB_DEV(vec3<T> vp, char c1='k', char c2=' ')
{
    printf("plot(%17.15f,%17.15f,'%c%c');\n", vp.x, vp.y, c1, c2);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDraw2DLineSegmentInMATLAB_DEV(vec3<T> v1, vec3<T> v2, char c1='k', char c2=' ', char c3=' ')
{
    printf("plot([%17.15f,%17.15f],[%17.15f,%17.15f],'%c%c%c');\n", v1.x, v2.x, v1.y, v2.y, c1, c2, c3);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDraw3DPointInMATLAB_DEV(vec3<T> vp, char c1='k', char c2=' ')
{
    printf("plot3(%17.15f,%17.15f,%17.15f,'%c%c');\n", vp.x, vp.y, vp.z, c1, c2);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDraw3DLineSegmentInMATLAB_DEV(vec3<T> v1, vec3<T> v2, char c1='k', char c2=' ', char c3=' ')
{
    printf("plot3([%17.15f,%17.15f],[%17.15f,%17.15f],[%17.15f,%17.15f],'%c%c%c');\n", v1.x, v2.x, v1.y, v2.y, v1.z, v2.z, c1, c2, c3);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDraw3DTriangleInMATLAB_DEV(vec3<T> v1, vec3<T> v2, vec3<T> v3, char c1='k', char c2=' ')
{
    printf("plot3([%17.15f,%17.15f,%17.15f,%17.15f],[%17.15f,%17.15f,%17.15f,%17.15f],[%17.15f,%17.15f,%17.15f,%17.15f],'%c%c');\n", v1.x, v2.x, v3.x, v1.x, v1.y, v2.y, v3.y, v1.y, v1.z, v2.z, v3.z, v3.y, c1, c2);
    
    return 0;
}

template <typename T>
__host__ __device__ __forceinline__
int DebugDrawFilled3DTriangleInMATLAB_DEV(vec3<T> v1, vec3<T> v2, vec3<T> v3, char c1='k', char c2=' ')
{
    printf("fill3([%17.15f,%17.15f,%17.15f],[%17.15f,%17.15f,%17.15f],[%17.15f,%17.15f,%17.15f],'%c%c');\n", v1.x, v2.x, v3.x, v1.y, v2.y, v3.y, v1.z, v2.z, v3.z, c1, c2);
    
    return 0;
}

#endif
