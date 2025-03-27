#include "mesh.h"

// Solution in FP32, geometry in FP32, in 2D/3D.
template class Mesh<float,float,2>;
template class Mesh<float,float,3>;

// Solution in FP32, geometry in FP64, in 2D/3D.
template class Mesh<float,double,2>;
template class Mesh<float,double,3>;

// Solution in FP64, geometry in FP32, in 2D/3D.
template class Mesh<double,float,2>;
template class Mesh<double,float,3>;

// Solution in FP64, geometry in FP64, in 2D/3D.
template class Mesh<double,double,2>;
template class Mesh<double,double,3>;
