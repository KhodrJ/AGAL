#include "geometry.h"

// Solution in FP32, geometry in FP32, in 2D/3D.
// template class Geometry<float,float,&AP2D_DEF>;
// template class Geometry<float,float,&AP3D_DEF>;
// 
// Solution in FP32, geometry in FP64, in 2D/3D.
// template class Geometry<float,double,&AP2D_DEF>;
// template class Geometry<float,double,&AP3D_DEF>;
// 
// Solution in FP64, geometry in FP32, in 2D/3D.
// template class Geometry<double,float,&AP2D_DEF>;
// template class Geometry<double,float,&AP3D_DEF>;
// 
// Solution in FP64, geometry in FP64, in 2D/3D.
// template class Geometry<double,double,&AP2D_DEF>;
// template class Geometry<double,double,&AP3D_DEF>;


// #include "geometry_init.cu"
// #include "geometry_dest.cu"
// 
// template Geometry<float,float,&AP2D_DEF>::Geometry(std::map<std::string, int> params_int,std::map<std::string, double> params_dbl,std::map<std::string, std::string> params_str);
// template Geometry<float,double,&AP2D_DEF>::Geometry(std::map<std::string, int> params_int,std::map<std::string, double> params_dbl,std::map<std::string, std::string> params_str);
// template Geometry<double,float,&AP2D_DEF>::Geometry(std::map<std::string, int> params_int,std::map<std::string, double> params_dbl,std::map<std::string, std::string> params_str);
// template Geometry<double,double,&AP2D_DEF>::Geometry(std::map<std::string, int> params_int,std::map<std::string, double> params_dbl,std::map<std::string, std::string> params_str);
// 
// template int Geometry<float,float,&AP2D_DEF>::G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
// template int Geometry<float,double,&AP2D_DEF>::G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
// template int Geometry<double,float,&AP2D_DEF>::G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
// template int Geometry<double,double,&AP2D_DEF>::G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
// 
// template int Geometry<float,float,&AP2D_DEF>::G_Init_Arrays_IndexLists_CPU(int i_dev);
// template int Geometry<float,double,&AP2D_DEF>::G_Init_Arrays_IndexLists_CPU(int i_dev);
// template int Geometry<double,float,&AP2D_DEF>::G_Init_Arrays_IndexLists_CPU(int i_dev);
// template int Geometry<double,double,&AP2D_DEF>::G_Init_Arrays_IndexLists_CPU(int i_dev);
// 
// template int Geometry<float,float,&AP2D_DEF>::G_Init_Arrays_CoordsList_CPU(int i_dev);
// template int Geometry<float,double,&AP2D_DEF>::G_Init_Arrays_CoordsList_CPU(int i_dev);
// template int Geometry<double,float,&AP2D_DEF>::G_Init_Arrays_CoordsList_CPU(int i_dev);
// template int Geometry<double,double,&AP2D_DEF>::G_Init_Arrays_CoordsList_CPU(int i_dev);
// 
// template int Geometry<float,float,&AP2D_DEF>::G_UpdateCounts(int i_dev);
// template int Geometry<float,double,&AP2D_DEF>::G_UpdateCounts(int i_dev);
// template int Geometry<double,float,&AP2D_DEF>::G_UpdateCounts(int i_dev);
// template int Geometry<double,double,&AP2D_DEF>::G_UpdateCounts(int i_dev);
// 
// template Geometry<float,float,&AP2D_DEF>::~Geometry();
// template Geometry<float,double,&AP2D_DEF>::~Geometry();
// template Geometry<double,float,&AP2D_DEF>::~Geometry();
// template Geometry<double,double,&AP2D_DEF>::~Geometry();
// 
// template int Geometry<float,float,&AP2D_DEF>::G_Dest();
// template int Geometry<float,double,&AP2D_DEF>::G_Dest();
// template int Geometry<double,float,&AP2D_DEF>::G_Dest();
// template int Geometry<double,double,&AP2D_DEF>::G_Dest();
// 
// template int Geometry<float,float,&AP2D_DEF>::G_Dest_Arrays_IndexLists(int i_dev);
// template int Geometry<float,double,&AP2D_DEF>::G_Dest_Arrays_IndexLists(int i_dev);
// template int Geometry<double,float,&AP2D_DEF>::G_Dest_Arrays_IndexLists(int i_dev);
// template int Geometry<double,double,&AP2D_DEF>::G_Dest_Arrays_IndexLists(int i_dev);
// 
// template int Geometry<float,float,&AP2D_DEF>::G_Dest_Arrays_CoordsList(int i_dev);
// template int Geometry<float,double,&AP2D_DEF>::G_Dest_Arrays_CoordsList(int i_dev);
// template int Geometry<double,float,&AP2D_DEF>::G_Dest_Arrays_CoordsList(int i_dev);
// template int Geometry<double,double,&AP2D_DEF>::G_Dest_Arrays_CoordsList(int i_dev);
