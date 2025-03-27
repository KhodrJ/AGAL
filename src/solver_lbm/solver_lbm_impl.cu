#include "solver_lbm.h"


template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_SetIC(int i_dev, int L)
{
	if (VS == VS_D2Q9)
		S_SetInitialConditions_D2Q9(i_dev,L);
	if (VS == VS_D3Q19)
		S_SetInitialConditions_D3Q19(i_dev,L);
	if (VS == VS_D3Q27)
		S_SetInitialConditions_D3Q27(i_dev,L);
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Collide(int i_dev, int L)
{
	if (VS == VS_D2Q9)
 		S_Collision_Original_D2Q9(i_dev,L);
 	if (VS == VS_D3Q19)
 		S_Collision_Original_D3Q19(i_dev,L);
 	if (VS == VS_D3Q27)
 		S_Collision_Original_D3Q27(i_dev,L);
	
// 	if (VS == VS_D2Q9)
// 	{
// 		if (CM == CM_BGK)
// 			S_Collide_BGK_D2Q9(i_dev,L);
// 		if (CM == CM_TRT)
// 			S_Collide_TRT_D2Q9(i_dev,L);
// 		if (CM == CM_MRT)
// 			S_Collide_MRT_D2Q9(i_dev,L);
// 	}
// 	if (VS == VS_D3Q19)
// 	{
// 		if (CM == CM_BGK)
// 			S_Collide_BGK_D3Q19(i_dev,L);
// 		if (CM == CM_TRT)
// 			S_Collide_TRT_D3Q19(i_dev,L);
// 		if (CM == CM_MRT)
// 			S_Collide_MRT_D3Q19(i_dev,L);
// 	}
// 	if (VS == VS_D3Q27)
// 	{
// 		if (CM == CM_BGK)
// 			S_Collide_BGK_D3Q27(i_dev,L);
// 		if (CM == CM_TRT)
// 			S_Collide_TRT_D3Q27(i_dev,L);
// 		if (CM == CM_MRT)
// 			S_Collide_MRT_D3Q27(i_dev,L);
// 	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Stream(int i_dev, int L)
{
 	if (VS == VS_D2Q9)
 		S_Stream_Original_D2Q9(i_dev,L);
 	if (VS == VS_D3Q19)
 		S_Stream_Original_D3Q19(i_dev,L);
 	if (VS == VS_D3Q27)
 		S_Stream_Original_D3Q27(i_dev,L);
	
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Interpolate(int i_dev, int L, int var)
{
	if (VS == VS_D2Q9)
	{
		if (IM == IM_LINEAR)
			S_Interpolate_Linear_Original_D2Q9(i_dev,L,var);
// 		if (IM == IM_QUADRATIC)
// 			S_Interpolate_Quadratic_D2Q9(i_dev,L,var);
// 		if (IM == IM_CUBIC)
// 			S_Interpolate_Cubic_D2Q9(i_dev,L,var);
	}
	if (VS == VS_D3Q19)
	{
		if (IM == IM_LINEAR)
			S_Interpolate_Linear_Original_D3Q19(i_dev,L,var);
// 		if (IM == IM_QUADRATIC)
// 			S_Interpolate_Quadratic_D3Q19(i_dev,L,var);
// 		if (IM == IM_CUBIC)
// 			S_Interpolate_Cubic_D3Q19(i_dev,L,var);
	}
	if (VS == VS_D3Q27)
	{
		if (IM == IM_LINEAR)
			S_Interpolate_Linear_Original_D3Q27(i_dev,L,var);
// 		if (IM == IM_QUADRATIC)
// 			S_Interpolate_Quadratic_D3Q27(i_dev,L,var);
// 		if (IM == IM_CUBIC)
// 			S_Interpolate_Cubic_D3Q27(i_dev,L,var);
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Average(int i_dev, int L, int var)
{
	if (VS == VS_D2Q9)
		S_Average_Original_D2Q9(i_dev,L,var);
	if (VS == VS_D3Q19)
		S_Average_Original_D3Q19(i_dev,L,var);
	if (VS == VS_D3Q27)
		S_Average_Original_D3Q27(i_dev,L,var);
	
	return 0;
}
