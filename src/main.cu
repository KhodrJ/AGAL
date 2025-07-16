#include "cppspec.h"
#include "geometry.h"
#include "mesh.h"
#include "solver_lbm.h"

#include "geometry_add.cu"
#include "geometry_bin_alt.cu"
#include "geometry_bin_alt_3D.cu"
#include "geometry_convert.cu"
#include "geometry_dest.cu"
#include "geometry_import.cu"
#include "geometry_init.cu"
#include "geometry_print.cu"
// 
#include "mesh_advance.cu"
#include "mesh_amr.cu"
#include "mesh_comm.cu"
#include "mesh_criterion_geometry_binned.cu"
#include "mesh_fill_binned.cu"
#include "mesh_criterion.cu"
#include "mesh_dest.cu"
#include "mesh_init.cu"
#include "mesh_output.cu"
#include "mesh_print_uniform.cu"
#include "mesh_print_vthb.cu"
#include "mesh_restart.cu"
//
#include "solver_lbm_advance.cu"
#include "solver_lbm_compute_macro.cu"
#include "solver_lbm_compute.cu"
#include "solver_lbm_criterion.cu"
// #include "solver_lbm_impl.cu"
#include "solver_lbm_init.cu"
#include "solver_lbm_identify_faces.cu"
//
#define USED2Q9
// #define USED3Q19
// #define USED3Q27
//
#ifdef USED2Q9
	#include "solver_lbm_setic_D2Q9.cu"
	#include "solver_lbm_collision_new_D2Q9.cu"
// 	#include "solver_lbm_collision_new_s1_D2Q9.cu"
// 	#include "solver_lbm_collision_new_s2_D2Q9.cu"
	#include "solver_lbm_imposebc_D2Q9.cu"
	#include "solver_lbm_stream_new_D2Q9.cu"
	#include "solver_lbm_interp_linear_original_D2Q9.cu"
	#include "solver_lbm_interp_cubic_original_D2Q9.cu"
	#include "solver_lbm_average_original_D2Q9.cu"
	#include "solver_lbm_debug_drawgeom_D2Q9.cu"
	#include "solver_lbm_compute_forces_mea_D2Q9.cu"
	#include "solver_lbm_compute_forces_cv_D2Q9.cu"
#endif
#ifdef USED3Q19
	#include "solver_lbm_setic_D3Q19.cu"
	#include "solver_lbm_collision_new_D3Q19.cu"
// 	#include "solver_lbm_collision_new_s1_D3Q19.cu"
// 	#include "solver_lbm_collision_new_s2_D3Q19.cu"
	#include "solver_lbm_imposebc_D3Q19.cu"
	#include "solver_lbm_stream_new_D3Q19.cu"
	#include "solver_lbm_interp_linear_original_D3Q19.cu"
	#include "solver_lbm_interp_cubic_original_D3Q19.cu"
	#include "solver_lbm_average_original_D3Q19.cu"
	#include "solver_lbm_debug_drawgeom_D3Q19.cu"
	#include "solver_lbm_compute_forces_mea_D3Q19.cu"
	#include "solver_lbm_compute_forces_cv_D3Q19.cu"
#endif
#ifdef USED3Q27
	#include "solver_lbm_setic_D3Q27.cu"
	#include "solver_lbm_collision_new_D3Q27.cu"
// 	#include "solver_lbm_collision_new_s1_D3Q27.cu"
// 	#include "solver_lbm_collision_new_s2_D3Q27.cu"
	#include "solver_lbm_imposebc_D3Q27.cu"
	#include "solver_lbm_stream_new_D3Q27.cu"
	#include "solver_lbm_interp_linear_original_D3Q27.cu"
	#include "solver_lbm_interp_cubic_original_D3Q27.cu"
	#include "solver_lbm_average_original_D3Q27.cu"
	#include "solver_lbm_debug_drawgeom_D3Q27.cu"
	#include "solver_lbm_compute_forces_mea_D3Q27.cu"
	#include "solver_lbm_compute_forces_cv_D3Q27.cu"
#endif

int ReadInputFile(std::string input_file_directory, std::map<std::string, int> &input_map_int, std::map<std::string, double> &input_map_dbl, std::map<std::string, std::string> &input_map_str);

// Define some mesh argument packs for the tests.
constexpr int M_BLOCK_C = 128;
constexpr int M_LBLOCK_C = 1;
constexpr int M_LWBLOCK_C = 24;
constexpr ArgsPack AP2D __attribute__((unused)) = ArgsPack(2,M_BLOCK_C,1,1,M_LBLOCK_C,M_LWBLOCK_C,2048);
constexpr ArgsPack AP3D __attribute__((unused)) = ArgsPack(3,M_BLOCK_C,1,1,M_LBLOCK_C,M_LWBLOCK_C,2048);

// Define some LBM argument packs for the tests.
constexpr LBMPack LP2D __attribute__((unused)) = LBMPack(&AP2D, VS_D2Q9, CM_BGK, IM_CUBIC);
constexpr LBMPack LP3D_1 __attribute__((unused)) = LBMPack(&AP3D, VS_D3Q19, CM_BGK, IM_CUBIC);
constexpr LBMPack LP3D_2 __attribute__((unused)) = LBMPack(&AP3D, VS_D3Q27, CM_BGK, IM_CUBIC);

// Typedefs and chosen packs.
typedef float REAL_s;
typedef float REAL_g;
constexpr ArgsPack APc = AP2D;
constexpr LBMPack LPc = LP2D;


int main(int argc, char *argv[])
{
	// Debug.
	// [-]   This block increases printf limit.
	//size_t size = 5 * 1024 * 1024;
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size);
	
	// Read input file and use map to make solver input.
	std::string input_file_directory = "../input/";
	std::map<std::string, int> input_map_int;
	std::map<std::string, double> input_map_dbl;
	std::map<std::string, std::string> input_map_str;
	ReadInputFile(input_file_directory, input_map_int, input_map_dbl, input_map_str);
	
	// Create a new geometry and import from input folder.
	Geometry<REAL_s,REAL_g,&APc> geometry(input_map_int, input_map_dbl, input_map_str);
	if (geometry.G_LOADTYPE == V_GEOMETRY_LOADTYPE_STL)
		geometry.G_ImportSTL_ASCII(geometry.G_FILENAME);
	else
	{
		geometry.G_ImportBoundariesFromTextFile(0);
		geometry.G_Convert_IndexListsToCoordList(0);
	}
	geometry.G_Init_Arrays_CoordsList_CPU(0);
	if (geometry.G_PRINT)
		geometry.G_PrintSTL(0);
	geometry.G_MakeBins(0);
	geometry.G_MakeBins3D(0);
	
	// Create a mesh.
	int enable_aux_data = 1;
	Mesh<REAL_s,REAL_g,&APc> mesh(input_map_int, input_map_dbl, input_map_str, LPc.N_Q, enable_aux_data, APc.N_DIM+1+1);
	mesh.M_AddGeometry(&geometry);
	//
	//mesh.geometry = &geometry;
	//geometry.mesh = &mesh;
	//mesh.geometry_init = 1;
	
	// Create a solver.
	Solver_LBM<REAL_s,REAL_g,&APc,&LPc> solver(&mesh, input_map_int, input_map_dbl, input_map_str);
	mesh.M_AddSolver(&solver);
	//
	//mesh.solver = &solver;
	
	// Solver loop (includes rendering and printing).
	mesh.M_AdvanceLoop();
	
	return 0;
}
