#include "cppspec.h"
#include "geometry.h"
#include "mesh.h"
#include "solver_lbm.h"

#include "geometry_add.cu"
#include "geometry_bin_alt.cu"
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
#include "solver_lbm_impl.cu"
#include "solver_lbm_init.cu"
//
#include "solver_lbm_setic_D2Q9.cu"
#include "solver_lbm_setic_D3Q19.cu"
#include "solver_lbm_setic_D3Q27.cu"
#include "solver_lbm_collision_original_D2Q9.cu"
#include "solver_lbm_collision_original_D3Q19.cu"
#include "solver_lbm_collision_original_D3Q27.cu"
#include "solver_lbm_stream_original_D2Q9.cu"
#include "solver_lbm_stream_original_D3Q19.cu"
#include "solver_lbm_stream_original_D3Q27.cu"
#include "solver_lbm_interp_linear_original_D2Q9.cu"
#include "solver_lbm_interp_linear_original_D3Q19.cu"
#include "solver_lbm_interp_linear_original_D3Q27.cu"
#include "solver_lbm_average_original_D2Q9.cu"
#include "solver_lbm_average_original_D3Q19.cu"
#include "solver_lbm_average_original_D3Q27.cu"

int ReadInputFile(std::string input_file_directory, std::map<std::string, int> &input_map_int, std::map<std::string, double> &input_map_dbl, std::map<std::string, std::string> &input_map_str);

// Define a D2Q9 LBM argument pack.
constexpr LBMPack LP2D __attribute__((unused)) = LBMPack(&AP2D_DEF, VS_D2Q9, CM_BGK, IM_LINEAR);
constexpr LBMPack LP3D_1 __attribute__((unused)) = LBMPack(&AP3D_DEF, VS_D3Q19, CM_BGK, IM_LINEAR);
constexpr LBMPack LP3D_2 __attribute__((unused)) = LBMPack(&AP3D_DEF, VS_D3Q27, CM_BGK, IM_LINEAR);

// Typedefs and chosen packs.
typedef float REAL_s;
typedef float REAL_g;
constexpr ArgsPack APc = AP3D_DEF;
constexpr LBMPack LPc = LP3D_1;


int main(int argc, char *argv[])
{
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
	
	// Create a mesh.
	Mesh<REAL_s,REAL_g,&APc> mesh(input_map_int, input_map_dbl, input_map_str, LPc.N_Q);
	mesh.geometry = &geometry;
	geometry.mesh = &mesh;
	
	// Create a solver.
	Solver_LBM<REAL_s,REAL_g,&APc,&LPc> solver(&mesh, input_map_int, input_map_dbl, input_map_str);
	mesh.solver = &solver;
	
	// Solver loop (includes rendering and printing).
	mesh.M_AdvanceLoop();
	
	return 0;
}
