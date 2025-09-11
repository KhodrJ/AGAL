#include "cppspec.h"
#include "parser.h"
#include "geometry.h"
#include "geometry_bins.h"
#include "mesh.h"
#include "solver_lbm.h"

#include "geometry_add.cu"
#include "geometry_bins.cu"
#include "geometry_bins_cpu.cu"
#include "geometry_bins_gpu.cu"
#include "geometry_bins_draw.cu"
#include "geometry_convert.cu"
#include "geometry_dest.cu"
#include "geometry_import.cu"
#include "geometry_init.cu"
#include "geometry_print.cu"
#include "geometry_refine.cu"
// 
#include "mesh_advance.cu"
#include "mesh_amr.cu"
#include "mesh_comm.cu"
#include "mesh_voxelizer.cu"
#include "mesh_criterion.cu"
#include "mesh_dest.cu"
#include "mesh_init.cu"
#include "mesh_output.cu"
#include "mesh_print_uniform.cu"
#include "mesh_print_vthb.cu"
#include "mesh_print_vthb_patch.cu"
#include "mesh_print_image.cu"
#include "mesh_restart.cu"
//
#include "solver_lbm_advance.cu"
#include "solver_lbm_compute_macro.cu"
#include "solver_lbm_compute.cu"
#include "solver_lbm_criterion.cu"
#include "solver_lbm_init.cu"
#include "solver_lbm_identify_faces.cu"
#include "solver_lbm_kernels.cu"
#include "solver_lbm_kernels_comm.cu"
#include "solver_lbm_kernels_forces.cu"
#include "solver_lbm_kernels_debug.cu"

//int ReadInputFile(std::string input_file_directory, std::map<std::string, int> &input_map_int, std::map<std::string, double> &input_map_dbl, std::map<std::string, std::string> &input_map_str);

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
typedef double REAL_s;
typedef double REAL_g;

#ifdef USED2Q9
    constexpr ArgsPack APc = AP2D;
    constexpr LBMPack LPc __attribute__((unused)) = LP2D;
#endif
#ifdef USED3Q19
    constexpr ArgsPack APc = AP3D;
    constexpr LBMPack LPc __attribute__((unused)) = LP3D_1;
#endif
#ifdef USED3Q27
    constexpr ArgsPack APc = AP3D;
    constexpr LBMPack LPc __attribute__((unused)) = LP3D_2;
#endif


int main(int argc, char *argv[])
{
    // Read input file and use map to make solver input.
    std::string input_file_directory = "../input/";
    Parser parser(input_file_directory);
    
    // Create a new geometry and import from input folder.
    Geometry<REAL_s,REAL_g,&APc> geometry(&parser);
    if (geometry.G_LOADTYPE == static_cast<int>(LoadType::STL))
        geometry.G_ImportSTL_ASCII(geometry.G_FILENAME);
    else
    {
        geometry.G_ImportBoundariesFromTextFile();
        geometry.G_Convert_IndexListsToCoordList();
    }
    geometry.G_Init_Arrays_CoordsList_CPU();
    if (geometry.G_PRINT)
        geometry.G_PrintSTL();
    geometry.G_InitBins(BinMake::GPU,0);
    
    // Create a mesh.
    //const int N_U = LPc.N_Q + (APc.N_DIM+1); // Size of solution field: N_Q DDFs + 1 density + N_DIM velocity.
    const int N_U = 1;
    Mesh<REAL_s,REAL_g,&APc> mesh(&parser, N_U);
    mesh.M_AddGeometry(&geometry);
    
    // Create a solver.
    Solver_LBM<REAL_s,REAL_g,&APc,&LPc> solver(&mesh);
    mesh.M_AddSolver(&solver);
    
    // Solver loop (includes rendering and printing).
    //mesh.M_AdvanceLoop();
    mesh.M_Advance_RefineNearWall();
    mesh.M_Advance_PrintData(0,0);
    
    return 0;
}
