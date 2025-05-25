#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "cppspec.h"

constexpr int V_GEOMETRY_LOADTYPE_STL               = 0;
constexpr int V_GEOMETRY_LOADTYPE_TXT               = 1;



template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Mesh;

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Geometry
{
	private:
	
	// o====================================================================================
	// | Internal parameters.
	// o====================================================================================
	
	int             init_index_lists        = 0;            ///< Indicates if index lists were initialized.
	int             init_coords_list        = 0;            ///< Indicates if face coordinates list was initialized.
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	int G_Init(std::map<std::string, int> params_int, std::map<std::string, double> params_dbl, std::map<std::string, std::string> params_str);
	int G_Dest();
	
	
	
	
	
	public:
	
	Mesh<ufloat_t,ufloat_g_t,AP> *mesh;
	
	// Constants.
	const int       N_DIM                   = AP->N_DIM;
	const int       N_Q_max                 = AP->N_Q_max;
	const int       Nqx                     = AP->Nqx;
	const int       N_CHILDREN              = AP->N_CHILDREN;
	const int       N_QUADS                 = AP->N_QUADS;
	const int       M_TBLOCK                = AP->M_TBLOCK;
	const int       M_HBLOCK                = AP->M_HBLOCK;
	const int       M_CBLOCK                = AP->M_CBLOCK;
	const int       M_LBLOCK                = AP->M_LBLOCK;
	const int       M_BLOCK                 = AP->M_BLOCK;
	const int       M_RNDOFF                = AP->M_RNDOFF;
	const int       N_DEV                   = AP->N_DEV;
	
	// o====================================================================================
	// | Geometry parameters.
	// o====================================================================================
	
	float           Lx                      = 1.0F;         ///< Length of domain in x-axis (in meters).
	float           Ly                      = 1.0F;         ///< Length of domain in y-axis as a fraction of @ref Lx.
	float           Lz                      = 1.0F;         ///< Length of domain in z-axis as a fraction of @ref Lx.
	int             Nx                      = 1;            ///< Number of cells along x-axis of domain.
	ufloat_g_t      dx                      = 1.0;          ///< Spatial step.
	int             MAX_LEVELS_WALL         = 1;            ///< Maximum number of grids for the domain boundary alone.          
	ufloat_g_t      G_NEAR_WALL_DISTANCE    = 1;            ///< The near-wall distance for refinement.
	int             n_bins_v                = 1;            ///< Number of bins to use for voxelization.
	int             n_bins_b                = 1;            ///< Number of bins to use for enforcing boundary conditions.
	int             G_LOADTYPE              = 0;            ///< The type of load to perform [0: from and STL file, 1: from the txt file].
	int             G_PRINT                 = 0;            ///< Indicates if the geometry should be printed in STL format after processing.
	int             G_BIN_DENSITY           = 1;            ///< The number of bins to divide the geometry surface.
	int             G_BIN_FRAC              = 1;            ///< Fraction of bin arrays to consider at a time.
	std::string     input_dir;                              ///< Input directory.
	std::string     output_dir;                             ///< Output directory.
	std::string     G_FILENAME;
	
	// o====================================================================================
	// | CPU parameters.
	// o====================================================================================
	
	//! Number of nodes.
	int             *n_nodes = new int[N_DEV];
	
	//! Number of vertices rounded to 32 for alignment.
	int             *n_nodes_a = new int[N_DEV];
	
	//! Number of faces.
	int             *n_faces = new int[N_DEV];
	
	//! Number of faces rounded to 32 for alignment.
	int             *n_faces_a = new int[N_DEV];
	
	//! Array of geometry node locations.
	ufloat_g_t      **geom_f_node_X = new ufloat_g_t*[N_DEV];
	
	//! Array of face indices.
	int             **geom_ID_face = new int*[N_DEV];
	
	//! Array of face vertices.
	ufloat_g_t      **geom_f_face_X = new ufloat_g_t*[N_DEV];
	
	//! Array of geometry face attributes attributes.
	ufloat_g_t      **geom_ID_face_attr = new ufloat_g_t*[N_DEV];
	
	int **bin_indicators_v = new int*[N_DEV];
	int **binned_face_ids_v = new int*[N_DEV];
	int **binned_face_ids_n_v = new int*[N_DEV];
	int **binned_face_ids_N_v = new int*[N_DEV];
	int **c_bin_indicators_v = new int*[N_DEV];
	int **c_binned_face_ids_v = new int*[N_DEV];
	int **c_binned_face_ids_n_v = new int*[N_DEV];
	int **c_binned_face_ids_N_v = new int*[N_DEV];
	
	int **bin_indicators_b = new int*[N_DEV];
	int **binned_face_ids_b = new int*[N_DEV];
	int **binned_face_ids_n_b = new int*[N_DEV];
	int **binned_face_ids_N_b = new int*[N_DEV];
	int **c_bin_indicators_b = new int*[N_DEV];
	int **c_binned_face_ids_b = new int*[N_DEV];
	int **c_binned_face_ids_n_b = new int*[N_DEV];
	int **c_binned_face_ids_N_b = new int*[N_DEV];
	
	std::vector<ufloat_g_t>     v_geom_f_node_X;
	std::vector<ufloat_g_t>     v_geom_f_node_Y;
	std::vector<ufloat_g_t>     v_geom_f_node_Z;
	std::vector<int>        v_geom_ID_face_1;
	std::vector<int>        v_geom_ID_face_2;
	std::vector<int>        v_geom_ID_face_3;
	std::vector<ufloat_g_t>     v_geom_f_face_1_X;
	std::vector<ufloat_g_t>     v_geom_f_face_1_Y;
	std::vector<ufloat_g_t>     v_geom_f_face_1_Z;
	std::vector<ufloat_g_t>     v_geom_f_face_2_X;
	std::vector<ufloat_g_t>     v_geom_f_face_2_Y;
	std::vector<ufloat_g_t>     v_geom_f_face_2_Z;
	std::vector<ufloat_g_t>     v_geom_f_face_3_X;
	std::vector<ufloat_g_t>     v_geom_f_face_3_Y;
	std::vector<ufloat_g_t>     v_geom_f_face_3_Z;
	
	// o====================================================================================
	// | GPU parameters.
	// o====================================================================================
	
	//! GPU counterpart of @ref geom_f_node_X.
	ufloat_g_t      **c_geom_f_node_X = new ufloat_g_t*[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face.
	int             **c_geom_ID_face = new int*[N_DEV];
	
	//! GPU counterpart of @ref geom_f_face_X.
	ufloat_g_t      **c_geom_f_face_X = new ufloat_g_t*[N_DEV];
	
	//! GPU counterpart of @ref geom_ID_face_attr.
	ufloat_g_t      **c_geom_ID_face_attr = new ufloat_g_t*[N_DEV];
	
	// o====================================================================================
	// | Routines.
	// o====================================================================================
	
	int G_ClearVectors();
	int G_ImportBoundariesFromTextFile(int i_dev);
	int G_ImportBoundariesFromSTLFile(int i_dev);
	int G_ImportSTL_ASCII(std::string filename);
	int G_PrintSTL(int i_dev);
	int G_PrintOBJ(int i_dev);
	
	int G_UpdateCounts(int i_dev);
	int G_Init_Arrays_IndexLists_CPU(int i_dev);
	int G_Init_Arrays_CoordsList_CPU(int i_dev);
	int G_Dest_Arrays_IndexLists(int i_dev);
	int G_Dest_Arrays_CoordsList(int i_dev);
	int G_Convert_IndexListsToCoordList(int i_dev);
	int G_Convert_CoordListToIndexLists(int i_dev);
	int G_CoordList_MachineEps(int i_dev);
	int G_MakeBins(int i_dev);
	int G_DrawBinsAndFaces(int i_dev);
	int G_MakeBins3D(int i_dev);
	int G_DrawBinsAndFaces3D(int i_dev);
	
	int G_AddBoundingBox(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz);
	int G_AddRectangle(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by);
	int G_AddCircle(int N, ufloat_g_t cx, ufloat_g_t cy, ufloat_g_t R);
	int G_AddPrism(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz);
	int G_AddSphere(int N1, int N2, ufloat_g_t cx, ufloat_g_t cy, ufloat_g_t cz, ufloat_g_t R);
	int G_AddNACA002D(int N, ufloat_g_t t, ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, int te);
	int G_AddNACA003D(int N, ufloat_g_t t, ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz, int te);
	
	
	
	
	
	Geometry(
		std::map<std::string, int> params_int,
		std::map<std::string, double> params_dbl,
		std::map<std::string, std::string> params_str
	)
	{
		G_Init(params_int, params_dbl, params_str);
		std::cout << "[-] Finished making geometry object." << std::endl << std::endl;
	}
	
	~Geometry()
	{
		G_Dest();
	}
};

// #include "geometry_add.cu"
// #include "geometry_bin.cu"
// #include "geometry_convert.cu"
// #include "geometry_dest.cu"
// #include "geometry_import.cu"
// #include "geometry_init.cu"
// #include "geometry_print.cu"

#endif
