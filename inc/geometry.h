#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "cppspec.h"
#include "parser.h"
#include "util.h"

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
    
    int             init_index_lists        = 0;                     ///< Indicates if index lists were initialized.
    int             init_coords_list        = 0;                     ///< Indicates if face coordinates list was initialized.
    
    // o====================================================================================
    // | Routines.
    // o====================================================================================
    
    int G_Init();
    int G_Dest();
    
    
    
    
    
    public:
    
    class Bins;    
    
    Parser *parser;
    Mesh<ufloat_t,ufloat_g_t,AP> *mesh;
    Bins *bins;
    
    
    // Constants.
    const int N_DIM                         = AP->N_DIM;             ///< Number of dimensions.
    const int N_Q_max                       = AP->N_Q_max;           ///< Neighbor-halo size (including self).
    const int Nqx                           = AP->Nqx;               ///< Number of sub-blocks along one axis.
    const int N_CHILDREN                    = AP->N_CHILDREN;        ///< Number of children per block.
    const int N_QUADS                       = AP->N_QUADS;           ///< Total number of sub-blocks per cell-block.
    const int M_TBLOCK                      = AP->M_TBLOCK;          ///< Number of threads per thread-block in primary-mode.
    const int M_CBLOCK                      = AP->M_CBLOCK;          ///< Number of cells per cell-block.
    const int M_LBLOCK                      = AP->M_LBLOCK;          ///< Number of cell-blocks processed per thread-block in primary-mode.
    const int M_WBLOCK                      = AP->M_WBLOCK;          ///< Number of threads working within a warp in uprimary-mode.
    const int M_LWBLOCK                     = AP->M_LWBLOCK;         ///< Number of cell-blocks processed per thread-block in uprimary-mode.
    const int M_BLOCK                       = AP->M_BLOCK;           ///< Number of threads per thread-block in secondary-mode.
    const int M_RNDOFF                      = AP->M_RNDOFF;          ///< Round-off constant for memory alignment.
    
    // o====================================================================================
    // | Geometry parameters.
    // o====================================================================================
    
    ufloat_g_t      Lx                      = 1.0F;                  ///< Length of domain in x-axis (in meters).
    ufloat_g_t      Ly                      = 1.0F;                  ///< Length of domain in y-axis as a fraction of @ref Lx.
    ufloat_g_t      Lz                      = 1.0F;                  ///< Length of domain in z-axis as a fraction of @ref Lx.
    int             Nx                      = 1;                     ///< Number of cells along x-axis of domain.
    ufloat_g_t      dx                      = 1.0;                   ///< Spatial step (x).
    ufloat_g_t      dy                      = 1.0;                   ///< Spatial step (y).
    ufloat_g_t      dz                      = 1.0;                   ///< Spatial step (z).
    int             MAX_LEVELS_WALL         = 1;                     ///< Maximum number of grids for the domain boundary alone.          
    ufloat_g_t      G_NEAR_WALL_DISTANCE    = 1;                     ///< The near-wall distance for refinement.
    int             G_LOADTYPE              = 0;                     ///< The type of load to perform [0: from and STL file, 1: from the txt file].
    int             G_PRINT                 = 0;                     ///< Indicates if the geometry should be printed in STL format\
                                                                          after processing.
    int             G_BIN_OVERLAP           = 10;                    ///< Factor accounting for the overlap of faces among bins
    int             G_BIN_DENSITY           = 1;                     ///< The number of bins to divide the geometry surface.
    int             G_BIN_FRAC              = 1;                     ///< Fraction of bin arrays to consider at a time.
    int             G_BIN_APPROACH          = 0;                     ///< The approach to take for classifying faces\
                                                                          [0: bounding-box, 1: intersection].
    int             G_BIN_LEVELS            = 1;                     ///< Number of bin levels to use.
    std::string     input_dir;                                       ///< Input directory.
    std::string     output_dir;                                      ///< Output directory.
    std::string     G_FILENAME;
    
    // o====================================================================================
    // | CPU parameters.
    // o====================================================================================
    
    // Geometry mesh data.
    int             n_nodes;                                         ///< Number of nodes.
    int             n_nodes_a;                                       ///< Number of vertices rounded to 32 for alignment.
    int             n_faces;                                         ///< Number of faces.
    int             n_faces_a;                                       ///< Number of faces rounded to 32 for alignment.
    int             *geom_ID_face;                                   ///< Array of face indices.
    ufloat_g_t      *geom_f_node_X;                                  ///< Array of geometry node locations.
    ufloat_g_t      *geom_f_face_X;                                  ///< Array of face data.
    ufloat_g_t      *geom_f_face_Xt;                                 ///< Array of face data but transposed.
    ufloat_g_t      *geom_ID_face_attr;                              ///< Array of geometry face attributes attributes.
    
    // Intermediate vectors used before construction of the coordinate list.
    std::vector<ufloat_g_t>     v_geom_f_node_X;
    std::vector<ufloat_g_t>     v_geom_f_node_Y;
    std::vector<ufloat_g_t>     v_geom_f_node_Z;
    std::vector<int>            v_geom_ID_face_1;
    std::vector<int>            v_geom_ID_face_2;
    std::vector<int>            v_geom_ID_face_3;
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
    ufloat_g_t      *c_geom_f_node_X;
    
    //! GPU counterpart of @ref geom_ID_face.
    int             *c_geom_ID_face;
    
    //! GPU counterpart of @ref geom_f_face_X.
    ufloat_g_t      *c_geom_f_face_X;
    
    //! GPU counterpart of @ref geom_f_face_X but transposed.
    ufloat_g_t      *c_geom_f_face_Xt;
    
    //! GPU counterpart of @ref geom_ID_face_attr.
    ufloat_g_t      *c_geom_ID_face_attr;
    
    // o====================================================================================
    // | Routines.
    // o====================================================================================
    
    // Import routines.
    int G_ImportBoundariesFromTextFile();
    int G_ImportBoundariesFromSTLFile();
    int G_ImportSTL_ASCII(std::string filename);
    int G_PrintSTL();
    int G_PrintOBJ();
    
    // Geometry initialization.
    int G_ClearVectors();
    int G_UpdateCounts();
    int G_Init_Arrays_IndexLists_CPU();
    int G_Init_Arrays_CoordsList_CPU();
    int G_Convert_IndexListsToCoordList();
    int G_Convert_CoordListToIndexLists();
    int G_CoordList_MachineEps();
    int G_RefineFaces_Length();
    
    // Adding primitives to the index lists.
    int G_AddBoundingBox(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz);
    int G_AddRectangle(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by);
    int G_AddCircle(int N, ufloat_g_t cx, ufloat_g_t cy, ufloat_g_t R);
    int G_AddPrism(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz);
    int G_AddSphere(int N1, int N2, ufloat_g_t cx, ufloat_g_t cy, ufloat_g_t cz, ufloat_g_t R);
    int G_AddNACA002D(int N, ufloat_g_t t, ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, int te);
    int G_AddNACA003D(int N, ufloat_g_t t, ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz, int te);
    
    // o====================================================================================
    // | Constructor.
    // o====================================================================================
    
    Geometry(Parser *parser_) : parser(parser_)
    {
        G_Init();
        std::cout << "[-] Finished making geometry object." << std::endl << std::endl;
    }
    
    ~Geometry()
    {
        G_Dest();
    }
};

#endif
