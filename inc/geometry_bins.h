#ifndef GEOMETRY_BIN_H
#define GEOMETRY_BIN_H

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Geometry;

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Geometry<ufloat_t,ufloat_g_t,AP>::Bins
{
    private:
    
    Parser *parser;
    Geometry *geometry;
    
    public:
    
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
    
    // Bin-related constants.
    int init_bins_2D = 0;
    int init_bins_3D = 0;
    int n_bin_density = 1;
    int n_binning_approach = 0;
    int n_levels = 1;
    
    // 2D bins.
    int *n_bins_2D;
    int **bin_indicators_2D;
    int **binned_face_ids_2D;
    int **binned_face_ids_n_2D;
    int **binned_face_ids_N_2D;
    int **c_bin_indicators_2D;
    int **c_binned_face_ids_2D;
    int **c_binned_face_ids_n_2D;
    int **c_binned_face_ids_N_2D;
    
    // 3D bins.
    int *n_bins_3D;
    int **bin_indicators_3D;
    int **binned_face_ids_3D;
    int **binned_face_ids_n_3D;
    int **binned_face_ids_N_3D;
    int **c_bin_indicators_3D;
    int **c_binned_face_ids_3D;
    int **c_binned_face_ids_n_3D;
    int **c_binned_face_ids_N_3D;
    
    // o====================================================================================
    // | Routines.
    // o====================================================================================
    
    int G_MakeBinsCPU(int L);
    int G_MakeBinsGPU(int L);
    int G_DrawBinsAndFaces(int L);
    
    // o====================================================================================
    // | Constructor.
    // o====================================================================================
    
    Bins(Parser *parser_, Geometry *geometry_) : parser(parser_), geometry(geometry_)
    {
        // Set useful parameters.
        Lx = parser->params_dbl["L_c"];
        Ly = parser->params_dbl["L_fy"]*Lx;
        Lz = parser->params_dbl["L_fz"]*Lx;
        Nx = parser->params_int["Nx"];
        n_bin_density = parser->params_int["G_BIN_DENSITY"];
        n_binning_approach = parser->params_int["G_BIN_APPROACH"];
        n_levels = parser->params_int["G_BIN_LEVELS"];
        
        // Derived parameters.
        dx = Lx/static_cast<ufloat_g_t>(Nx);
        dy = Ly/static_cast<ufloat_g_t>((static_cast<int>(Nx*(Ly/Lx))));
        dz = Lz/static_cast<ufloat_g_t>((static_cast<int>(Nx*(Ly/Lx))));
        
        // Initialize 2D bins.
        n_bins_2D = new int[n_levels];
        bin_indicators_2D = new int*[n_levels];
        binned_face_ids_2D = new int*[n_levels];
        binned_face_ids_n_2D = new int*[n_levels];
        binned_face_ids_N_2D = new int*[n_levels];
        c_bin_indicators_2D = new int*[n_levels];
        c_binned_face_ids_2D = new int*[n_levels];
        c_binned_face_ids_n_2D = new int*[n_levels];
        c_binned_face_ids_N_2D = new int*[n_levels];
        
        // Initialize 3D bins.
        n_bins_3D = new int[n_levels];
        bin_indicators_3D = new int*[n_levels];
        binned_face_ids_3D = new int*[n_levels];
        binned_face_ids_n_3D = new int*[n_levels];
        binned_face_ids_N_3D = new int*[n_levels];
        c_bin_indicators_3D = new int*[n_levels];
        c_binned_face_ids_3D = new int*[n_levels];
        c_binned_face_ids_n_3D = new int*[n_levels];
        c_binned_face_ids_N_3D = new int*[n_levels];
        
        std::cout << "[-] Finished making bins object." << std::endl << std::endl;
    }
    
    ~Bins()
    {
        if (init_bins_2D)
        {
            for (int j = 0; j < n_levels; j++)
            {
                delete[] bin_indicators_2D[j];
                delete[] binned_face_ids_2D[j];
                delete[] binned_face_ids_n_2D[j];
                delete[] binned_face_ids_N_2D[j];
                gpuErrchk( cudaFree(c_bin_indicators_2D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_2D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_n_2D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_N_2D[j]) )
            }
        }
        delete[] bin_indicators_2D;
        delete[] binned_face_ids_2D;
        delete[] binned_face_ids_n_2D;
        delete[] binned_face_ids_N_2D;
        delete[] c_bin_indicators_2D;
        delete[] c_binned_face_ids_2D;
        delete[] c_binned_face_ids_n_2D;
        delete[] c_binned_face_ids_N_2D;
        
        if (init_bins_3D)
        {
            for (int j = 0; j < n_levels; j++)
            {
                delete[] bin_indicators_3D[j];
                delete[] binned_face_ids_3D[j];
                delete[] binned_face_ids_n_3D[j];
                delete[] binned_face_ids_N_3D[j];
                gpuErrchk( cudaFree(c_bin_indicators_3D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_3D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_n_3D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_N_3D[j]) )
            }
        }
        delete[] bin_indicators_3D;
        delete[] binned_face_ids_3D;
        delete[] binned_face_ids_n_3D;
        delete[] binned_face_ids_N_3D;
        delete[] c_bin_indicators_3D;
        delete[] c_binned_face_ids_3D;
        delete[] c_binned_face_ids_n_3D;
        delete[] c_binned_face_ids_N_3D;
    }
};

#endif
