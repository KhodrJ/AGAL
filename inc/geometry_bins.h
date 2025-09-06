/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef GEOMETRY_BIN_H
#define GEOMETRY_BIN_H

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
class Geometry<ufloat_t,ufloat_g_t,AP>::Bins
{
    private:
    
    Geometry<ufloat_t,ufloat_g_t,AP> *geometry;
    
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
    int n_bin_density_root = 1;
    int n_bin_approach = 0;
    int n_levels = 1;
    int n_max_levels_wall = 1;
    int Nx;
    int Ny;
    int Nz;
    int *Nxi_L;
    ufloat_g_t Lx;
    ufloat_g_t Ly;
    ufloat_g_t Lz;
    ufloat_g_t dx;
    ufloat_g_t dy;
    ufloat_g_t dz;
    
    // Bin-related arrays.
    int *n_bin_density;
    ufloat_g_t *dxf_vec;
    ufloat_g_t *Lx0g_vec;
    
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
    
    Bins(Geometry<ufloat_t,ufloat_g_t,AP> *geometry_, const int &make_type, const int &draw_bins) : geometry(geometry_)
    {
        // make_type: [0: CPU, 1: GPU]
        // draw_bins: [0: don't, 1: do]
        
        // Set useful parameters.
        Lx = geometry->parser->params_dbl["L_c"];
        Ly = geometry->parser->params_dbl["L_fy"]*Lx;
        Lz = geometry->parser->params_dbl["L_fz"]*Lx;
        Nx = geometry->parser->params_int["Nx"];
        Ny = (int)(Nx*(Ly/Lx));
        Nz = (int)(Nx*(Lz/Lx));
        n_bin_density_root = geometry->parser->params_int["G_BIN_DENSITY"];
        n_bin_approach = geometry->parser->params_int["G_BIN_APPROACH"];
        n_levels = geometry->parser->params_int["G_BIN_LEVELS"];
        n_max_levels_wall = geometry->parser->params_int["MAX_LEVELS_WALL"];
        
        // Derived parameters.
        dx = Lx/static_cast<ufloat_g_t>(Nx);
        dy = Ly/static_cast<ufloat_g_t>((static_cast<int>(Nx*(Ly/Lx))));
        dz = Lz/static_cast<ufloat_g_t>((static_cast<int>(Nx*(Ly/Lx))));
        n_bin_density = new int[n_levels];
        Nxi_L = new int[3*n_levels];
        dxf_vec = new ufloat_g_t[3*n_levels];
        Lx0g_vec = new ufloat_g_t[3*n_levels];
        
        // Cap n_levels using n_max_levels_wall.
        n_levels = std::min(n_levels, n_max_levels_wall);
        
        // Fill vectors based on n_levels.
        n_bin_density[0] = n_bin_density_root;
        dxf_vec[0 + 0*n_levels] = dx;
        dxf_vec[0 + 1*n_levels] = dy;
        dxf_vec[0 + 2*n_levels] = dz;
        Lx0g_vec[0 + 0*n_levels] = Lx/static_cast<ufloat_g_t>(n_bin_density_root);
        Lx0g_vec[0 + 1*n_levels] = Ly/static_cast<ufloat_g_t>(n_bin_density_root);
        Lx0g_vec[0 + 2*n_levels] = Lz/static_cast<ufloat_g_t>(n_bin_density_root);
        Nxi_L[0 + 0*n_levels] = Nx;
        Nxi_L[0 + 1*n_levels] = Ny;
        Nxi_L[0 + 2*n_levels] = Nz;
        for (int L = 1; L < n_levels; L++)
        {
            n_bin_density[L] = n_bin_density[L-1]*2;
            for (int d = 0; d < N_DIM; d++)
            {
                dxf_vec[L + d*n_levels] = dxf_vec[(L-1) + d*n_levels]*static_cast<ufloat_g_t>(0.5);
                Lx0g_vec[L + d*n_levels] = Lx0g_vec[(L-1) + d*n_levels]*static_cast<ufloat_g_t>(0.5);
                Nxi_L[L + d*n_levels] = Nxi_L[(L-1) + d*n_levels]*2;
            }
        }
        
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
        
        // Consider up to n_levels.
        for (int k = 0; k < n_levels; k++)
        {
            // Make the bins based on the specified make_type.
            if (make_type == static_cast<int>(BinMake::CPU)) G_MakeBinsCPU(k);
            if (make_type == static_cast<int>(BinMake::GPU)) G_MakeBinsGPU(k);
            
            // Draw the bins, if specified to do so.
            if (draw_bins == 1) G_DrawBinsAndFaces(k);
        }
        
        std::cout << "[-] Finished making bins object." << std::endl << std::endl;
    }
    
    ~Bins()
    {
        delete[] n_bin_density;
        delete[] dxf_vec;
        
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
