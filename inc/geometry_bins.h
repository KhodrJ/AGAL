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
    const int N_VERTEX_DATA                 = 9;                     ///< Maximum of nine data elements for vertices (3 coordinates x 3 vertices).
    const int N_VERTEX_DATA_PADDED          = 16;                    ///< Number of data elements after padding to ensure alignement.
    
    // Bin-related constants.
    int init_bins_2D = 0;
    int init_bins_3D = 0;
    int init_bins_MD = 0;
    int n_bin_density_root = 1;
    int n_bin_approach = 0;
    int n_bin_levels = 1;
    int n_bin_spec;
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
    int **binned_face_ids_2D;
    int **binned_face_ids_n_2D;
    int **binned_face_ids_N_2D;
    int **c_binned_face_ids_2D;
    int **c_binned_face_ids_n_2D;
    int **c_binned_face_ids_N_2D;
    
    // 3D bins.
    int *n_bins_3D;
    int **binned_face_ids_3D;
    int **binned_face_ids_n_3D;
    int **binned_face_ids_N_3D;
    int **c_binned_face_ids_3D;
    int **c_binned_face_ids_n_3D;
    int **c_binned_face_ids_N_3D;
    
    // Multi-directional 3D bins (single level for now).
    int n_bins_MD;
    int *binned_face_ids_MD;
    int *binned_face_ids_n_MD;
    int *binned_face_ids_N_MD;
    int *c_binned_face_ids_MD;
    int *c_binned_face_ids_n_MD;
    int *c_binned_face_ids_N_MD;    
    
    // o====================================================================================
    // | Routines.
    // o====================================================================================
    
    template <bool make_2D=true> int G_MakeBinsCPU(int L);
    template <bool make_2D=true> int G_MakeBinsGPU(int L);
    int G_MakeBinsGPU_MD(int L);
    int G_DrawBinsAndFaces(int L);
    
    // o====================================================================================
    // | Constructor.
    // o====================================================================================
    
    Bins(Geometry<ufloat_t,ufloat_g_t,AP> *geometry_, const BinMake &make_type, const int &draw_bins) : geometry(geometry_)
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
        n_bin_levels = geometry->parser->params_int["G_BIN_LEVELS"];
        n_bin_spec = geometry->parser->params_int["G_BIN_SPEC"];
        n_max_levels_wall = geometry->parser->params_int["MAX_LEVELS_WALL"];
        
        // Derived parameters.
        dx = Lx/static_cast<ufloat_g_t>(Nx);
        dy = Ly/static_cast<ufloat_g_t>((static_cast<int>(Nx*(Ly/Lx))));
        dz = Lz/static_cast<ufloat_g_t>((static_cast<int>(Nx*(Ly/Lx))));
        n_bin_density = new int[n_bin_levels];
        Nxi_L = new int[3*n_bin_levels];
        dxf_vec = new ufloat_g_t[3*n_bin_levels];
        Lx0g_vec = new ufloat_g_t[3*n_bin_levels];
        
        // Cap n_bin_levels using n_max_levels_wall.
        if (n_bin_levels > n_max_levels_wall)
        {
            std::cout << "Capping number of bin levels to " << n_max_levels_wall << "..." << std::endl;
            n_bin_levels = n_max_levels_wall;
        }
        if (n_max_levels_wall > n_bin_levels)
        {
            std::cout << "Capping specified near-wall bin level to n_bin_levels=" << n_bin_levels << "..." << std::endl;
            n_max_levels_wall = n_bin_levels;
        }
        
        
        // Fill vectors based on n_bin_levels.
        n_bin_density[0] = n_bin_density_root;
        dxf_vec[0 + 0*n_bin_levels] = dx;
        dxf_vec[0 + 1*n_bin_levels] = dy;
        dxf_vec[0 + 2*n_bin_levels] = dz;
        Lx0g_vec[0 + 0*n_bin_levels] = Lx/static_cast<ufloat_g_t>(n_bin_density_root);
        Lx0g_vec[0 + 1*n_bin_levels] = Ly/static_cast<ufloat_g_t>(n_bin_density_root);
        Lx0g_vec[0 + 2*n_bin_levels] = Lz/static_cast<ufloat_g_t>(n_bin_density_root);
        Nxi_L[0 + 0*n_bin_levels] = Nx;
        Nxi_L[0 + 1*n_bin_levels] = Ny;
        Nxi_L[0 + 2*n_bin_levels] = Nz;
        for (int L = 1; L < n_bin_levels; L++)
        {
            n_bin_density[L] = n_bin_density[L-1]*2;
            for (int d = 0; d < N_DIM; d++)
            {
                dxf_vec[L + d*n_bin_levels] = dxf_vec[(L-1) + d*n_bin_levels]*static_cast<ufloat_g_t>(0.5);
                Lx0g_vec[L + d*n_bin_levels] = Lx0g_vec[(L-1) + d*n_bin_levels]*static_cast<ufloat_g_t>(0.5);
                Nxi_L[L + d*n_bin_levels] = Nxi_L[(L-1) + d*n_bin_levels]*2;
            }
        }
        
        // Initialize 2D bins.
        n_bins_2D = new int[n_bin_levels];
        binned_face_ids_2D = new int*[n_bin_levels];
        binned_face_ids_n_2D = new int*[n_bin_levels];
        binned_face_ids_N_2D = new int*[n_bin_levels];
        c_binned_face_ids_2D = new int*[n_bin_levels];
        c_binned_face_ids_n_2D = new int*[n_bin_levels];
        c_binned_face_ids_N_2D = new int*[n_bin_levels];
        
        // Initialize 3D bins.
        n_bins_3D = new int[n_bin_levels];
        binned_face_ids_3D = new int*[n_bin_levels];
        binned_face_ids_n_3D = new int*[n_bin_levels];
        binned_face_ids_N_3D = new int*[n_bin_levels];
        c_binned_face_ids_3D = new int*[n_bin_levels];
        c_binned_face_ids_n_3D = new int*[n_bin_levels];
        c_binned_face_ids_N_3D = new int*[n_bin_levels];
        
        // Consider up to n_bin_levels.
        for (int k = 0; k < n_bin_levels; k++)
        {
            // Make the bins based on the specified make_type.
            if (make_type == BinMake::CPU) G_MakeBinsCPU(k);
            if (make_type == BinMake::GPU) G_MakeBinsGPU(k);
            
            // Draw the bins, if specified to do so.
            if (draw_bins == 1) G_DrawBinsAndFaces(k);
        }
        
        // Also, make the MD bins at the specified wall level.
        G_MakeBinsGPU_MD(n_max_levels_wall-1);
        
        std::cout << "[-] Finished making bins object." << std::endl << std::endl;
    }
    
    ~Bins()
    {
        delete[] n_bin_density;
        delete[] dxf_vec;
        
        // 2D bins.
        std::cout << "Deleting 2D bins..." << std::endl;
        if (init_bins_2D)
        {
            for (int j = 0; j < n_bin_levels; j++)
            {
                delete[] binned_face_ids_2D[j];
                delete[] binned_face_ids_n_2D[j];
                delete[] binned_face_ids_N_2D[j];
                gpuErrchk( cudaFree(c_binned_face_ids_2D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_n_2D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_N_2D[j]) )
            }
        }
        delete[] n_bins_2D;
        delete[] binned_face_ids_2D;
        delete[] binned_face_ids_n_2D;
        delete[] binned_face_ids_N_2D;
        delete[] c_binned_face_ids_2D;
        delete[] c_binned_face_ids_n_2D;
        delete[] c_binned_face_ids_N_2D;
        
        // 3D bins.
        std::cout << "Deleting 3D bins..." << std::endl;
        if (init_bins_3D)
        {
            for (int j = 0; j < n_bin_levels; j++)
            {
                delete[] binned_face_ids_3D[j];
                delete[] binned_face_ids_n_3D[j];
                delete[] binned_face_ids_N_3D[j];
                gpuErrchk( cudaFree(c_binned_face_ids_3D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_n_3D[j]) )
                gpuErrchk( cudaFree(c_binned_face_ids_N_3D[j]) )
            }
        }
        delete[] n_bins_3D;
        delete[] binned_face_ids_3D;
        delete[] binned_face_ids_n_3D;
        delete[] binned_face_ids_N_3D;
        delete[] c_binned_face_ids_3D;
        delete[] c_binned_face_ids_n_3D;
        delete[] c_binned_face_ids_N_3D;
        
        // MD bins.
        std::cout << "Deleting MD bins..." << std::endl;
        if (init_bins_MD)
        {
            delete[] binned_face_ids_MD;
            delete[] binned_face_ids_n_MD;
            delete[] binned_face_ids_N_MD;
            gpuErrchk( cudaFree(c_binned_face_ids_MD) )
            gpuErrchk( cudaFree(c_binned_face_ids_n_MD) )
            gpuErrchk( cudaFree(c_binned_face_ids_N_MD) )
        }
    }
};

#endif
