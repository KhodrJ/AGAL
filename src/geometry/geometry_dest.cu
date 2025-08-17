/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_Dest()
{
    std::cout << " o====================================================================================" << std::endl;
    std::cout << " | Deleting: Geometry Object                                                          " << std::endl;
    std::cout << " o====================================================================================" << std::endl;
    
    // If an index list was constructed, free those arrays.
    if (init_index_lists)
    {
        // Free all the memory allocated on the CPU.
        delete[] geom_f_node_X;
        delete[] geom_ID_face;
        delete[] geom_ID_face_attr;
        
        // Free all the memory allocated on the GPU.
        gpuErrchk( cudaFree(c_geom_f_node_X) );
        gpuErrchk( cudaFree(c_geom_ID_face) );
        gpuErrchk( cudaFree(c_geom_ID_face_attr) );
        
        init_index_lists = 0;
    }
    
    // If a coordinate list was constructed, free those arrays.
    if (init_coords_list)
    {
        // Free all the memory allocated on the CPU.
        delete[] geom_f_face_X;
        delete[] geom_f_face_Xt;
        
        // Free all the memory allocated on the GPU.
        gpuErrchk( cudaFree(c_geom_f_face_X) );
        gpuErrchk( cudaFree(c_geom_f_face_Xt) );
        
        init_coords_list = 0;
    }
    
    return 0;
}
