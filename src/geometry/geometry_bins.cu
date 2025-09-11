/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_InitBins(BinMake make_type, int draw_bins)
{
    std::cout << "Initializing bins..." << std::endl;
    if (init_bins == 0)
    {
        bins = new Bins(this, make_type, draw_bins);
        //bins.reserve(G_BIN_LEVELS);
        //for (int L = 0; L < G_BIN_LEVELS; L++)
        //    bins.emplace_back(this, make_type, draw_bins);
        init_bins = 1;
    }
    else
        std::cout << "Bins have already been initialized..." << std::endl;
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_DestBins()
{
    std::cout << "Destroying bins..." << std::endl;
    if (init_bins == 1)
    {
        delete bins;
        init_bins = 0;
    }
    else
        std::cout << "Bins are already un-initialized..." << std::endl;
    
    return 0;
}
