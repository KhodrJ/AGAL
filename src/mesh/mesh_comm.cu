#include "mesh.h"
#include "solver.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Interpolate(int i_dev, int L, int var)
{
    //if (L < MAX_LEVELS-1)
    //    solver->S_Interpolate(i_dev, L, var);

    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Average(int i_dev, int L, int var)
{
    //if (L < MAX_LEVELS-1)
    //    solver->S_Average(i_dev, L, var);

    return 0;
}
