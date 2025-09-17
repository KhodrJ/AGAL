/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef UTIL_MATH
#define UTIL_MATH

// o====================================================================================
// | Enums.
// o====================================================================================

// Order of the derivative (note: not order of accuracy).
enum FinDiffOrder
{
    First,
    Second
};

// Type of stencil to use.
enum FinDiffType
{
    Forward,
    Central,
    Backward
};

// o====================================================================================
// | Interpolation/Extrapolation.
// o====================================================================================

template <int N_DIM, int incr=1, int wid=1>
__device__ __forceinline__
int Cu_Halo(const int &Ip, const int &Jp, const int &Kp)
{
    // Simplified version of Cu_HaloCellId in index_mapper.h.
    // Doesn't depend on constant memory, but indices need to be incremented in advance.
    int nbr_kap_h = (Ip+wid) + (4+2*wid)*(Jp+wid);
    if (N_DIM==3)
        nbr_kap_h += (4+2*wid)*(4+2*wid)*(Kp+wid);
    
    return nbr_kap_h;
}

template <typename T>
__host__ __device__ __forceinline__
T ExtrapolateFourthOrder
(
    const T &d1,
    const T &d2,
    const T &d3,
    const T &d4
)
{
    // Forward:  di = d_{n+i}
    // Backward: di = d_{n-i}
    
    return
        static_cast<T>(+4.0)*d1 +
        static_cast<T>(-6.0)*d2 +
        static_cast<T>(+4.0)*d3 +
        static_cast<T>(-1.0)*d4;
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
void FillHalo(T *__restrict__ s_u, const T &d, const int &I, const int &J, const int &K)
{
    s_u[Cu_Halo<N_DIM>(I,J,K)] = d;
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
void ExtrapolateToHalo(T *__restrict__ s_u, const int &I, const int &J, const int &K)
{
    // Extrapolates data in a 4x4(x4) grid (so indices 0 <= I,J,K < 4) to a single-cell surrounding halo.
    // Assumes s_u in the interior of the grid is already filled.
    
    if (I==0)
    {
        s_u[Cu_Halo<N_DIM>(I-1,J,K)] = ExtrapolateFourthOrder(
            s_u[Cu_Halo<N_DIM>(I,J,K)],
            s_u[Cu_Halo<N_DIM>(I+1,J,K)],
            s_u[Cu_Halo<N_DIM>(I+2,J,K)],
            s_u[Cu_Halo<N_DIM>(I+3,J,K)]
        );
        s_u[Cu_Halo<N_DIM>(I+4,J,K)] = ExtrapolateFourthOrder(
            s_u[Cu_Halo<N_DIM>(I+3,J,K)],
            s_u[Cu_Halo<N_DIM>(I+2,J,K)],
            s_u[Cu_Halo<N_DIM>(I+1,J,K)],
            s_u[Cu_Halo<N_DIM>(I,J,K)]
        );
    }
    if (J==0)
    {
        s_u[Cu_Halo<N_DIM>(I,J-1,K)] = ExtrapolateFourthOrder(
            s_u[Cu_Halo<N_DIM>(I,J,K)],
            s_u[Cu_Halo<N_DIM>(I,J+1,K)],
            s_u[Cu_Halo<N_DIM>(I,J+2,K)],
            s_u[Cu_Halo<N_DIM>(I,J+3,K)]
        );
        s_u[Cu_Halo<N_DIM>(I,J+4,K)] = ExtrapolateFourthOrder(
            s_u[Cu_Halo<N_DIM>(I,J+3,K)],
            s_u[Cu_Halo<N_DIM>(I,J+2,K)],
            s_u[Cu_Halo<N_DIM>(I,J+1,K)],
            s_u[Cu_Halo<N_DIM>(I,J,K)]
        );
    }
    if (N_DIM==3 && K==0)
    {
        s_u[Cu_Halo<N_DIM>(I,J,K-1)] = ExtrapolateFourthOrder(
            s_u[Cu_Halo<N_DIM>(I,J,K)],
            s_u[Cu_Halo<N_DIM>(I,J,K+1)],
            s_u[Cu_Halo<N_DIM>(I,J,K+2)],
            s_u[Cu_Halo<N_DIM>(I,J,K+3)]
        );
        s_u[Cu_Halo<N_DIM>(I,J,K+4)] = ExtrapolateFourthOrder(
            s_u[Cu_Halo<N_DIM>(I,J,K+3)],
            s_u[Cu_Halo<N_DIM>(I,J,K+2)],
            s_u[Cu_Halo<N_DIM>(I,J,K+1)],
            s_u[Cu_Halo<N_DIM>(I,J,K)]
        );
    }
}

// o====================================================================================
// | Finite differences.
// o====================================================================================

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
T FinDiff_D1_Central(const T &dx, const T &dm1, const T &d, const T &dp1)
{
    return (dp1 - dm1)/(static_cast<T>(2.0)*dx);
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
T FinDiff_D1_Forward(const T &dx, const T &dm1, const T &d, const T &dp1)
{
    return (dp1 - d)/dx;
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
T FinDiff_D1_Backward(const T &dx, const T &dm1, const T &d, const T &dp1)
{
    return (d - dm1)/dx;
}

template <typename T, int N_DIM>
__host__ __device__ __forceinline__
T FinDiff_D2_Central(const T &dx, const T &dm1, const T &d, const T &dp1)
{
    return (dp1 + dm1 - static_cast<T>(2.0)*d)/(dx*dx);
}



// uX = (s_u[(I+1 +1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - s_u[(I+1 -1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
// uY = (s_u[(I+1)+(4+2)*(J+1 +1)+(4+2)*(4+2)*(K+1)] - s_u[(I+1)+(4+2)*(J+1 -1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
// uZ = (s_u[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 +1)] - s_u[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 -1)])/(2*dx_L);
// vX = (s_v[(I+1 +1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - s_v[(I+1 -1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
// vY = (s_v[(I+1)+(4+2)*(J+1 +1)+(4+2)*(4+2)*(K+1)] - s_v[(I+1)+(4+2)*(J+1 -1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
// vZ = (s_v[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 +1)] - s_v[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 -1)])/(2*dx_L);
// wX = (s_w[(I+1 +1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)] - s_w[(I+1 -1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
// wY = (s_w[(I+1)+(4+2)*(J+1 +1)+(4+2)*(4+2)*(K+1)] - s_w[(I+1)+(4+2)*(J+1 -1)+(4+2)*(4+2)*(K+1)])/(2*dx_L);
// wZ = (s_w[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 +1)] - s_w[(I+1)+(4+2)*(J+1)+(4+2)*(4+2)*(K+1 -1)])/(2*dx_L);

#endif
