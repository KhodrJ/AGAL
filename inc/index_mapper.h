/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef INDEX_MAPPER
#define INDEX_MAPPER

template <int N_DIM>
struct Coords : public vec3<int>
{
    __host__ __device__ Coords() : vec3<int>(0,0,0) {}
    __host__ __device__ Coords(int x_) : vec3<int>(x_,0,0) {}
    __host__ __device__ Coords(int x_, int y_) : vec3<int>(x_,y_,0) {}
    __host__ __device__ Coords(int x_, int y_, int z_) : vec3<int>(x_,y_,z_) {}
    __host__ __device__ Coords(uint t)
    {
        x = t % 4;
        y = (t / 4) % 4;
        z = 0;
        if (N_DIM==3)
            z = (t / 4) / 4;
    }
    
    // Useful routines.
    template <int N_DIM_>
    __device__ static Coords<N_DIM_> Increment(Coords<N_DIM_> coords, int p, int incr=1)
    {
        return Coords<N_DIM_>(
            coords.x + incr*V_CONN_ID[p+0*27],
            coords.y + incr*V_CONN_ID[p+1*27],
            coords.z + incr*V_CONN_ID[p+2*27]
        );
    }
};

template <int N_DIM>
__host__ __device__ __forceinline__
int Cu_NbrMap(int I, int J, int K)
{
    int S = 0;
    
    //if (I == -1) S = 0;
    if (I >= 4) S = 2;
    if (I >= 0 && I < 4) S = 1;
    
    //if (J == -1) S += 3*(0);
    if (J >= 4) S += 3*(2);
    if (J >= 0 && J < 4) S += 3*(1);
    
    if (N_DIM==3)
    {
        //if (K == -1) S += 9*(0);
        if (K >= 4) S += 9*(2);
        if (K >= 0 && K < 4) S += 9*(1);
    }
    
    return S;
}

template <int N_DIM>
__host__ __device__ __forceinline__
int Cu_NbrCellId(int Ip, int Jp, int Kp)
{
    // I,J,K are incremented by direction, and can be in {-1,0,1,2,3,4}.
    Ip = (4 + (Ip % 4)) % 4;
    Jp = (4 + (Jp % 4)) % 4;
    if (N_DIM==2)
        Kp = 0;
    if (N_DIM==3)
        Kp = (4 + (Kp % 4)) % 4;
    return Ip + 4*Jp + 16*Kp;
}

template <int N_DIM>
__device__ __forceinline__
int Cu_HaloCellId(int p, int incr, int wid, int I, int J, int K)
{
    // First, increment indices along pth direction.
    int Ip = I + V_CONN_ID[p + 0*27];
    int Jp = J + V_CONN_ID[p + 1*27];
    int Kp = 0;
    if (N_DIM==3)
        Kp = K + V_CONN_ID[p + 2*27];
    
    // Store the resulting halo index.
    int nbr_kap_h = (Ip+incr+1) + (4+2*wid)*(Jp+incr+1);
    if (N_DIM==3)
        nbr_kap_h += (4+2*wid)*(4+2*wid)*(Kp+incr+1);
    
    return nbr_kap_h;
}

template <int N_DIM, int get_halo=0>
__device__ __forceinline__
int Cu_GetNbrIndices
(
    const int p,
    const int incr,
    int *nbr_kap_b,
    int *nbr_kap_c,
    int *nbr_kap_h,
    bool *halo_change,
    const int I,
    const int J,
    const int K,
    int *__restrict__ s_ID_nbr
)
{
    // IMPORTANT:
    // Assumes that the shared memory array s_ID_nbr has already been set up.
    // Do not call this routine unless the kernel has something like the code below:
    //
    // if (threadIdx.x==0)
    // {
    //     for (int p = 0; p < N_Q_max; p++)
    //         s_ID_nbr[p] = cblock_ID_nbr[i_kap_b + V_CONN_MAP[p]*n_maxcblocks];
    // }
    //__syncthreads();
    
    // First, increment indices along pth direction. Store the resulting halo index.
    int Ip = I + incr*V_CONN_ID[p + 0*27];
    int Jp = J + incr*V_CONN_ID[p + 1*27];
    int Kp = 0;
    if (N_DIM==3)
        Kp = K + incr*V_CONN_ID[p + 2*27];
    if (get_halo == 1 && nbr_kap_h != nullptr)
    {
        *nbr_kap_h = (Ip+incr) + (4+2*incr)*(Jp+incr);
        if (N_DIM==3)
            *nbr_kap_h += (4+2*incr)*(4+2*incr)*(Kp+incr);
    }
    
    // Then, identify the appropriate neighbor block to store the retrieved cell masks.
    *nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
    Ip = (4 + (Ip % 4)) % 4;
    Jp = (4 + (Jp % 4)) % 4;
    if (N_DIM==3)
        Kp = (4 + (Kp % 4)) % 4;
    *nbr_kap_c = Ip + 4*Jp + 16*Kp;
    
    if (get_halo == 1 && halo_change != nullptr)
    {
        *halo_change = (Ip != I+incr*V_CONN_ID[p + 0*27] || V_CONN_ID[p + 0*27]==0) && (Jp != J+incr*V_CONN_ID[p + 1*27] || V_CONN_ID[p + 1*27]==0) && (Kp != K+incr*V_CONN_ID[p + 2*27] || V_CONN_ID[p + 2*27]==0);
    }
    
    return 0;
}

#endif
