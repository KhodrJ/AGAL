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
int Cu_NbrMap(const int &I, const int &J, const int &K)
{
    // Default case (I==-1 [==0] or J==-1 [+=3*0] or K==-1 [+=9*0])
    int S = 0;
    
    // X
    if (I >= 4) S = 2;
    if (I >= 0 && I < 4) S = 1;
    
    // Y
    if (J >= 4) S += 3*(2);
    if (J >= 0 && J < 4) S += 3*(1);
    
    if (N_DIM==3)
    {
        // Z
        if (K >= 4) S += 9*(2);
        if (K >= 0 && K < 4) S += 9*(1);
    }
    
    return S;
}

template <int N_DIM, int wid=4>
__host__ __device__ __forceinline__
int Cu_GridId(const int &I, const int &J, const int &K)
{
    // I,J,K are the indices of cells in a 4x4(x4) grid (so 0 <= I,J,K < 4).
    if (N_DIM==2) return I + wid*J;
    return I + wid*J + wid*wid*K;
}

template <int N_DIM>
__host__ __device__ __forceinline__
int Cu_CellId(const int &I, const int &J, const int &K)
{
    // I,J,K are the indices of cells in a 4x4(x4) grid (so 0 <= I,J,K < 4).
    if (N_DIM==2)
        return I + 4*J;
    else
        return I + 4*J + 16*K;
}

template <int N_DIM>
__host__ __device__ __forceinline__
int Cu_NbrCellId(int Ip, int Jp, int Kp)
{
    // I,J,K are incremented by direction, and can be in {-1,0,1,2,3,4}.
    Ip = (4 + (Ip % 4)) % 4;
    Jp = (4 + (Jp % 4)) % 4;
    if (N_DIM==3)
        Kp = (4 + (Kp % 4)) % 4;
    return Cu_CellId<N_DIM>(Ip,Jp,Kp);
}

template <int N_DIM, int incr=1, int wid=1>
__device__ __forceinline__
int Cu_HaloCellId(const int &p, const int &I, const int &J, const int &K)
{
    // First, increment indices along pth direction.
    int Ip = I;
    int Jp = J;
    int Kp = K;
    if (incr != 0)
    {
        Ip = I + incr*V_CONN_ID[p + 0*27];
        Jp = J + incr*V_CONN_ID[p + 1*27];
        if (N_DIM==3)
            Kp = K + incr*V_CONN_ID[p + 2*27];
    }
    
    // Store the resulting halo index.
    int nbr_kap_h = (Ip+wid) + (4+2*wid)*(Jp+wid);
    if (N_DIM==3)
        nbr_kap_h += (4+2*wid)*(4+2*wid)*(Kp+wid);
    
    return nbr_kap_h;
}

template <int N_DIM, int incr=1>
__device__ __forceinline__
int Cu_GetNbrIndices
(
    const int &p,
    int *nbr_kap_b,
    int *nbr_kap_c,
    const int &I,
    const int &J,
    const int &K,
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
    
    // incr is how much to increment along a particular direction (1 means move one cell along dir. p etc.).
    
    // First, increment indices along pth direction. Store the resulting halo index.
    int Ip = I + incr*V_CONN_ID[p + 0*27];
    int Jp = J + incr*V_CONN_ID[p + 1*27];
    int Kp = 0;
    if (N_DIM==3)
        Kp = K + incr*V_CONN_ID[p + 2*27];
    
    // Then, identify the appropriate neighbor block to store the retrieved cell masks.
    *nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
    *nbr_kap_c = Cu_NbrCellId<N_DIM>(Ip,Jp,Kp);
    
    return 0;
}

template <int N_DIM, int incr=1, int wid=1>
__device__ __forceinline__
int Cu_GetNbrIndicesH
(
    const int &p,
    int *nbr_kap_b,
    int *nbr_kap_c,
    int *nbr_kap_h,
    bool *halo_change,
    const int &I,
    const int &J,
    const int &K,
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
    
    // incr is how much to increment along a particular direction (1 means move one cell along dir. p etc.).
    // wid is the width of the halo (one cell layer needs 1, two cell layer needs 2 etc.)
    
    // First, increment indices along pth direction. Store the resulting halo index.
    int Ip = I + incr*V_CONN_ID[p + 0*27];
    int Jp = J + incr*V_CONN_ID[p + 1*27];
    int Kp = 0;
    if (N_DIM==3)
        Kp = K + incr*V_CONN_ID[p + 2*27];
    
    // Get halo index, if applicable.
    if (nbr_kap_h != nullptr)
    {
        *nbr_kap_h = 
        *nbr_kap_h = (Ip+wid) + (4+2*wid)*(Jp+wid);
        if (N_DIM==3)
            *nbr_kap_h += (4+2*wid)*(4+2*wid)*(Kp+wid);
    }
    
    // Then, identify the appropriate neighbor block to store the retrieved cell masks.
    *nbr_kap_b = s_ID_nbr[Cu_NbrMap<N_DIM>(Ip,Jp,Kp)];
    Ip = (4 + (Ip % 4)) % 4;
    Jp = (4 + (Jp % 4)) % 4;
    if (N_DIM==3)
        Kp = (4 + (Kp % 4)) % 4;
    *nbr_kap_c = Ip + 4*Jp + 16*Kp;
    
    // Get halo change indicator, if applicable.
    if (halo_change != nullptr)
    {
        *halo_change = (Ip != I+incr*V_CONN_ID[p + 0*27] || V_CONN_ID[p + 0*27]==0) && (Jp != J+incr*V_CONN_ID[p + 1*27] || V_CONN_ID[p + 1*27]==0) && (Kp != K+incr*V_CONN_ID[p + 2*27] || V_CONN_ID[p + 2*27]==0);
    }
    
    return 0;
}

#endif
