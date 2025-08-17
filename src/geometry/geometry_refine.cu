/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_g_t, int N_DIM>
int RecursiveRefineFace
(
    int Nref,
    const ufloat_g_t &l_spec,
    const ufloat_g_t vx1,
    const ufloat_g_t vy1,
    const ufloat_g_t vz1,
    const ufloat_g_t vx2,
    const ufloat_g_t vy2,
    const ufloat_g_t vz2,
    const ufloat_g_t vx3,
    const ufloat_g_t vy3,
    const ufloat_g_t vz3,
    std::vector<ufloat_g_t> &v_geom_f_face_1_X,
    std::vector<ufloat_g_t> &v_geom_f_face_1_Y,
    std::vector<ufloat_g_t> &v_geom_f_face_1_Z,
    std::vector<ufloat_g_t> &v_geom_f_face_2_X,
    std::vector<ufloat_g_t> &v_geom_f_face_2_Y,
    std::vector<ufloat_g_t> &v_geom_f_face_2_Z,
    std::vector<ufloat_g_t> &v_geom_f_face_3_X,
    std::vector<ufloat_g_t> &v_geom_f_face_3_Y,
    std::vector<ufloat_g_t> &v_geom_f_face_3_Z,
    bool &replaced_first,
    const int &index_first
)
{
    if (N_DIM==2)
    {
        ufloat_g_t lM_p = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1));
        
        if (lM_p > l_spec)
        {
            ufloat_g_t mx = (ufloat_g_t)0.5 * (vx1+vx2);
            ufloat_g_t my = (ufloat_g_t)0.5 * (vy1+vy2);
            
            RecursiveRefineFace<ufloat_g_t,N_DIM>
            (
                Nref+1,l_spec,vx1,vy1,(ufloat_g_t)0.0,mx,my,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,
                v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
                v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
                v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
                replaced_first, index_first
            );
            RecursiveRefineFace<ufloat_g_t,N_DIM>
            (
                Nref+1,l_spec,mx,my,(ufloat_g_t)0.0,vx2,vy2,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,(ufloat_g_t)0.0,
                v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
                v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
                v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
                replaced_first, index_first
            );
        }
        if (lM_p <= l_spec && Nref > 0)
        {
            if (replaced_first==false)
            {
                v_geom_f_face_1_X[index_first] = vx1;
                v_geom_f_face_1_Y[index_first] = vy1;
                v_geom_f_face_1_Z[index_first] = (ufloat_g_t)0.0;
                v_geom_f_face_2_X[index_first] = vx2;
                v_geom_f_face_2_Y[index_first] = vy2;
                v_geom_f_face_2_Z[index_first] = (ufloat_g_t)0.0;
                v_geom_f_face_3_X[index_first] = (ufloat_g_t)0.0;
                v_geom_f_face_3_Y[index_first] = (ufloat_g_t)0.0;
                v_geom_f_face_3_Z[index_first] = (ufloat_g_t)0.0;
                replaced_first = true;
            }
            else
            {
                v_geom_f_face_1_X.push_back(vx1);
                v_geom_f_face_1_Y.push_back(vy1);
                v_geom_f_face_1_Z.push_back((ufloat_g_t)0.0);
                v_geom_f_face_2_X.push_back(vx2);
                v_geom_f_face_2_Y.push_back(vy2);
                v_geom_f_face_2_Z.push_back((ufloat_g_t)0.0);
                v_geom_f_face_3_X.push_back((ufloat_g_t)0.0);
                v_geom_f_face_3_Y.push_back((ufloat_g_t)0.0);
                v_geom_f_face_3_Z.push_back((ufloat_g_t)0.0);
            }
        }
    }
    else // N_DIM==3
    {
        ufloat_g_t l1 = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
        ufloat_g_t l2 = sqrt((vx3-vx2)*(vx3-vx2) + (vy3-vy2)*(vy3-vy2) + (vz3-vz2)*(vz3-vz2));
        ufloat_g_t l3 = sqrt((vx1-vx3)*(vx1-vx3) + (vy1-vy3)*(vy1-vy3) + (vz1-vz3)*(vz1-vz3));
        ufloat_g_t lM_p = std::max({l1,l2,l3});
        
        if (lM_p > l_spec)
        {
            ufloat_g_t mx1 = (ufloat_g_t)0.5 * (vx1+vx2);
            ufloat_g_t my1 = (ufloat_g_t)0.5 * (vy1+vy2);
            ufloat_g_t mz1 = (ufloat_g_t)0.5 * (vz1+vz2);
            ufloat_g_t mx2 = (ufloat_g_t)0.5 * (vx2+vx3);
            ufloat_g_t my2 = (ufloat_g_t)0.5 * (vy2+vy3);
            ufloat_g_t mz2 = (ufloat_g_t)0.5 * (vz2+vz3);
            ufloat_g_t mx3 = (ufloat_g_t)0.5 * (vx3+vx1);
            ufloat_g_t my3 = (ufloat_g_t)0.5 * (vy3+vy1);
            ufloat_g_t mz3 = (ufloat_g_t)0.5 * (vz3+vz1);
            
            RecursiveRefineFace<ufloat_g_t,N_DIM>
            (
                Nref+1,l_spec,vx1,vy1,vz1,mx1,my1,mz1,mx3,my3,mz3,
                v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
                v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
                v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
                replaced_first, index_first
            );
            RecursiveRefineFace<ufloat_g_t,N_DIM>
            (
                Nref+1,l_spec,mx1,my1,mz1,vx2,vy2,vz2,mx2,my2,mz2,
                v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
                v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
                v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
                replaced_first, index_first
            );
            RecursiveRefineFace<ufloat_g_t,N_DIM>
            (
                Nref+1,l_spec,mx3,my3,mz3,mx2,my2,mz2,vx3,vy3,vz3,
                v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
                v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
                v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
                replaced_first, index_first
            );
            RecursiveRefineFace<ufloat_g_t,N_DIM>
            (
                Nref+1,l_spec,mx1,my1,mz1,mx2,my2,mz2,mx3,my3,mz3,
                v_geom_f_face_1_X, v_geom_f_face_1_Y, v_geom_f_face_1_Z,
                v_geom_f_face_2_X, v_geom_f_face_2_Y, v_geom_f_face_2_Z,
                v_geom_f_face_3_X, v_geom_f_face_3_Y, v_geom_f_face_3_Z,
                replaced_first, index_first
            );
        }
        if (lM_p <= l_spec && Nref > 0)
        {
            if (replaced_first==false)
            {
                v_geom_f_face_1_X[index_first] = vx1;
                v_geom_f_face_1_Y[index_first] = vy1;
                v_geom_f_face_1_Z[index_first] = vz1;
                v_geom_f_face_2_X[index_first] = vx2;
                v_geom_f_face_2_Y[index_first] = vy2;
                v_geom_f_face_2_Z[index_first] = vz2;
                v_geom_f_face_3_X[index_first] = vx3;
                v_geom_f_face_3_Y[index_first] = vy3;
                v_geom_f_face_3_Z[index_first] = vz3;
                replaced_first = true;
            }
            else
            {
                v_geom_f_face_1_X.push_back(vx1);
                v_geom_f_face_1_Y.push_back(vy1);
                v_geom_f_face_1_Z.push_back(vz1);
                v_geom_f_face_2_X.push_back(vx2);
                v_geom_f_face_2_Y.push_back(vy2);
                v_geom_f_face_2_Z.push_back(vz2);
                v_geom_f_face_3_X.push_back(vx3);
                v_geom_f_face_3_Y.push_back(vy3);
                v_geom_f_face_3_Z.push_back(vz3);
            }
        }
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_RefineFaces_Length()
{
    if (v_geom_f_face_1_X.size() > 0)
    {
        G_UpdateCounts();
        constexpr int N_DIMg = AP->N_DIM;
        ufloat_g_t l_spec = (ufloat_g_t)(0.95*1.0) * ((ufloat_g_t)Lx/(ufloat_g_t)G_BIN_DENSITY);
        
        // This needs to be done before calling G_Init_Arrays_CoordsList_CPU to make sure that n_faces is finalized
        // before GPU memory is allocated.
        for (int p = n_faces-1; p >= 0; p--)
        {
            ufloat_g_t vx1 = v_geom_f_face_1_X[p];
            ufloat_g_t vy1 = v_geom_f_face_1_Y[p];
            ufloat_g_t vz1 = v_geom_f_face_1_Z[p];
            ufloat_g_t vx2 = v_geom_f_face_2_X[p];
            ufloat_g_t vy2 = v_geom_f_face_2_Y[p];
            ufloat_g_t vz2 = v_geom_f_face_2_Z[p];
            ufloat_g_t vx3 = v_geom_f_face_3_X[p];
            ufloat_g_t vy3 = v_geom_f_face_3_Y[p];
            ufloat_g_t vz3 = v_geom_f_face_3_Z[p];
            
            ufloat_g_t l1 = sqrt((vx2-vx1)*(vx2-vx1) + (vy2-vy1)*(vy2-vy1) + (vz2-vz1)*(vz2-vz1));
            ufloat_g_t l2 = sqrt((vx3-vx2)*(vx3-vx2) + (vy3-vy2)*(vy3-vy2) + (vz3-vz2)*(vz3-vz2));
            ufloat_g_t l3 = sqrt((vx1-vx3)*(vx1-vx3) + (vy1-vy3)*(vy1-vy3) + (vz1-vz3)*(vz1-vz3));
            ufloat_g_t lM_p = std::max({l1,l2,l3});
            
            bool replaced_first = false;
            RecursiveRefineFace<ufloat_g_t,N_DIMg>
            (
                0,l_spec,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3,
                v_geom_f_face_1_X,
                v_geom_f_face_1_Y,
                v_geom_f_face_1_Z,
                v_geom_f_face_2_X,
                v_geom_f_face_2_Y,
                v_geom_f_face_2_Z,
                v_geom_f_face_3_X,
                v_geom_f_face_3_Y,
                v_geom_f_face_3_Z,
                replaced_first,
                p
            );
        }
        G_UpdateCounts();
        std::cout << "Refined. There are now: " << n_faces << " faces..." << std::endl;
    }
    else
    {
        std::cout << "ERROR: There are no faces to refine..." << std::endl;
    }
    
    return 0;
}
