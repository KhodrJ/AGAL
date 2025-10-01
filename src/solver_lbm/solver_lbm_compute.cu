#include "mesh.h"
#include "solver_lbm.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out)
{
    // Density and velocity computations.    
    for (int kap_i = 0; kap_i < M_TBLOCK; kap_i++)
    {
//         ufloat_t rho = 0;
//         ufloat_t u = 0;
//         ufloat_t v = 0;
//         ufloat_t w = 0;
//         S_ComputeMacroProperties(i_dev, i_kap, i_Q, kap_i, rho, u, v, w);
//         out[kap_i + 0*M_TBLOCK] = rho;
//         out[kap_i + 1*M_TBLOCK] = u;
//         out[kap_i + 2*M_TBLOCK] = v;
//         out[kap_i + 3*M_TBLOCK] = w;
        
        out[kap_i + 0*M_TBLOCK] = (double)mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + (N_Q+0)*n_maxcells];
        out[kap_i + 1*M_TBLOCK] = (double)mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + (N_Q+1)*n_maxcells];
        out[kap_i + 2*M_TBLOCK] = (double)mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + (N_Q+2)*n_maxcells];
        out[kap_i + 3*M_TBLOCK] = 0.0;
        if (N_DIM==3)
            out[kap_i + 3*M_TBLOCK] = (double)mesh->cells_f_F[i_dev][i_kap*M_CBLOCK + kap_i + (N_Q+3)*n_maxcells];
    }

    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ComputeOutputProperties(int i_dev, int i_Q, int i_kap, ufloat_t dx_L, double *out)
{
    // Load rho, u, v, w.
    S_ComputeProperties(i_dev, i_Q, i_kap, dx_L, out);
    
    // Get this block's neighbor indices.
    int s_ID_nbr[N_Q_max];
    for (int p = 0; p < N_Q_max; p++)
        s_ID_nbr[p] = mesh->cblock_ID_nbr[i_dev][i_kap + V_CONN_MAP_H[p]*n_maxcblocks];
    
    // Vorticity calculation.
    for (int kap_i = 0; kap_i < M_TBLOCK; kap_i++)
    {
        constexpr int N_DIMg = AP->N_DIM;
        int I = kap_i % 4;
        int J = (kap_i / 4) % 4;
        int K = (kap_i / 4) / 4;
        
        out[kap_i + 4*M_TBLOCK] = 0.0;
        out[kap_i + 5*M_TBLOCK] = 0.0;
        out[kap_i + 6*M_TBLOCK] = 0.0;
        
        int nbr_kap_b_xm = N_SKIPID;
        int nbr_kap_b_xp = N_SKIPID;
        int nbr_kap_b_ym = N_SKIPID;
        int nbr_kap_b_yp = N_SKIPID;
        int nbr_kap_b_zm = N_SKIPID;
        int nbr_kap_b_zp = N_SKIPID;
        int nbr_kap_c = -1;
        double u = out[kap_i + 1*M_TBLOCK];
        double v = out[kap_i + 2*M_TBLOCK];
        double w = out[kap_i + 3*M_TBLOCK];
        double u_xp = 0.0;
        double u_xm = 0.0;
        double u_ym = 0.0;
        double u_yp = 0.0;
        double u_zm __attribute__((unused)) = 0.0;
        double u_zp __attribute__((unused)) = 0.0;
        double v_xm = 0.0;
        double v_yp = 0.0;
        double v_ym = 0.0;
        double v_xp = 0.0;
        double v_zm __attribute__((unused)) = 0.0;
        double v_zp __attribute__((unused)) = 0.0;
        double w_xm __attribute__((unused)) = 0.0;
        double w_xp __attribute__((unused)) = 0.0;
        double w_ym __attribute__((unused)) = 0.0;
        double w_yp __attribute__((unused)) = 0.0;
        double w_zp __attribute__((unused)) = 0.0;
        double w_zm __attribute__((unused)) = 0.0;
        
        nbr_kap_b_xm = s_ID_nbr[Cu_NbrMap<N_DIMg>(I-1,J,K)];
        nbr_kap_c = Cu_NbrCellId<N_DIMg>(I-1,J,K);
        if (nbr_kap_b_xm > -1)
        {
            u_xm = (double)mesh->cells_f_F[i_dev][nbr_kap_b_xm*M_CBLOCK + nbr_kap_c + (N_Q+1)*n_maxcells];
            v_xm = (double)mesh->cells_f_F[i_dev][nbr_kap_b_xm*M_CBLOCK + nbr_kap_c + (N_Q+2)*n_maxcells];
            if (N_DIM==3) w_xm = (double)mesh->cells_f_F[i_dev][nbr_kap_b_xm*M_CBLOCK + nbr_kap_c + (N_Q+3)*n_maxcells];
        }
        
        nbr_kap_b_xp = s_ID_nbr[Cu_NbrMap<N_DIMg>(I+1,J,K)];
        nbr_kap_c = Cu_NbrCellId<N_DIMg>(I+1,J,K);
        if (nbr_kap_b_xp > -1)
        {
            u_xp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_xp*M_CBLOCK + nbr_kap_c + (N_Q+1)*n_maxcells];
            v_xp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_xp*M_CBLOCK + nbr_kap_c + (N_Q+2)*n_maxcells];
            if (N_DIM==3) w_xp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_xp*M_CBLOCK + nbr_kap_c + (N_Q+3)*n_maxcells];
        }
        
        nbr_kap_b_ym = s_ID_nbr[Cu_NbrMap<N_DIMg>(I,J-1,K)];
        nbr_kap_c = Cu_NbrCellId<N_DIMg>(I,J-1,K);
        if (nbr_kap_b_ym > -1)
        {
            u_ym = (double)mesh->cells_f_F[i_dev][nbr_kap_b_ym*M_CBLOCK + nbr_kap_c + (N_Q+1)*n_maxcells];
            v_ym = (double)mesh->cells_f_F[i_dev][nbr_kap_b_ym*M_CBLOCK + nbr_kap_c + (N_Q+2)*n_maxcells];
            if (N_DIM==3) w_ym = (double)mesh->cells_f_F[i_dev][nbr_kap_b_ym*M_CBLOCK + nbr_kap_c + (N_Q+3)*n_maxcells];
        }
        
        nbr_kap_b_yp = s_ID_nbr[Cu_NbrMap<N_DIMg>(I,J+1,K)];
        nbr_kap_c = Cu_NbrCellId<N_DIMg>(I,J+1,K);
        if (nbr_kap_b_yp > -1)
        {
            u_yp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_yp*M_CBLOCK + nbr_kap_c + (N_Q+1)*n_maxcells];
            v_yp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_yp*M_CBLOCK + nbr_kap_c + (N_Q+2)*n_maxcells];
            if (N_DIM==3) w_yp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_yp*M_CBLOCK + nbr_kap_c + (N_Q+3)*n_maxcells];
        }
        
        if (N_DIM==3)
        {
            nbr_kap_b_zm = s_ID_nbr[Cu_NbrMap<N_DIMg>(I,J,K-1)];
            nbr_kap_c = Cu_NbrCellId<N_DIMg>(I,J,K-1);
            if (nbr_kap_b_zm > -1)
            {
                u_zm = (double)mesh->cells_f_F[i_dev][nbr_kap_b_zm*M_CBLOCK + nbr_kap_c + (N_Q+1)*n_maxcells];
                v_zm = (double)mesh->cells_f_F[i_dev][nbr_kap_b_zm*M_CBLOCK + nbr_kap_c + (N_Q+2)*n_maxcells];
                w_zm = (double)mesh->cells_f_F[i_dev][nbr_kap_b_zm*M_CBLOCK + nbr_kap_c + (N_Q+3)*n_maxcells];
            }
        
            nbr_kap_b_zp = s_ID_nbr[Cu_NbrMap<N_DIMg>(I,J,K+1)];
            nbr_kap_c = Cu_NbrCellId<N_DIMg>(I,J,K+1);
            if (nbr_kap_b_zp > -1)
            {
                u_zp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_zp*M_CBLOCK + nbr_kap_c + (N_Q+1)*n_maxcells];
                v_zp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_zp*M_CBLOCK + nbr_kap_c + (N_Q+2)*n_maxcells];
                w_zp = (double)mesh->cells_f_F[i_dev][nbr_kap_b_zp*M_CBLOCK + nbr_kap_c + (N_Q+3)*n_maxcells];
            }
        }
        
        double dudx = 0.5*(u_xp-u_xm);
        double dudy = 0.5*(u_yp-u_ym);
        double dudz = 0.5*(u_zp-u_zm);
        double dvdx = 0.5*(v_xp-v_xm);
        double dvdy = 0.5*(v_yp-v_ym);
        double dvdz = 0.5*(v_zp-v_zm);
        double dwdx = 0.5*(w_xp-w_xm);
        double dwdy = 0.5*(w_yp-w_ym);
        double dwdz = 0.5*(w_zp-w_zm);
        if (nbr_kap_b_xm < 0) { dudx = (u_xp-u); dvdx = (v_xp-v); dwdx = (w_xp-w); }
        if (nbr_kap_b_xp < 0) { dudx = (u-u_xm); dvdx = (v-v_xm); dwdx = (w-w_xm); }
        if (nbr_kap_b_ym < 0) { dudy = (u_yp-u); dvdy = (v_yp-v); dwdy = (w_yp-w); }
        if (nbr_kap_b_yp < 0) { dudy = (u-u_ym); dvdy = (v-v_ym); dwdy = (w-w_ym); }
        if (nbr_kap_b_zm < 0) { dudz = (u_zp-u); dvdz = (v_zp-v); dwdz = (w_zp-w); }
        if (nbr_kap_b_zp < 0) { dudz = (u-u_zm); dvdz = (v-v_zm); dwdz = (w-w_zm); }
        
        if (N_DIM==3) out[kap_i + 4*M_TBLOCK] = (dwdy - dvdz)/dx_L;
        if (N_DIM==3) out[kap_i + 5*M_TBLOCK] = (dudz - dwdx)/dx_L;
        out[kap_i + 6*M_TBLOCK] = (dvdx - dudy)/dx_L;
        if (N_DIM==3) out[kap_i + 7*M_TBLOCK] = -0.5*(dudx*dudx + dvdy*dvdy + dwdz*dwdz + 2.0*(dudy*dvdx + dudz*dwdx + dwdy*dvdz))/(dx_L*dx_L);
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_ReportForces(int i_dev, int L, int iter, double t, int START)
{
    if (START == 1 && L == -1)
    {
        // Synchronize GPU.
        cudaDeviceSynchronize();
        
        // Identify the time step to use in the denominator.
        double factor = 1.0;
        if (S_FORCE_TYPE==0) factor = 1.0/dxf_vec[MAX_LEVELS-1];
        if (S_FORCE_TYPE==1) factor = 1.0/dxf_vec[N_LEVEL_START];
        
        // Perform reductions using Thrust.
        int cblocks_id_max = mesh->id_max[i_dev][MAX_LEVELS];
        double Fpx = thrust::reduce(
            thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][0*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][0*n_maxcblocks] + cblocks_id_max, 0.0
        );
        double Fmx = thrust::reduce(
            thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][1*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][1*n_maxcblocks] + cblocks_id_max, 0.0
        );
        double Fpy = thrust::reduce(
            thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][2*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][2*n_maxcblocks] + cblocks_id_max, 0.0
        );
        double Fmy = thrust::reduce(
            thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][3*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][3*n_maxcblocks] + cblocks_id_max, 0.0
        );
        double Fpz __attribute__((unused)) = 0.0;
        double Fmz __attribute__((unused)) = 0.0;
        if (N_DIM==3)
        {
            Fpz = thrust::reduce(
                thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][4*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][4*n_maxcblocks] + cblocks_id_max, 0.0
            );
            Fmz = thrust::reduce(
                thrust::device, &mesh->c_cblock_f_Ff_dptr[i_dev][5*n_maxcblocks], &mesh->c_cblock_f_Ff_dptr[i_dev][5*n_maxcblocks] + cblocks_id_max, 0.0
            );
        }
        double Fx = factor*(Fpx - Fmx);
        double Fy = factor*(Fpy - Fmy);
        double Fz __attribute__((unused)) = factor*(Fpz - Fmz);
        double Dp = 1.0/64.0;
        double uin = 0.05;
        std::cout << "Report:" << std::endl;
        std::cout << "CD: " << 2.0*Fx / (uin*uin*(Dp)) << "   " << 8.0*Fx / (uin*uin*(M_PI*Dp*Dp)) << std::endl;
        std::cout << "CL: " << 2.0*Fy / (uin*uin*(Dp)) << std::endl;
        mesh->to.force_printer << iter << " " << t << " " << 2.0*Fx / (uin*uin*(Dp)) << " " << 2.0*Fy / (uin*uin*(Dp)) << " " << 8.0*Fx / (uin*uin*(M_PI*Dp*Dp)) << " " << 8.0*Fy / (uin*uin*(M_PI*Dp*Dp)) << std::endl;
        
        // Reset the forces array in-case some blocks were coarsened.
        Cu_ResetToValue<<<(M_BLOCK+6*n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, mesh->streams[i_dev]>>>(6*n_maxcblocks, mesh->c_cblock_f_Ff[i_dev], (ufloat_t)0);
    }
    
    return 0;
}
