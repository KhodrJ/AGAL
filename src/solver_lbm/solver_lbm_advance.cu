#include "mesh.h"
#include "solver_lbm.h"

int global_Liter = 0;

/*
 .d8888b.           888                           
d88P  Y88b          888                           
Y88b.               888                           
 "Y888b.    .d88b.  888 888  888  .d88b.  888d888 
    "Y88b. d88""88b 888 888  888 d8P  Y8b 888P"   
      "888 888  888 888 Y88  88P 88888888 888     
Y88b  d88P Y88..88P 888  Y8bd8P  Y8b.     888     
 "Y8888P"   "Y88P"  888   Y88P    "Y8888  888     
*/

#if (P_SHOW_ADVANCE==1)
    #define TIMER(code, arg1)                \
    do                                       \
    {                                        \
        tic_simple("");                  \
        code;                            \
        cudaDeviceSynchronize();         \
        arg1 += toc_simple("",T_US,0);   \
    } while (0)
#else
    #define TIMER(code,arg1)
#endif

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Advance(int i_dev, int L, double *tmp)
{
    // NOTE: tmp should be of length MAX_LEVELS*4.
    
    if (MAX_LEVELS>1 && (MAX_LEVELS!=N_LEVEL_START+1))
    {
        if (L == N_LEVEL_START)
        {
            global_Liter = 0;
//             if (S_FORCE_TYPE==1)
//             {
//                 for (int K = N_LEVEL_START; K < MAX_LEVELS-1; K++)
//                     TIMER( S_Interpolate(i_dev, K, V_INTERP_ADVANCE); ,   tmp[4 + L*6] );
//             }
            if (S_FORCE_TYPE==1)
            {
                for (int K = N_LEVEL_START; K < MAX_LEVELS; K++)
                    TIMER( S_ComputeForcesCV(i_dev, K, 0); ,                      tmp[4 + L*6] );
            }
            
            TIMER( S_Interpolate(i_dev, L, V_INTERP_ADVANCE); ,                   tmp[0 + L*6] );
            TIMER( S_Collide(i_dev, L); ,                                         tmp[1 + L*6] );
            TIMER( S_ImposeBC(i_dev, L); ,                                        tmp[5 + L*6] );
            TIMER( S_Stream(i_dev, L); ,                                          tmp[2 + L*6] );
            
            S_Advance(i_dev, L+1, tmp);
            
            TIMER( S_Average(i_dev, L, V_AVERAGE_ADVANCE); ,                      tmp[3 + L*6] );
            
            if (S_FORCE_TYPE==1)
            {
                for (int K = MAX_LEVELS-2; K >= N_LEVEL_START; K--)
                    TIMER( S_Average(0,K,V_AVERAGE_GRID); ,                       tmp[4 + L*6] );
                for (int K = N_LEVEL_START; K < MAX_LEVELS-1; K++)
                    TIMER( S_Interpolate(i_dev, K, V_INTERP_ADVANCE); ,           tmp[4 + L*6] );
                for (int K = N_LEVEL_START; K < MAX_LEVELS; K++)
                    TIMER( S_ComputeForcesCV(i_dev, K, 1); ,                      tmp[4 + L*6] );
            }

            
            
#if (P_SHOW_ADVANCE==1)
            double tot_time = 0.0;
            long int tot_cells = 0;
            int multip = 1;
            for (int Lp = N_LEVEL_START; Lp < 6*MAX_LEVELS; Lp++)
            {
                tot_time += tmp[Lp];
                mesh->to.adv_printer << tmp[Lp] << " ";
            }
            for (int Lp = N_LEVEL_START; Lp < MAX_LEVELS-1; Lp++)
            {
                tot_cells += multip*(mesh->n_ids[i_dev][Lp] - mesh->n_ids[i_dev][Lp+1]/N_CHILDREN);
                multip *= 2;
            }
            tot_cells += multip*(mesh->n_ids[i_dev][MAX_LEVELS-1]);
            mesh->to.adv_printer << (1.0 / tot_time)*(double)(tot_cells*M_CBLOCK);
            mesh->to.adv_printer << std::endl;
#endif
        }
        else
        {
            // Interpolate.
            if (L < MAX_LEVELS-1)
            {
                TIMER( S_Interpolate(i_dev, L, V_INTERP_ADVANCE); ,   tmp[0 + L*6] );
            }
            
            // First fine step.
            TIMER( S_Collide(i_dev, L); ,                             tmp[1 + L*6] );
            TIMER( S_ImposeBC(i_dev, L); ,                            tmp[5 + L*6] );
            TIMER( S_Stream(i_dev, L); ,                              tmp[2 + L*6] );
            if (L == MAX_LEVELS-1) global_Liter++;
            
            // Average, then interpolate again.
            if (L < MAX_LEVELS-1)
            {
                S_Advance(i_dev, L+1, tmp);
                TIMER( S_Average(i_dev, L, V_AVERAGE_ADVANCE); ,      tmp[3 + 6*L] );
                TIMER( S_Interpolate(i_dev, L, V_INTERP_ADVANCE); ,   tmp[0 + 6*L] );
            }
            
            // Second fine step.
            TIMER( S_Collide(i_dev, L); ,                             tmp[1 + L*6] );
            if (S_FORCE_TYPE==0)
            {
                if (global_Liter == (1<<(MAX_LEVELS-1))-1)
                    TIMER( S_ComputeForcesMEA(i_dev, L, 0); ,         tmp[4 + L*6] );
            }
            TIMER( S_ImposeBC(i_dev, L); ,                            tmp[5 + L*6] );
            TIMER( S_Stream(i_dev, L); ,                              tmp[2 + L*6] );
            if (S_FORCE_TYPE==0)
            {
                if (global_Liter == (1<<(MAX_LEVELS-1))-1)
                    TIMER( S_ComputeForcesMEA(i_dev, L, 1); ,          tmp[4 + L*6] );
            }
            if (L == MAX_LEVELS-1) global_Liter++;
            
            // Average.
            if (L < MAX_LEVELS-1)
            {
                S_Advance(i_dev, L+1, tmp);
                TIMER( S_Average(i_dev, L, V_AVERAGE_ADVANCE); ,      tmp[3 + 4*L] );
            }
        }
    }
    else
    {
        if (S_FORCE_TYPE==1)
            TIMER( S_ComputeForcesCV(i_dev, L, 0); ,                  tmp[4] );
        TIMER( S_Collide(i_dev, L); ,                                 tmp[1] );
        if (S_FORCE_TYPE==0)
            TIMER( S_ComputeForcesMEA(i_dev, L, 0); ,                 tmp[4] );
        TIMER( S_ImposeBC(i_dev, L); ,                                tmp[5] );
        TIMER( S_Stream(i_dev, L); ,                                  tmp[2] );
        if (S_FORCE_TYPE==0)
            TIMER( S_ComputeForcesMEA(i_dev, L, 1); ,                 tmp[4] );
        if (S_FORCE_TYPE==1)
            TIMER( S_ComputeForcesCV(i_dev, L, 1); ,                  tmp[4] );
        
        
        
#if (P_SHOW_ADVANCE==1)
        double tot_time = 0.0;
        long int tot_cells = 0;
        int multip = 1;
        for (int Lp = N_LEVEL_START; Lp < 6*MAX_LEVELS; Lp++)
        {
            tot_time += tmp[Lp];
            mesh->to.adv_printer << tmp[Lp] << " ";
        }
        for (int Lp = N_LEVEL_START; Lp < MAX_LEVELS-1; Lp++)
        {
            tot_cells += multip*(mesh->n_ids[i_dev][Lp] - mesh->n_ids[i_dev][Lp+1]/N_CHILDREN);
            multip *= 2;
        }
        tot_cells += multip*(mesh->n_ids[i_dev][MAX_LEVELS-1]);
        mesh->to.adv_printer << (1.0 / tot_time)*(double)(tot_cells*M_CBLOCK);
        mesh->to.adv_printer << std::endl;
#endif
    }
    
    return 0;
}


template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>
int Solver_LBM<ufloat_t,ufloat_g_t,AP,LP>::S_Debug(int i_dev, int L, int var)
{
    S_Debug_DrawGeometry(i_dev, L);
    
    return 0;
}
