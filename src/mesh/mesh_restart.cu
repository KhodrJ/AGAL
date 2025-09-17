/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

// Assumptions:
// - Same confmake parameters (must be restarting the exact same simulation).
//
// Format Details:
// - Little Endian.
// - One array for all data processed sequentially.
//
// Format:
// - Data
//    + Size data
//       * n_maxcblocks (int)
//       * n_ids (*int)
//       * n_gaps (int)
//       * id_max (*int)
//       * cblocks_id_max (int)
//       * cells_id_max (long int)
//       * n_ids_probed (int)
//       * iter (int)
//    + Mesh data
//       * id_set
//       * gap_set
//       * cells_ID_mask
//       * cblock_f_X
//       * cblock_ID_mask
//       * cblock_ID_nbr
//       * cblock_ID_nbr_child
//       * cblock_ID_onb
//       * cblock_ID_ref
//       * cblock_level
//    + Solver data
//       * cells_f_F
//       * cells_f_U_probed_tn
//       * cells_f_U_mean

#include "mesh.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Restart(int i_dev, int var, int *iter)
{
    std::string filename = output_dir + std::string("restart.bin");
    
    if (var == V_MESH_RESTART_SAVE)
    {
        // Open file.
        std::ofstream restart_file = std::ofstream(filename, std::ios::binary);
        int cblocks_id_max = id_max[i_dev][MAX_LEVELS];
        long int cells_id_max = id_max[i_dev][MAX_LEVELS]*M_CBLOCK;
        
        // Size data.
        restart_file.write(reinterpret_cast<const char*>(&n_maxcblocks), sizeof(int));
        restart_file.write((char*)&n_ids[i_dev][0], (MAX_LEVELS+1)*sizeof(int));
        restart_file.write(reinterpret_cast<const char*>(&n_gaps[i_dev]), sizeof(int));
        restart_file.write((char*)&id_max[i_dev][0], (MAX_LEVELS+1)*sizeof(int));
        restart_file.write(reinterpret_cast<const char*>(&cblocks_id_max), sizeof(int));
        restart_file.write(reinterpret_cast<const char*>(&cells_id_max), sizeof(long int));
        restart_file.write(reinterpret_cast<const char*>(&n_ids_probed[i_dev]), sizeof(int));
        restart_file.write((char*)&iter[0], sizeof(int));
        
        // Mesh data.
            // Id and Gap Sets
        for (int L = 0; L < MAX_LEVELS; L++)
            restart_file.write((char*)&id_set[i_dev][L*n_maxcblocks], n_ids[i_dev][L]*sizeof(int));
        restart_file.write((char*)&gap_set[i_dev][0], n_gaps[i_dev]*sizeof(int));
            // Cell data.
        restart_file.write((char*)&cells_ID_mask[i_dev][0], cells_id_max*sizeof(int));
            // Block data.
        for (int d = 0; d < N_DIM; d++)
            restart_file.write((char*)&cblock_f_X[i_dev][d*n_maxcblocks], cblocks_id_max*sizeof(ufloat_t));
        restart_file.write((char*)&cblock_ID_mask[i_dev][0], cblocks_id_max*sizeof(int));
        for (int p = 0; p < N_Q_max; p++)
        {
            restart_file.write((char*)&cblock_ID_nbr[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int));
            restart_file.write((char*)&cblock_ID_nbr_child[i_dev][p*n_maxcblocks], cblocks_id_max*sizeof(int));
        }
        restart_file.write((char*)&cblock_ID_onb[i_dev][0], cblocks_id_max*sizeof(int));
        restart_file.write((char*)&cblock_ID_ref[i_dev][0], cblocks_id_max*sizeof(int));
        restart_file.write((char*)&cblock_level[i_dev][0], cblocks_id_max*sizeof(int));
        
        // Solver data.
            // GPU-side.
        for (int p = 0; p < N_Q; p++)
            restart_file.write((char*)&cells_f_F[i_dev][p*n_maxcells], cells_id_max*sizeof(ufloat_t));
            // CPU-side.
        for (int d = 0; d < N_DIM; d++)
            restart_file.write((char*)&cells_f_U_probed_tn[i_dev][d*n_ids_probed[i_dev]*M_CBLOCK], n_ids_probed[i_dev]*M_CBLOCK*sizeof(ufloat_t));
        for (int d = 0; d < N_DIM+1; d++)
            restart_file.write((char*)&cells_f_U_mean[i_dev][d*n_ids[i_dev][0]*M_CBLOCK], n_ids[i_dev][0]*M_CBLOCK*sizeof(ufloat_t));
        
        // Close file.
        restart_file.close();
    }
    else if (var == V_MESH_RESTART_LOAD)
    {
        // Open file.
        std::ifstream restart_file = std::ifstream(filename, std::ios::binary);
        long int init_read_length = (2*(MAX_LEVELS+1)+5)*sizeof(int) + sizeof(long int);
        char *buffer_init = new char[init_read_length];
        int cblocks_id_max = 1;
        long int cells_id_max = 1;
        int n_maxcblocks_old = 1;
        
        // Size data.
        long int pos = 0;
        restart_file.read(buffer_init, init_read_length);
        memcpy(&n_maxcblocks_old, &buffer_init[pos], sizeof(int));               pos += sizeof(int);
        memcpy(n_ids[i_dev], &buffer_init[pos], (MAX_LEVELS+1)*sizeof(int));     pos += (MAX_LEVELS+1)*sizeof(int);
        memcpy(&n_gaps[i_dev], &buffer_init[pos], sizeof(int));                  pos += sizeof(int);
        memcpy(id_max[i_dev], &buffer_init[pos], (MAX_LEVELS+1)*sizeof(int));    pos += (MAX_LEVELS+1)*sizeof(int);
        memcpy(&cblocks_id_max, &buffer_init[pos], sizeof(int));                 pos += sizeof(int);
        memcpy(&cells_id_max, &buffer_init[pos], sizeof(long int));              pos += sizeof(long int);
        memcpy(&n_ids_probed[i_dev], &buffer_init[pos], sizeof(int));            pos += sizeof(int);
        memcpy(iter, &buffer_init[pos], sizeof(int));                            pos += sizeof(int);
        //if (n_maxcblocks_old > n_maxcblocks)
        if (id_max[i_dev][MAX_LEVELS] > n_maxcblocks)
        {
            std::cout << "Insufficient memory allocation..." << std::endl;
            return 1; // Exit with failure due to insufficient memory allocation.
        }
            
        std::cout << "Read the following:" << std::endl;
        std::cout << "\tn_maxcblocks (old) = " << n_maxcblocks_old << std::endl;
        std::cout << "\tn_maxcblocks = " << n_maxcblocks << std::endl;
        for (int L = 0; L < MAX_LEVELS; L++)
            std::cout << "\tn_ids[" << L << "] = " << n_ids[i_dev][L] << std::endl;
        std::cout << "\tn_ids[MAX_LEVELS] = " << n_ids[i_dev][MAX_LEVELS] << std::endl;
        std::cout << "\tn_gaps = " << n_gaps[i_dev] << std::endl;
        for (int L = 0; L < MAX_LEVELS; L++)
            std::cout << "\tid_max[" << L << "] = " << id_max[i_dev][L] << std::endl;
        std::cout << "\tid_max[MAX_LEVELS] = " << id_max[i_dev][MAX_LEVELS] << std::endl;
        std::cout << "\tcblocks_id_max = " << cblocks_id_max << std::endl;
        std::cout << "\tcells_id_max = " << cells_id_max << std::endl;
        std::cout << "\tn_ids_probed = " << n_ids_probed[i_dev] << std::endl;
        std::cout << "\titer = " << *iter << std::endl;
            
        // Remaining data preparation.
        long int data_read_length = 0;
        for (int L = 0; L < MAX_LEVELS; L++)
        {
            data_read_length +=
                MAX_LEVELS*n_ids[i_dev][L]*sizeof(int)             // id_set
            ;
        }
        data_read_length += (
            n_gaps[i_dev]*sizeof(int) +                                // gap_set
            cells_id_max*sizeof(int) +                                 // cells_ID_mask
            N_DIM*cblocks_id_max*sizeof(ufloat_t) +                    // cblock_f_X
            cblocks_id_max*sizeof(int) +                               // cblock_ID_mask
            N_Q_max*cblocks_id_max*sizeof(int) +                       // cblock_ID_nbr
            N_Q_max*cblocks_id_max*sizeof(int) +                       // cblock_ID_nbr_child
            cblocks_id_max*sizeof(int) +                               // cblock_ID_onb
            cblocks_id_max*sizeof(int) +                               // cblock_ID_ref
            cblocks_id_max*sizeof(int) +                               // cblock_level
            N_Q*cells_id_max*sizeof(ufloat_t) +                        // cells_f_F
            N_DIM*n_ids_probed[i_dev]*M_CBLOCK*sizeof(ufloat_t) +      // cells_f_U_probed_tn
            (N_DIM+1)*n_ids[i_dev][0]*M_CBLOCK*sizeof(ufloat_t)        // cells_f_U_mean
        );
        char *buffer_data = new char[data_read_length];
        restart_file.read(buffer_data, data_read_length);
        
        // Mesh data.
        pos = 0;
        for (int L = 0; L < MAX_LEVELS; L++)
        {
            memcpy(&id_set[i_dev][L*n_maxcblocks], &buffer_data[pos], n_ids[i_dev][L]*sizeof(int));
            pos += n_ids[i_dev][L]*sizeof(int);
        }
        memcpy(tmp_1[i_dev], &buffer_data[pos], n_gaps[i_dev]*sizeof(int));
        pos += n_gaps[i_dev]*sizeof(int);
        if (n_maxcblocks > n_maxcblocks_old) // More gaps available.
            memcpy(&gap_set[i_dev][n_maxcblocks-n_maxcblocks_old], tmp_1[i_dev], n_gaps[i_dev]*sizeof(int));
        else // Less gaps available.
            memcpy(gap_set[i_dev], &tmp_1[i_dev][n_maxcblocks_old-n_maxcblocks], (n_gaps[i_dev]-(n_maxcblocks_old-n_maxcblocks))*sizeof(int));
        n_gaps[i_dev] += n_maxcblocks-n_maxcblocks_old;
        memcpy(cells_ID_mask[i_dev], &buffer_data[pos], cells_id_max*sizeof(int));           pos += cells_id_max*sizeof(int);
        for (int d = 0; d < N_DIM; d++)
        {
            memcpy(&cblock_f_X[i_dev][d*n_maxcblocks], &buffer_data[pos], cblocks_id_max*sizeof(ufloat_t));
            pos += cblocks_id_max*sizeof(ufloat_t);
        }
        memcpy(cblock_ID_mask[i_dev], &buffer_data[pos], cblocks_id_max*sizeof(int));        pos += cblocks_id_max*sizeof(int);
        for (int p = 0; p < N_Q_max; p++)
        {
            memcpy(&cblock_ID_nbr[i_dev][p*n_maxcblocks], &buffer_data[pos], cblocks_id_max*sizeof(int));
            pos += cblocks_id_max*sizeof(int);
            memcpy(&cblock_ID_nbr_child[i_dev][p*n_maxcblocks], &buffer_data[pos], cblocks_id_max*sizeof(int));
            pos += cblocks_id_max*sizeof(int);
        }
        memcpy(cblock_ID_onb[i_dev], &buffer_data[pos], cblocks_id_max*sizeof(int));         pos += cblocks_id_max*sizeof(int);
        memcpy(cblock_ID_ref[i_dev], &buffer_data[pos], cblocks_id_max*sizeof(int));         pos += cblocks_id_max*sizeof(int);
        memcpy(cblock_level[i_dev], &buffer_data[pos], cblocks_id_max*sizeof(int));          pos += cblocks_id_max*sizeof(int);
        
        // Solver data.
        for (int p = 0; p < N_Q; p++)
        {
            memcpy(&cells_f_F[i_dev][p*n_maxcells], &buffer_data[pos], cells_id_max*sizeof(ufloat_t));
            pos += cells_id_max*sizeof(ufloat_t);
        }
        for (int d = 0; d < N_DIM; d++)
        {
            memcpy(&cells_f_U_probed_tn[i_dev][d*n_ids_probed[i_dev]*M_CBLOCK], &buffer_data[pos], n_ids_probed[i_dev]*M_CBLOCK*sizeof(ufloat_t));
            pos += n_ids_probed[i_dev]*M_CBLOCK*sizeof(ufloat_t);
        }
        for (int d = 0; d < N_DIM+1; d++)
        {
            memcpy(&cells_f_U_mean[i_dev][d*n_ids[i_dev][0]*M_CBLOCK], &buffer_data[pos], n_ids[i_dev][0]*M_CBLOCK*sizeof(ufloat_t));
            pos += n_ids[i_dev][0]*M_CBLOCK*sizeof(ufloat_t);
        }
        
        // Close file.
        delete[] buffer_init;
        delete[] buffer_data;
        restart_file.close();
    }
    else
        std::cout << "Invalid restart option selected..." << std::endl;

    return 0;
}
