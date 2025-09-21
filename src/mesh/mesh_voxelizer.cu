/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"
#include "mesh_voxelizer_voxelize.cu"
#include "mesh_voxelizer_propagate.cu"
#include "mesh_voxelizer_refine.cu"
#include "mesh_voxelizer_masks.cu"

// Since there are many kernels involved, I've divided them into four files:
// - voxelize: Various versions of ray cast surface voxelization.
// - propagate: Various versions of external mask propagation.
// - refine: Kernels dealing with refinement mark setting and propagation.
// - masks: Kernels dealing with checks on and updates to cell/block masks.

/*
        888     888                            888 d8b                                  
        888     888                            888 Y8P                                  
        888     888                            888                                      
        Y88b   d88P  .d88b.  888  888  .d88b.  888 888 88888888  .d88b.  888d888        
         Y88b d88P  d88""88b `Y8bd8P' d8P  Y8b 888 888    d88P  d8P  Y8b 888P"          
          Y88o88P   888  888   X88K   88888888 888 888   d88P   88888888 888            
           Y888P    Y88..88P .d8""8b. Y8b.     888 888  d88P    Y8b.     888            
88888888    Y8P      "Y88P"  888  888  "Y8888  888 888 88888888  "Y8888  888   88888888 
*/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_Voxelize_S1(int i_dev, int L)
{
    if (n_ids[i_dev][L] > 0)
    {
        // Calculate constants.
        ufloat_g_t R = geometry->G_NEAR_WALL_DISTANCE/pow(2.0,(ufloat_g_t)L);
        //int Nprop_i = (int)(R/(ufloat_g_t)(4.0*sqrt(2.0)*dxf_vec[L])) + 1;   // For filling-in the interior.
        int Nprop_d = (int)(R/(ufloat_g_t)(4.0*sqrt(2.0)*dxf_vec[L])) + 1;   // For satisfying the near-wall distance criterion.
        bool hit_max = L==MAX_LEVELS-1;
        int Lbin = std::min(L, geometry->bins->n_bin_levels-1);
        std::cout << "Using bin level " << Lbin << "..." << std::endl;
        
        // Voxelize the solid, filling in all solid cells.
        tic_simple("");
        //Cu_Voxelize_V2_WARP<ufloat_t,ufloat_g_t,AP> <<<(32+(32*n_ids[i_dev][L])-1)/32,32,0,streams[i_dev]>>>(
        Cu_Voxelize_V1<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks, dxf_vec[L],
            c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_f_X[i_dev], c_cblock_ID_nbr[i_dev],
            geometry->n_faces, geometry->n_faces_a, geometry->c_geom_f_face_Xt,
            geometry->bins->c_binned_face_ids_n_3D[Lbin], geometry->bins->c_binned_face_ids_N_3D[Lbin], geometry->bins->c_binned_face_ids_3D[Lbin], geometry->bins->n_bin_density[Lbin],
            hit_max
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", Voxelize"; toc_simple("",T_US,1);
        thrust::device_ptr<int> mask_ptr = thrust::device_pointer_cast(c_cells_ID_mask[i_dev]);
        int ns1 = thrust::count_if(thrust::device, mask_ptr, mask_ptr + id_max[i_dev][L]*M_CBLOCK, is_equal_to(V_CELLMASK_SOLID));

        // Patch any gaps that show up due to round-off errors during computations.
        tic_simple("");
        Cu_Voxelize_PatchGaps<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
            c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", VoxelizePatchGaps (1)"; toc_simple("",T_US,1);
        
        //if (L < MAX_LEVELS-1)
        {
            // Perform an internal propagation of cell flags prior to external propagation.
            tic_simple("");
            Cu_Voxelize_Propagate_Internal<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
                n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
                c_cells_ID_mask[i_dev]
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", VoxelizePropagateInternal"; toc_simple("",T_US,1);
            
            // Propagate preliminary solid masks within the interior from both sides.
            tic_simple("");
            Cu_Voxelize_Propagate_Right<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
                n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
                c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev],
                L==N_LEVEL_START
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", VoxelizePropagate (+x)"; toc_simple("",T_US,1);
            if (L > N_LEVEL_START)
            {
                tic_simple("");
                Cu_Voxelize_Propagate_Left<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
                    n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
                    c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev]
                );
                cudaDeviceSynchronize();
                std::cout << "MESH_VOXELIZE | L=" << L << ", VoxelizePropagate (-x)"; toc_simple("",T_US,1);
            }
            int ns2 = thrust::count_if(thrust::device, mask_ptr, mask_ptr + id_max[i_dev][L]*M_CBLOCK, is_equal_to(V_CELLMASK_SOLID));
            
            // Propagate preliminary solid masks within the interior from both sides.
            tic_simple("");
            //Cu_Voxelize_UpdateMasks_WARP<ufloat_t,ufloat_g_t,AP> <<<(32+(32*n_ids[i_dev][L])-1)/32,32,0,streams[i_dev]>>>(
            Cu_Voxelize_UpdateMasks<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
                n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks],
                c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev]
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", VoxelizeUpdateMasks"; toc_simple("",T_US,1);
            
            // Identify the cell-blocks on the boundary of the solid. Mark them for refinement, if eligible.
            tic_simple("");
            Cu_MarkBlocks_MarkBoundary<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
                id_max[i_dev][L], n_maxcblocks, hit_max, L,
                c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", MarkBoundary"; toc_simple("",T_US,1);
            
            // Now, mark regular blocks adjacent to these solid-boundary cell-blocks for refinement, if eligible.
            tic_simple("");
            Cu_MarkBlocks_MarkExterior<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
                id_max[i_dev][L], n_maxcblocks, hit_max, L,
                c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev]
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", MarkExterior"; toc_simple("",T_US,1);
            
            // Also, mark appropriate solid blocks adjacent to these solid-boundary cell-blocks for refinement, if eligible.
    //         tic_simple("");
    //         Cu_MarkBlocks_MarkInterior<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
    //             n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
    //             c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_ref[i_dev], hit_max
    //         );
    //         cudaDeviceSynchronize();
    //         std::cout << "MESH_VOXELIZE | L=" << L << ", MarkInterior"; toc_simple("",T_US,1);
            
            // Propagate these latter marks until the specified near-wall refinement criterion is approximately reached.
            for (int j = 0; j < Nprop_d; j++)
            {
                tic_simple("");
                Cu_MarkBlocks_Propagate<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
                    id_max[i_dev][L], n_maxcblocks, L,
                    c_cblock_ID_ref[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_level[i_dev], c_tmp_1[i_dev], j
                );
                cudaDeviceSynchronize();
                std::cout << "MESH_VOXELIZE | L=" << L << ", Propagate (" << j << ")"; toc_simple("",T_US,1);
            }
            
            // Finalize the intermediate marks for refinement.
            tic_simple("");
            Cu_MarkBlocks_Finalize<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
                id_max[i_dev][L], c_cblock_ID_ref[i_dev], c_tmp_1[i_dev]
            );
            cudaDeviceSynchronize();
            std::cout << "MESH_VOXELIZE | L=" << L << ", Finalize"; toc_simple("",T_US,1);
            
            
            std::cout << "Counted (pre) " << ns1 << " solid cells..." << std::endl;
            std::cout << "Counted (post) " << ns2 << " solid cells..." << std::endl;
            cudaDeviceSynchronize();
            gpuErrchk( cudaPeekAtLastError() );
        }
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_Voxelize_S2(int i_dev, int L)
{
    if (n_ids[i_dev][L] > 0)
    {
        // Reset one of the intermediate arrays in preparation for copying.
        // Note: I can use id_max instead of n_maxcblocks since refinement and coarsening is finished.
        if (L == 0)
        {
            Cu_ResetToValue<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(id_max[i_dev][MAX_LEVELS], c_tmp_1[i_dev], 0);
            Cu_ResetToValue<<<(M_BLOCK+id_max[i_dev][MAX_LEVELS]-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(id_max[i_dev][MAX_LEVELS], c_tmp_2[i_dev], 0);
        }
        
        // Update solid-adjacent cell masks and indicate adjacency of blocks to the geometry boundary.
        cudaDeviceSynchronize();
        tic_simple("");
        Cu_MarkBlocks_CheckMasks<AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks], n_maxcblocks,
            c_cells_ID_mask[i_dev], c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev], c_cblock_ID_onb[i_dev],
            c_tmp_1[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_CHECKMASKS | L=" << L << ", CheckMasks"; toc_simple("",T_US,1);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );;
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Geometry_Voxelize_S3(int i_dev)
{
    // Declare an 'old' number of solid cell-blocks prior to adjustment via padding.
    int n_solidb_old = 0;
    
    // Compute the number of solid-adjacent cells, and the number of blocks these cells occupy.
    n_solida = thrust::reduce(thrust::device, c_tmp_1_dptr[i_dev], c_tmp_1_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], 0);
    n_solidb = thrust::count_if(thrust::device, c_tmp_1_dptr[i_dev], c_tmp_1_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], is_positive());
    n_solidb_old = n_solidb;
    n_solidb = ((n_solidb + 128) / 128) * 128;
    n_maxcells_b = n_solidb*M_CBLOCK;
    std::cout << "Counted " << n_solida << " cells adjacent to the solid boundary (" << (double)n_solida / (double)n_maxcells << ", in " << n_solidb << " blocks or " << n_maxcells_b << " cells)..." << std::endl;
    
    // If there were solid-adjacent cells, then allocate memory for face-cell linkages.
    if (n_solidb > 0)
    {
        // Allocate memory for the solid cell linkage data.
        cells_ID_mask_b[i_dev] = new int[n_maxcells_b*N_Q_max];
        cells_f_X_b[i_dev] = new ufloat_g_t[n_maxcells_b*N_Q_max];
        cblock_ID_onb_solid[i_dev] = new int[n_maxcblocks];
        gpuErrchk( cudaMalloc((void **)&c_cells_ID_mask_b[i_dev], n_maxcells_b*N_Q_max*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_cells_f_X_b[i_dev], n_maxcells_b*N_Q_max*sizeof(ufloat_g_t)) );
        gpuErrchk( cudaMalloc((void **)&c_cblock_ID_onb_solid[i_dev], n_maxcblocks*sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&c_cblock_ID_face[i_dev], n_solidb*N_Q_max*sizeof(int)) );
        
        // Reset some arrays. Make a device pointer to the new cblock_ID_onb_solid array.
        Cu_ResetToValue<<<(M_BLOCK+n_maxcblocks-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_maxcblocks, c_cblock_ID_onb_solid[i_dev], -1);
        Cu_ResetToValue<<<(M_BLOCK+(n_solidb*N_Q_max)-1)/M_BLOCK, M_BLOCK, 0, streams[i_dev]>>>(n_solidb*N_Q_max, c_cblock_ID_face[i_dev], -1);
        thrust::device_ptr<int> *c_cblock_ID_onb_solid_dptr = new thrust::device_ptr<int>[N_DEV];
        c_cblock_ID_onb_solid_dptr[i_dev] = thrust::device_pointer_cast(c_cblock_ID_onb_solid[i_dev]);
        
        // Now create the map from block Ids in their usual order to the correct region in the linkage data arrays.
        // Note: Make sure c_tmp_1 still has the number of solid-adjacent cells for each block.
        //
        // Copy the indices of cell-blocks with a positive number of solid-adjacent cells so that they are contiguous.
        thrust::copy_if(thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + id_max[i_dev][MAX_LEVELS], c_tmp_1_dptr[i_dev], c_tmp_2_dptr[i_dev], is_positive());
        //
        // Now scatter the addresses of these copied Ids so that cell-blocks know where to find the data of their solid-adjacent cells
        // in the new arrays.
        thrust::scatter(thrust::device, c_tmp_counting_iter_dptr[i_dev], c_tmp_counting_iter_dptr[i_dev] + n_solidb_old, c_tmp_2_dptr[i_dev], c_cblock_ID_onb_solid_dptr[i_dev] );
        cudaDeviceSynchronize();
        
        // Report available memory after these new allocations.
        cudaMemGetInfo(&free_t, &total_t);
        std::cout << "[-] After allocations:\n";
        std::cout << "    Free: " << free_t*CONV_B2GB << "GB, " << "Total: " << total_t*CONV_B2GB << " GB" << std::endl;
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );;
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_UpdateMasks_Vis(int i_dev, int L)
{
    if (n_ids[i_dev][L] > 0)
    {
        tic_simple("");
        Cu_MarkBlocks_UpdateSolidChildren<AP> <<<(M_BLOCK+id_max[i_dev][L]-1)/M_BLOCK,M_BLOCK>>>(
            id_max[i_dev][L], n_maxcblocks, L,
            c_cblock_ID_mask[i_dev], c_cblock_ID_nbr[i_dev], c_cblock_ID_nbr_child[i_dev], c_cblock_level[i_dev]
        );
        cudaDeviceSynchronize();
        std::cout << "MESH_VOXELIZE | L=" << L << ", UpdateSolidChildren"; toc_simple("",T_US,1);
        
        Cu_Voxelize_UpdateMasks_Vis<ufloat_t,ufloat_g_t,AP> <<<n_ids[i_dev][L],M_TBLOCK,0,streams[i_dev]>>>(
            n_ids[i_dev][L], &c_id_set[i_dev][L*n_maxcblocks],
            c_cells_ID_mask[i_dev], c_cblock_ID_nbr_child[i_dev]
        );
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
    }
    
    return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_IdentifyFaces(int i_dev, int L)
{
    // Use the solver-defined face-cell link calculation procedure (might move this back to the Mesh since it doesn't seem to depend on the solver).
    if (n_solidb > 0)
    {
        cudaDeviceSynchronize();
        tic_simple("");
        solver->S_IdentifyFaces(0,L);
        cudaDeviceSynchronize();
        std::cout << "MESH_IDENTIFYFACES | L=" << L << ", IdentifyFaces"; toc_simple("",T_US,1);
    }
    
    return 0;
}
