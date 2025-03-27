/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

int PrintAGAL();

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance(int i_dev, int L, double *tmp)
{
	solver->S_Advance(i_dev, L, tmp);
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_InitTextOutput()
{
	if (to.init == 0)
	{
		// Always open a "forces.txt" file.
		to.force_printer.open(output_dir + "forces.txt");
		
		// If specified to show advance execution times, open the required files and add a header.
#if (P_SHOW_ADVANCE==1)
		to.iter_printer.open(output_dir + "iter_counter.txt");
		to.adv_printer.open(output_dir + "time_counter.txt");
		to.adv_printer << "iter ";
		for (int L = 0; L < MAX_LEVELS; L++) to.adv_printer << "n" << L << " ";
		for (int L = 0; L < MAX_LEVELS; L++) to.adv_printer << L << "-Interp " << L << "-Collide " << L << "-Stream " << L << "-Average ";
		to.adv_printer << "MLUPS" << std::endl;
#endif
		
		// If specified to show refinement execution times, open the required file and add a header.
#if (P_SHOW_REFINE==1)
		to.ref_printer.open(output_dir + "refine_counter.txt");
		to.ref_printer << "iter mAve mIntComp mRed iter nR nC Pre S1 S2 S3 nCu S4 S5 S6 S7 S8 total\n";
#endif
		
		// Indicate that the TextOutput has been initialized for this mesh.
		to.init = 1;
	}
	else
		std::cout << "[-] Mesh TextOutput already initialized..." << std::endl;
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_RefineNearWall()
{
	// o====================================================================================
	// | Near-wall refinement calls.
	// o====================================================================================
	
	// If starting on a deeper level, but not uniform:
	// - Refine the root grid to the deeper level.
	// - Then, perform additional near-wall refinement starting on the deeper level.
	if (MAX_LEVELS>1 && (MAX_LEVELS!=N_LEVEL_START+1))
	{
		// Initial uniform refinement up to N_LEVEL_START. Initialize only this level.
		for (int L = 0; L < N_LEVEL_START; L++)
		{
			std::cout << "Refining to get to starting level [L=" << L << "]..." << std::endl;
			M_ComputeRefCriteria(0,L,V_MESH_REF_UNIFORM);
			M_RefineAndCoarsenBlocks(0);
		}
		solver->S_SetIC(0,N_LEVEL_START);
		
		// Near-wall refinement, starting from N_LEVEL_START.
		for (int L = N_LEVEL_START; L < MAX_LEVELS_WALL; L++)
		{
			std::cout << "Near wall refinement #" << L+1 << std::endl;
			if (L < MAX_LEVELS-1)
				M_ComputeRefCriteria(0,L,V_MESH_REF_NW_CASES);
			cudaDeviceSynchronize();
			tic_simple("");
			//M_ComputeRefCriteria(0,L,V_MESH_REF_NW_GEOMETRY);
			cudaDeviceSynchronize();
			toc_simple("",T_MS);
			std::cout << "Filling nodes." << std::endl;
			tic_simple("");
			M_Geometry_FillBinned(0,L);
			cudaDeviceSynchronize();
			toc_simple("",T_MS);

			tic_simple("");
			M_RefineAndCoarsenBlocks(0);
			cudaDeviceSynchronize();
			std::cout << "(Refine and coarsen time: " << toc_simple("",T_MS,0) << std::endl;;
			solver->S_SetIC(0,L);
		}
		
		// Freeze mesh: these new near-wall cells are not eligible for coarsening.
		M_FreezeRefinedCells(0);
	}
	
	// If starting on a deeper level, but uniform:
	// - Refine only to the deeper level.
	// - Do not refine further.
	if (MAX_LEVELS>1 && (MAX_LEVELS==N_LEVEL_START+1))
	{
		// Uniform refinement up to N_LEVEL_START. Initialize only this level.
		for (int L = 0; L < MAX_LEVELS-1; L++)
		{
			std::cout << "Refining to get to starting level [L=" << L << "]..." << std::endl;
			M_ComputeRefCriteria(0,L,V_MESH_REF_UNIFORM);
			M_RefineAndCoarsenBlocks(0);
		}
		solver->S_SetIC(0,N_LEVEL_START);
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_LoadRestartFile(int &iter_s)
{
	if (M_Restart(0, V_MESH_RESTART_LOAD, &iter_s) == 1)
	{
		std::cout << "Error: Exited during restart." << std::endl;
		return 0;
	}
	M_LoadToGPU();
	iter_s += 1;
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_PrintIter(int i, int iter_s)
{
	// o====================================================================================
	// | Print the current iteration to the console/output file.
	// o====================================================================================
	
	std::cout << "Iteration " << i << ", t = " << i*dxf_vec[0] << std::endl;
#if (P_SHOW_ADVANCE==1)
	to.iter_printer << "Iteration " << i << ", t = " << i*dx << " | ";
	for (int L = 0; L < MAX_LEVELS; L++)
		to.iter_printer << "N" << L << "=" << n_ids[0][L] << ", ";
	to.iter_printer << std::endl;
#endif
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_Step(int i, int iter_s, int iter_mult)
{
	// o====================================================================================
	// | Perform a step in time (relative to N_LEVEL_START).
	// o====================================================================================
	
	for (int j = 0; j < iter_mult; j++)
	{
		if (N_LEVEL_START > 0)
			std::cout << "    Sub-Iteration: " << j << ", t = " << i*dxf_vec[0] + j*dxf_vec[N_LEVEL_START] << std::endl;
		
#if (P_SHOW_ADVANCE==1)
		double tmp_arr[4*MAX_LEVELS]; for (int L = 0; L < 4*MAX_LEVELS; L++) tmp_arr[L] = 0.0;
		to.adv_printer << i << " ";
		for (int L = 0; L < MAX_LEVELS; L++)
			to.adv_printer << n_ids[0][L] << " ";
		
		// Advance w.r.t the coarse grid.
		solver->S_Advance(0,N_LEVEL_START, tmp_arr);
		cudaDeviceSynchronize();
#else
		//solver->S_Debug(0,N_LEVEL_START, 0);
		solver->S_Advance(0,N_LEVEL_START, 0);
#endif
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_RefineWithSolution(int i, int iter_s)
{
	// o====================================================================================
	// | Solution-based refinement calls.
	// o====================================================================================
	
	if (N_PROBE_AVE == 0 || (N_PROBE_AVE==1 && (i+1) <= N_PROBE_AVE_START))
	{
		// Output to refinement time counter (includes grid hierarchy sizes of last iteration).
		std::cout << "Refining... " << i+1 << ", t = " << (i+1)*dxf_vec[0] << std::endl;
		std::cout << "(last) "; for (int L = 0; L < MAX_LEVELS; L++) std::cout << n_ids[0][L] << " "; std::cout << std::endl;
		
		// Global average so that data is safely interpolated to new cells.
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		std::cout << "Grid average..." << std::endl;
		tic_simple("");
#endif
		// |
		for (int L = MAX_LEVELS-2; L >= N_LEVEL_START; L--)
			solver->S_Average(0,L,V_AVERAGE_GRID);
		// |
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		to.ref_printer << i+1 << " " << toc_simple("",T_US) << " ";
		
		// Interpolate to ghost cells so refinement criterion is not spoiled.
		std::cout << "Interpolating to ghost cells..." << std::endl;
		tic_simple("");
		// |
#endif
		for (int L = N_LEVEL_START; L < MAX_LEVELS-1; L++)
		{	
			if (L < MAX_LEVELS-1)
				solver->S_Interpolate(0,L,V_INTERP_INTERFACE);
		}
		// |
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		to.ref_printer << toc_simple("",T_US) << " ";
		
		// Compute the refinement criterion on all cells.
		std::cout << "Reducing..." << std::endl;
		tic_simple("");
#endif
		// |
		solver->S_ComputeRefCriteria(0,0,1);
		// |
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
		to.ref_printer << toc_simple("",T_US) << " ";
#endif
		
		// Refine.
		M_RefineAndCoarsenBlocks(i+1);
#if (P_SHOW_REFINE==1)
		cudaDeviceSynchronize();
#endif
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_Probe(int i, int iter_s)
{
	// o====================================================================================
	// | Probe the current solution for convergence.
	// o====================================================================================
	
	if (i%N_PROBE_FREQUENCY == 0 && i > 0)
	{
		ufloat_t ctol = (ufloat_t)0.0;
	
		M_RetrieveFromGPU();
		std::cout << "Checking convergence..." << std::endl;
		ctol = M_CheckConvergence(0);
		if (ctol < V_PROBE_TOL)
		{
			std::cout << "Converged..." << std::endl;
			i = N_ITER_TOTAL;
		}
		else
			std::cout << "Not converged...(tol = " << ctol << ")" << std::endl;
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_ProbeAverage(int i, int iter_s, int &N_iters_ave)
{
	// o====================================================================================
	// | Probe the time-averaged solution for convergece.
	// o====================================================================================
	
	if (i%N_PROBE_AVE_FREQUENCY == 0 && i > N_PROBE_AVE_START)
	{
		std::cout << "Updating averages..." << std::endl;
		// Ensure data is valid on all cells. Global average, then interpolate to ghost cells.
		for (int L = MAX_LEVELS-1; L >= 0; L--)
			solver->S_Average(0,L,V_AVERAGE_GRID);
		for (int L = 0; L < MAX_LEVELS; L++)
			solver->S_Interpolate(0,L,V_INTERP_INTERFACE);
		
		M_RetrieveFromGPU();
		M_UpdateMeanVelocities(0, N_iters_ave);
		N_iters_ave++;
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_PrintData(int i, int iter_s)
{
	// o====================================================================================
	// | Print the solution to the output directory.
	// o====================================================================================
	
	std::cout << "Printing after iteration " << i << " (t = " << i*dxf_vec[N_LEVEL_START] << ")..." << std::endl;
	
	// Ensure data is valid on all cells. Global average, then interpolate to ghost cells.
	for (int L = MAX_LEVELS-2; L >= 0; L--)
		solver->S_Average(0,L,V_AVERAGE_GRID);
	for (int L = 0; L < MAX_LEVELS-1; L++)
		solver->S_Interpolate(0,L,V_INTERP_INTERFACE);
	
	// Retrieve data from the GPU.
	M_RetrieveFromGPU();
	
	// Write restart file if at the end of the simulation.
	if ((i+1)%(iter_s+N_ITER_TOTAL) == 0 && i > iter_s)
	{
		std::cout << "Preparing restart file..." << std::endl;
		M_Restart(0, V_MESH_RESTART_SAVE, &i);
		std::cout << "Finished printing restart file..." << std::endl;
	}
	
	if (N_PRINT_LEVELS > 0)
	{
		// Add to .bin file.
		std::cout << "Writing output..." << std::endl;
		M_RenderAndPrint_Uniform(0, i+1);
		std::cout << "Finished printing..." << std::endl;
	}
	if (N_PRINT_LEVELS_LEGACY > 0)
	{
		std::cout << "Writing legacy output..." << std::endl;
		// Print to .vthb file.
		M_Print_VTHB(0, i);
		std::cout << "Finished legacy printing..." << std::endl;
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_Advance_PrintForces(int i, int iter_s)
{
	// o====================================================================================
	// | Perform a force calculation around the designated solid obstacles.
	// o====================================================================================
	
	if (i%N_PROBE_F_FREQUENCY == 0 && i > N_PROBE_F_START)
	{
		std::cout << "Printing forces..." << std::endl;
		for (int L = MAX_LEVELS-1; L >= N_LEVEL_START; L--)
			solver->S_Average(0,L,V_AVERAGE_GRID);
		
		M_RetrieveFromGPU();
		M_ComputeForces(0, std::max(0,MAX_LEVELS_INTERIOR-1));
	}
	
	return 0;
}





template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Mesh<ufloat_t,ufloat_g_t,AP>::M_AdvanceLoop()
{
	// o====================================================================================
	// | Pre-processing.
	// o====================================================================================
	
	// Parameters.
	int           iter_mult          = pow(2.0, (double)N_LEVEL_START);
	int           iter_s             = 0;
	int           N_iters_ave        = 0;
	
	
	// o====================================================================================
	// | Initialization and restart.
	// o====================================================================================
	
	// Set initial conditions on the root grid.
	//solver->S_SetIC(0,0);
	
	// Prepare statistics counters.
	M_Advance_InitTextOutput();
	
	// Perform initial near-wall refinements depending on the current choice of mesh.
	M_Advance_RefineNearWall();
	
	// If restart was selected, retrieve data.
	if (N_RESTART==1)
		M_Advance_LoadRestartFile(iter_s);
	
	
	// o====================================================================================
	// | Solver loop.
	// o====================================================================================
	
	// Primary loop. Total of N_PRINT*P_PRINT iterations (number of prints x number of iterations per print, scaled depending on value of N_LEVEL_START).
	for (int i = iter_s; i < iter_s + N_ITER_TOTAL; i++)
	{
		// Print iteration.
		M_Advance_PrintIter(i, iter_s);
		
		
		// Reset advancement time counters, then output the grid hierarchy sizes for computation later.
		M_Advance_Step(i, iter_s, iter_mult);
		
		
		// Refinement stage, performed every P_REFINE iterations.
		if (MAX_LEVELS > 1 && (MAX_LEVELS!=N_LEVEL_START+1) && (i+1)%P_REFINE == 0 && i > 0)
			M_Advance_RefineWithSolution(i, iter_s);
		
		
		// Convergence check, performed every N_PROBE_FREQ iterations.
		if (N_PROBE==1)
			M_Advance_Probe(i, iter_s);
		
		
		// Update the time-average solution, if selected.
		if (N_PROBE_AVE==1)
			M_Advance_ProbeAverage(i, iter_s, N_iters_ave);
		

		// Printing stage, performed every P_PRINT iterations.
		if ((i+1)%P_OUTPUT == 0 && i > iter_s+N_OUTPUT_START)
			M_Advance_PrintData(i, iter_s);
		
		
		// Print lift and drag forces to output if applicable. Temporary measure, will be refined later.
		if (N_DIM==2 && N_PROBE_FORCE==1)
			M_Advance_PrintForces(i, iter_s);
	}
	
	return 0;
}





int PrintAGAL()
{
	std::cout << std::endl;
	std::cout << "*************************************************************************************************************" << std::endl;
	std::cout << "*                                                                                                           *" << std::endl;
	std::cout << "*   *****************************************************************************************************   *" << std::endl;
	std::cout << "*   *                                                                                                   *   *" << std::endl;
	std::cout << "*   *                                                                          1003                     *   *" << std::endl;
	std::cout << "*   *                                                                       0000                        *   *" << std::endl;
	std::cout << "*   *                                                                   40001                           *   *" << std::endl;
	std::cout << "*   *                                                                 004                               *   *" << std::endl;
	std::cout << "*   *                                                5                                                  *   *" << std::endl;
	std::cout << "*   *                             03             7000                                                   *   *" << std::endl;
	std::cout << "*   *                            500          8000                                                      *   *" << std::endl;
	std::cout << "*   *                            000      30003                                                         *   *" << std::endl;
	std::cout << "*   *                            600    704                                         7404                *   *" << std::endl;
	std::cout << "*   *                            7003                                          5000000000         501   *   *" << std::endl;
	std::cout << "*   *                             006                                       20000000000008      0000    *   *" << std::endl;
	std::cout << "*   *                             000            407                       000000000000003    00000     *   *" << std::endl;
	std::cout << "*   *                             800       5001     700000000002     18  0006         00   000007      *   *" << std::endl;
	std::cout << "*   *                             400 30009        700000000000000000004 0005         100000000         *   *" << std::endl;
	std::cout << "*   *                          74000087           800000000000000000000 100000000000000000008           *   *" << std::endl;
	std::cout << "*   *                   300000002 500            400       0000000005   0000000000000000008             *   *" << std::endl;
	std::cout << "*   *               4000004       500            02     00000001         0000000000000000               *   *" << std::endl;
	std::cout << "*   *            80008            500           04   40000007                     900009                *   *" << std::endl;
	std::cout << "*   *         70003               400           0  00000000                     10000 9                 *   *" << std::endl;
	std::cout << "*   *        008                  8006         86000000  00                   10000   0                 *   *" << std::endl;
	std::cout << "*   *      909                    0000         000000    001                900001    6                 *   *" << std::endl;
	std::cout << "*   *     80                      000007    7000000      0002            4000004     7                  *   *" << std::endl;
	std::cout << "*   *    20                      1000000000000008        00000057  15000000006       9                  *   *" << std::endl;
	std::cout << "*   *    01                      00 000000000006          000000000000000004         4                  *   *" << std::endl;
	std::cout << "*   *   60                      007  0000000  9            000000000000007                              *   *" << std::endl;
	std::cout << "*   *   00                     000            0              100000087              6                   *   *" << std::endl;
	std::cout << "*   *   00                    000             0                                                         *   *" << std::endl;
	std::cout << "*   *   002                 4000              0                                                         *   *" << std::endl;
	std::cout << "*   *   0003              90000                                                                         *   *" << std::endl;
	std::cout << "*   *   800001         8000005                                                                          *   *" << std::endl;
	std::cout << "*   *    8000000000000000000                     30                                                     *   *" << std::endl;
	std::cout << "*   *      000000000000005                      90000                                                   *   *" << std::endl;
	std::cout << "*   *        900000009                         00000                                                    *   *" << std::endl;
	std::cout << "*   *                                           008                                                     *   *" << std::endl;
	std::cout << "*   *                                                                                                   *   *" << std::endl;
	std::cout << "*   *****************************************************************************************************   *" << std::endl;
	std::cout << "*                                                                                                           *" << std::endl;
	std::cout << "*************************************************************************************************************" << std::endl;
	std::cout << std::endl;

	return 0;
}
