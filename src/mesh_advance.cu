#include "mesh.h"
#include "solver.h"

int Mesh::M_Advance(int i_dev, int L, std::ofstream *file, double *tmp)
{
	solver->S_Advance(i_dev, L, file, tmp);
	
	return 0;
}

int PrintAGAL();

int Mesh::M_AdvanceLoop()
{
	// o====================================================================================
	// | Pre-processing.
	// o====================================================================================
	
	// Parameters.
	int           iter_mult          = pow(2.0, (double)N_LEVEL_START);
	int           iter_s             = 0;
	int           N_iters_ave        = 0;
	std::ofstream iter_printer;
	std::ofstream ref_printer;
	std::ofstream adv_printer;
	std::ofstream force_printer;
	
	
	// Set initial conditions on the coarse grid.
	solver->S_SetIC(0,0);
	
	
	// Prepare statistics counters.
#if (P_PRINT_ADVANCE==1)
	iter_printer.open(output_dir + "iter_counter.txt");
	adv_printer.open(output_dir + "time_counter.txt");
	adv_printer << "iter ";
	for (int L = 0; L < MAX_LEVELS; L++) adv_printer << "n" << L << " ";
	for (int L = 0; L < MAX_LEVELS; L++) adv_printer << L << "-Interp " << L << "-Collide " << L << "-Stream " << L << "-Average ";
	adv_printer << "MLUPS" << std::endl;
	force_printer.open(output_dir + "forces.txt");
#endif
#if (P_SHOW_REFINE==1)
	ref_printer.open(output_dir + "refine_counter.txt");
	ref_printer << "iter mAve mIntComp mRed iter nR nC Pre S1 S2 S3 nCu S4 S5 S6 S7 S8 total\n";
#endif
	
	
	// Near-wall refinement.
	if (MAX_LEVELS>1 && (MAX_LEVELS!=N_LEVEL_START+1))
	{
		// Refine near walls.
		for (int L = 0; L < N_LEVEL_START; L++)
		{
			std::cout << "Refining to get to starting level [L=" << L << "]..." << std::endl;
			
			// Mark cells near the wall within certain distance.
			solver->S_ComputeRefCriteria(0,L,2);
			
			// Refine.
			M_RefineAndCoarsenCells(0, &ref_printer);
		}
		solver->S_SetIC(0,N_LEVEL_START);
		
		for (int L = N_LEVEL_START; L < (MAX_LEVELS)-1; L++)
		{
			std::cout << "Near wall refinement #" << L+1 << std::endl;
			
			// Mark cells near the wall within certain distance.
			solver->S_ComputeRefCriteria(0,L,0);
			
			// Refine.
			M_RefineAndCoarsenCells(0, &ref_printer);
			
			// Initialize data on higher grid levels.
			solver->S_SetIC(0,L+1);
		}
		
		// Freeze mesh: these new near-wall cells are not eligible for coarsening.
		M_FreezeRefinedCells(0);
	}
	if (MAX_LEVELS>1 && (MAX_LEVELS==N_LEVEL_START+1))
	{
		for (int L = 0; L < MAX_LEVELS-1; L++)
		{
			std::cout << "Refining to get to starting level [L=" << L << "]..." << std::endl;
			
			// Mark cells near the wall within certain distance.
			solver->S_ComputeRefCriteria(0,L,2);
			
			// Refine.
			M_RefineAndCoarsenCells(0, &ref_printer);
		}
		solver->S_SetIC(0,N_LEVEL_START);
	}
	
	
	// If restart was selected, retrieve data.
	if (N_RESTART==1)
	{
		if (M_Restart(0, V_MESH_RESTART_LOAD, &iter_s) == 1)
		{
			std::cout << "Error: Exited during restart." << std::endl;
			return 0;
		}
		M_LoadToGPU();
		iter_s += 1;
	}
	
	
	// o====================================================================================
	// | Solver loop.
	// o====================================================================================
	
	// Primary loop. Total of N_PRINT*P_PRINT iterations (number of prints x number of iterations per print, scaled depending on value of N_LEVEL_START).
	//PrintAGAL();
	for (int i = iter_s; i < iter_s + P_OUTPUT*N_OUTPUT*iter_mult; i++)
	//for (int i = 0; i < 1; i++) // Uncomment for debugging.
	{
		// Print iteration.
		//if (i%16 == 0)
			std::cout << "Iteration " << i << ", t = " << i*dxf_vec[N_LEVEL_START] << std::endl;
#if (P_PRINT_ADVANCE==1)
		iter_printer << "Iteration " << i << ", t = " << i*dx << " | ";
		for (int L = 0; L < MAX_LEVELS; L++)
			iter_printer << "N" << L << "=" << n_ids[0][L] << ", ";
		iter_printer << std::endl;
#endif
		
		
		
		// Reset advancement time counters, then output the grid hierarchy sizes for computation later.
#if (P_PRINT_ADVANCE==1)
		double tmp_arr[4*MAX_LEVELS]; for (int L = 0; L < 4*MAX_LEVELS; L++) tmp_arr[L] = 0.0;
		adv_printer << i << " ";
		for (int L = 0; L < MAX_LEVELS; L++)
			adv_printer << n_ids[0][L] << " ";
		
		// Advance w.r.t the coarse grid.
		solver->S_Advance(0,N_LEVEL_START, &adv_printer, tmp_arr);
		cudaDeviceSynchronize();
#else
		solver->S_Advance(0,N_LEVEL_START, 0, 0);
#endif
		
		
		
		// Refinement stage, performed every P_REFINE iterations.
		if (MAX_LEVELS > 1 && (MAX_LEVELS!=N_LEVEL_START+1) && i%P_REFINE == 0 && i > 0)
		{
			if (N_PROBE_AVE == 0 || (N_PROBE_AVE==1 && i <= N_PROBE_AVE_START))
			{
				// Output to refinement time counter (includes grid hierarchy sizes of last iteration).
				std::cout << "Refining... " << i << ", t = " << i*dxf_vec[N_LEVEL_START] << std::endl;
				std::cout << "(last) "; for (int L = 0; L < MAX_LEVELS; L++) std::cout << n_ids[0][L] << " "; std::cout << std::endl;
				
				// Global average so that data is safely interpolated to new cells.
#if (P_SHOW_REFINE==1)
				cudaDeviceSynchronize();
				std::cout << "Grid average..." << std::endl;
				tic_simple("");
#endif
				// |
				for (int L = MAX_LEVELS-1; L >= N_LEVEL_START; L--)
					solver->S_Average(0,L,V_AVERAGE_GRID);
				// |
#if (P_SHOW_REFINE==1)
				cudaDeviceSynchronize();
				ref_printer << i << " " << toc_simple("",T_US) << " ";
				
				// Interpolate to ghost cells so refinement criterion is not spoiled.
				std::cout << "Interpolating to ghost cells..." << std::endl;
				tic_simple("");
				// |
#endif
				for (int L = N_LEVEL_START; L < MAX_LEVELS; L++)
				{	
					if (L < MAX_LEVELS-1)
						solver->S_Interpolate(0,L,V_INTERP_INTERFACE);
				}
				// |
#if (P_SHOW_REFINE==1)
				cudaDeviceSynchronize();
				ref_printer << toc_simple("",T_US) << " ";
				
				// Compute the refinement criterion on all cells.
				std::cout << "Reducing..." << std::endl;
				tic_simple("");
#endif
				// |
				solver->S_ComputeRefCriteria(0,0,1);
				// |
#if (P_SHOW_REFINE==1)
				cudaDeviceSynchronize();
				ref_printer << toc_simple("",T_US) << " ";
#endif
				
				// Refine.
				M_RefineAndCoarsenCells(i, &ref_printer);
#if (P_SHOW_REFINE==1)
				cudaDeviceSynchronize();
#endif
			}
		}
		
		
		
		// Convergence check, performed every N_PROBE_FREQ iterations.
		if (N_PROBE==1)
		{
			if (i%N_PROBE_FREQUENCY == 0 && i > 0)
			{
				ufloat_t ctol = N_Pf(0.0);
			
				M_RetrieveFromGPU();
				std::cout << "Checking convergence..." << std::endl;
				ctol = M_CheckConvergence(0);
				if (ctol < V_PROBE_TOL)
				{
					std::cout << "Converged..." << std::endl;
					i = P_OUTPUT*N_OUTPUT;
				}
				else
					std::cout << "Not converged...(tol = " << ctol << ")" << std::endl;
			}
		}
		
		
		
		// Update the time-average solution, if selected.
		if (N_PROBE_AVE==1)
		{
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
		}
		
		

		// Printing stage, performed every P_PRINT iterations.
		if ((i+1)%P_OUTPUT == 0 && i > N_OUTPUT_START)
		{
			std::cout << "Printing after iteration " << i << " (t = " << i*dxf_vec[N_LEVEL_START] << ")..." << std::endl;
			
			// Ensure data is valid on all cells. Global average, then interpolate to ghost cells.
			for (int L = MAX_LEVELS-1; L >= 0; L--)
				solver->S_Average(0,L,V_AVERAGE_GRID);
			for (int L = 0; L < MAX_LEVELS; L++)
				solver->S_Interpolate(0,L,V_INTERP_INTERFACE);
			
			// Retrieve data from the GPU.
			M_RetrieveFromGPU();
			
			// Write restart file if at the end of the simulation.
			std::cout << "Preparing restart file..." << std::endl;
			if ((i+1)%(iter_s+P_OUTPUT*N_OUTPUT) == 0 && i > iter_s)
				M_Restart(0, V_MESH_RESTART_SAVE, &i);
			std::cout << "Finished printing restart file..." << std::endl;
			
			// Print to .vti file.
			std::cout << "Writing output..." << std::endl;
			M_RenderAndPrint_Uniform(0, i);
			std::cout << "Finished printing..." << std::endl;
			
			// Print to .vthb file.
			//std::cout << "Writing output..." << std::endl;
			//mesh.M_Print_VTHB(0, i);
			//std::cout << "Finished printing..." << std::endl;
		}
		
		
		
		// Print lift and drag forces to output if applicable. Temporary measure, will be refined later.
		if (N_DIM==2 && N_PROBE_FORCE==1)
		{
			if (i%N_PROBE_F_FREQUENCY == 0 && i*dxf_vec[N_LEVEL_START] > 140)
			{
				std::cout << "Printing forces..." << std::endl;
				for (int L = MAX_LEVELS-1; L >= N_LEVEL_START; L--)
					solver->S_Average(0,L,V_AVERAGE_GRID);
				
				M_RetrieveFromGPU();
				M_ComputeForces(0, std::max(0,MAX_LEVELS_INTERIOR-1), &force_printer);
			}
		}
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
