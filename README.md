# AGAL: Adaptive GPU-native Algorithm for Lattice-based simulations
AGAL (or عَجَل, meaning 'haste') is an open-source C++ code implementing a GPU-native algorithm (AMR) for adaptive mesh refinement applied to Lattice Boltzmann Simulations. A simple square/cubic domain is fixed, and boundary conditions in the streaming function are designed for the lid-driven cavity problem (simple bounce-back, positive lid velocity $u_{\text{lid}}$). The code also implements three velocity sets: D2Q9, D3Q19 and D3Q27. This version of the code was used for the case studies reported in the corresponding manuscript "[GPU-Native Adaptive Mesh Refinement with Application to Lattice Boltzmann Simulations](https://arxiv.org/abs/2308.08085)." 

# Instructions for Compilation and Usage

## Compilation

The code uses the VTK library for output; an vtkOverlappingAMR dataset is used during data collection and written via the vtkXMLUniformGridAMRWriter. The Thrust library is employed for many common array operations, and Octave is required to run the code generators that tailor the solver for the supplied initial and boundary conditions. The correct path to VTK (built for the machine being used) needs to be adjusted in the Makefile. Three options are assumed: 1) in the 'lib' directory, 2) in "/usr/local/", 3) standard software directory on a GPU cluster, these can be commented/commented-out accordingly. If the 'lib' directory is used, make sure that VTK include and lib files are available via the paths "./lib/vtk/include/" and "./lib/vtk/lib/", respectively. Note that VTK versions 9.1/2/3 have been used to successfully compile the code. The input files defining initial and boundary conditions are located in "src/generators/input/".

## Usage

The manuscript detailing the GPU-native algorithm is under review. Validation was conducted with the lid-driven cavity in 2D and 3D and flow past a square cylinder in 2D. Reynolds number is specified by modification of the kinematic viscosity during calculation of the relaxation rate. This allows for specification of the following (in the cppspec.h file): u_lid (lid velocity), Re_c (Reynolds number), L_c (characteristic length, usually left as 1.0 m). If any of these are modified, the program must be recompiled to take effect. Lid velocity should be left small to ensure validity of the Lattice Boltzmann simulation (keep in mind $\text{Ma} = u_{\text{lid}} / c_s$).

The "confmake.sh" file defines compilation options and runs the Makefile automatically. The suggested procedure is detailed below:
1. Define initial and boundary conditions in "src/generators/input/".
2. Choose the velocity set by modifying N_Q (9=D2Q9, 19=D3Q19, 27=D3Q27). Number of dimensions is automatically adjusted.
3. Choose level of precision by modifying N_PRECISION (0=single-precision, 1=double-precision).
4. Choose the grid hierarchy size by modifying MAX_LEVELS. A value of 1 corresponds to a uniform grid.
5. Adjust the kinematic viscosity v0 according to the characteristic velocity and length scale. L_c defines the domain length along the x-axis.
6. Adjust the resolution of the coarsest grid with Nx.
7. If replicating the examples in the preprint, adjust N_CASE (1=LDC, 2=FPSC).
8. If the root grid is too fine, the output may become unacceptably large and N_LEVEL_START should be adjusted. For example, setting N_Q=19, MAX_LEVELS=4, N_LEVEL_START=1 (i.e., start on grid 1 instead of 0) and Nx=64 results in a solver with root grid resolution 128^3, refinement up to effective resolution 512 and an output with resolution 64^3.
9. The chosen refinement parameters were fixed for all test cases, but these can be modified freely. N_CONN_TYPE is unused for now. Set P_SHOW_REFINE to 0 to disable printing of execution times to console.
10. Three kinds of probing are available:
    - N_PROBE probes the solution on the root grid with probe density N_PROBE_DENSITY at a frequency of N_PROBE_FREQUENCY iterations until the solution converges according to tolerance V_PROBE_TOL.
    - N_PROBE_FORCE performs force calculation via the momentum exchange algorithm with frequency N_PROBE_F_FREQUENCY, printed to "forces.txt" in the output directory.
    - N_PROBE_AVE performs a time average on the root grid with frequency N_PROBE_AVE_FREQUENCY, starting after a number of iterations N_PROBE_AVE_START has passed.
11. Printing is controlled as follows:
    - The output directory path is defined by P_DIR_NAME (default is "../out/results").
    - The solution is printed every P_PRINT iterations, and the solution is run for a total of N_PRINT\*P_PRINT iterations. The test cases in the preprint relied on a single print (N_PRINT=1) with P_PRINT scaled with Nx. 
    - One second in solution time corresponds to 1\*Nx iterations.
    - N_PRINT_LEVELS controls the number of grids in the hierarchy to be included in the output (e.g., MAX_LEVELS=4 and N_PRINT_LEVELS=2 results in levels 0 and 1 being printed, excluding levels 2 and 3).
    - P_SHOW_ADVANCE shows the recursive calls on the console for debugging, P_SHOW_ADVANCE is unused.
12. If the initial and/or boundary conditions were modified, N_REGEN should be set to 1 so that the code generators are called before compilation. Make sure Octave is available in the file path.

Once all of these steps have been checked, simply run `./confmake.sh` from the 'src' directory. This will produce an executable 'a.out' which is executed with `./a.out`. Once the simulation is terminated, the solution is printed and can be viewed with software such as Paraview (make sure that your choice of software can read .vthb files).

For those interested in modifying the domain to include obstacles, this needs to be done manually in the "mesh/init_grid_data.cu" file for now. Locate the condition that defines the square cylinder (line 47) and modify as needed. Make sure that N_CASE is set to 1 so that it's activated.

# Remarks and Feedback

I intend to develop this code further as research with the Turbulence Research Lab progresses. Please do not hesitate to contact me by email (khodr.jaber@mail.utoronto.ca) if you have any comments/recommendations, come across any issues or intend to use this code with your research (I'd love to see how this algorithm is applied in other contexts). I'm also open to collaboration. Make sure the term 'AGAL' is present in the email subject, otherwise I might miss it. Keep in mind that the streaming routine will need to be customized as boundary conditions in this version have been tailored for a lid-driven cavity problem. 

# Citation

Please use the following BibTeX citation if you use this code in your research:

    @misc{jaber2023gpunative,
    title={GPU-Native Adaptive Mesh Refinement with Application to Lattice Boltzmann Simulations}, 
    author={Khodr Jaber and Ebenezer Essel and Pierre Sullivan},
    year={2023},
    eprint={2308.08085},
    archivePrefix={arXiv},
    primaryClass={physics.comp-ph}
    }

This work is still in peer-review, so only the pre-print is available for now.
