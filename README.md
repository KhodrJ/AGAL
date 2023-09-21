# AGAL: Adaptive GPU-native Algorithm for Lattice-based simulations
AGAL (or عَجَل, meaning 'haste') is an open-source C++ code implementing a GPU-native algorithm (AMR) for adaptive mesh refinement applied to Lattice Boltzmann Simulations. A simple square/cubic domain is fixed, and boundary conditions in the streaming function are designed for the lid-driven cavity problem (simple bounce-back, positive lid velocity $u_{\text{lid}}$). The code also implements three velocity sets: D2Q9, D3Q19 and D3Q27. This version of the code was used for the case studies reported in the corresponding manuscript "[GPU-Native Adaptive Mesh Refinement with Application to Lattice Boltzmann Simulations](https://arxiv.org/abs/2308.08085)." 

# Instructions for Compilation and Usage

## Compilation

The code uses the VTK library for output; an vtkOverlappingAMR dataset is used during data collection and written via the vtkXMLUniformGridAMRWriter. The Thrust library is employed for many common array operations. The correct path to VTK (built for the machine being used) needs to be adjusted in the Makefile. Two options are assumed: 1) in the 'lib' directory, 2) in /usr/local/, these can be commented/commented-out accordingly. If the 'lib' directory is used, make sure that VTK include and lib files are available via the paths ./lib/vtk/include/ and ./lib/vtk/lib/, respectively.

## Usage

The manuscript detailing the GPU-native algorithm is under review. Validation was restricted to lid-driven cavity benchmarks in 2D and 3D. Reynolds number is specified by modification of the kinematic viscosity during calcluation of the relaxation-rate. This allows for specification of the following (in the cppspec.h file): u_lid (lid velocity), Re_c (Reynolds number), L_c (characteristic length, usually left as 1.0 m). If any of these are modified, the program must be recompiled to take effect. Lid velocity should be left small to ensure validity of the Lattice Boltzmann simulation (keep in mind $\text{Ma} = u_{\text{lid}} / c_s$). Velocity sets are chosen in 'lbm.h' by modifying the ld_q variable. This step can be skipped for 2D simulations since only D2Q9 is available right now.
   
There are four implemented compilation options:
1. N_PRECISION (0 is for single precision, 1 is double precision)
2. N_DIM (2 or 3)
3. MAX_LEVELS (determines the size of the grid hierarchy)
4. P_DIR_NAME (where output is directed, default is  ../out/).

Other options that must be modified in 'cppspec.h' are (I'll be turning these into compilation options eventually):
- Nx (coarse grid resolution)
- P_REFINE (how many iterations should pass between calls to refinement/coarsening)
- P_PRINT (how many iterations should pass before simulation is terminated and solution is printed)
- N_PRINT_LEVELS (how many grid levels should be printed)
- P_SHOW_REFINE (whether (1) or not (0) refinement/coarsening info should be printed to console)

A sample command for compilation is:
`make N_PRECISION=0 N_DIM=2 MAX_LEVELS=3`. Run this command in the 'src' directory. This will produce an executable 'a.out' which is simply run with `./a.out`. Once the simulation is terminated, the solution is printed and can be viewed with software such as Paraview (make sure that your choice of software can read .vthb files).

# Remarks and Feedback

I intend to develop this code further as research with the Turbulence Research Lab progresses. Please do not hesistate to contact me by email (khodr.jaber@mail.utoronto.ca) if you have any comments/recommendations, come across any issues or intend to use this code with your research (I'd love to see how this algorithm is applied in other contexts). I'm also open to collaboration. Make sure the term 'AGAL' is present in the email subject, otherwise I might miss it. Keep in mind that the streaming routine will need to be customized as boundary conditions in this version have been tailored for a lid-driven cavity problem. 

# Citation

Please use the following BibTeX citation if you use this code or AMR methodology in your research:

    @misc{jaber2023gpunative,
    title={GPU-Native Adaptive Mesh Refinement with Application to Lattice Boltzmann Simulations}, 
    author={Khodr Jaber and Ebenezer Essel and Pierre Sullivan},
    year={2023},
    eprint={2308.08085},
    archivePrefix={arXiv},
    primaryClass={physics.comp-ph}
    }

This work is still in peer-review, so only the pre-print is available for now.
