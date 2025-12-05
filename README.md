
# AGAL: <u>A</u>daptive <u>G</u>PU-native <u>A</u>lgorithm for <u>L</u>attice Boltzmann Simulations
AGAL (or عَجَل, meaning "hurry") is an open-source C++/CUDA code implementing a GPU-native algorithm (AMR) for adaptive mesh refinement applied to Lattice Boltzmann Simulations. This repository is still in-development; it supports a simple, fixed square/cubic domain that implements either the lid-driven cavity or external flow over an arbitrary user-supplied geometry. The code also implements three velocity sets: D2Q9, D3Q19 and D3Q27. It is current restricted to execution on a single GPU. Expect large changes to the codebase as I improve readability and performance over time (though I believe that the header library format will be retained in the long term).

# Instructions for Compilation and Usage

## Compilation

This is a header library, so it does not need to be built. Simply add the requires includes to a main script and compile that. The code uses the VTK library for output; an vtkOverlappingAMR dataset is used during data collection and written via the vtkXMLUniformGridAMRWriter. The Thrust library is employed for many standard parallel algorithms. The correct path to VTK (built for the machine being used) needs to be adjusted in the Makefile. Three options are assumed: 1) in the 'lib' directory, 2) in "/usr/local/", 3) standard software directory on a GPU cluster, these can be commented/commented-out accordingly. If the 'lib' directory is used, make sure that VTK include and lib files are available via the paths "./lib/vtk/include/" and "./lib/vtk/lib/", respectively. Note that VTK versions 9.1-9.5 have been used to successfully compile the code. I have now verified that the codebase can be compiled in the Windows Subsystem for Linux (WSL) as well.

## Usage

Detailed instructions for setting input parameters and specifying boundary conditions are being written. Please check back shortly.

# Remarks and Feedback

I intend to develop this code further as research with the Turbulence Research Lab progresses. Please do not hesitate to contact me by email (khodr.jaber@mail.utoronto.ca) if you have any comments/recommendations, come across any issues or intend to use this code with your research (I'd love to see how this algorithm is applied in other contexts). I'm also open to collaboration. Make sure the term 'AGAL' is present in the email subject, otherwise I might miss it.

# Citation

The original manuscript [1] detailing the GPU-native algorithm was accepted by Computer Physics Communications. A manuscript detailing an updated framework for embedding complex geometries [2] is now in review.

Please use the following BibTeX citation if you use this code in your research, or build on my framework:
@article{Jaber2025,
title = {GPU-native adaptive mesh refinement with application to lattice Boltzmann simulations},
journal = {Computer Physics Communications},
volume = {311},
pages = {109543},
year = {2025},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2025.109543},
url = {https://www.sciencedirect.com/science/article/pii/S0010465525000463},
author = {Khodr Jaber and Ebenezer E. Essel and Pierre E. Sullivan},
}

[1] K. Jaber, E. E. Essel, and P. E. Sullivan, ‘GPU-native adaptive mesh refinement with application to lattice Boltzmann simulations’, Computer Physics Communications, vol. 311, p. 109543, 2025.
[2] K. Jaber, E. E. Essel, and P. E. Sullivan, ‘GPU-native Embedding of Complex Geometries in Adaptive Octree Grids Applied to the Lattice Boltzmann Method’, arXiv [cs.CE]. 2025.
