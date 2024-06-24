clc
clear all
close all

addpath("./general/");
addpath("./math/");
addpath("./comm/");
addpath("./lbm/");
addpath("./input/");

fileID = fopen("./out_test/test.txt",'w');


conds = "(interp_type == 0 && cells_ID_mask[(i_kap_bc+%i)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1)";
vars_s1 = {"cells_f_F[(i_kap_bc+%i)*M_CBLOCK + threadIdx.x + %i*n_maxcells]", "cells_f_Feq[(i_kap_bc+%i)*M_CBLOCK + threadIdx.x + %i*n_maxcells]"};
vars_s2 = {"s_F","s_F"};


interp_linear_tree(fileID, 2, 3, vars_s1, vars_s2, conds);



fclose(fileID);
