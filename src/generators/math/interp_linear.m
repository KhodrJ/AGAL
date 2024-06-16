function [] = interp_linear(fileID, n_ind, n_d, indices, var_s1, var_s2)
% Linear interpolation in either 2D or 3D for a fixed set of indices.
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - n_d:        Number of dimensions (integer).
% - indices:    The indices in shared memory to use for interpolation.
% - var_s1:     The destination variable (string).
% - var_s2:     The variable being interpolated (string).

% Construct indentations.
ind = "";
for i = 1:n_ind
	ind = strcat(ind,"\t");
end

% Cases depends on dimension.
if (n_d == 2)
	fprintf(fileID, "%s%s = %s[%d] + (%s[%d]-%s[%d])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) + (%s[%d]-%s[%d])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) + (%s[%d]-%s[%d]-%s[%d]+%s[%d])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5));\n",
		ind, var_s1,
		var_s2, indices(1,1),   % a0
		var_s2, indices(2,1), var_s2, indices(1,1),   % a1 
		var_s2, indices(1,2), var_s2, indices(1,1),   % a2
		var_s2, indices(2,2), var_s2, indices(1,2), var_s2, indices(2,1), var_s2, indices(1,1)   % a3
	);
else
	fprintf(fileID, "%s%s = %s[%d] + (%s[%d] - %s[%d])*(N_Pf(-0.25) + I_kap*N_Pf(0.5)) + (%s[%d] - %s[%d])*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) + (%s[%d] - %s[%d])*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) + (%s[%d] - %s[%d] - %s[%d] + %s[%d])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5)) + (%s[%d] - %s[%d] - %s[%d] + %s[%d])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) + (%s[%d] - %s[%d] - %s[%d] + %s[%d])*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5)) + (%s[%d] + %s[%d] + %s[%d] + %s[%d] - %s[%d] - %s[%d] - %s[%d] - %s[%d])*(N_Pf(-0.25) + I_kap*N_Pf(0.5))*(N_Pf(-0.25) + J_kap*N_Pf(0.5))*(N_Pf(-0.25) + K_kap*N_Pf(0.5));\n",
		ind, var_s1,
		var_s2, indices(1,1,1),   % a0
		var_s2, indices(2,1,1), var_s2, indices(1,1,1),   % a1 
		var_s2, indices(1,2,1), var_s2, indices(1,1,1),   % a2
		var_s2, indices(1,1,2), var_s2, indices(1,1,1),   % a3
		var_s2, indices(2,2,1), var_s2, indices(2,1,1), var_s2, indices(1,2,1), var_s2, indices(1,1,1),   % a4
		var_s2, indices(2,1,2), var_s2, indices(2,1,1), var_s2, indices(1,1,2), var_s2, indices(1,1,1),   % a5
		var_s2, indices(1,2,2), var_s2, indices(1,2,1), var_s2, indices(1,1,2), var_s2, indices(1,1,1),   % a6
		var_s2, indices(2,2,2), var_s2, indices(2,1,1), var_s2, indices(1,2,1), var_s2, indices(1,1,2), var_s2, indices(2,2,1), var_s2, indices(2,1,2), var_s2, indices(1,2,2), var_s2, indices(1,1,1)   % a7
	);
end

endfunction
