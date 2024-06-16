function [] = load_to_shared_halo(fileID, n_ind, n_d, var_s1, var_s2)
% Load a variable into shared memory arranged for a 4x4(x4) block with a halo.
% Assumes existence of {I/J/K}_kap variables.
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - n_d:        Number of dimensions.
% - var_s1:     The destination variable (string).
% - var_s2:     The variable being loaded (string).

s_indices = "(I_kap+1)+(Nbx+2)*(J_kap+1)";
if (n_d == 3)
	s_indices = "(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)";
end	
add_statement(fileID, n_ind, sprintf("%s[%s] = %s", var_s1, s_indices, var_s2), true);

endfunction
