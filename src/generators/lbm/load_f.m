function [] = load_f(fileID, n_ind, n_d, l_dqs, pb)
% Print the expressions for loading DDFs into register memory.
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - n_d:        Number of dimensions (integer).
% - l_dqs:      Number of elements in velocity set (integer).
% - pb:         Reverse-direction indices (array).

% Print.
for P = 1:l_dqs
	add_statement(fileID, n_ind, sprintf("f_%i = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells]", P-1, P-1), true);
end

endfunction
