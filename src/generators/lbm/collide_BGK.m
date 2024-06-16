function [] = collide_BGK(fileID, n_ind, n_d, l_dqs, c, w)
% Print the expressions for collision under the BGK (equivalently, the SRT) model.
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - n_d:        Number of dimensions (integer).
% - l_dqs:      Number of elements in velocity set (integer).
% - c:          Discrete particle velocity vectors (array).
% - w:          Weights corresponding to particle velocity vectors (array).

for P = 1:l_dqs
	add_statement(fileID, n_ind, sprintf("cdotu = %s", get_cdotu(c(:,P), n_d)), true);
	add_statement(fileID, n_ind, sprintf("f_%i = f_%i*omegp + ( N_Pf(%17.15f)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg", P-1, P-1, w(P)), true);
end

endfunction
