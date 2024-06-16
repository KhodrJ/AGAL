function [str_cond_total_BC, str_cond_total_nBC, str_index_total_BC] = get_conditions_nbrs(v_id, n_d)
% Building conditions for accessing DDFs based on neighbor index. Return these conditions for later reads or writes from these neighbors.
% Input:
% - v_id:                      Index of opposing DDF velocity vector (file ID).
% - n_d:                       Number of dimensions (integer).
% Output:
% - str_cond_total_BC:         Conditions for placement of internal f* DDFs in halo for BC (string).
% - str_cond_total_nBC:        Conditions for placement of neighbor f* DDFs in halo for non-BC (string).
% - str_index_total_BC:        Indices for placement of f* DDFs in halo (string).

str_i_cond_BC = ""; str_i_cond_nBC = ""; str_i_index_BC = "(I_kap+1)";
str_j_cond_BC = ""; str_i_cond_nBC = ""; str_j_index_BC = "(J_kap+1)";
str_k_cond_BC = ""; str_i_cond_nBC = ""; str_k_index_BC = "(K_kap+1)";
str_cond_total_BC = ""; str_cond_total_nBC = ""; str_index_total_BC = "";
if (v_id(1)==1)
	
	%fprintf(fileID, "(I_kap == 0) && ");
	str_i_cond_BC = "(I_kap == 0)";
	str_i_cond_nBC = "(I_kap == Nbx-1)";
	str_i_index_BC = "(0)";
end
if (v_id(1)==-1)
	%fprintf(fileID, "(I_kap == Nbx-1) && ");
	str_i_cond_BC = "(I_kap == Nbx-1)";
	str_i_cond_nBC = "(I_kap == 0)";
	str_i_index_BC = "(Nbx+2 - 1)";
end
if (v_id(1)==0)
	%fprintf(fileID, "(I_kap>=0 && I_kap<Nbx) && ");
	str_i_cond_BC = "(I_kap>=0 && I_kap<Nbx)";
	str_i_cond_nBC = "(I_kap>=0 && I_kap<Nbx)";
end
if (v_id(2)==1)
	%fprintf(fileID, "(J_kap == 0) && ");
	str_j_cond_BC = "(J_kap == 0)";
	str_j_cond_nBC = "(J_kap == Nbx-1)";
	str_j_index_BC = "(0)";
end
if (v_id(2)==-1)
	%fprintf(fileID, "(J_kap == Nbx-1) && ");
	str_j_cond_BC = "(J_kap == Nbx-1)";
	str_j_cond_nBC = "(J_kap == 0)";
	str_j_index_BC = "(Nbx+2 - 1)";
end
if (v_id(2)==0)
	%fprintf(fileID, "(J_kap>=0 && J_kap<Nbx) && ");
	str_j_cond_BC = "(J_kap>=0 && J_kap<Nbx)";
	str_j_cond_nBC = "(J_kap>=0 && J_kap<Nbx)";
end
if (n_d == 3)
	if (v_id(3)==1)
		%fprintf(fileID, "(K_kap == 0) )\n");
		str_k_cond_BC = "(K_kap == 0)";
		str_k_cond_nBC = "(K_kap == Nbx-1)";
		str_k_index_BC = "(0)";
	end
	if (v_id(3)==-1)
		%fprintf(fileID, "(K_kap == Nbx-1) )\n");
		str_k_cond_BC = "(K_kap == Nbx-1)";
		str_k_cond_nBC = "(K_kap == 0)";
		str_k_index_BC = "(Nbx+2 - 1)";
	end
	if (v_id(3)==0)
		%fprintf(fileID, "(K_kap>=0 && K_kap<Nbx) )\n");
		str_k_cond_BC = "(K_kap>=0 && K_kap<Nbx)";
		str_k_cond_nBC = "(K_kap>=0 && K_kap<Nbx)";
	end
end
if (n_d == 2)
	str_cond_total_BC = [str_i_cond_BC, " && ", str_j_cond_BC];
	str_cond_total_nBC = [str_i_cond_nBC, " && ", str_j_cond_nBC];
	str_index_total_BC = [str_i_index_BC, " + (Nbx+2)*", str_j_index_BC];
else
	str_cond_total_BC = [str_i_cond_BC, " &&  ", str_j_cond_BC, " && ", str_k_cond_BC];
	str_cond_total_nBC = [str_i_cond_nBC, " &&  ", str_j_cond_nBC, " && ", str_k_cond_nBC];
	str_index_total_BC = [str_i_index_BC, " + (Nbx+2)*", str_j_index_BC, " + (Nbx+2)*(Nbx+2)*", str_k_index_BC];
end
endfunction 
