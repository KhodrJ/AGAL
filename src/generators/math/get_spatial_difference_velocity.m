function [s] = get_spatial_difference_velocity(i, j, n_d)
% Get expression for the first-order spatial finite-difference.
% Input:
% - i,j:    Indices for del u_i / del x_j.
% - n_d:    Number of dimensions.  

s_us = {'u','v','w'};
s_pis = {'+1','',''};
s_pjs = {'','+1',''};
s_pks = {'','','+1'};
s_mis = {'-1','',''};
s_mjs = {'','-1',''};
s_mks = {'','','-1'};

% i is the velocity component, j is the spatial component, K is the problem dimension.
if (n_d == 2)
	s = sprintf("(s_%s[(I_kap%s+1)+(Nbx+2)*(J_kap%s+1)] - s_%s[(I_kap%s+1)+(Nbx+2)*(J_kap%s+1)])", s_us{i},s_pis{j},s_pjs{j}, s_us{i},s_mis{j},s_mjs{j});
else
	s = sprintf("(s_%s[(I_kap%s+1)+(Nbx+2)*(J_kap%s+1)+(Nbx+2)*(Nbx+2)*(K_kap%s+1)] - s_%s[(I_kap%s+1)+(Nbx+2)*(J_kap%s+1)+(Nbx+2)*(Nbx+2)*(K_kap%s+1)])", s_us{i},s_pis{j},s_pjs{j},s_pks{j}, s_us{i},s_mis{j},s_mjs{j},s_mks{j});
end

endfunction
