function [] = load_macro_props(fileID, n_ind, n_d, l_dqs, c)
% Print the expression to load density and velocity from DDFs in register memory.
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - n_d:        Number of dimensions (integer).
% - l_dqs:      Number of elements in velocity set (integer).
% - c:          Discrete particle velocity vectors (array).

% Compute density.
s_density = "";
for P = 1:l_dqs
	s_density = strcat(s_density,  sprintf(" +f_%i", P-1));
end

% Compute velocity.
s_u = "";
s_v = "";
s_w = "";
for P = 1:l_dqs
	if (c(1,P) == 1)
		s_u = strcat(s_u,  sprintf(" +f_%i", P-1));
	end
	if (c(1,P) == -1)
		s_u = strcat(s_u,  sprintf(" -f_%i", P-1));
	end
	
	if (c(2,P) == 1)
		s_v = strcat(s_v,  sprintf(" +f_%i", P-1));
	end
	if (c(2,P) == -1)
		s_v = strcat(s_v,  sprintf(" -f_%i", P-1));
	end
	
	if (n_d == 3)
		if (c(3,P) == 1)
			s_w = strcat(s_w,  sprintf(" +f_%i", P-1));
		end
		if (c(3,P) == -1)
			s_w = strcat(s_w,  sprintf(" -f_%i", P-1));
		end
	end
end

% Print macroscopic properties.
add_statement(fileID, n_ind, sprintf("rho_kap =%s", s_density), true);
add_statement(fileID, n_ind, sprintf("u_kap = (%s) / rho_kap", s_u), true);
add_statement(fileID, n_ind, sprintf("v_kap = (%s) / rho_kap", s_v), true);
if (n_d == 3)
	add_statement(fileID, n_ind, sprintf("w_kap = (%s) / rho_kap", s_w), true);
end

% Print udotu while we're at it.
if (n_d == 2)
	add_statement(fileID, n_ind, "udotu = u_kap*u_kap + v_kap*v_kap", true);
else
	add_statement(fileID, n_ind, "udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap", true);
end

endfunction
