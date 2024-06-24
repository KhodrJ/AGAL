function [] = LES_builder(fileID, n_ind, n_d, l_dqs, c, w, build_type)
% Assembles the code for computing LES turbulent viscosity according to a specified subgrid-scale model.
% Input:
% - fileID:           The file being updated (file ID).
% - n_ind:            Number of indentations to start with (integer).
% - n_d:              Number of dimensions (integer).
% - l_dqs:      Number of elements in velocity set (integer).
% - c:                Discrete particle velocity vectors (array).
% - w:                Weights corresponding to particle velocity vectors (array).
% - build_type:       The LBM operation under consideration (integer).
% - comm_type:        .
% 
% Build types:
% 1- Collision.
% 2- Interpolation.
% 3- Averaging.



% LES input file.
T = textread("LES.txt");
nz = 2+T(2);
LES_tab = T(1:nz);



if (LES_tab(1) > 0)
COMMENT(fileID, n_ind, "Get turublent viscosity for Large Eddy Simulation.");
add_statement(fileID, 0, "#if (S_LES==1)", false);

% LES Models.
if (LES_tab(1) == 1)
%
%
%
%	Constant Smagorisnky.
%
%
%
	% Loop and compute second moment tensor. 
	COMMENT(fileID, n_ind, "Compute the second moment tensor.")'
	add_statement(fileID, n_ind, "tmp_k = N_Pf(0.0)", true);
	for P = 1:l_dqs
		COMMENT(fileID, n_ind+1, sprintf("p = %i.", P-1));
		add_statement(fileID, n_ind, sprintf("cdotu = %s", get_cdotu(c(:,P), n_d)), true);
		add_statement(fileID, n_ind, sprintf("tmp_i = f_%i - N_Pf(%17.15f)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)", P-1, w(P)), true);
		for Q = 1:l_dqs
			add_statement(fileID, n_ind, sprintf("cdotu = %s", get_cdotu(c(:,Q), n_d)), true);
			add_statement(fileID, n_ind, sprintf("tmp_i = f_%i - N_Pf(%17.15f)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)", Q-1, w(Q)), true);
		
			% Get coefficient.
			C_ab = 0;
			for PQ_beta = 1:n_d
				for PQ_alpha = 1:n_d
					C_ab = C_ab + c(PQ_alpha,P)*c(PQ_beta,P)*c(PQ_alpha,Q)*c(PQ_beta,Q);
				end
			end
			add_statement(fileID, n_ind, sprintf("tmp_k += N_Pf(%17.15f)*tmp_i*tmp_j = %s", C_ab), true);
		end
	end
	add_line(fileID);
	
	
	% Compute effect relaxation rate, store appropriately.
	% - If during collision operation, store in register memory.
	% - If during communication (interpolation/averaging), store in shared memory.
	COMMENT(fileID, n_ind, "Compute t_eff.");
	if (build_type == 1)
		add_statement(fileID, n_ind, sprintf("omeg = dx_L / (   N_Pf(1.5)*v0 + N_Pf(0.25)*dx_L + N_Pf(0.5)*sqrt( N_Pf(9.0)*v0*v0 + N_Pf(3.0)*v0*dx_L + dx_L*dx_L*(N_Pf(0.25) + N_Pf(36.0)*(N_Pf(%17.15f))*sqrt(tmp_k)/(sqrt(N_Pf(2.0))*rho_kap)) )   )", LES_tab(3)*LES_tab(3)), true);
		add_statement(fileID, n_ind, "omegp = N_Pf(1.0) - omeg", true);
		add_statement(fileID, n_ind, "tau_ratio = N_Pf(0.25) + (N_Pf(0.75)*tau_L - N_Pf(0.25)*dx_L)*(omeg/dx_L)", true);
	end
	if (build_type==2 || build_type==3)
		add_statement(fileID, n_ind, sprintf("s_tau[threadIdx.x] = N_Pf(1.5)*v0 + N_Pf(0.25)*dx_L + N_Pf(0.5)*sqrt( N_Pf(9.0)*v0*v0 + N_Pf(3.0)*v0*dx_L + dx_L*dx_L*(N_Pf(0.25) + N_Pf(36.0)*(N_Pf(%17.15f))*sqrt(tmp_k)/(sqrt(N_Pf(2.0))*rho_kap)) )", LES_tab(3)*LES_tab(3)), true);
	end
elseif (LES_tab(1) == 2)
%
%
%
%	Wall-Adapting Local-Eddy (WALE).
%
%
%
	% Extrapolate velocities to fill in shared memory at block edges. These are needed for second-order differences.
	COMMENT(fileID, n_ind, "Extrapolate macroscopic properties to block edges.");
	extrapolate_halos(fileID, n_ind, n_d, {"s_u","s_v","s_w"}(1:n_d));
	add_statement(fileID, n_ind, "__syncthreads()", true);
	add_line(fileID);
	
	
	% Compute the turbulence viscosity, add to physical viscosity and update relaxation rate.
	COMMENT(fileID, n_ind, "Compute turbulent viscosity, storing S_{ij}^d in tmp_j, S_{ij} in tmp_k.");
	add_statement(fileID, n_ind, "tmp_j = N_Pf(0.0)", true);
	add_statement(fileID, n_ind, "tmp_k = N_Pf(0.0)", true);
	for xbeta = 1:n_d
		for xalpha = 1:n_d
			% Compute S_{alpha,beta}^d.
			add_line(fileID);
			COMMENT(fileID, n_ind+1, sprintf("(%i,%i)", xalpha, xbeta));
			s = "tmp_i = (N_Pf(0.5)*(";
			% uixk*ukxj
			s = strcat(s, "( ");
			for xgamma = 1:n_d
				s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(xalpha,xgamma,n_d),  get_spatial_difference_velocity(xgamma,xbeta,n_d)));
			end
			s = strcat(s, " ) + (");
			% ujxk*ukxi
			for xgamma = 1:n_d
				s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(xbeta,xgamma,n_d),  get_spatial_difference_velocity(xgamma,xalpha,n_d)));
			end
			s = strcat(s, " ))");
			% delta_{ij}*ulxkukxl
			if (xalpha==xbeta)
				s = strcat(s, sprintf(" - (N_Pf(%17.15f))*(", 1/3));
				for xdelta = 1:n_d
					for xgamma = 1:n_d
						s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(xdelta,xgamma,n_d),  get_spatial_difference_velocity(xgamma,xdelta,n_d)));
					end
				end
				s = strcat(s, ")");
			end
			s = strcat(s, ")/(N_Pf(4.0)*dx_L*dx_L)");
			add_statement(fileID, n_ind, s, true);
			
			% Add S_{alpha,beta}^d S_{alpha,beta}^d to tmp_j.
			add_statement(fileID, n_ind, "tmp_j += tmp_i*tmp_i", true);
			
			
			% Compute S_{alpha,beta}.
			s = "tmp_i = N_Pf(0.5)*(";
			s = strcat(s, sprintf("%s + %s", get_spatial_difference_velocity(xalpha,xbeta,n_d),  get_spatial_difference_velocity(xbeta,xalpha,n_d)));
			s = strcat(s, ")/(N_Pf(2.0)*dx_L)");
			add_statement(fileID, n_ind, s, true);
			
			% Add S_{alpha,beta} S_{alpha,beta} to tmp_k.
			add_statement(fileID, n_ind, "tmp_k += tmp_i*tmp_i", true);
		end
	end
	add_line(fileID);
	
	
	% Compute effect relaxation rate, store appropriately.
	% - If during collision operation, store in register memory.
	% - If during communication (interpolation/averaging), store in shared memory.
	COMMENT(fileID, n_ind, "Compute t_eff.");
	add_statement(fileID, n_ind, sprintf("tmp_i = dx_L*dx_L*(N_Pf(%17.15f))*(pow(tmp_j,N_Pf(1.5)))/(pow(tmp_k,N_Pf(2.5))+pow(tmp_j,N_Pf(1.25)))", LES_tab(3)*LES_tab(3)), true);
	n_ind = add_condition(fileID, n_ind, "isnan(tmp_i)", false);
		add_statement(fileID, n_ind, "tmp_i = N_Pf(0.0)", true);
	n_ind = add_condition(fileID, n_ind, "", true);
	if (build_type == 1)
		add_statement(fileID, n_ind, sprintf("omeg = dx_L / (   N_Pf(3.0)*(v0 + tmp_i) + N_Pf(0.5)*dx_L   )", LES_tab(3)*LES_tab(3)), true);
		add_statement(fileID, n_ind, "omegp = N_Pf(1.0) - omeg", true);
		add_statement(fileID, n_ind, "tau_ratio = N_Pf(0.25) + (N_Pf(0.75)*tau_L - N_Pf(0.25)*dx_L)*(omeg/dx_L)", true);
	end
	if (build_type==2 || build_type==3)
		add_statement(fileID, n_ind, sprintf("s_tau[threadIdx.x] = N_Pf(3.0)*(v0 + tmp_i) + N_Pf(0.5)*dx_L", LES_tab(3)*LES_tab(3)), true);
	end
elseif (LES_tab(1)==3)
%
%
%
%	Vreman Model.
%
%
%
	% Extrapolate velocities to fill in shared memory at block edges. These are needed for second-order differences.
	COMMENT(fileID, n_ind, "Extrapolate macroscopic properties to block edges.");
	extrapolate_halos(fileID, n_ind, n_d, {"s_u","s_v","s_w"}(1:n_d));
	add_statement(fileID, n_ind, "__syncthreads()", true);
	add_line(fileID);

	
	% Compute the turbulence viscosity, add to physical viscosity and update relaxation rate.
	COMMENT(fileID, n_ind, "Compute turbulent viscosity, storing S_{ij}^d in tmp_j, S_{ij} in tmp_k.");
	add_statement(fileID, n_ind, "tmp_i = N_Pf(0.0)", true);
	add_statement(fileID, n_ind, "tmp_j = N_Pf(0.0)", true);
	add_statement(fileID, n_ind, "tmp_k = N_Pf(0.0)", true);
	
	
	% Compute B_beta.
	% + 11 22
	COMMENT(fileID, n_ind+1, "(11,22)");
	s = "tmp_i += ((";
	for xgamma = 1:n_d
		s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(1,xgamma,n_d),  get_spatial_difference_velocity(1,xgamma,n_d)));
	end
	s = strcat(s, ")*(");
	for xgamma = 1:n_d
		s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(2,xgamma,n_d),  get_spatial_difference_velocity(2,xgamma,n_d)));
	end
	s = strcat(s, "))");
	add_statement(fileID, n_ind, s, true);
	%
	if (n_d == 3)
		% + 11 33
		COMMENT(fileID, n_ind+1, "(11,33)");
		s = "tmp_i += ((";
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(1,xgamma,n_d),  get_spatial_difference_velocity(1,xgamma,n_d)));
		end
		s = strcat(s, ")*(");
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(3,xgamma,n_d),  get_spatial_difference_velocity(3,xgamma,n_d)));
		end
		s = strcat(s, "))");
		add_statement(fileID, n_ind, s, true);
		%
		% + 22 33
		COMMENT(fileID, n_ind+1, "(22,33)");
		s = "tmp_i += ((";
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(2,xgamma,n_d),  get_spatial_difference_velocity(2,xgamma,n_d)));
		end
		s = strcat(s, ")*(");
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(3,xgamma,n_d),  get_spatial_difference_velocity(3,xgamma,n_d)));
		end
		s = strcat(s, "))");
		add_statement(fileID, n_ind, s, true);
	end
	%
	% + 12 12
	COMMENT(fileID, n_ind+1, "(12,12)");
	s = "tmp_j += ((";
	for xgamma = 1:n_d
		s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(1,xgamma,n_d),  get_spatial_difference_velocity(2,xgamma,n_d)));
	end
	s = strcat(s, ")*(");
	for xgamma = 1:n_d
		s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(1,xgamma,n_d),  get_spatial_difference_velocity(2,xgamma,n_d)));
	end
	s = strcat(s, "))");
	add_statement(fileID, n_ind, s, true);
	%
	if (n_d == 3)
		% - 13 13
		COMMENT(fileID, n_ind+1, "(13,13)");
		s = "tmp_j += ((";
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(1,xgamma,n_d),  get_spatial_difference_velocity(3,xgamma,n_d)));
		end
		s = strcat(s, ")*(");
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(1,xgamma,n_d),  get_spatial_difference_velocity(3,xgamma,n_d)));
		end
		s = strcat(s, "))");
		add_statement(fileID, n_ind, s, true);
		%
		% - 23 23
		COMMENT(fileID, n_ind+1, "(23,23)");
		s = "tmp_j += ((";
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(2,xgamma,n_d),  get_spatial_difference_velocity(3,xgamma,n_d)));
		end
		s = strcat(s, ")*(");
		for xgamma = 1:n_d
			s = strcat(s, sprintf(" + %s*%s", get_spatial_difference_velocity(2,xgamma,n_d),  get_spatial_difference_velocity(3,xgamma,n_d)));
		end
		s = strcat(s, "))");
		add_statement(fileID, n_ind, s, true);
	end
	%
	add_statement(fileID, n_ind, "tmp_j = N_Pf(0.25)*(tmp_i - tmp_j)", true);
	add_line(fileID);
	
	
	% Compute alpha_{ij} alpha_{ij}.
	COMMENT(fileID, n_ind, "Denominator.");
	for xdelta = 1:n_d
		for xgamma = 1:n_d
			add_statement(fileID, n_ind, sprintf("tmp_k += %s*%s", get_spatial_difference_velocity(xdelta,xgamma,n_d),  get_spatial_difference_velocity(xdelta,xgamma,n_d)), true);
		end
	end
	add_line(fileID);
	
	
	% Compute effect relaxation rate, store appropriately.
	% - If during collision operation, store in register memory.
	% - If during communication (interpolation/averaging), store in shared memory.
	COMMENT(fileID, n_ind, "Compute t_eff.");
	add_statement(fileID, n_ind, "tmp_k = tmp_k/(N_Pf(4.0)*dx_L*dx_L)", true);
	add_statement(fileID, n_ind, sprintf("tmp_i = (N_Pf(%17.15f))*sqrt(tmp_j/tmp_k)", LES_tab(3)), true);
	n_ind = add_condition(fileID, n_ind, "isnan(tmp_i)", false);
		add_statement(fileID, n_ind, "tmp_i = N_Pf(0.0)", true);
	n_ind = add_condition(fileID, n_ind, "", true);
	if (build_type == 1)
		add_statement(fileID, n_ind, sprintf("omeg = dx_L / (   N_Pf(3.0)*(v0 + tmp_i) + N_Pf(0.5)*dx_L   )", LES_tab(3)*LES_tab(3)), true);
		add_statement(fileID, n_ind, "omegp = N_Pf(1.0) - omeg", true);
		add_statement(fileID, n_ind, "tau_ratio = N_Pf(0.25) + (N_Pf(0.75)*tau_L - N_Pf(0.25)*dx_L)*(omeg/dx_L)", true);
	end
	if (build_type==2 || build_type==3)
		add_statement(fileID, n_ind, sprintf("s_tau[threadIdx.x] = N_Pf(3.0)*(v0 + tmp_i) + N_Pf(0.5)*dx_L", LES_tab(3)*LES_tab(3)), true);
	end
else
	
end

add_statement(fileID, 0, "#endif", false);
add_line(fileID);
end



endfunction
