function [] = extrapolate_halos(fileID, n_ind, n_d, vars_s1)
% Load a variable into shared memory arranged for a 4x4(x4) block with a halo.
% Assumes existence of {I/J/K}_kap variables.
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - n_d:        Number of dimensions (integer).
% - var_s1:     The destination variables (cell array).

n_1 = length(vars_s1);

if (n_d == 2)
	n_ind = add_condition(fileID, n_ind, "I_kap==0", false);
	% Left.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[0+(Nbx+2)*(J_kap+1)] = 4*%s[1+(Nbx+2)*(J_kap+1)] - 6*%s[2+(Nbx+2)*(J_kap+1)] + 4*%s[3+(Nbx+2)*(J_kap+1)] - %s[4+(Nbx+2)*(J_kap+1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	% Right.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[5+(Nbx+2)*(J_kap+1)] = 4*%s[4+(Nbx+2)*(J_kap+1)] - 6*%s[3+(Nbx+2)*(J_kap+1)] + 4*%s[2+(Nbx+2)*(J_kap+1)] - %s[1+(Nbx+2)*(J_kap+1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	n_ind = add_condition(fileID, n_ind, "", true);


	n_ind = add_condition(fileID, n_ind, "J_kap==0", false);
	% Down.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[(I_kap+1)+(Nbx+2)*(0)] = 4*%s[(I_kap+1)+(Nbx+2)*(1)] - 6*%s[(I_kap+1)+(Nbx+2)*(2)] + 4*%s[(I_kap+1)+(Nbx+2)*(3)] - %s[(I_kap+1)+(Nbx+2)*(4)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	% Up.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[(I_kap+1)+(Nbx+2)*(5)] = 4*%s[(I_kap+1)+(Nbx+2)*(4)] - 6*%s[(I_kap+1)+(Nbx+2)*(3)] + 4*%s[(I_kap+1)+(Nbx+2)*(2)] - %s[(I_kap+1)+(Nbx+2)*(1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	n_ind = add_condition(fileID, n_ind, "", true);
else
	n_ind = add_condition(fileID, n_ind, "I_kap==0", false);
	% Left.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[0+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*%s[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*%s[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*%s[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - %s[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	% Right.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[5+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*%s[4+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*%s[3+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*%s[2+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - %s[1+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	n_ind = add_condition(fileID, n_ind, "", true);


	n_ind = add_condition(fileID, n_ind, "J_kap==0", false);
	% Back.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[(I_kap+1)+(Nbx+2)*(0)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*%s[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*%s[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*%s[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - %s[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	% Front.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[(I_kap+1)+(Nbx+2)*(5)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = 4*%s[(I_kap+1)+(Nbx+2)*(4)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - 6*%s[(I_kap+1)+(Nbx+2)*(3)+(Nbx+2)*(Nbx+2)*(K_kap+1)] + 4*%s[(I_kap+1)+(Nbx+2)*(2)+(Nbx+2)*(Nbx+2)*(K_kap+1)] - %s[(I_kap+1)+(Nbx+2)*(1)+(Nbx+2)*(Nbx+2)*(K_kap+1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	n_ind = add_condition(fileID, n_ind, "", true);
	
	
	n_ind = add_condition(fileID, n_ind, "K_kap==0", false);
	% Bottom.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(0)] = 4*%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)] - 6*%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] + 4*%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] - %s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	% Top.
	for j = 1:n_1
		add_statement(fileID, n_ind, sprintf("%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(5)] = 4*%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(4)] - 6*%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(3)] + 4*%s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(2)] - %s[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(1)]", vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}, vars_s1{j}), true);
	end
	n_ind = add_condition(fileID, n_ind, "", true);
end

endfunction
