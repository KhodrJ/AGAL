function [s_final] = get_minimal_BC_cond(n_d, l_dqs, P, nbr, c)
% Gets the exact condition needed for implementing the appropriate boundary condition.
% Input:
% - n_d:        Number of dimensions (integer).
% - l_dqs:      Number of elements in velocity set (integer).
% - P:          DDF index (integer).
% - nbr:        Neighbor index (integer).
% - c:          Discrete particle velocity vectors (array).
% Output:
% - s_final:    The minimal BC condition (string).

s = {};
s_final = "";
cP = c(:,P);
cnbr = c(:,nbr);

% In X.
s_x = "";
if (cP(1)==1)
	s_x = [s_x, "(I_kap+1"];
	%s(end+1) = {"(I_kap+1==Nbx)"};
elseif (cP(1)==-1)
	s_x = [s_x, "(I_kap-1"];
	%s(end+1) = {"(I_kap-1==-1)"};
end
if (cnbr(1)==1)
	s_x = [s_x, "==Nbx)"];
	%s(end+1) = {"(I_kap+1==Nbx)"};
elseif (cnbr(1)==-1)
	s_x = [s_x, "==-1)"];
	%s(end+1) = {"(I_kap-1==-1)"};
else
	if (cP(1)==1)
		s_x = [s_x, "< Nbx)"];
	end
	if (cP(1)==-1)
		s_x = [s_x, ">= 0)"];
	end
end
if (cP(1)~=0 && ~strcmp(s_x,""))
	s(end+1) = {s_x};
end

% In Y.
s_y = "";
if (cP(2)==1)
	s_y = [s_y, "(J_kap+1"];
	%s(end+1) = {"(J_kap+1"};
elseif (cP(2)==-1)
	s_y = [s_y, "(J_kap-1"];
	%s(end+1) = {"(J_kap-1"};
end
if (cnbr(2)==1)
	s_y = [s_y, "==Nbx)"];
	%s(end+1) = {"(I_kap+1==Nbx)"};
elseif (cnbr(2)==-1)
	s_y = [s_y, "==-1)"];
	%s(end+1) = {"(I_kap-1==-1)"};
else
	if (cP(2)==1)
		s_y = [s_y, "< Nbx)"];
	end
	if (cP(2)==-1)
		s_y = [s_y, ">= 0)"];
	end
end
if (cP(2)~=0 && ~strcmp(s_y,""))
	s(end+1) = {s_y};
end

if (n_d == 3)
	% In Z.
	s_z = "";
	if (cP(3)==1)
		s_z = [s_z, "(K_kap+1"];
		%s(end+1) = {"(K_kap+1==Nbx)"};
	elseif (cP(3)==-1)
		s_z = [s_z, "(K_kap-1"];
		%s(end+1) = {"(K_kap-1==-1)"};
	end
	if (cnbr(3)==1)
		s_z = [s_z, "==Nbx)"];
		%s(end+1) = {"(I_kap+1==Nbx)"};
	elseif (cnbr(3)==-1)
		s_z = [s_z, "==-1)"];
		%s(end+1) = {"(I_kap-1==-1)"};
	else
		if (cP(3)==1)
			s_z = [s_z, "< Nbx)"];
		end
		if (cP(3)==-1)
			s_z = [s_z, ">= 0)"];
		end
	end
	if (cP(3)~=0 && ~strcmp(s_z,""))
		s(end+1) = {s_z};
	end
end

s_final = s{1};
for i = 2:length(s)
	s_final = [s_final, " && ", s{i}];
end

endfunction 
