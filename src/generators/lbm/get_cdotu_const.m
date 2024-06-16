function [s] = get_cdotu_const(ci,n_d)
% Get the proper expression for (c dot u), depending on the dimension and particle velocity.
% Input:
% - ci:    Particle velocity vector (array).
% - n_d:   Number of dimensions (integer).
% Output:
% - s:     Expression for c dot u (string).

s = "";

if (sum(abs(ci))==0)
	s = "N_Pf(0.0)";
end

if (ci(1) == 1)
	s = [s,"+u_t0"];
elseif (ci(1) == -1)
	s = [s,"-u_t0"];
end
if (ci(2) == 1)
	s = [s,"+v_t0"];
elseif (ci(2) == -1)
	s = [s,"-v_t0"];
end
if (n_d == 3)
	if (ci(3) == 1)
		s = [s,"+w_t0"];
	elseif (ci(3) == -1)
		s = [s,"-w_t0"];
	end
end

endfunction
