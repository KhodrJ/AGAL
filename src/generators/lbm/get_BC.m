% Reads the BC text files and produces a table of boundary types, Ids and specified variables (only velocity for now).
function [H] = get_BC(n_d)
% Reads the appropriate BC input files (BC_{2/3}D.txt) and produces a table of parameters to be interpreted later.
% Input:
% - n_d:        Number of dimensions (integer).
%
% Format of the BC input files:
% Definitions:
% - b_id                      -> Boundary Id being specified.
% - u_{x/y/z}                 -> Velocity vector.
%
% Types:
% - (0) bounce-back           -> b_id 0 u_x u_y u_z
% - (1) open [outlet]         -> b_id 1

H = [];

if (n_d == 2)
	T = textread("BC_2D.txt");
else
	T = textread("BC_3D.txt");
end

cont = true;
counter = 1;
while (cont)
	b_id = T(counter);
	if (isnan(b_id) == false)
		b_type = T(counter+1);   % Not to be confused with B_TYPE (which tells us if non-no-slip boundary conditions might be present). This tells us which type of boundary condition we're dealing with.
		if (b_type == 0)
			% Read velocities.
			ux = T(counter+2);
			uy = T(counter+3);
			uz = 0;
			if (n_d == 3)
				uz = T(counter+4);
			end
			
			H = [H; [b_id,b_type,ux,uy,uz]];
			
			% Update counter (depends on K).
			counter = counter + 5;
			if (n_d == 2)
				counter = counter - 1;
			end
		elseif (b_type == 1)
			H = [H; [b_id,b_type,0,0,0]];
			
			% Update counter.
			counter = counter + 2;
		end
	else
		cont = false;
	end
end

endfunction 
