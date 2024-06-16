function [ids] = get_combos(v,c)
% Combinations of DDF indices and parent-block retrieval indices for precise information transfer.
% Input:
% - v:          Index of opposing DDF velocity vector (file ID).
% - c:          Discrete particle velocity vectors (array).
% Output:
% - ids:        Ids from which to obtain neighboring DDFs (array).

M = v;
if (length(v)==2)
	M_m = [
		v(1),0;
		0,v(2)
	];
	for q = 1:2
		if (M_m(1,q)~=0 || M_m(2,q)~=0)
			M = [M,M_m(:,q)];
		end
	end
else
	M_m = [
		v(1),0,v(1),0,v(1),0;
		0,v(2),v(2),0,0,v(2);
		0,0,0,v(3),v(3),v(3) 
	];
	for q = 1:6
		if (M_m(1,q)~=0 || M_m(2,q)~=0 || M_m(3,q)~=0)
			M = [M,M_m(:,q)];
		end
	end
end

M = unique(M', 'rows')';

[rc, cc] = size(c);
[rM, cM] = size(M);
ids = [];
for K = 1:cM
	for k = 1:cc
		if (isequal(M(:,K), c(:,k)))
			ids = [ids,k-1];
		end
	end
end

ids = sort(ids,'descend');

endfunction 
