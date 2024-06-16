function [l_dqs_indexed, p_indexed, nbr_quad_indexed] =  esoteric_data()
% Constructs the velocity set (particle velocity, weight, reversed-direction index) data structures.
% Output:
% - l_dqs_indexed:      Number of DDF pairs to be processed (array).
% - p_indexed:          Indices of the first DDF among the pairs (cell array).
% - nbr_quad_indexed:   Indices of the block neighbors involved in the DDF pair exchange (cell array).

l_dqs_indexed = [4, 9, 13];
p_indexed = {
		[1, 2, 5, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1],
		[1, 3, 5, 7, 9, 11, 13, 15, 17, -1, -1, -1, -1],
		[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 26]
};
nbr_quad_indexed = {
		[0, 1, 2, 5, -1, -1, -1, -1],
		[0, 1, 3, 5, 7, 9, 11, -1],
		[0, 1, 3, 5, 7, 9, 11, 19]
};

endfunction
