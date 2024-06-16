function [s1, s2, s3] = get_BC_string(n_d, Hk, P, c, w)
% Build the string expression for boundary conditions loaded from BC_2D.txt and BC_3D.txt in standardized format. For now, these are bounce-back conditions for specification of velocity [no-slip, inlet], and anti-bounce-back conditions for specification of pressure [outlet]. Periodic boundary conditions are applied naturally without special conditions (neighbors wrap around).
% Input:
% - n_d:        Number of dimensions (integer).
% - indices:    The indices in shared memory to use for interpolation.
% - var_s1:     The variable being interpolated (string).
% - var_s2:     The destination variable (string).
% - l_dqs:      Number of elements in velocity set (integer).
% - c:          Discrete particle velocity vectors (array).
% - pb:         Reverse-direction indices (array).
% - w:          Weights corresponding to particle velocity vectors (array).
% Output:
% - s1:         Condition for the boundary Id (string).
% - s2:         Expression for cdotu based on BC values (string).
% - s3:         Expression for the DDF (string).

% Get the boundary ID for the condition and boundary type for constructing the BC. Get values for the associated BC as well.
b_id = Hk(1);
b_type = Hk(2);
u = Hk(3:5);

% Get the correct expression for cdotu based on BC values.
cdotu = 0.0;
if (n_d == 2)
	cdotu = dot(c(:,P), u(1:2));
else
	cdotu = dot(c(:,P), u(1:3));
end

% Build the string.
s1 = "";
s2 = "";
s3 = "";
%
% - Case 1: Non-zero specified wall velocity.
%
if ( b_type == 0 && (sum(abs(u))>0) )
	s1 = sprintf("nbr_kap_b == %i", b_id);
	s2 = sprintf("cdotu = N_Pf(%17.15f)", cdotu);
	s3 = sprintf("f_%i = f_%i - N_Pf(2.0)*N_Pf(%17.15f)*N_Pf(3.0)*cdotu", P-1, P-1, w(P));
%
% - Case 2: Outflow condition.
%
elseif (b_type == 1)
	s1 = sprintf("nbr_kap_b == %i", b_id);
	s2 = sprintf("cdotu = %s", get_cdotu(c(:,P), n_d));
	s3 = sprintf("f_%i = -f_%i + N_Pf(2.0)*N_Pf(%17.15f)*(N_Pf(1.0) + cdotu*cdotu*N_Pf(4.5) - udotu*N_Pf(1.5))", P-1, P-1, w(P));
end

endfunction 
