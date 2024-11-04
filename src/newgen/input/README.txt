NOTES

INFOR: For-loop within code generator.
args: <string, s1> <int, N> <int, r_{m.0}> <int, r_{M,0}> <int, r_{i,0}> <int, r_{m,1}> <int, r_{M,1}> <int, r_{i,1}> ... <int, r_{m,N-1}> <int, r_{M,N-1}> <int, r_{i,N-1}>
- where:
    + s1 is a string whose substrings form the loop indices.
    + N is the length of each substring.
    + r_{m,k}, r_{M,k}, r_{i,k} are the respective loop lower bound (inclusive), upper bound (exclusive), and increment for 0 <= k < N.

INIF: If-condition within code generator.
args: <string, cond>
- where:
    + cond is the condition to be parsedto determine whether or not to include the subsequent block of code up till END_INIF.

OUTFOR: For-loop in the code.
args: <string, s1> <int, N> <int, r_{m.0}> <int, r_{M,0}> <int, r_{i,0}> <int, r_{m,1}> <int, r_{M,1}> <int, r_{i,1}> ... <int, r_{m,N-1}> <int, r_{M,N-1}> <int, r_{i,N-1}>
- where:
    + s1 is a string whose substrings form the loop indices.
    + N is the length of each substring.
    + r_{m,k}, r_{M,k}, r_{i,k} are the respective loop lower bound (inclusive), upper bound (exclusive), and increment for 0 <= k < N.



Arithmetic: ^< <string, expression> >^
- where expression is the string to be processed (if this fails, the full string will remain untouched in the output).

In-line sums/products: SUM/PROD<<char, c> <int, r_m> <int, r_M> <int, r_i> <string, expression>>END_SUM/PROD
- where:
    + c is the loop index character.
    + r_m is the loop lower bound (inclusive).
    + r_M is the loop upper bound (exclusive).
    + r_i is the loop increment.
    + expression is the string which is being summed, and in which index substrings "<c>" are replaced with their respective values.
      e.g. SUM<i 0 3 1 f_<i>>END_SUM becomes f_0+f_1+f_2.
      note: make sure to enclose nested loop indices with ^<>^ if an arithmetic operation is required.

