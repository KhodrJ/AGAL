#!/bin/bash

N=110  # Change this to your desired upper limit

for ((k=0; k<=N; k++)); do
    echo ${k};
    #./a_rmc.out > "./data/out_rmc_${k}.txt"
    ./a_n.out > "./data/out_n_${k}.txt"
    ./a_r.out > "./data/out_r_${k}.txt"
    ./a_c.out > "./data/out_c_${k}.txt"
    ./a_rc.out > "./data/out_rc_${k}.txt"
done
