#!/bin/bash

N=110  # Change this to your desired upper limit

for ((k=0; k<=N; k++)); do
    echo ${k};
    #./a_vv1_soa.out > "./data/out_vv1_soa_${k}.txt"
    #./a_vv2.out > "./data/out_vv2_${k}.txt"
    #./a_vv1w_soa.out > "./data/out_vv1w_soa_${k}.txt"
    #./a_vv2w.out > "./data/out_vv2w_${k}.txt"
    #./a_vv1_aos.out > "./data/out_vv1_aos_${k}.txt"
    #./a_vv1w_aos.out > "./data/out_vv1w_aos_${k}.txt"
    #./a_cm_v0.out > "./data/out_cm_v0_${k}.txt"
    #./a_cm_v1.out > "./data/out_cm_v1_${k}.txt"
    #./a_shfl_n.out > "./data/out_shfl_n_${k}.txt"
    #./a_shfl_y.out > "./data/out_shfl_y_${k}.txt"
    #./a_bsz.out > "./data/out_bsz08_${k}.txt"
    ./a.out > "./data/out_dragon_time_${k}.txt"
done
