#!/bin/bash
# v3.2

# arg 1 -> training setup
# arg 2 -> split list file

logs_dir=logs
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

data_dir=/data/project/inm3_svm/data/

# print the .submit header
printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 4
request_memory = 24G

# Execution
initial_dir    = /data/project/inm3_svm/server_scripts/cluster_python_scripts
executable     = python_script_wrapper.sh
\n"

# loop through each run

while IFS="," read -r split_id split_file remaining_col; do
    printf "arguments = cluster_svm_permutation_training.py ${data_dir}$1 $(dirname $(realpath $2))/${split_file}\n"
    printf "log       = ../${logs_dir}/\$(Cluster)_\$(Process)_svm_permutations_${split_id}.log\n"
    printf "output    = ../${logs_dir}/\$(Cluster)_\$(Process)_svm_permutations_${split_id}.out\n"
    printf "error     = ../${logs_dir}/\$(Cluster)_\$(Process)_svm_permutations_${split_id}.err\n"
    printf "Queue\n\n"
done < <(tail -n +2 $2)
