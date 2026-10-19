#!/bin/bash
# v3.2

# arg 1 -> runlist file

# loop through each run
while IFS="," read -r setup_file run_split_list; do
    bash run_permutation_jobs.sh "${setup_file}" "../data/${run_split_list}"|condor_submit
done < <(tail -n +2 $1)