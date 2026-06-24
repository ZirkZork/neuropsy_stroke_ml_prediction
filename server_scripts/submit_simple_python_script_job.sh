#!/bin/bash
# v3.2

logs_dir=logs
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

# print the .submit header
printf "
# The environment
universe       = vanilla
getenv         = True
request_cpus   = 1
request_memory = 10G

# Execution
initial_dir    = /data/project/inm3_svm/server_scripts/cluster_python_scripts
executable     = python_script_wrapper.sh

arguments = $*
log       = ../${logs_dir}/\$(Cluster)_\$(Process)_$(basename $1 .py).log
output    = ../${logs_dir}/\$(Cluster)_\$(Process)_$(basename $1 .py).out
error     = ../${logs_dir}/\$(Cluster)_\$(Process)_$(basename $1 .py).err
Queue
"
