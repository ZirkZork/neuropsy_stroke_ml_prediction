 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility_functions import settings_handler as seth
from utility_functions import UI_windows as uiiai
from utility_functions import lesion_operations as lops
import numpy as np
import pandas as pd
import pickle as pkl

filepath_dict = {
    "setup_file": "file",
    "unpermuted_training_output": "file",
    "permutation_split_list": "file",
    "permutation_training_outputs": "files",
}

filepath_dict = uiiai.data_choose_file_paths(filepath_dict)

setup_file_name = filepath_dict["setup_file"]
with open(setup_file_name, "rb") as f:
    training_setup = pkl.load(f)

analysis_tag = training_setup["analysis_tag"]

unpermuted_output_file_name = filepath_dict["unpermuted_training_output"]
with open(unpermuted_output_file_name, "rb") as f:
    unpermuted_output = pkl.load(f)

assert unpermuted_output["analysis_tag"] == analysis_tag

unpermuted_intercept = unpermuted_output["intercept"]
unpermuted_coefficients = unpermuted_output["coeficients"]
unpermuted_metrics = unpermuted_output["metrics"]

selected_metrics = unpermuted_metrics.keys()
n_features = len(unpermuted_coefficients)

permutation_split_list_file_name = filepath_dict["permutation_split_list"]
permutation_split_list = pd.read_csv(permutation_split_list_file_name)
permutation_split_ids = permutation_split_list["split_id"]

n_permutations = training_setup["n_permutations"]

permutation_intercepts = np.zeros(n_permutations)
permutation_coefficients = np.zeros((n_permutations, n_features))
permutation_metrics = {}
for metric in selected_metrics:
    permutation_metrics[metric] = np.zeros(n_permutations)

permuted_output_file_names = filepath_dict["permutation_training_outputs"].split("; ")
for file_name in permuted_output_file_names:
    with open(file_name, "rb") as f:
        permutation_split_output = pkl.load(f)
        assert permutation_split_output["analysis_tag"] == analysis_tag

        split_id = permutation_split_output["run_id"]
        split_row = permutation_split_list.query("split_id == @split_id").iloc[0]

        start_index = split_row["start_index"]
        end_index = split_row["end_index"]

        permutation_intercepts[start_index:end_index] = permutation_split_output["permutation_intercepts"]        
        permutation_coefficients[start_index:end_index] = permutation_split_output["permutation_coefficients"]
        for metric in selected_metrics:
            permutation_metrics[metric][start_index:end_index] = permutation_split_output["permutation_metrics"][metric]

comprehensive_output = {
    "analysis_tag": analysis_tag,

    "unpermuted_intercept": unpermuted_intercept,
    "permutation_intercepts": permutation_intercepts,

    "unpermuted_coefficients": unpermuted_coefficients,
    "permutation_coefficients": permutation_coefficients,

    "unpermuted_metrics": unpermuted_metrics,
    "permutation_metrics": permutation_metrics,

    "brain_affine": training_setup["brain_affine"],
    "brain_dimensions": training_setup["brain_dimensions"],
    "feature_coordinates": training_setup["feature_coordinates"],
}

#output_folder = filepath_dict["output_folder"]
output_folder = os.path.dirname(unpermuted_output_file_name)

output_file_name = os.path.join(output_folder, f"{analysis_tag}_comprehensive_output.pkl")
with open(output_file_name, "wb") as f:
    pkl.dump(comprehensive_output, f)