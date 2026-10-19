
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle as pkl

import numpy as np

from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.simpledialog import askinteger
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility_functions import UI_windows as uiiai

setup_file_name = askopenfilename()
with open(setup_file_name, "rb") as f:
    permutation_setup = pkl.load(f)

try:
    unpermuted_results_file_name = askopenfilename()
    with open(unpermuted_results_file_name, "rb") as f:
        unpermuted_results = pkl.load(f)
    best_c_param = unpermuted_results["selected_c_param"]
except:
    best_c_param = uiiai.generic_get_number("c parameter", "Choose a c parameter for the permutations.")
    
analysis_tag = permutation_setup["analysis_tag"]

n_permutations = permutation_setup["n_permutations"]
n_samples = len(permutation_setup["targets"])

n_splits = askinteger("How many fractions?", "Into how many fractions should the permutations be split into?")

random_number_seed = permutation_setup["rng_seed"]
random_generator = np.random.default_rng(random_number_seed)
permutations_indices = random_generator.permuted (np.arange(n_samples).reshape((1,-1)).repeat(n_permutations, axis=0), axis=1)

fraction_lengths = - (permutations_indices.shape[0] // - n_splits)
print(f"Splitting the permutation list into {n_splits} fractions with up to {fraction_lengths} permutations.")

output_folder = os.path.join(os.path.dirname(setup_file_name), "data_permutation_splits")
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

with open(os.path.join(output_folder, f"split_list_{analysis_tag}.csv"), "w") as f:
    f.write(",".join(["split_id", "file_name", "length", "start_index", "end_index"]))  
    f.write("\n")
    
    for i in range(n_splits):
        start_index = i*fraction_lengths
        end_index = min((i+1)*fraction_lengths, permutations_indices.shape[0])
        permutation_fraction = permutations_indices[start_index:end_index]
        split_id = f"{analysis_tag}_perm_split_{i+1}_of_{n_splits}"
        split_file_name = f"{split_id}_setup.pkl"

        print(split_id, split_file_name, permutation_fraction.shape, start_index, end_index)
        f.write(",".join([split_id, split_file_name, str(permutation_fraction.shape[0]), str(start_index), str(end_index)]))
        f.write("\n")
        split_svm_permutation_setup_dict = {
            "run_id": split_id,
            "permutations_indices": permutation_fraction,
            "c_param": best_c_param
        }

        with open(os.path.join(output_folder, split_file_name), "wb") as f_:
            pkl.dump(split_svm_permutation_setup_dict, f_)