
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import pandas as pd

from tkinter.filedialog import askopenfilename, askdirectory

original_data_output_filepath = askopenfilename(initialdir = "../../Data Repositories/Classifier Output")
permutation_split_list_filepath = askopenfilename(initialdir = "cluster_mirror/input")
permutation_split_output_folderpath = askdirectory(initialdir = "cluster_mirror/output")

permutation_split_list = pd.read_csv(permutation_split_list_filepath)

with open(original_data_output_filepath, "rb") as f:
    svm_original_data_output_dict = pickle.load(f)

run_id = svm_original_data_output_dict["run_id"]
settings = svm_original_data_output_dict["settings"]
n_permutations = settings.training.n_permutations
n_features = svm_original_data_output_dict["svm_real_data_coefficients_retrained"].size

svm_permutation_intercept_retrained = np.zeros(n_permutations)
svm_permutation_coefficients_retrained = np.zeros((n_permutations, n_features))
svm_permutation_intercept = np.zeros(n_permutations)
svm_permutation_coefficients = np.zeros((n_permutations, n_features))
svm_permutation_accuracy = np.zeros(n_permutations)
svm_permutation_sensitivity = np.zeros(n_permutations)
svm_permutation_specificity = np.zeros(n_permutations)

total_train_duration = 0
for i, row in permutation_split_list.iterrows():
    try:
        with open(os.path.join(permutation_split_output_folderpath, f"{row["split_id"]}_svm_permutations.pickle"), "rb") as f:
            svm_permutation_output_dict = pickle.load(f)
            permutation_split_list.loc[i, "result_found"] = True
    except:
        permutation_split_list.loc[i, "result_found"] = False
        continue
    assert (run_id in svm_permutation_output_dict ["run_id"])
    start_index = row.start_index
    end_index = row.end_index

    svm_permutation_intercept_retrained[start_index:end_index] = svm_permutation_output_dict ["svm_permutation_intercept_retrained"]
    svm_permutation_coefficients_retrained[start_index:end_index] = svm_permutation_output_dict ["svm_permutation_coefficients_retrained"]
    svm_permutation_intercept[start_index:end_index] = svm_permutation_output_dict ["svm_permutation_intercept"]
    svm_permutation_coefficients[start_index:end_index] = svm_permutation_output_dict["svm_permutation_coefficients"]
    svm_permutation_accuracy[start_index:end_index] = svm_permutation_output_dict["svm_permutation_accuracy"]
    svm_permutation_sensitivity[start_index:end_index] = svm_permutation_output_dict["svm_permutation_sensitivity"]
    svm_permutation_specificity[start_index:end_index] = svm_permutation_output_dict["svm_permutation_specificity"]

    total_train_duration += svm_permutation_output_dict["train_duration"]


assert np.all(permutation_split_list.result_found)

svm_original_data_output_dict["svm_permutation_intercept_retrained"] = svm_permutation_intercept_retrained
svm_original_data_output_dict["svm_permutation_coefficients_retrained"] = svm_permutation_coefficients_retrained
svm_original_data_output_dict["svm_permutation_intercept"] = svm_permutation_intercept
svm_original_data_output_dict["svm_permutation_coefficients"] = svm_permutation_coefficients
svm_original_data_output_dict["svm_permutation_accuracy"] = svm_permutation_accuracy
svm_original_data_output_dict["svm_permutation_sensitivity"] = svm_permutation_sensitivity
svm_original_data_output_dict["svm_permutation_specificity"] = svm_permutation_specificity
    
svm_original_data_output_dict["train_duration"] = total_train_duration

output_folder = os.path.expanduser(settings.misc.output_folder)
with open(os.path.join(output_folder, f"svm_out_{run_id}.pickle"), "wb") as f:
    pickle.dump(svm_original_data_output_dict, f)




