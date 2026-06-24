
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%% 
### Package import

import os
import time
import pickle as pkl
import sys

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility_functions import settings_handler as seth
from utility_functions import UI_windows as uiiai
from utility_functions import lesion_operations as lops

import pandas as pd

from tkinter.filedialog import askopenfilename


filepath_dict = {
    "setup_file": "file",
    "data_file": "file",
}

filepath_dict = uiiai.data_choose_file_paths(filepath_dict)

setup_file_name = filepath_dict["setup_file"]
with open(setup_file_name, "rb") as f:
    training_setup = pkl.load(f)

run_dir = os.path.abspath(os.path.join(os.path.dirname(setup_file_name), os.pardir))

analysis_tag = training_setup["analysis_tag"]
predictive_features = training_setup["features"]
prediction_targets = training_setup["targets"]

n_features = predictive_features.shape[1]

non_zero_features = ~ np.isclose(predictive_features, 0)

data_file_path = filepath_dict["data_file"]
loaded_data_frame = pd.read_excel(data_file_path)
included_rows = loaded_data_frame["excluded"] == False

feature_rows = loaded_data_frame["lesion_mostly_in_mask"] == True



loaded_data_frame[["lesion_voxels_in_mask", "lesion_in_mask_proportion", "mask_covered_proportion", "lesion_mostly_in_mask"]] = pd.NA

lesion_voxels_in_mask = non_zero_features.sum(axis=1)

loaded_data_frame.loc[feature_rows, "lesion_voxels_in_mask"] = lesion_voxels_in_mask
loaded_data_frame.loc[feature_rows, "lesion_in_mask_proportion"] = lesion_voxels_in_mask / loaded_data_frame.loc[feature_rows, "lesion_size"]
loaded_data_frame.loc[feature_rows, "mask_covered_proportion"] = lesion_voxels_in_mask / n_features

lesion_in_mask = loaded_data_frame.loc[feature_rows, "lesion_in_mask_proportion"] > 0

loaded_data_frame.loc[included_rows, "lesion_mostly_in_mask"] = lesion_in_mask

new_predictive_features = predictive_features[lesion_in_mask]   
new_prediction_targets = prediction_targets[lesion_in_mask]

new_training_setup = training_setup.copy()

new_training_setup["features"] = new_predictive_features
new_training_setup["targets"] = new_prediction_targets

loaded_data_frame.to_excel(os.path.join(run_dir, f"{analysis_tag}_processed_data_set.xlsx"))

with open(setup_file_name, "wb") as f:
    pkl.dump(new_training_setup, f)