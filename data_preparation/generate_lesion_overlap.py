 
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
    "data_file": "file",
    "lesion_map_folder": "dir",
}
filepath_dict = uiiai.data_choose_file_paths(filepath_dict)

data_filepath = filepath_dict["data_file"]
data_frame = pd.read_excel(data_filepath)

subject_id_variable = "sub_id"
included_samples = data_frame.query("excluded == False")[subject_id_variable]

slice_spacing = 5
slice_offset = -2

lesion_map_folder = filepath_dict["lesion_map_folder"]
lesion_maps, map_dimensions, map_affine, found_map_ids = lops.load_lesion_maps(list(included_samples), lesion_map_folder, slice_spacing, slice_offset)

lesion_overlap = lesion_maps.sum(axis=0) / len(found_map_ids)

output_folder = os.path.dirname(data_filepath)
output_filepath = os.path.join(output_folder, "lesion_overlap.nii")

lops.save_as_nifti(lesion_overlap ,map_affine, output_filepath)