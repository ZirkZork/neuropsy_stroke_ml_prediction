
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility_functions import lesion_operations as lops
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pickle as pkl
import pandas as pd
import nibabel as nib

output_file_name = askopenfilename()

with open(output_file_name, "rb") as f:
    output = pkl.load(f)

affine = output["brain_affine"]
dimensions = output["brain_dimensions"]

feature_coordinates = output["feature_coordinates"]
n_features = feature_coordinates.shape[0]

reconstructed_mask = lops.transform_feature_space_to_brain(np.ones(n_features), feature_coordinates, dimensions)
lops.save_as_nifti (reconstructed_mask.astype(float), affine, asksaveasfilename())
    
