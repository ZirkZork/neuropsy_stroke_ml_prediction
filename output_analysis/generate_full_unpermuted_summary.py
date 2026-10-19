
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
from tkinter.filedialog import askdirectory
import pickle as pkl
import pandas as pd
import numpy as np

voi_file_re = re.compile(r".*unpermuted_run_output.*\.pkl")

data_folder = askdirectory()

found_files = []
for path, subdirs, files in os.walk(data_folder):
    for file_name in files:
        file_re_match = voi_file_re.fullmatch(file_name)
        if file_re_match is None:
            continue

        found_files.append(os.path.join(path, file_name))

output_dict = {}

for file_name in found_files:
    with open(file_name, "rb") as f:
        loaded_output = pkl.load(f)
        output_dict[loaded_output["analysis_tag"]] = loaded_output

metric_dict = {}
for id, analysis_output in output_dict.items():
    metric_dict[id] = analysis_output["metrics"]
    n_voxels = len(analysis_output["coeficients"])
    metric_dict[id]["n_voxels"] = n_voxels
    n_samples = analysis_output["predictions"].shape[1]
    metric_dict[id]["n_samples"] = n_samples
    try:
        values, counts = np.unique(analysis_output["best_c_params"], return_counts=True)
        metric_dict[id]["best_c"] = values[np.argmax(counts)]
    except:
        pass
metrics_frame = pd.DataFrame(metric_dict).T

for row in metrics_frame.index:
    print(row + ": " + "; ".join([f"{metric}: {metrics_frame.loc[row, metric]:.3}" for metric in metrics_frame]))

metrics_frame.to_excel(os.path.join(data_folder, "model_summary.xlsx"))