 
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
from tkinter.filedialog import askopenfilename

data_file_path = askopenfilename()
data_frame = pd.read_excel(data_file_path)

included_data_frame = data_frame.query("lesion_mostly_in_mask == True")

sample_count = len(included_data_frame)

selected_demo_variables = uiiai.data_select_demo_variables(list(included_data_frame.columns))
print(f"{sample_count} samples found.")
for var in selected_demo_variables:
    column = included_data_frame[var]
    try:
        mean = column.mean()
        std = column.std()
        median = column.median()
        iqr = column.quantile(0.75) - column.quantile(0.25)
        print(f"{var} - mean: {mean}; std: {std}; median: {median}; iqr: {iqr}")
    except:
        print(f"no statistics for \"{var}\".")