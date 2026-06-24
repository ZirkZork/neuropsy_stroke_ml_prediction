
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%% 
### Package import

import os
import time
import pickle as pkl
import copy
import math
import numpy as np
import sys
import functools as funky
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility_functions import settings_handler as seth
from utility_functions import UI_windows as uiiai
from utility_functions import lesion_operations as lops

from sklearn import svm, metrics
from sklearn import model_selection as modsel
import sklearn.model_selection as sklms
import imblearn.pipeline as pl
import imblearn.under_sampling as imbus

from tkinter.filedialog import askopenfilename

filepath_dict = {
    "setup": "file",
    "output": "files"
}

filepath_dict = uiiai.data_choose_file_paths(filepath_dict)

setup_file_name = filepath_dict["setup"]
with open(setup_file_name, "rb") as f:
    setup = pkl.load(f)

output_file_name = filepath_dict["output"]
with open(output_file_name, "rb") as f:
    output = pkl.load(f)

prediction_targets = setup["targets"]
predictions = output["predictions"]

generated_predictions = ~ np.isnan(predictions.astype(float))

decision_boundary = 18
categories = np.asarray(["negative", "positive"])

binary_prediction_targets = categories[(prediction_targets >= decision_boundary).astype(int)]
binary_predictions = predictions

binary_predictions[generated_predictions] = categories[(predictions[generated_predictions] >= decision_boundary).astype(int)]

n_splits = predictions.shape[0]
model_cv_metrics={
    "f1": np.zeros(n_splits),
    "accuracy": np.zeros(n_splits),
    "sensitivity": np.zeros(n_splits),
    "specificity": np.zeros(n_splits)
}

for i, (split_predictions, split_generated_predictions) in enumerate(zip(binary_predictions, generated_predictions)):
    predicted_categories = split_predictions[split_generated_predictions]
    target_categories = binary_prediction_targets[split_generated_predictions]

    model_cv_metrics["f1"][i] = metrics.f1_score(target_categories, predicted_categories, pos_label="positive")
    model_cv_metrics["accuracy"][i] = metrics.accuracy_score(target_categories, predicted_categories)
    model_cv_metrics["sensitivity"][i] = metrics.recall_score(target_categories, predicted_categories, pos_label="positive")
    model_cv_metrics["specificity"][i] = metrics.recall_score(target_categories, predicted_categories, pos_label="negative")

mean_model_metrics = [model_cv_metrics[metric].mean() for metric in model_cv_metrics.keys()]

print(f"Training complete. {"; ".join([f"mean {metric}: {mean_model_metrics[i]:.3f}" for i,metric in enumerate(model_cv_metrics.keys())])}")

