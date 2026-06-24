
from sklearn import decomposition as deco
from tkinter.filedialog import askopenfilename
import numpy as np
import pickle as pkl
import copy
import sys
import os

setup_file_name = sys.argv[1]
with open(setup_file_name, "rb") as f:
    training_setup = pkl.load(f)

predictive_features = training_setup["features"]
#predictive_features = (~ np.isclose(predictive_features, 0)).astype(float)
prediction_targets = training_setup["targets"]    

selections = deco.PCA()
print("fitting")

new_features = selections.fit_transform(predictive_features, prediction_targets)



new_training_setup = copy.deepcopy(training_setup)

new_training_setup["analysis_tag"] = "pca_" + training_setup["analysis_tag"]
new_training_setup["features"] = new_features
new_training_setup["feature_coordinates"] = None

new_setup_directory = os.path.dirname(setup_file_name) + "_pca"
if not os.path.isdir(new_setup_directory):
    os.makedirs(new_setup_directory)

new_setup_file_name = os.path.basename(setup_file_name).replace(".pkl", "_pca.pkl")

with open(os.path.join(new_setup_directory, new_setup_file_name), "wb") as f:
    pkl.dump(new_training_setup, f)
