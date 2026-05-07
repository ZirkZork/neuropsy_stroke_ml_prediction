
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%% 
### Package import

import os
import time
import pickle as pkl

import numpy as np
import pandas as pd

from sklearn import svm, metrics
import sklearn.model_selection as sklms
import imblearn.pipeline as pl
import imblearn.under_sampling as imbus

from tkinter.filedialog import askopenfilename

## settings
verbose = True

setup_file_name = askopenfilename()
with open(setup_file_name, "rb") as f:
    training_setup = pkl.load(f)

training_parameters = training_setup["parameters"]

prediction_target_variable_type = training_setup["prediction_type"]

predictive_features = training_setup["features"]
prediction_targets = training_setup["targets"]    

if prediction_target_variable_type == "cat_binary":
    prediction_positive_category = training_setup["prediction_positive_category"]
    prediction_negative_category = [var for var in np.unique(prediction_targets) if var != prediction_positive_category].pop()

isCategoricalPrediction = prediction_target_variable_type in ["cat", "cat_binary"]

n_cv_folds = training_parameters["n_cv_folds"]
n_cv_split_repeats = training_parameters["n_cv_split_repetitions"]


random_number_seed = training_setup["rng_seed"]
random_generator = np.random.default_rng(random_number_seed)


analysis_tag = training_setup["analysis_tag"]

n_samples, n_features = predictive_features.shape

sample_prediction_tracker = np.zeros(n_samples)

## training cycles

# timer for training
print(f"Starting classifier training.")
start_train_time = time.time()

## cross validation setup

# split generator

split_generator_rng_seed = random_generator.integers(0, 2**32 - 1)

rskf = sklms.RepeatedStratifiedKFold(n_splits=n_cv_folds, n_repeats=n_cv_split_repeats, random_state=split_generator_rng_seed)
n_splits= rskf.get_n_splits()

# arrays for CV results
if prediction_target_variable_type == "cat_binary":
    selected_metrics = ["f1", "accuracy", "sensitivity", "specificity"]
if prediction_target_variable_type == "cat":
    selected_metrics = []
if prediction_target_variable_type == "num":
    selected_metrics = []

model_cv_metrics = {}
for metric in selected_metrics:
    model_cv_metrics[metric] = np.zeros(n_splits)


cycle_start_time = time.time()

if(isCategoricalPrediction):
    training_model_template = lambda c_param: pl.make_pipeline(imbus.RandomUnderSampler(random_state=random_generator.integers(0, 2**32 - 1)), svm.LinearSVC(C=c_param, verbose=verbose-1))
    estimator_name = "linearsvc"
else:
    training_model_template = lambda c_param: pl.make_pipeline(svm.LinearSVR(C=c_param, verbose=verbose-1))
    estimator_name = "linearsvr"



## cross validation loop
for j, (training_samples_indices, test_samples_indices) in enumerate(rskf.split(predictive_features, prediction_targets)):
    
    ## setup for training in the split
    split_start_time = time.time()

    c_param = 0.01

    # training set for the split
    inner_training_features = predictive_features[training_samples_indices]
    inner_training_targets = prediction_targets[training_samples_indices]
    
    training_model = training_model_template(c_param)
    training_model.fit(inner_training_features, inner_training_targets)    

    ## scoring

    # test set for the split
    inner_test_features = predictive_features[test_samples_indices]
    inner_test_labels = prediction_targets[test_samples_indices]

    # prediction with trained model
    estimator = training_model.named_steps[estimator_name]
    inner_model_prediction = estimator.predict(inner_test_features)

    # prediction_scoring
    for metric in selected_metrics:
        if prediction_target_variable_type == "cat_binary":
            if metric == "f1":
                model_cv_metrics[metric][j] = metrics.f1_score(inner_test_labels, inner_model_prediction, pos_label=prediction_positive_category)
                continue
            elif metric == "sensitivity":
                model_cv_metrics[metric][j] = metrics.recall_score(inner_test_labels, inner_model_prediction, pos_label=prediction_positive_category)
                continue
            elif metric == "specificity":
                model_cv_metrics[metric][j] = metrics.recall_score(inner_test_labels, inner_model_prediction, pos_label=prediction_negative_category)
                continue
            elif metric == "accuracy":
                model_cv_metrics[metric][j] = metrics.accuracy_score(inner_test_labels, inner_model_prediction)
                continue

    

    if verbose:
        print(f"Fold {j+1}: Trained in {time.time() - split_start_time:.1f} seconds. {"; ".join([f"{metric}: {model_cv_metrics[metric][j]:.2%}" for metric in selected_metrics])}")


## retraining the classifier on the whole dataset

# pipeline for balancing (undersampling) and classifier training
retraining_model = training_model_template(c_param)
retraining_model.fit(predictive_features, prediction_targets)    

# get intercept and coeficients of trained classifier from the pipeline
retrained_estimator = retraining_model.named_steps[estimator_name]
intercept_retrained = retrained_estimator.intercept_[0]
coefficients_retrained = retrained_estimator.coef_.flatten()

# averaging metrics over the cross validation loops and saving for each iteration
mean_model_metrics = [model_cv_metrics[metric].mean() for metric in selected_metrics]

print(f"Training complete. {"; ".join([f"mean {metric}: {mean_model_metrics[i]:.2%}" for i,metric in enumerate(selected_metrics)])}")

## report total training time

training_duration = time.time() - start_train_time

training_duration_seconds = int(training_duration % 60)
training_duration_minutes = int(training_duration % 3600 // 60)
training_duration_hours = int(training_duration // 3600)

print(f"training has finished in {training_duration_hours:d} hours, {training_duration_minutes:d} minutes, and {training_duration_seconds:d} seconds.")

output_dict = {
    "intercept": intercept_retrained,
    "coeficients": coefficients_retrained
}

for i,metric in enumerate(selected_metrics):
    output_dict[metric] = mean_model_metrics[i]

## Save results to disk as serialized dict object
output_fike_name = os.path.join(os.path.dirname(setup_file_name), "unpermuted_run_output.pkl")

with open(output_fike_name, "wb") as f:
    pkl.dump(output_dict, f)
