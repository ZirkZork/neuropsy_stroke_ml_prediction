
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%% 
### Package import

import os
import time
import pickle as pkl
import copy
import numpy as np
import functools as funky

from sklearn import svm, metrics
from sklearn import model_selection as modsel
import sklearn.model_selection as sklms
import imblearn.pipeline as pl
import imblearn.under_sampling as imbus

from tkinter.filedialog import askopenfilename

## settings
verbose = 1

setup_file_name = askopenfilename()
with open(setup_file_name, "rb") as f:
    training_setup = pkl.load(f)

training_parameters = training_setup["parameters"]

prediction_target_variable_type = training_setup["prediction_type"]

predictive_features = training_setup["features"]
prediction_targets = training_setup["targets"]    

remove_featureless_samples = True
if remove_featureless_samples:
    non_zero_samples = predictive_features.sum(axis=1) != 0
    predictive_features = predictive_features[non_zero_samples]
    prediction_targets = prediction_targets[non_zero_samples]

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

print (f"loaded {n_samples} samples, with {n_features} features.")

# center data so we can use a zero intercept
no_intercept = False
if no_intercept:
    predictive_features = predictive_features - predictive_features.mean(axis=0)


sample_prediction_tracker = np.zeros(n_samples)

## training cycles

# timer for training
print(f"Starting classifier training.")
start_train_time = time.time()

## cross validation setup

# split generator

split_generator_rng_seed = random_generator.integers(0, 2**32 - 1)

if(isCategoricalPrediction):
    split_generator = sklms.RepeatedStratifiedKFold(n_splits=n_cv_folds, n_repeats=n_cv_split_repeats, random_state=split_generator_rng_seed)
else:
    split_generator = sklms.RepeatedKFold(n_splits=n_cv_folds, n_repeats=n_cv_split_repeats, random_state=split_generator_rng_seed)   

n_splits = split_generator.get_n_splits()
print (f"runnign {n_splits} splits.")


# arrays for CV results
if prediction_target_variable_type == "cat_binary":
    selected_metrics = ["f1", "accuracy", "sensitivity", "specificity"]
if prediction_target_variable_type == "cat":
    selected_metrics = ["accuracy"]
if prediction_target_variable_type == "num":
    selected_metrics = ["r2", "mse", "rmse"]

model_cv_metrics = {}
for metric in selected_metrics:
    model_cv_metrics[metric] = np.zeros(n_splits)

predictions = np.empty((n_splits, n_samples), dtype=object)
predictions[:,:] = np.nan

cycle_start_time = time.time()

do_hypa_optimisation = True
c_param = 0.01
possible_c_params = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = {"C": possible_c_params}

if do_hypa_optimisation:
    print(f"running nested cv hyperparameter optimisation. possible c parameters: {", ".join(map(str, param_grid["C"]))}")
else:
    print(f"using a fixed c parameter: {c_param}")

pipeline_steps = []

if(isCategoricalPrediction):
    pipeline_steps.append(imbus.RandomUnderSampler(random_state=random_generator.integers(0, 2**32 - 1)))

    estimator = svm.LinearSVC(max_iter=2000, fit_intercept= not no_intercept)
    scorer = metrics.make_scorer(metrics.f1_score, pos_label=prediction_positive_category)
    #scorer = metrics.get_scorer("accuracy")
    
    estimator_name = "linearsvc"
else:
    estimator = svm.LinearSVR(verbose=max(0, verbose-1))
    #scorer = metrics.get_scorer("r2")
    scorer = metrics.get_scorer("neg_root_mean_squared_error")

    estimator_name = "linearsvr"

original_estimator_name = estimator_name

if (do_hypa_optimisation):
    estimator_step = modsel.GridSearchCV(copy.deepcopy(estimator), param_grid, verbose=max(0, (verbose-2)*3), scoring=scorer, n_jobs=-1)
    estimator_name = "gridsearchcv"
else:
    estimator.C = c_param
    estimator_step = copy.deepcopy(estimator)

training_model_template = lambda: pl.make_pipeline(*(pipeline_steps + [copy.deepcopy(estimator_step)]))

c_params = np.ones(n_splits) * c_param

## cross validation loop
for i, (training_samples_indices, test_samples_indices) in enumerate(split_generator.split(predictive_features, prediction_targets)):
    
    ## setup for training in the split
    split_start_time = time.time()

    # training set for the split
    inner_training_features = predictive_features[training_samples_indices]
    inner_training_targets = prediction_targets[training_samples_indices]
    
    training_model = training_model_template()
    training_model.fit(inner_training_features, inner_training_targets)    
    if do_hypa_optimisation:
        c_params[i] = training_model.named_steps[estimator_name].best_estimator_.C
    ## scoring

    # test set for the split
    inner_test_features = predictive_features[test_samples_indices]
    inner_test_labels = prediction_targets[test_samples_indices]

    if(isCategoricalPrediction and verbose):
        print(f"training - total samples: {len(inner_training_targets)}, positive samples: {sum(inner_training_targets == prediction_positive_category)}, negative samples: {sum(inner_training_targets == prediction_negative_category)}")
        print(f"testing - total samples: {len(inner_test_labels)}, positive samples: {sum(inner_test_labels == prediction_positive_category)}, negative samples: {sum(inner_test_labels == prediction_negative_category)}")


    # prediction with trained model
    estimator_step = training_model.named_steps[estimator_name]
    inner_model_prediction = estimator_step.predict(inner_test_features)

    predictions[i, test_samples_indices] = inner_model_prediction

    # prediction_scoring
    for metric in selected_metrics:
        if prediction_target_variable_type == "cat_binary":
            if metric == "f1":
                model_cv_metrics[metric][i] = metrics.f1_score(inner_test_labels, inner_model_prediction, pos_label=prediction_positive_category)
                continue
            elif metric == "sensitivity":
                model_cv_metrics[metric][i] = metrics.recall_score(inner_test_labels, inner_model_prediction, pos_label=prediction_positive_category)
                continue
            elif metric == "specificity":
                model_cv_metrics[metric][i] = metrics.recall_score(inner_test_labels, inner_model_prediction, pos_label=prediction_negative_category)
                continue
            elif metric == "accuracy":
                model_cv_metrics[metric][i] = metrics.accuracy_score(inner_test_labels, inner_model_prediction)
                continue
        if metric == "r2":
            model_cv_metrics[metric][i] = metrics.r2_score(inner_test_labels, inner_model_prediction)
            continue
        if metric == "mse":
            model_cv_metrics[metric][i] = metrics.mean_squared_error(inner_test_labels, inner_model_prediction)
            continue
        if metric == "rmse":
            model_cv_metrics[metric][i] = metrics.root_mean_squared_error(inner_test_labels, inner_model_prediction)
            continue

    if verbose:
        print(f"Fold {i+1}: Trained in {time.time() - split_start_time:.1f} seconds. {"; ".join([f"{metric}: {model_cv_metrics[metric][i]:.2f}" for metric in selected_metrics])}")

## retraining the classifier on the whole dataset

# pipeline for balancing (undersampling) and classifier training
cs, counts = np.unique(c_params, return_counts=True)
best_c = cs[np.argmax(counts)]
estimator.C = best_c
retraining_model = pl.make_pipeline(*(pipeline_steps + [copy.deepcopy(estimator)]))

retraining_model.fit(predictive_features, prediction_targets)    

# get intercept and coeficients of trained classifier from the pipeline
retrained_estimator = retraining_model.named_steps[original_estimator_name]
try:
    intercept_retrained = retrained_estimator.intercept_[0]
except:
    intercept_retrained = retrained_estimator.intercept_
coefficients_retrained = retrained_estimator.coef_.flatten()

# averaging metrics over the cross validation loops and saving for each iteration
mean_model_metrics = [model_cv_metrics[metric].mean() for metric in selected_metrics]

print(f"Training complete. {"; ".join([f"mean {metric}: {mean_model_metrics[i]:.3f}" for i,metric in enumerate(selected_metrics)])}")

## report total training time

training_duration = time.time() - start_train_time

training_duration_seconds = int(training_duration % 60)
training_duration_minutes = int(training_duration % 3600 // 60)
training_duration_hours = int(training_duration // 3600)

print(f"training has finished in {training_duration_hours:d} hours, {training_duration_minutes:d} minutes, and {training_duration_seconds:d} seconds.")

output_dict = {
    "analysis_tag": analysis_tag,
    "intercept": intercept_retrained,
    "coeficients": coefficients_retrained,
    "predictions": predictions,
    "best_c_params": c_params,
    "selected_c_param": best_c,
    "metrics": {}
}

for i,metric in enumerate(selected_metrics):
    output_dict["metrics"][metric] = mean_model_metrics[i]

## Save results to disk as serialized dict object
output_dir = os.path.join(os.path.dirname(setup_file_name), "output")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

output_file_name = os.path.join(output_dir, "unpermuted_run_output.pkl")

with open(output_file_name, "wb") as f:
    pkl.dump(output_dict, f)
