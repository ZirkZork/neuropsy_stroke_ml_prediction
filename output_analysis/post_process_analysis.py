
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle as pkl
import sys
import os
from tkinter.filedialog import askopenfilename, askdirectory
import numpy as np
from sklearn import cluster as clst
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility_functions import lesion_operations as lops


file_name = askopenfilename()
with open(file_name, "wb") as f:
    comprehensive_output = pkl.load(f)

analysis_tag = comprehensive_output["analysis_tag"]

unpermuted_intercept = comprehensive_output["unpermuted_intercept"]
permutation_intercepts = comprehensive_output["permutation_intercepts"]

unpermuted_coefficients = comprehensive_output["unpermuted_coefficients"]
permutation_coefficients = comprehensive_output["permutation_coefficients"]

unpermuted_metrics = comprehensive_output["unpermuted_metrics"]
permutation_metrics = comprehensive_output["permutation_metrics"]

brain_affine = comprehensive_output["brain_affine"]
brain_dimensions = comprehensive_output["brain_dimensions"]
feature_coordinates = comprehensive_output["feature_coordinates"]


n_permutations = permutation_intercepts.size
n_features = unpermuted_coefficients.size

intercept_p_value = np.sum(np.abs(unpermuted_intercept) <= np.abs(permutation_intercepts), axis=0) / n_permutations

single_threshold_correction = True

if single_threshold_correction:
    coefficient_p_values = np.sum(np.abs(unpermuted_coefficients) <= np.max(np.abs(permutation_coefficients), axis=1), axis=0) / n_permutations
else:
    coefficient_p_values = np.sum(np.abs(unpermuted_coefficients) <= np.abs(permutation_coefficients), axis=0) / n_permutations

coefficent_sign_mask = np.sign(unpermuted_coefficients)

##
alpha_threshold = 0.001
##

significant_coefficents = coefficient_p_values < alpha_threshold

##
cluster_max_gap = 1
##

positive_super_threshold_voxels = significant_coefficents & coefficent_sign_mask == 1
negative_super_threshold_voxels = significant_coefficents & coefficent_sign_mask == -1

cluster_pos_core_samples, cluster_pos_labels = clst.dbscan(feature_coordinates[positive_super_threshold_voxels], eps=cluster_max_gap)
cluster_neg_core_samples, cluster_neg_labels = clst.dbscan(feature_coordinates[negative_super_threshold_voxels], eps=cluster_max_gap)

cluster_neg_labels[cluster_neg_labels != -1] += max(cluster_pos_labels) + 1

cluster_labels = np.zeros(n_features) - 1
cluster_labels[positive_super_threshold_voxels] = cluster_pos_labels
cluster_labels[negative_super_threshold_voxels] = cluster_neg_labels

n_clusters = max(cluster_labels) + 1

nifti_folder = askdirectory()

brain_cluster_labels = lops.transform_feature_space_to_brain(cluster_labels + 1, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_cluster_labels, brain_affine, os.path.join(nifti_folder, "cluster_labels.nii.gz"))

brain_coefficients = lops.transform_feature_space_to_brain(unpermuted_coefficients, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_coefficients, brain_affine, os.path.join(nifti_folder, "voxel_coefficients.nii.gz"))

brain_pvalues = lops.transform_feature_space_to_brain(coefficient_p_values, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_pvalues, brain_affine, os.path.join(nifti_folder, "voxel_pvalues.nii.gz"))

all_cluster_coefficients = np.zeros(n_features)
all_cluster_coefficients[cluster_labels >= 0] = unpermuted_coefficients[cluster_labels >= 0]
brain_all_cluster_coefficients = lops.transform_feature_space_to_brain(all_cluster_coefficients, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_all_cluster_coefficients, brain_affine, os.path.join(nifti_folder, "all_clusters_voxel_coefficients.nii.gz"))

all_cluster_pvalues = np.zeros(n_features)
all_cluster_pvalues[cluster_labels >= 0] = coefficient_p_values[cluster_labels >= 0]
brain_all_cluster_pvalues = lops.transform_feature_space_to_brain(all_cluster_pvalues, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_all_cluster_pvalues, brain_affine, os.path.join(nifti_folder, "all_clusters_voxel_pvalues.nii.gz"))

cluster_nifti_folder = os.path.join()
if not os.path.isdir(cluster_nifti_folder):
    os.mkdir(cluster_nifti_folder)

for i in range(n_clusters):
    cluster_voxels = cluster_labels == i
    
    cluster_coefficients = np.zeros(n_features)
    cluster_coefficients[cluster_voxels] = unpermuted_coefficients[cluster_voxels]
    brain_cluster_coefficients = lops.transform_feature_space_to_brain(cluster_coefficients, feature_coordinates, brain_dimensions)
    lops.save_as_nifti(brain_cluster_coefficients, brain_affine, os.path.join(cluster_nifti_folder, f"cluster{i}_voxel_coefficients.nii.gz"))
    
    cluster_pvalues = np.zeros(n_features)
    cluster_pvalues[cluster_voxels] = coefficient_p_values[cluster_voxels]
    brain_cluster_pvalues = lops.transform_feature_space_to_brain(cluster_pvalues, feature_coordinates, brain_dimensions)
    lops.save_as_nifti(brain_cluster_pvalues, brain_affine, os.path.join(cluster_nifti_folder, f"cluster{i}_voxel_pvalues.nii.gz"))
