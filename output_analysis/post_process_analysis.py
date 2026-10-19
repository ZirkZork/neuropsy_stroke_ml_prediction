
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle as pkl
import sys
import os
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn import cluster as clst
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility_functions import lesion_operations as lops
import pandas as pd


file_name = askopenfilename()
with open(file_name, "rb") as f:
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

p_value_divider = (n_permutations + 1)

metric_p_values = {}
for metric in unpermuted_metrics.keys():
    metric_p_values[metric] = np.sum(unpermuted_metrics[metric] <= permutation_metrics[metric], axis=0) / p_value_divider
    print(f"{metric}: {unpermuted_metrics[metric]:.3f}; p={metric_p_values[metric]:.4f}")

intercept_p_value = np.sum(np.abs(unpermuted_intercept) <= np.abs(permutation_intercepts), axis=0) / p_value_divider

single_threshold_correction = False

if single_threshold_correction:
    voxel_by_permutation = np.abs(unpermuted_coefficients) <= np.max(np.abs(permutation_coefficients), axis=1).reshape((-1,1))
else:
    voxel_by_permutation = np.abs(unpermuted_coefficients) <= np.abs(permutation_coefficients)
                                  
coefficient_p_values = np.sum(voxel_by_permutation, axis=0) / p_value_divider


coefficent_sign_mask = np.sign(unpermuted_coefficients)

##
alpha_threshold = 0.001
cluster_size_threshhold = 50
##

significant_coefficents = coefficient_p_values < alpha_threshold

##
cluster_max_gap = 1
##

positive_super_threshold_voxels = np.logical_and(significant_coefficents,coefficent_sign_mask == 1)
negative_super_threshold_voxels = np.logical_and(significant_coefficents, coefficent_sign_mask == -1)

cluster_labels = np.zeros(n_features) - 1

if np.any(positive_super_threshold_voxels):
    cluster_pos_core_samples, cluster_pos_labels = clst.dbscan(feature_coordinates[positive_super_threshold_voxels], eps=cluster_max_gap)
    cluster_labels[positive_super_threshold_voxels] = cluster_pos_labels

if np.any(negative_super_threshold_voxels):
    cluster_neg_core_samples, cluster_neg_labels = clst.dbscan(feature_coordinates[negative_super_threshold_voxels], eps=cluster_max_gap)
    cluster_neg_labels[cluster_neg_labels != -1] += max(cluster_pos_labels) + 1
    cluster_labels[negative_super_threshold_voxels] = cluster_neg_labels

cluster_names, cluster_sizes = np.unique(cluster_labels, return_counts=True)
n_clusters = int(max(cluster_labels) + 1)
cluster_size_dict = dict(zip(cluster_names, cluster_sizes))

supra_threshold_clusters = [label for label, size in zip(cluster_names, cluster_sizes) if size >= cluster_size_threshhold and label != -1]

nifti_folder = os.path.join(os.path.dirname(file_name), "niftis")
if not os.path.isdir(nifti_folder):
    os.mkdir(nifti_folder)

brain_cluster_labels = lops.transform_feature_space_to_brain(cluster_labels + 1, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_cluster_labels, brain_affine, os.path.join(nifti_folder, "cluster_labels.nii.gz"))

brain_coefficients = lops.transform_feature_space_to_brain(unpermuted_coefficients, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_coefficients, brain_affine, os.path.join(nifti_folder, "voxel_coefficients.nii.gz"))

brain_pvalues = lops.transform_feature_space_to_brain(coefficient_p_values, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_pvalues, brain_affine, os.path.join(nifti_folder, "voxel_pvalues.nii.gz"))

supra_threshold_cluster_mask = [cluster_label in supra_threshold_clusters for cluster_label in cluster_labels]

all_cluster_coefficients = np.zeros(n_features)
all_cluster_coefficients[supra_threshold_cluster_mask] = unpermuted_coefficients[supra_threshold_cluster_mask]
brain_all_cluster_coefficients = lops.transform_feature_space_to_brain(all_cluster_coefficients, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_all_cluster_coefficients, brain_affine, os.path.join(nifti_folder, "all_clusters_voxel_coefficients.nii.gz"))

all_cluster_pvalues = np.zeros(n_features)
all_cluster_pvalues[supra_threshold_cluster_mask] = coefficient_p_values[supra_threshold_cluster_mask]
brain_all_cluster_pvalues = lops.transform_feature_space_to_brain(all_cluster_pvalues, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_all_cluster_pvalues, brain_affine, os.path.join(nifti_folder, "all_clusters_voxel_pvalues.nii.gz"))

all_cluster_mask = np.zeros(n_features)
all_cluster_mask[supra_threshold_cluster_mask] = 1
brain_all_cluster_mask = lops.transform_feature_space_to_brain(all_cluster_mask, feature_coordinates, brain_dimensions)
lops.save_as_nifti(brain_all_cluster_mask, brain_affine, os.path.join(nifti_folder, "all_clusters_voxel_mask.nii.gz"))

cluster_nifti_folder = os.path.join(nifti_folder, "clusters")
if not os.path.isdir(cluster_nifti_folder):
    os.mkdir(cluster_nifti_folder)

cluster_info_dict = {}

for i, id in enumerate(supra_threshold_clusters):
    cluster_voxels = cluster_labels == id
    cluster_tag = f"{analysis_tag}_cluster{i+1}"

    brain_cluster_mask = lops.transform_feature_space_to_brain(cluster_voxels, feature_coordinates, brain_dimensions)
    lops.save_as_nifti(brain_cluster_mask, brain_affine, os.path.join(cluster_nifti_folder, f"{cluster_tag}_voxel_mask.nii.gz"))

    cluster_coefficients = np.zeros(n_features)
    cluster_coefficients[cluster_voxels] = unpermuted_coefficients[cluster_voxels]
    brain_cluster_coefficients = lops.transform_feature_space_to_brain(cluster_coefficients, feature_coordinates, brain_dimensions)
    lops.save_as_nifti(brain_cluster_coefficients, brain_affine, os.path.join(cluster_nifti_folder, f"{cluster_tag}_voxel_coefficients.nii.gz"))
    
    cluster_pvalues = np.zeros(n_features)
    cluster_pvalues[cluster_voxels] = coefficient_p_values[cluster_voxels]
    brain_cluster_pvalues = lops.transform_feature_space_to_brain(cluster_pvalues, feature_coordinates, brain_dimensions)
    lops.save_as_nifti(brain_cluster_pvalues, brain_affine, os.path.join(cluster_nifti_folder, f"{cluster_tag}_voxel_pvalues.nii.gz"))

    min_p_value = min(coefficient_p_values[cluster_voxels])
    cluster_peak_voxels = np.zeros(n_features)
    cluster_peak_voxels[cluster_voxels] = (coefficient_p_values[cluster_voxels] == min_p_value).astype(float)
    brain_cluster_peak_voxels = lops.transform_feature_space_to_brain(cluster_peak_voxels, feature_coordinates, brain_dimensions)
    lops.save_as_nifti(brain_cluster_peak_voxels, brain_affine, os.path.join(cluster_nifti_folder, f"{cluster_tag}_peak_voxels.nii.gz"))

    n_voxels = sum(cluster_voxels)

    peak_mni_com = lops.apply_affine(np.asarray(feature_coordinates[cluster_voxels][coefficient_p_values[cluster_voxels] == min_p_value].mean(axis =0)).reshape(1, -1), brain_affine).flatten()
    peak_n_voxels = sum(coefficient_p_values[cluster_voxels] == min_p_value)

    cluster_info_dict[cluster_tag] = [n_voxels, min_p_value, peak_n_voxels, peak_mni_com[0], peak_mni_com[1], peak_mni_com[2]]

    with np.printoptions(precision=1):
        print(f"{cluster_tag}: n voxels: {n_voxels}; peak p: {min_p_value:.4f}; n peak voxels: {peak_n_voxels}; peak COM: {peak_mni_com}")
    
cluster_info_frame = pd.DataFrame.from_dict(cluster_info_dict, orient="index", columns=["n_voxels", "peak_p", "n_peak_voxels", "peak_com_x", "peak_com_y", "peak_com_z"])
cluster_info_frame.to_excel(os.path.join(cluster_nifti_folder, f"{analysis_tag}_cluster_voxel_pvalues.xlsx"))
