 
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
    "settings": "file",
    "data_files": "files",
    "lesion_map_folder": "dir",
    "output_folder": "dir"
}


filepath_dict = uiiai.data_choose_file_paths(filepath_dict)

# load settings
settings_path = filepath_dict["settings"]
settings = seth.load_settings(settings_path)

analysis_tag = settings["cfg_tag"]

variables = settings["variables"]
variable_names = [var["name"].replace(" ", "_") for var in variables]

# get data from excel sheets
data_input_files = filepath_dict["data_files"].split("; ")
combine_datasets = True

if(len(data_input_files) == 1):
    single_data_file_frame = pd.read_excel(data_input_files.pop())
    if np.all([var in single_data_file_frame.columns for var in variable_names]):
        combine_datasets = False
        combined_data_frame = single_data_file_frame


if(combine_datasets):
    combined_data_frame = pd.DataFrame(columns=variable_names)

    # select and map columns from the xcel sheets
    for filepath in data_input_files:

        excel_frame = pd.read_excel(filepath)

        # select correct columns and empty values
        column_mapping, empty_value_mapping = uiiai.data_variable_matching(variable_names, excel_frame)

        empty_columns = [key for key, column in column_mapping.items() if column == "NA"]
        for column in empty_columns:
            column_mapping.pop(column)
        corresponding_columns = [column_mapping[column] for column in variable_names if column in column_mapping.keys()]
        stripped_excel_frame = excel_frame.loc[:, corresponding_columns]
        stripped_excel_frame.columns = [column for column in variable_names if column not in empty_columns]

        stripped_excel_frame.replace(empty_value_mapping, pd.NA, inplace=True)

        combined_data_frame = pd.concat([combined_data_frame, stripped_excel_frame])
   

# remap variable values
for var in variables:
    column_name = var["name"].replace(" ", "_")

    # match categories between origin and data table
    if var["type"] == "cat":
        found_categories = list(np.unique(combined_data_frame[column_name].dropna()))
        if len (found_categories) == 0:
            continue
        if np.all([cat in var["options"] for cat in found_categories]):
            continue
        remapping = uiiai.data_remap_categorical_variables(var["name"], found_categories, var["options"])
        combined_data_frame = combined_data_frame.replace({column_name: remapping})
    # convert dates to date format
    elif var["type"] == "date":
        combined_data_frame[column_name] = pd.to_datetime(combined_data_frame[column_name], dayfirst=True)

# calculate missing values
for var in variables:
    column_name = var["name"].replace(" ", "_")
    if "formular" in var:
        missing_rows = combined_data_frame[column_name].isna()
        if missing_rows.sum() == 0: 
            continue

        formular = var["formular"]

        # select categories from numerical values
        if var["type"] == "cat":
            basis, rest = formular.split(":")
            assignments = rest.split(";")
            intervals = [assignment.split("->")[0] for assignment in assignments]
            bin_labels = [assignment.split("->")[1] for assignment in assignments]

            conds = [combined_data_frame.loc[missing_rows, basis].between (eval(interval.split("to")[0]), eval(interval.split("to")[1])) for interval in intervals]

            combined_data_frame.loc[missing_rows, column_name] = np.select(conds, bin_labels, pd.NA)

        # calculate variables from two others
        else:

            fill_in_values = combined_data_frame.loc[missing_rows].eval(formular)

            # if variable is a time intervall either convert to whole days or years
            if fill_in_values.dtype == 'timedelta64[ns]':
                if var["unit"].lower() in ["day", "days", "d"]:
                    fill_in_values = fill_in_values.dt.days
                elif var["unit"].lower() in ["year", "years", "y"]:
                    fill_in_values = fill_in_values.dt.days//365.25

            combined_data_frame.loc[missing_rows, column_name] = fill_in_values

if(combine_datasets):
    # save combined data table to excelfile
    combined_data_frame.to_excel(os.path.join(filepath_dict["output_folder"], "combined_data_set.xlsx"), index=False)

# load and process lesion maps

subject_id_variable = settings["variable_selection"]["sample_id"]
prediction_target_variable = settings["variable_selection"]["prediction_target"]

prediction_variable = [var for var in settings["variables"] if var ["name"] == prediction_target_variable].pop()
prediction_type = prediction_variable["type"]
if prediction_type not in ["int", "flow", "cat"]:
    raise Exception("Prediction target needs to be number or category.")

prediction_positive_category = None

if prediction_type == "cat" and len(prediction_variable["options"])==2:
    prediction_type = "cat_binary"
    prediction_positive_category = uiiai.data_choose_positive_category_ui(prediction_variable["options"])
if prediction_type in ["int", "float"]:
    prediction_type = "num"

lesion_map_settings = settings["lesionmap_processing"]
lesion_map_folder = filepath_dict["lesion_map_folder"]
lesion_maps, map_dimensions, map_affine, found_map_ids = lops.load_lesion_maps(list(combined_data_frame[subject_id_variable]), lesion_map_folder, lesion_map_settings["slice_spacing"], lesion_map_settings["slice_offset"])

mapped_rows = combined_data_frame[subject_id_variable].apply(lambda x: any([i in x.lower() for i in found_map_ids]))

lesion_sizes = lesion_maps.sum(axis=(1,2,3))

combined_data_frame["map_found"] = False
combined_data_frame.loc[mapped_rows, "map_found"] = True

combined_data_frame.loc[mapped_rows, "lesion_size"] = lesion_sizes

# set inclusion criteria and filter data 

filter_variables = settings["variable_selection"]["filter_variables"]

missing_values_excluded_rows = [combined_data_frame[subject_id_variable].isna(), combined_data_frame[prediction_target_variable].isna()]
filter_excluded_rows = []

for filter in filter_variables:
    if np.any([keyword in filter for keyword in ["median", "mean", "quartile"]]):
        eval_var = [column for column in combined_data_frame.columns if column in filter].pop()

        filter = filter.replace("median", str(combined_data_frame[eval_var].median()))
        filter = filter.replace("mean", str(combined_data_frame[eval_var].mean()))
        filter = filter.replace("lower_quartile", str(combined_data_frame[eval_var].quantile(0.25)))
        filter = filter.replace("upper_quartile", str(combined_data_frame[eval_var].quantile(0.75)))
        
    filter_excluded_rows.append(~ combined_data_frame.eval(filter))

combined_data_frame["excluded"] = False
combined_data_frame["exclusion_reason"] = ""

for exclusion in missing_values_excluded_rows:
    combined_data_frame.loc[exclusion, "excluded"] = True
    combined_data_frame.loc[exclusion, "exclusion_reason"] += "missing values; "

for exclusion in filter_excluded_rows:
    combined_data_frame.loc[exclusion, "excluded"] = True
    combined_data_frame.loc[exclusion, "exclusion_reason"] += "filtered out; "

combined_data_frame.loc[~ combined_data_frame["map_found"], "excluded"] = True
combined_data_frame.loc[~ combined_data_frame["map_found"], "exclusion_reason"] += "no map; "

included_rows = combined_data_frame.eval("excluded==False")
n_included_subjects = sum(included_rows)

included_mapped_rows = combined_data_frame[mapped_rows].eval("excluded==False")
included_lesion_maps = lesion_maps[included_mapped_rows]

assert(n_included_subjects == sum(included_mapped_rows))

# lesion masking

lesion_overlap_theshold = lesion_map_settings["lesion_overlap_theshold"]
lesion_overlap_proportion = (included_lesion_maps>=1).sum(axis=0) / n_included_subjects

voxel_mask = lesion_overlap_proportion >= lesion_overlap_theshold
voxel_mask_size = np.sum(voxel_mask)

print(f"{voxel_mask_size} voxels included in lesion overlap mask that are lesioned in at least {lesion_overlap_theshold:.0%} of patients")

lesion_voxels_in_mask, lesion_in_mask_proportion, mask_covered_proportion = zip(*[lops.check_overlap(lesion_map, voxel_mask) for lesion_map in included_lesion_maps])

combined_data_frame.loc[included_rows, "lesion_voxels_in_mask"] = lesion_voxels_in_mask
combined_data_frame.loc[included_rows, "lesion_in_mask_proportion"] = lesion_in_mask_proportion
combined_data_frame.loc[included_rows, "mask_covered_proportion"] = mask_covered_proportion

mask_overlap_cutoff = lesion_map_settings["mask_overlap_cutoff"]
lesion_mostly_in_mask = combined_data_frame.loc[included_rows, "lesion_in_mask_proportion"] >= mask_overlap_cutoff

combined_data_frame.loc[included_rows, "lesion_mostly_in_mask"] = lesion_mostly_in_mask

# save output
output_directory = os.path.join(filepath_dict["output_folder"], f"{analysis_tag}_LSM_prediction")
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

run_directory = os.path.join(output_directory, "runs")
if not os.path.isdir(run_directory):
    os.makedirs(run_directory)

sub_run_directory = os.path.join(run_directory, f"{analysis_tag}_initial_run")
if not os.path.isdir(sub_run_directory):
    os.makedirs(sub_run_directory)

# save a copy of the settings
seth.save_settings(settings, os.path.join(output_directory, f"{analysis_tag}_settings.cfg"))

# save final data table to excelfile
combined_data_frame.to_excel(os.path.join(output_directory, "processed_data_set.xlsx"), index=False)

# save lesion mask as nifti
lops.save_as_nifti (voxel_mask.astype(float), map_affine, os.path.join(output_directory, "lesion_overlap_voxel_mask.nii"))
    
# save training features and targets

training_feature_vectors = included_lesion_maps[lesion_mostly_in_mask][:, voxel_mask]
feature_coordinates = np.asarray(np.nonzero(voxel_mask)).T

do_lesion_volume_control = lesion_map_settings["do_volume_control"]

if(do_lesion_volume_control):
    training_feature_vectors *= 1 / np.sqrt(np.sum(training_feature_vectors, axis=1)).reshape(-1,1)

    assert (np.all(np.isclose(np.linalg.norm(training_feature_vectors, axis=1), 1)))

training_targets = np.asarray(combined_data_frame.query("lesion_mostly_in_mask == True")[prediction_target_variable])

training_setup = {
    "analysis_tag": analysis_tag,
    "rng_seed": settings["rng_seed"],
    "features": training_feature_vectors,
    "brain_dimensions": map_dimensions,
    "brain_affine": map_affine,
    "feature_coordinates": feature_coordinates,
    "targets": training_targets,
    "prediction_type": prediction_type,
    "prediction_positive_category": prediction_positive_category,
    "parameters": settings["training_parameters"],
    "n_permutations": settings["training_parameters"]["n_permutations"],
}


with open(os.path.join(sub_run_directory, f"{analysis_tag}_training_data_setup.pkl"), "wb") as f:
    pkl.dump(training_setup, f)
