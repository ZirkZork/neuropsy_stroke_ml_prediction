#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
import random
import functools as funky
import yaml
from tkinter.filedialog import askopenfilename

import utility_functions.UI_windows as ui
import utility_functions.variable_parser_util as varpar
'''
Settings I need:

prediction variable
sample_id_variable

slice_offset = -2
slice_spacing = 5

overlap_cutoff

retrain_alpha_cutoffs
retrain_overlap_cutoff

n_cv_folds
n_cv_split_repetitions

n_permutations

cluster_alpha_threshhold
min_cluster_size

rng_seed

'''

def get_settings():
    return load_settings(askopenfilename())

def load_settings(path):
    with open(path, "r") as f:
        settings_dict = yaml.full_load(f)

    return settings_dict

def save_settings(settings_dict, path):
    with open(path, "w")as f:
        yaml.dump(settings_dict, f)

def setup_settings(tag=None, path=None):
    if tag == None:
        tag = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    settings_dict = {
        "cfg_tag": tag,
        "variable_selection":{
            "prediction_target": "",
            "sample_id": "",
            "filter_variables": []
        },
        "lesionmap_processing":{
            "slice_offset": -2,
            "slice_spacing": 5,
            "lesion_overlap_theshold": 0.1,
            "mask_overlap_cutoff": 0.1
        },
        "retraining":{
            "retrain_mask_overlap_cutoff": 0,
            "retrain_alpha_cutoffs": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
        },
        "training_parameters":{
            "n_cv_folds": 4,
            "n_cv_split_repetitions": 25,
            "n_permutations": 10000            
        },
        "variables":[]
    }

    if path != None:
        save_settings(settings_dict, path)

    return settings_dict

def setup_variables (settings_dict):
    n_vars = ui.settings_get_number_variables()
    
    settings_dict["variables"] += [ui.settings_setup_variable() for i in range(n_vars)]

def remove_variables (vars_to_remove:list|str, settings_dict):
    if type(vars_to_remove) is str:
        vars_to_remove = [vars_to_remove]
    
    for var in settings_dict["variables"]:
        if (var["name"] in vars_to_remove):
            settings_dict["variables"].remove(var)

def remove_all_variables (settings_dict):
    settings_dict["variables"] = []

def set_var_formular(settings_dict):
    variables = settings_dict["variables"]
    variable_names = [var["name"] for var in variables]
    selected_variables = ui.settings_select_variables(variable_names)

    for var in variables:
        if var["name"] not in selected_variables:
            continue

        if var["type"] == "cat":
            var["formular"] = set_category_formular(var, variables)
        elif var["type"] not in ["date", "label"]:
            var["formular"] = set_arithmetic_formular(var, variables)

def set_category_formular (variable_dict, all_variables):
    passed_validation = funky.partial(varpar.validate_expression, target_variable = variable_dict, variables = all_variables)

    basis_options = [var["name"] for var in all_variables if var["type"] in ["int", "float"]]

    return ui.settings_set_category_formular(variable_dict["name"], variable_dict["options"], basis_options, passed_validation)

def set_arithmetic_formular (variable_dict, all_variables):
    passed_validation = funky.partial(varpar.validate_expression, target_variable = variable_dict, variables = all_variables)

    basis_options = [var["name"] for var in all_variables if var["type"] in ["int", "float", "date"]]

    return ui.settings_set_arithmetic_formular(variable_dict["name"], basis_options, passed_validation)

def get_variable_by_name(settings_dict, name):
    for var in settings_dict["variables"]:
        if var["name"] == name:
            return var