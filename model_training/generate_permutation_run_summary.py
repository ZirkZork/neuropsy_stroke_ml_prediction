#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from tkinter.filedialog import askdirectory

input_folder = askdirectory()

split_list_file_re = re.compile(r"split_list_(?P<run_id>.*)\.csv")
training_setup_file_re = re.compile(r"(?P<run_id>.*)_training_data_setup\.pkl")

setup_files = {}
run_lists = {}

for path, subdirs, files in os.walk(input_folder):
    for file_name in files:
        split_file_re_match = split_list_file_re.fullmatch(file_name)
        setup_file_re_match = training_setup_file_re.fullmatch(file_name)
        if setup_file_re_match is not None:
            run_id = setup_file_re_match.group("run_id")
            setup_files[run_id] = os.path.join(path, file_name)
        if split_file_re_match is not None:
            run_id = split_file_re_match.group("run_id")
            run_lists[run_id] = os.path.join(path, file_name)

found_run_ids = [id for id in setup_files.keys() if id in run_lists.keys()]

with open(os.path.join(input_folder, "runlist.csv"), "w") as f:
    f.write(",".join(["setup_file", "run_split_list"]))
    f.write("\n")
    for id in found_run_ids:
        f.write(",".join([setup_files[id].replace(input_folder, "."), run_lists[id].replace(input_folder, ".")]))
        f.write("\n")