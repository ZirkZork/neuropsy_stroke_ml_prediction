#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import tkinter.filedialog as fidi

class data_choose_file_paths_ui:
    def __init__(self, file_request_dict:dict, initial_dir = None):
        self.file_request_dict = file_request_dict
        
        self.root = tk.Tk()

        self.root.title("Choose File Paths")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.file_path_entry_texts = {}

        for name, path_type in self.file_request_dict.items():
            file_path_frame = ttk.Frame(self.frame)
            file_path_frame.pack()

            file_path_entry_label = ttk.Label(file_path_frame, text=f"{name.capitalize().replace("_", " ")}: ")
            file_path_entry_label.pack(side="left")

            self.file_path_entry_texts[name] = tk.StringVar()

            file_path_entry = ttk.Entry(file_path_frame, textvariable=self.file_path_entry_texts[name])
            file_path_entry.pack(side="left")

            if path_type == "file":
                get_path_command = lambda path_name = name: self.file_path_entry_texts[path_name].set(fidi.askopenfilename(initialdir=initial_dir)) 
            elif path_type == "files":
                get_path_command = lambda path_name = name: self.file_path_entry_texts[path_name].set("; ".join(fidi.askopenfilenames(initialdir=initial_dir))) 
            elif path_type == "dir":
                get_path_command = lambda path_name = name: self.file_path_entry_texts[path_name].set(fidi.askdirectory(initialdir=initial_dir)) 
            elif path_type == "save_file":
                get_path_command = lambda path_name = name: self.file_path_entry_texts[path_name].set(fidi.asksaveasfilename(initialdir=initial_dir)) 
            else:
                raise Exception("neither file nor dir path requested.")

            button = ttk.Button(file_path_frame, text="...", command=get_path_command)
            button.pack(side="left")

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()

    def submit(self):
        for name in self.file_request_dict.keys():
            self.file_request_dict[name] = self.file_path_entry_texts[name].get()
        self.root.destroy()

def data_choose_file_paths(file_request_dict:dict, initial_dir = None):

    data_choose_file_paths_ui(file_request_dict, initial_dir = initial_dir)

    return file_request_dict

class data_variable_matching_ui:
    def __init__(self, column_mapping:dict, empty_value_mapping:dict, variables, excel_file:pd.DataFrame):
        self.column_mapping = column_mapping
        self.empty_value_mapping = empty_value_mapping
        excel_file_columns = list(excel_file.columns)
        self.root = tk.Tk()

        #self.root.geometry("800x500")
        self.root.title("Column mapping")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.dropdowns = {}
        self.empty_value_entry = {}
        
        def update_empty_value_list(column_name, new_associated_column):
            column_name =  column_name.split(".")[-1]
            self.empty_value_entry[column_name]["values"] = [""]
            if new_associated_column in excel_file.columns:
                print(f"{column_name} now mapped to {new_associated_column}")
                self.empty_value_entry[column_name]["values"] = [""] + list(np.unique(excel_file[new_associated_column]))
            return True
        self.update_empty_value_list_cmd = self.root.register(update_empty_value_list)

        for column in variables:
            column_frame = ttk.Frame(self.frame)
            column_frame.pack()

            column_label = ttk.Label(column_frame, text=f"{column}:")
            column_label.pack(side='left')



            self.dropdowns[column] = ttk.Combobox(column_frame, name=column ,state="readonly", values=["NA"] + excel_file_columns, validate="all", validatecommand=(self.update_empty_value_list_cmd, '%W', '%P'))
            self.dropdowns[column].current(0)
            self.dropdowns[column].pack(side='left')

            empty_value_label = ttk.Label(column_frame, text="empty value:")
            empty_value_label.pack(side='left')

            self.empty_value_entry[column] = ttk.Combobox(column_frame, state="readonly", values=[""])
            self.empty_value_entry[column].pack(side='left')

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()
    
    def submit(self):
        for key, item in self.dropdowns.items():
            self.column_mapping[key] = item.get()
        for key, item in self.empty_value_entry.items():
            empty_value_string = item.get()
            if empty_value_string != "":
                try:
                    empty_value_string = ast.literal_eval(empty_value_string)
                except:
                    pass
                self.empty_value_mapping[key] = empty_value_string
        self.root.destroy()

def data_variable_matching(variables, excel_file):
    column_mapping = {}
    empty_value_mapping = {}

    data_variable_matching_ui(column_mapping, empty_value_mapping, variables, excel_file)

    return column_mapping, empty_value_mapping

class data_remap_categorical_variables_ui:
    def __init__(self, column_name, unique_values: list, mapping_options: list, remapping:dict):
        self.remapping = remapping

        self.root = tk.Tk()

        #self.root.geometry("800x500")
        self.root.title(f"{column_name.capitalize()} remapping")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.remappings = {}
        
        for value in unique_values:
            column_frame = ttk.Frame(self.frame)
            column_frame.pack()
            remapping_label = ttk.Label(column_frame, text=f"{value}:")
            remapping_label.pack(side='left')
            self.remappings[value] = tk.StringVar()
            for option in mapping_options:
                ttk.Radiobutton(column_frame, text=option, variable=self.remappings[value], value=option).pack(side='left')

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()
    
    def submit(self):
        for key, item in self.remappings.items():
            self.remapping[key] = item.get()
            print(key, item.get())
        self.root.destroy()

def data_remap_categorical_variables(column_name, unique_values: list, mapping_options: list):
    remapping = {}

    data_remap_categorical_variables_ui(column_name, unique_values, mapping_options, remapping)

    return remapping

class generic_number_prompt_ui:
    def __init__(self, title, text, number_out):
        self.number_out = number_out
        
        self.root = tk.Tk()

        self.root.title(title)

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.prompt_label = ttk.Label(self.frame, text=text)
        self.prompt_label.pack()

        vcmd = (self.frame.register(self.number_entry_validation))
        self.number_entry = ttk.Entry(self.frame, validate='all', validatecommand=(vcmd, '%P'))
        self.number_entry.pack()

        self.entered_number = tk.StringVar()
        self.number_entry["textvariable"] = self.entered_number

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()

    def number_entry_validation(self, P):
        return str.isdigit(P.replace(".", "", 1)) or P == ""
    
    def submit(self):
        try:
            self.number_out["num"] = int(self.entered_number.get())
        except:
            self.number_out["num"] = float(self.entered_number.get())

        self.root.destroy()

def settings_get_number_variables():
    number_out = {"num": 0}
    generic_number_prompt_ui("Number of Variable", "How many variables do you want to set up?", number_out)
    return number_out["num"]

def generic_get_number(title, text):
    number_out = {"num": 0}
    generic_number_prompt_ui(title, text, number_out)
    return number_out["num"]

class generic_list_item_selection:
    def __init__(self, title, text, selection_options, selection_out):
        self.selection_out = selection_out
        
        self.root = tk.Tk()

        self.root.title(title)

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.prompt_label = ttk.Label(self.frame, text=text)
        self.prompt_label.pack()

        self.selected_item = tk.StringVar()

        self.radio_button_frame = ttk.Frame(self.frame)
        self.radio_button_frame.pack()

        for option in selection_options:
            radiobutton = ttk.Radiobutton(self.radio_button_frame, text=str(option), variable=self.selected_item, value=str(option))
            radiobutton.pack(side='left')

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()
    
    def submit(self):
        self.selection_out["selection"] = self.selected_item.get()
        self.root.destroy()

def data_choose_positive_category_ui(selection_options):
    selection_out = {"selection": None}
    generic_list_item_selection("Choose Prediction Target", 'Which category is the "positive" prediction?', selection_options, selection_out)
    return selection_out["selection"]

class generic_list_items_selection:
    def __init__(self, title, text, selection_options, selection_out):
        self.selection_out = selection_out
        
        self.root = tk.Tk()

        self.root.title(title)

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.prompt_label = ttk.Label(self.frame, text=text)
        self.prompt_label.pack()

        self.check_button_frame = ttk.Frame(self.frame)
        self.check_button_frame.pack()

        self.option_checks = []
        for option in selection_options:
            option_var = tk.StringVar()
            ttk.Checkbutton(self.check_button_frame, text=str(option), variable=option_var, onvalue=str(option), offvalue=None).pack()
            self.option_checks.append(option_var)

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()
    
    def submit(self):
        self.selection_out += [option_var.get() for option_var in self.option_checks if option_var.get() != ""]
        self.root.destroy()

def training_select_runs_ui(selection_options):
    selection_out = []
    generic_list_items_selection("Choose Runs", 'Which permutation runs should be run?', selection_options, selection_out)
    return selection_out

def data_select_demo_variables(selection_options):
    selection_out = []
    generic_list_items_selection("Select variables", 'Choose demographic variables to get statistics for:', selection_options, selection_out)
    return selection_out

class settings_variable_setup_ui:
    def __init__(self, variable_dict):
        self.variable_dict = variable_dict
        
        self.root = tk.Tk()

        self.root.title(f"Create Variable")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        # variable name
        self.variable_name_label = ttk.Label(self.frame, text="variable name:")
        self.variable_name_label.pack()

        self.variable_name_entry = ttk.Entry(self.frame)
        self.variable_name_entry.pack()

        self.entered_variable_name = tk.StringVar()
        self.variable_name_entry["textvariable"] = self.entered_variable_name

        # variable type
        self.variable_type_label = ttk.Label(self.frame, text="variable type:")
        self.variable_type_label.pack()

        self.radio_button_frame = ttk.Frame(self.frame)
        self.radio_button_frame.pack()

        self.selected_type = tk.StringVar(value="int")
        
        def update_variable_type(a=None,b=None,c=None):
            self.cat_options_frame.pack_forget()
            self.unit_option_frame.pack_forget()
            self.submit_button.pack_forget()
            if self.selected_type.get() == "cat":
                self.cat_options_frame.pack()
            elif self.selected_type.get() in ["int", "float"]:
                self.unit_option_frame.pack()
            self.submit_button.pack()

        self.selected_type.trace("w", update_variable_type)

        self.radiobutton_int = ttk.Radiobutton(self.radio_button_frame, text="int", variable=self.selected_type, value="int")
        self.radiobutton_int.pack(side='left')
        self.radiobutton_float = ttk.Radiobutton(self.radio_button_frame, text="float", variable=self.selected_type, value="float")
        self.radiobutton_float.pack(side='left')
        self.radiobutton_cat = ttk.Radiobutton(self.radio_button_frame, text="cat", variable=self.selected_type, value="cat")
        self.radiobutton_cat.pack(side='left')
        self.radiobutton_label = ttk.Radiobutton(self.radio_button_frame, text="label", variable=self.selected_type, value="label")
        self.radiobutton_label.pack(side='left')
        self.radiobutton_date = ttk.Radiobutton(self.radio_button_frame, text="date", variable=self.selected_type, value="date")
        self.radiobutton_date.pack(side='left')

        # variable options - unit (numerical)
        self.unit_option_frame = ttk.Frame(self.frame)
        self.unit_option_frame.pack()

        self.unit_option_label = ttk.Label(self.unit_option_frame, text="unit (optional):")
        self.unit_option_label.pack(side='left')

        self.unit_option_entry = ttk.Entry(self.unit_option_frame)
        self.unit_option_entry.pack(side='left')

        # variable options (cat)
        self.cat_options_frame = ttk.Frame(self.frame)
        self.cat_options_frame.pack()

        self.cat_option_label = ttk.Label(self.cat_options_frame, text="options:")
        self.cat_option_label.pack()

        self.cat_option_number_subframe = ttk.Frame(self.cat_options_frame)
        self.cat_option_number_subframe.pack()

        self.cat_option_number_label = ttk.Label(self.cat_option_number_subframe, text="number of options:")
        self.cat_option_number_label.pack(side="left")

        def update_number_of_categories(a=None,b=None,c=None):
            number_of_categories = int(self.category_number.get())
            if(number_of_categories < 2):
                self.category_number.set("2")
            elif(number_of_categories > 5):
                self.category_number.set("5")
            for i, entry in enumerate(self.cat_option_entries):
                entry.pack_forget()
                if i < number_of_categories or i < 2:
                    entry.pack()

        vcmd = (self.frame.register(self.number_entry_validation))
        self.cat_option_number_entry = ttk.Entry(self.cat_option_number_subframe, validate='all', validatecommand=(vcmd, '%P'))
        self.cat_option_number_entry.pack(side="left")

        self.category_number = tk.StringVar(value="2")
        self.cat_option_number_entry["textvariable"] = self.category_number
        self.category_number.trace("w", update_number_of_categories)

        self.cat_options_subframe = ttk.Frame(self.cat_options_frame)
        self.cat_options_subframe.pack()

        self.cat_option_entries = []
        max_options = 5
        for i in range (max_options):
            cat_option_entry = ttk.Entry(self.cat_options_subframe)
            cat_option_entry.pack()
            self.cat_option_entries.append(cat_option_entry)


        # submit button
        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        update_variable_type()
        update_number_of_categories()

        self.root.mainloop()
    
    def number_entry_validation(self, P):
        return str.isdigit(P) or P == ""

    def submit(self):
        self.variable_dict["name"] = self.entered_variable_name.get()
        self.variable_dict["type"] = self.selected_type.get()

        if self.variable_dict["type"] == "cat":
            category_options = []
            for i in range(int(self.category_number.get())):
                category_options.append (self.cat_option_entries[i].get())

            self.variable_dict["options"] = category_options
        elif self.variable_dict["type"] in ["int", "float"]:
            unit = self.unit_option_entry.get().strip()
            if unit != "":
                self.variable_dict["unit"] = unit
        self.root.destroy()

def settings_setup_variable():
    variable_dict = {}
    settings_variable_setup_ui(variable_dict)
    return variable_dict

class settings_select_variable_ui:
    def __init__(self, variable_names, selected_variable):
        self.selected_variable = selected_variable

        self.root = tk.Tk()

        self.root.title("Select a variable")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        for name in variable_names:
            ttk.Radiobutton(self.frame, text=name, variable=self.selected_variable, value=name).pack()

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()
        
    def submit(self):
        self.root.destroy()

def settings_select_variable(variable_names):
    selected_variable = tk.StringVar()
    
    settings_select_variable_ui(variable_names, selected_variable)

    return selected_variable.get()

class settings_select_variables_ui:
    def __init__(self, variable_names, selected_variables):
        self.selected_variables = selected_variables

        self.root = tk.Tk()

        self.root.title("Select a variable")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.variable_checks = []
        for name in variable_names:
            check_var = tk.StringVar()
            ttk.Checkbutton(self.frame, text=name, variable=check_var, onvalue=name, offvalue=None).pack()
            self.variable_checks.append(check_var)

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()
        
    def submit(self):
        self.selected_variables += [check_var.get() for check_var in self.variable_checks]
        self.root.destroy()

def settings_select_variables(variable_names):
    selected_variables = []
    
    settings_select_variables_ui(variable_names, selected_variables)

    return selected_variables

class settings_set_category_formular_ui:
    def __init__(self, expression_dict, variable_name, category_names, basis_variable_options, validation_func):
        self.category_names = category_names
        self.expression_dict = expression_dict
        self.formular_validation_func = validation_func

        self.root = tk.Tk()

        self.root.title(f"Set category formulars for {variable_name}")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.main_label = ttk.Label(self.frame, text=f"Set the ranges for the categories of {variable_name}")
        self.main_label.pack()

        # selection for variable to base the categories on
        self.basis_var_selection_frame = ttk.Frame(self.frame)
        self.basis_var_selection_frame.pack()

        self.basis_var_selection_label = ttk.Label(self.basis_var_selection_frame)
        self.basis_var_selection_label.pack(side="left")

        self.basis_var_selection_selection = ttk.Combobox(self.basis_var_selection_frame, state="readonly", values=[""] + basis_variable_options)
        self.basis_var_selection_selection.pack(side="left")

        self.category_ranges = {}

        # frames to set the category ranges
        for cat_name in self.category_names:
            category_frame = ttk.Frame(self.frame)
            category_frame.pack()

            category_title_frame = ttk.Frame(category_frame)
            category_title_frame.pack()

            category_title_label = ttk.Label(category_title_frame, text=f"{cat_name}:")
            category_title_label.pack()

            category_range_frame = ttk.Frame(category_frame)
            category_range_frame.pack()

            category_range_from_label = ttk.Label(category_range_frame, text="from:")
            category_range_from_label.pack(side="left")

            category_range_from_entry = ttk.Entry(category_range_frame)
            category_range_from_entry.pack(side="left")

            category_range_to_label = ttk.Label(category_range_frame, text="to:")
            category_range_to_label.pack(side="left")

            category_range_to_entry = ttk.Entry(category_range_frame)
            category_range_to_entry.pack(side="left")

            self.category_ranges[cat_name] = {
                "from": category_range_from_entry,
                "to": category_range_to_entry
            }


        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()


    def submit(self):
        basis_vriable = self.basis_var_selection_selection.get()

        expression = "".join([f"{basis_vriable}:",  ";".join([f"{self.category_ranges[cat]["from"].get()}to{self.category_ranges[cat]["to"].get()}->{cat}" for cat in self.category_names])])


        if self.formular_validation_func(expression):
            self.expression_dict["expression"] = expression
            self.root.destroy()

def settings_set_category_formular(variable_name, category_names, basis_variable_options, validation_func):
    expression_dict = {}
    settings_set_category_formular_ui(expression_dict, variable_name, category_names, basis_variable_options, validation_func)

    return expression_dict["expression"]

class settings_set_arithmetic_formular_ui:
    def __init__(self, expression_dict, variable_name, basis_variable_options, validation_func):
        self.expression_dict = expression_dict
        self.formular_validation_func = validation_func

        self.root = tk.Tk()

        self.root.title(f"Set category formulars for {variable_name}")

        self.frame = ttk.Frame(self.root)
        self.frame.pack()

        self.main_label = ttk.Label(self.frame, text=f"Set the ranges for the categories of {variable_name}")
        self.main_label.pack()

        # selection for variables and calculations
        self.formular_selection_frame = ttk.Frame(self.frame)
        self.formular_selection_frame.pack()

        self.basis_var_1_selection = ttk.Combobox(self.formular_selection_frame, state="readonly", values=[""] + basis_variable_options)
        self.basis_var_1_selection.pack(side="left")

        self.operator_selection = ttk.Combobox(self.formular_selection_frame, state="readonly", values=["+", "-", "*", "/"])
        self.operator_selection.pack(side="left")

        self.basis_var_2_selection = ttk.Combobox(self.formular_selection_frame, state="readonly", values=[""] + basis_variable_options)
        self.basis_var_2_selection.pack(side="left")

        self.submit_button = ttk.Button(self.frame, command=self.submit, text="Submit")
        self.submit_button.pack()

        self.root.mainloop()


    def submit(self):
        basis_variable_1 = self.basis_var_1_selection.get()
        basis_variable_2 = self.basis_var_2_selection.get()
        basis_operator = self.operator_selection.get()

        expression = "".join([basis_variable_1, basis_operator, basis_variable_2])


        if self.formular_validation_func(expression):
            self.expression_dict["expression"] = expression
            self.root.destroy()

def settings_set_arithmetic_formular(variable_name, basis_variable_options, validation_func):
    expression_dict = {}
    settings_set_arithmetic_formular_ui(expression_dict, variable_name, basis_variable_options, validation_func)

    return expression_dict["expression"]