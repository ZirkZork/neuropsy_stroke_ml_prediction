#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utility_functions.settings_handler as settings

settings_dict = settings.setup_settings("test_settings")
settings.setup_variables(settings_dict)
#settings.save_settings(settings_dict, os.path.join(os.path.dirname(__file__), "test_settings.cfg"))

#settings_dict = settings.load_settings(os.path.join(os.path.dirname(__file__), "test_settings.cfg"))
settings.set_var_formular(settings_dict)

settings.save_settings(settings_dict, os.path.join(os.path.dirname(__file__), "test_settings.cfg"))
