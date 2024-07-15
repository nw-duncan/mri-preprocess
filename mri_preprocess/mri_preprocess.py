#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:51:05 2024

@author: niall
"""

from mri_preprocess import utils
from mri_preprocess.anat_prepro import run_anat_preprocess
from mri_preprocess.func_prepro import run_func_preprocess


def mri_preprocess(subject, settings_file):

    # Initiate preprocessing settings
    settings = utils.initiate_settings()

    # Update settings with user defined ones
    settings = utils.update_settings(settings, settings_file)

    # Check that the root directory follows BIDs style
    utils.check_bids_style(settings['root_dir'])

    # Do directory management stuff
    settings = utils.prepare_directories(subject, settings)

    ###########################
    # Anatomical preprocessing
    ###########################
    
    if settings['process_anat']:
        run_anat_prepro(subject, settings)

    ######################################################
    # Functional preprocessing
    # Assumes anatomical preprocessing has been done
    ######################################################
    
    if settings['process_func']:

        for run_number in range(settings['number_func_runs']):
            run_number += 1
            run_func_preprocess(subject, settings, run_number)

                
                
        