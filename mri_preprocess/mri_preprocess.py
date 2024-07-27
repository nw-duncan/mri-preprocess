#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run basic preprocessing on anatomical and functional MRI data.

"""

from joblib import Parallel, delayed
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
        run_anat_preprocess(subject, settings)

    ######################################################
    # Functional preprocessing
    # Assumes anatomical preprocessing has been done
    ######################################################
    
    if settings['process_func']:
        if settings['number_of_runs'] > 1:
            # Run multiple runs simultaneously
            if settings['process_multi_runs']:
                settings = utils.check_thread_no(settings)
                Parallel(n_jobs=settings['number_of_runs'], verbose=0)(delayed(run_func_preprocess)(subject, settings, run_number) for run_number in settings['run_numbers'])

            # Run multiple runs sequentially
            else:
                for run_number in settings['run_numbers']:
                    run_func_preprocess(subject, settings, run_number)
        else:
            run_func_preprocess(subject, settings, settings['run_numbers'])

