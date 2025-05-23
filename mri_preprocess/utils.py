#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper functions for preprocessing.

"""

import json
import numpy as np
from os import path, mkdir, cpu_count
from templateflow import api as tflow


def initiate_settings():
    """
    Create the settings object with all default values.

    Returns
    -------
    settings: dict
            Dictionary holding all settings.

    """
    settings = dict(root_dir="",
                    output_dir_name="",
                    session_number=None,
                    num_threads=1,
                    process_anat=True,
                    process_func=True,
                    template_name="MNI152NLin2009cAsym",
                    template_resolution=1,
                    T1_brain_extract_type="BET",
                    task_name="",
                    number_of_runs=1,
                    run_numbers=1,
                    process_multi_runs=False,
                    bold_TR=None,
                    drop_nonsteady_vols=True,
                    bold_reference_type="median",
                    bold_to_anat_cost="bbr",
                    reference_image=None,
                    run_slice_timing=True,
                    slice_time_ref=0.5,
                    slice_encoding_direction='k',
                    slice_acquisition_order=None,
                    detrend_degree=3,
                    number_nonsteady_vols=None,
                    number_usable_vols=None,
                    smoothing_fwhm=None,
                    run_melodic_ica=False,
                    apply_fieldmap=False,
                    scanner_manufacturer=None,
                    ants_reg_params={"transforms": ["Affine", "SyN"],
                                        "metric": ["Mattes", "Mattes"],
                                        "shrink_factors": [[2, 1], [3, 2, 1]],
                                        "smoothing_sigmas": [[1, 0], [2, 1, 0]],
                                        "number_of_iterations": [[1500, 200], [100, 50, 30]],
                                        "interpolation": "Linear",
                                        "float": True,
                                        "winsorize_lower_quantile": 0.02,
                                        "winsorize_upper_quantile": 0.98,
                                        "metric_weight": [1, 1],
                                        "radius_or_number_of_bins": [32, 32]},
                    overwrite_directories=False,
                    anat_in=None,
                    anat_out=None,
                    func_in=None,
                    func_out=None,
                    fmap_in=None,
                    fmap_out=None
                    )
    return settings


def update_settings(settings, settings_file):
    """

    Parameters
    ----------
    settings: dict
            Settings object
    settings_file: str
            Path to the user defined settings file.

    Returns
    -------
    settings: dict
        Dictionary holding all settings.

    """
    # Load in user's setting file
    with open(settings_file, 'r') as json_file:
        new_settings = json.load(json_file)

    # Update relevant settings
    for item in new_settings.keys():
        # Settings other than the ANTs ones
        if not item == 'ants_reg_params':
            settings[item] = new_settings[item]
        # ANTs specific settings
        elif item == 'ants_reg_params':
            for ants_item in new_settings['ants_reg_params']:
                settings['ants_reg_params'][ants_item] = new_settings['ants_reg_params'][ants_item]

    # Check that the set number of threads isn't bigger than the number available
    if settings['num_threads'] > cpu_count():
        settings['num_threads'] = cpu_count()

    # Handle run numbers for functional data
    if type(settings['run_numbers']) is int:
        settings['number_of_runs'] = 1
    elif type(settings['run_numbers']) is list:
        settings['number_of_runs'] = len(settings['run_numbers'])
        # If there's just one run number in a list then turn it into an integer for ease of use
        if settings['number_of_runs'] == 1:
            settings['run_numbers'] = settings['run_numbers'][0]

    return settings


def check_bids_style(root_dir):
    """
    Check that the root directory that the user specifies follows the required BIDS layout (i.e., has rawdata folder
    and a derivatives folder).

    Parameters
    ----------
    root_dir: str
            Path to root directory for data.

    Returns
    -------
    None

    """
    if not path.isdir(path.join(root_dir, 'rawdata')):
        raise IsADirectoryError('The root directory does not contain a rawdata folder')
    if not path.isdir(path.join(root_dir, 'derivatives')):
        raise IsADirectoryError('The root directory does not contain a derivatives folder')


def create_output_directories(subject, settings):
    # Create output directories, overwriting any that are present
    if settings['overwrite_directories']:
        print('Overwriting any existing directories.')
        # Create overall output directory
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name']))
        # Create subject output directory
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject))
        # If there are sessions
        if settings['session_number']:
            # Create session output directory
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f"ses-{settings['session_number']:02d}"))
        return

    # Create directories, not overwriting any
    # Create overall output directory
    if not path.isdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'])):
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name']))
    else:
        print('Output directory already exists. Not overwriting.')
    # Create subject output directory
    if not path.isdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject)):
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject))
    else:
        print('Subject output directory already exists. Not overwriting.')
    # If there are sessions
    if settings['session_number']:
        if not path.isdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f"ses-{settings['session_number']:02d}")):
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f"ses-{settings['session_number']:02d}"))
        else:
            print('Session directory already exists. Not overwriting.')


def create_directory(settings, dir_type):
    # Create directories, overwriting any existing ones
    if settings['overwrite_directories']:
        mkdir(settings[f'{dir_type}_out'])
        return

    # Create directory, not overwriting
    if not path.isdir(settings[f'{dir_type}_out']):
        mkdir(settings[f'{dir_type}_out'])


def prepare_directories(subject, settings):
    """
    Complete all steps to prepare the required directories and return their paths for subsequent use.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dict
            Dictionary holding all settings.

    Returns
    -------
    settings: dict
            Dictionary holding all settings.

    """
    # Define paths
    if settings['session_number']:
        # We set the session number in the directory here
        settings['anat_in'] = path.join(settings['root_dir'], 'rawdata', subject, f"ses-{settings['session_number']:02d}", 'anat')
        settings['func_in'] = path.join(settings['root_dir'], 'rawdata', subject, f"ses-{settings['session_number']:02d}", 'func')
        settings['fmap_in'] = path.join(settings['root_dir'], 'rawdata', subject, f"ses-{settings['session_number']:02d}", 'fmap')

        settings['anat_out'] = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f"ses-{settings['session_number']:02d}", 'anat')
        settings['func_out'] = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f"ses-{settings['session_number']:02d}", 'func')

    else:
        settings['anat_in'] = path.join(settings['root_dir'], 'rawdata', subject, 'anat')
        settings['func_in'] = path.join(settings['root_dir'], 'rawdata', subject, 'func')
        settings['fmap_in'] = path.join(settings['root_dir'], 'rawdata', subject, 'fmap')

        settings['anat_out'] = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'anat')
        settings['func_out'] = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'func')


    # Create root output directories
    create_output_directories(subject, settings)

    # Create anatomical directory, if requrired
    if settings['process_anat']:
        create_directory(settings, 'anat')

    # Create functional directory, if required
    if settings['process_func']:
        create_directory(settings, 'func')

    return settings


def check_template_file(template_name, resolution, desc=None, suffix=None):
    """
    Ensure that the template requested by the user is available from TemplateFlow. Downloads the template if it is
    available and isn't yet on the local system.

    Parameters
    ----------
    template_name: str
            Name of the template
    resolution: int
            Resolution of the template
    desc
    suffix

    Returns
    -------
    template_path: str
            Path to the downloaded template file

    """
    if desc:
        template_path = tflow.get(template_name,
                              resolution=resolution,
                              extension='nii.gz',
                              desc=desc)
    elif suffix:
        template_path = tflow.get(template_name,
                              resolution=resolution,
                              extension='nii.gz',
                              suffix=suffix,
                              desc=desc)
    return template_path
        

def check_thread_no(settings):
    """
    When the simultaneous processing of multiple functional runs is requested, this will ensure that the total number of
    threads does not exceed what is avaialable in the system.

    Parameters
    ----------
    settings: dict
            Dictionary holding all settings.

    Returns
    -------
        settings: dict
            Dictionary holding all settings.

    """

    # Check when processing multiple runs that the combined number isn't bigger than available
    if cpu_count()/settings['number_of_runs'] < settings['num_threads']:
        settings['num_threads'] = int(np.floor(cpu_count()/settings['number_of_runs']))

    return settings


