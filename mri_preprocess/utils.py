#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:58:25 2024

@author: niall
"""

import json
from os import path, mkdir
from templateflow import api as tflow


def initiate_settings():
    settings = dict(root_dir="",
                    output_dir_name="",
                    num_threads=1,
                    process_anat=True,
                    process_func=True,
                    template_name="MNI152NLin2009cAsym",
                    template_resolution=1,
                    T1_brain_extract_type="BET",
                    task_name="",
                    number_of_runs=None,
                    number_of_sessions=None,
                    drop_nonsteady_bold=True,
                    bold_ref_type="median",
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
                    overwrite_directories=False
                    )
    return settings


def update_settings(settings, settings_file):
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

    return settings

def check_bids_style(root_dir):
    if not path.isdir(path.join(root_dir, 'rawdata')):
        raise IsADirectoryError('The root directory does not contain a rawdata folder')
    if not path.isdir(path.join(root_dir, 'derivatives')):
        raise IsADirectoryError('The root directory does not contain a derivatives folder')


def create_output_dirs(subject, settings, session=None):
    # Directory structure when there is a session name
    if settings['overwrite_directories']:
        print('Overwriting any existing directories.')
        # Create overall output directory
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name']))
        # Create subject output directory
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject))
        # If there are sessions
        if session:
            # Create session output directory
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, session))
            # Create modality sepcific directories
            if settings['process_anat']:
                mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, session, 'anat'))
            if settings['process_func']:
                mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, session, 'func'))
            return
        # If there are no sessions
        if settings['process_anat']:
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'anat'))
        if settings['process_func']:
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'func'))
        return

    # Directory structure when there is no session name
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
    # Create anatomical output directory
    if settings['process_anat']:
        if not path.isdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'anat')):
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'anat'))
        else:
            print('Anatomical output directory already exists. Not overwriting')
    # Create functional output directory
    if settings['process_func']:
        if not path.isdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'func')):
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'func'))
        else:
            print('Functional output directory already exists. Not overwriting')


def define_directories(subject, settings, session=None):
    if session:
        anat_in = path.join(settings['root_dir'], 'rawdata', subject, session, 'anat')
        func_in = path.join(settings['root_dir'], 'rawdata', subject, session, 'func')

        anat_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, session, 'anat')
        func_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, session, 'func')

    else:
        anat_in = path.join(settings['root_dir'], 'rawdata', subject, 'anat')
        func_in = path.join(settings['root_dir'], 'rawdata', subject, 'func')

        anat_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'anat')
        func_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'func')

    return anat_in, func_in, anat_out, func_out


def prepare_directories(subject, settings, session=None):
    # Create directories and define relevant paths
    if session:
        create_output_dirs(subject, settings, session=session)
        anat_in, func_in, anat_out, func_out = define_directories(subject, settings, session=session)
    else:
        create_output_dirs(subject, settings)
        anat_in, func_in, anat_out, func_out = define_directories(subject, settings)

    # Set required paths in settings object
    if settings['process_anat']:
        settings['anat_in'] = anat_in
        settings['anat_out'] = anat_out
    if settings['process_func']:
        settings['func_in'] = func_in
        settings['func_out'] = func_out

    return settings


def check_template_file(template_name, resolution, desc=None, suffix=None):
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
        
    