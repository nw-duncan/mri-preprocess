#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:58:25 2024

@author: niall
"""

import json
import numpy as np
from os import path, mkdir
from templateflow import api as tflow


def initiate_settings():
    settings = dict(root_dir="",
                    output_dir_name="",
                    number_of_sessions=None,
                    num_threads=1,
                    process_anat=True,
                    process_func=True,
                    template_name="MNI152NLin2009cAsym",
                    template_resolution=1,
                    T1_brain_extract_type="BET",
                    task_name="",
                    number_func_runs=1,
                    bold_TR=None,
                    drop_nonsteady_vols=True,
                    bold_reference_type="median",
                    reference_image=None,
                    slice_time_ref=0.5,
                    slice_encoding_direction='k',
                    slice_order=None,
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
                    func_out=None
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


def create_output_dirs(subject, settings):
    # Create directories, overwriting any existing ones
    if settings['overwrite_directories']:
        print('Overwriting any existing directories.')
        # Create overall output directory
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name']))
        # Create subject output directory
        mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject))
        # If there are sessions
        if settings['number_of_sessions']:
            for ses in np.arange(1, settings['number_of_sessions']+1):
                # Create session output directory
                mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}'))
                # Create modality sepcific directories
                if settings['process_anat']:
                    mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}', 'anat'))
                if settings['process_func']:
                    mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}', 'func'))
            return
            # If there are no sessions
        if settings['process_anat']:
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'anat'))
        if settings['process_func']:
            mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'func'))
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
    if settings['number_of_sessions']:
        for ses in np.arange(1, settings['number_of_sessions'] + 1):
            if not path.isdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}')):
                mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}'))
            else:
                print('Session directory already exists. Not overwriting.')
            # Create anatomical output directory
            if settings['process_anat']:
                if not path.isdir(
                        path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}', 'anat')):
                    mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}', 'anat'))
                else:
                    print('Anatomical output directory already exists. Not overwriting')
            # Create functional output directory
            if settings['process_func']:
                if not path.isdir(
                        path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}', 'func')):
                    mkdir(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, f'ses-{str(ses).zfill(2)}', 'func'))
                else:
                    print('Functional output directory already exists. Not overwriting')
        return
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


def define_directories(subject, settings):
    if settings['number_of_sessions']:
        # We set it to the first session here - change the session name elsewhere
        anat_in = path.join(settings['root_dir'], 'rawdata', subject, 'ses-01', 'anat')
        func_in = path.join(settings['root_dir'], 'rawdata', subject, 'ses-01', 'func')

        anat_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'ses-01', 'anat')
        func_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'ses-01', 'func')

    else:
        anat_in = path.join(settings['root_dir'], 'rawdata', subject, 'anat')
        func_in = path.join(settings['root_dir'], 'rawdata', subject, 'func')

        anat_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'anat')
        func_out = path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], subject, 'func')

    return anat_in, func_in, anat_out, func_out


def prepare_directories(subject, settings):
    # Create directories and define relevant paths
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
        
    