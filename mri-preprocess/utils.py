#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:58:25 2024

@author: niall
"""

from os import path, mkdir
from templateflow import api as tflow


def check_bids_style(root_dir):
    if not path.isdir(path.join(root_dir, 'rawdata')):
        raise IsADirectoryError('The root directory does not contain a rawdata folder')
    if not path.isdir(path.join(root_dir, 'derivatives')):
        raise IsADirectoryError('The root directory does not contain a derivatives folder')


def create_output_dirs(subject, root_dir, out_name, process_anat, process_func):
    # Create overall output directory
    if not path.isdir(path.join(root_dir, 'derivatives', out_name)):
        mkdir(path.join(root_dir, 'derivatives', out_name))
    else:
        print('Output directory already exists. Not overwriting.')
    # Create subject output directory
    if not path.isdir(path.join(root_dir, 'derivatives', out_name, subject)):
        mkdir(path.join(root_dir, 'derivatives', out_name, subject))
    else:
        print('Subject output directory already exists. Not overwriting.')
    # Create anatomical output directory
    if process_anat:
        if not path.isdir(path.join(root_dir, 'derivatives', out_name, subject, 'anat')):
            mkdir(path.join(root_dir, 'derivatives', out_name, subject, 'anat'))
        else:
            print('Anatomical output directory already exists. Not overwriting')
    # Create functional output directory
    if process_func:
        if not path.isdir(path.join(root_dir, 'derivatives', out_name, subject, 'func')):
            mkdir(path.join(root_dir, 'derivatives', out_name, subject, 'func'))
        else:
            print('Functional output directory already exists. Not overwriting')
        
        
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
        
    