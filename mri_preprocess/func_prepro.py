#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:06:37 2024

@author: niall
"""

import subprocess
import nibabel as nib
import numpy as np
from nipype.interfaces import fsl, afni
from nipype.algorithms.confounds import NonSteadyStateDetector
from os import path
from shutil import copyfile


def reorient_bold_to_standard(subject, settings, run_number=None):
    if run_number:
        in_file = path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{str(run_number).zfill(2)}_bold.nii.gz")
        out_root = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{str(run_number).zfill(2)}_")
    else:
        in_file = path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_bold.nii.gz")
        out_root = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_")
    # Calculate the transform between original orientation and standard orientation
    with open(path.join(out_root, 'orig2std.mat'), 'w') as fd:
        subprocess.run(['fslreorient2std', in_file], stdout=fd)
    # Calculate the inverse of that transform
    subprocess.run(['convert_xfm',
                    '-inverse',
                    path.join(out_root, 'orig2std.mat'),
                    '-omat',
                    path.join(out_root, 'std2orig.mat')])
    # Create new image
    subprocess.run(['fslreorient2std', in_file,
                    path.join(out_root, 'preproc-bold.nii.gz')])


def detect_nonsteady(subject, settings, run_number=None):
    # Run if the filename has a run number in it
    if run_number:
        steady = NonSteadyStateDetector(in_file=path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{str(run_number).zfill(2)}_bold.nii.gz"))
        steady.run()
        n_vols = str(steady.outputs)
        n_vols = n_vols.split('=')[-1]
        return int(n_vols)
    # Run if there is no run number in the filename
    steady = NonSteadyStateDetector(in_file=path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_bold.nii.gz"))
    steady.run()
    n_vols = str(steady.outputs)
    n_vols = n_vols.split('=')[-1]
    return int(n_vols)
    

def nonsteady_reference(subject, settings, n_vols, run_number=None):
    # Run if the filename has a run number in it
    if run_number:
        in_img = nib.load(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{str(run_number).zfill(2)}_bold.nii.gz"))
        if n_vols == 1:
            in_data = in_img.get_fdata()[:, :, :, 0]
            out_img = nib.Nifti1Image(in_data, in_img.affine)
        elif n_vols > 1:
            in_data = in_img.get_fdata()[:, :, :, 0:n_vols]
            out_img = nib.Nifti1Image(np.mean(in_data, axis=-1), in_img.affine)
        out_img.to_filename(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{str(run_number).zfill(2)}_bold-reference.nii.gz"))
        return
    # Run if there is no run number in the filename
    else:
        in_img = nib.load(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_bold.nii.gz"))
        if n_vols == 1:
            in_data = in_img.get_fdata()[:, :, :, 0]
            out_img = nib.Nifti1Image(in_data, in_img.affine)
        elif n_vols > 1:
            in_data = in_img.get_fdata()[:, :, :, 0:n_vols]
            out_img = nib.Nifti1Image(np.mean(in_data, axis=-1), in_img.affine)
        out_img.to_filename(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_bold-reference.nii.gz"))
        
        

def external_reference(subject, root_dir, out_dir, task_name, run_number):
    
    
    
    
def prepare_bold(subject, root_dir, out_dir, task_name, run_number):
    
    # Reorient to standard
    
    
    
    
    
                                        