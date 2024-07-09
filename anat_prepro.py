#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:20:27 2024

@author: niall
"""


import subprocess
import nibabel as nib
import numpy as np
from os import path
from nipype.interfaces import ants
from nipype.interfaces import fsl
from shutil import copyfile, move
from mri_preprocess import utils


def reorient_t1_to_standard(subject, root_dir, out_dir):
    t1_file = path.join(root_dir, 'rawdata', subject, 'anat', f'{subject}_T1w.nii.gz')
    # Make a copy of the original file for reference
    copyfile(t1_file,
             path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_orig.nii.gz'))
    # Calculate the transform between original orientation and standard orientation
    with open(path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_orig2std.mat'), 'w') as fd:
        subprocess.run(['fslreorient2std', t1_file], stdout=fd)
    # Calculate the inverse of that transform
    subprocess.run(['convert_xfm',
                    '-inverse',
                    path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_orig2std.mat'),
                    '-omat',
                    path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std2orig.mat')])
    # Create new image
    subprocess.run(['fslreorient2std', t1_file,
                    path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std.nii.gz')])
    
    

def reduce_fov(subject, root_dir, out_dir):
    subprocess.run(['robustfov',
                    '-i',
                    path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std.nii.gz'),
                    '-m',
                    path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_fullfov2crop.mat'),
                    '-r',
                    path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_crop.nii.gz')])
    
    
def create_rough_mask(subject, root_dir, out_dir):
    bet = fsl.BET(in_file = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_crop.nii.gz'),
                  frac = 0.1,
                  mask = True,
                  no_output = True,
                  out_file = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', 'temp_brain.nii.gz'))
    bet.run()

    

def bias_correct(subject, root_dir, out_dir, num_threads):
    
    t1_in = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_crop.nii.gz')
    mask_in = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', 'temp_brain_mask.nii.gz')
    # Set up the bias correction function
    bias_corr = ants.N4BiasFieldCorrection(dimension = 3,
                                           mask_image = mask_in,
                                           output_image = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                                           bias_image = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_biasfield.nii.gz'),
                                           num_threads = num_threads)
                                           
    # Ensure there are no negative values within the brain area
    in_img = nib.load(t1_in)
    mask_img = nib.load(mask_in).get_fdata()
    img_min = np.min(in_img.get_fdata()[mask_img==1])
    if img_min < 0:
        # If nonnegative values are seen get rid of them and run the bias correction
        out_img = in_img.get_fdata() + (img_min*-1)
        out_img = nib.Nifti1Image(out_img, in_img.affine)
        out_img.to_filename(path.join(root_dir, 'derivatives', out_dir, subject, 'anat', 'temp_nonneg.nii.gz'))
        bias_corr.inputs.input_image=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', 'temp_nonneg.nii.gz')
        bias_corr.run()
    else:
        bias_corr.inputs.input_image=t1_in
        bias_corr.run()

    
def align_to_template(subject, root_dir, out_dir, template_name, template_resolution, num_threads, ants_reg_params):
    # Get path to template image through templateflow
    template_path = utils.check_template_file(template_name, template_resolution, desc=None, suffix='T1w')
    # Set up registration parameters
    ants_reg = ants.Registration(fixed_image = template_path,
                                 moving_image = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                                 output_transform_prefix = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_to_{template_name}_'),
                                 output_warped_image = path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_{template_name}_res-{str(template_resolution).zfill(2)}.nii.gz'),
                                 dimension = 3,
                                 num_threads = num_threads,
                                 metric = ants_reg_params['metric'],
                                 metric_weight = ants_reg_params['metric_weight'],
                                 shrink_factors = ants_reg_params['shrink_factors'],
                                 smoothing_sigmas = ants_reg_params['smoothing_sigmas'],
                                 number_of_iterations = ants_reg_params['number_of_iterations'],
                                 winsorize_lower_quantile = ants_reg_params['winsorize_lower_quantile'],
                                 winsorize_upper_quantile = ants_reg_params['winsorize_upper_quantile'],
                                 transforms = ants_reg_params['transforms'],
                                 radius_or_number_of_bins = ants_reg_params['radius_or_number_of_bins'],
                                 transform_parameters = [(2.0,), (0.25, 3.0, 0.0)])
    ants_reg.run()


def brain_extract(subject, root_dir, out_dir, extract_type, num_threads):
    if extract_type == 'ANTs':
        # Get the necessary template and tissue probability map 
        template_path = str(utils.check_template_file('MNI152NLin2009cAsym', 1, desc=None, suffix='T1w'))
        temp = utils.check_template_file('MNI152NLin2009cAsym', 1, desc='brain', suffix=None)
        prob_available = False
        for item in temp:
            if 'probseg' in str(item):
                brain_prob_path = str(item)
                prob_available = True
        # Check the tissue map is available - if not switch to BET
        if prob_available:
            # Set up ANTs extraction
            ants_extract = ants.BrainExtraction(dimension=3,
                                                anatomical_image=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                                                brain_template=template_path,
                                                brain_probability_mask=brain_prob_path,
                                                num_threads=num_threads,
                                                out_prefix=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_'))
    
            ants_extract.run()

            # Rename files 
            move(path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_BrainExtractionBrain.nii.gz'),
                 path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_brain.nii.gz'))
            
            move(path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_BrainExtractionMask.nii.gz'),
                 path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_brain_mask.nii.gz'))
            
            return
        else:
            print('No tissue probability map available. Switing to BET for brain extraction.')
            move()
            
    # Set up BET extraction
    bet_extract = fsl.BET(in_file=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                          mask=True,
                          skull=True,
                          reduce_bias=False,
                          out_file=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_brain.nii.gz'))
    
    bet_extract.run()
    
            
def tissue_segment(subject, root_dir, out_dir):
    fast = fsl.FAST(in_files=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_brain.nii.gz'),
                    img_type=1,
                    no_bias=True,
                    no_pve=False,
                    probability_maps=False)
                    #out_basename=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_brain.nii.gz'))
    
    fast.run()
    
    