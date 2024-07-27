#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for preprocessing T1 anatomical data.

"""


import subprocess
import nibabel as nib
import numpy as np
from os import path
from nipype.interfaces import ants
from nipype.interfaces import fsl
from shutil import copyfile, move
from mri_preprocess import utils


def reorient_t1_to_standard(subject, settings):
    """
    Rearanges the anatomical data so that it follows the standard orientation used by FSL.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    t1_file = path.join(settings['anat_in'], f'{subject}_T1w.nii.gz')
    # Make a copy of the original file for reference
    copyfile(t1_file,
             path.join(settings['anat_out'], f'{subject}_T1w_orig.nii.gz'))
    # Calculate the transform between original orientation and standard orientation
    with open(path.join(settings['anat_out'], f'{subject}_T1w_orig2std.mat'), 'w') as fd:
        subprocess.run(['fslreorient2std', t1_file], stdout=fd)
    # Calculate the inverse of that transform
    subprocess.run(['convert_xfm',
                    '-inverse',
                    path.join(settings['anat_out'], f'{subject}_T1w_orig2std.mat'),
                    '-omat',
                    path.join(settings['anat_out'], f'{subject}_T1w_std2orig.mat')])
    # Create new image
    subprocess.run(['fslreorient2std', t1_file,
                    path.join(settings['anat_out'], f'{subject}_T1w_std.nii.gz')])
    
    

def reduce_fov(subject, settings):
    """
    Use FSL's robustfov to remove excess neck area from the image.

    Does this with the default settings so may not be suitable for non-standard FoVs or small brains (e.g., children).

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    subprocess.run(['robustfov',
                    '-i',
                    path.join(settings['anat_out'], f'{subject}_T1w_std.nii.gz'),
                    '-m',
                    path.join(settings['anat_out'], f'{subject}_T1w_std_fullfov2crop.mat'),
                    '-r',
                    path.join(settings['anat_out'], f'{subject}_T1w_std_crop.nii.gz')])
    
    
def create_rough_mask(subject, settings):
    """
    Helper function to make a temporary brain mask.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    bet = fsl.BET(in_file = path.join(settings['anat_out'], f'{subject}_T1w_std_crop.nii.gz'),
                  frac = 0.1,
                  mask = True,
                  no_output = True,
                  out_file = path.join(settings['anat_out'], 'temp_brain.nii.gz'))
    bet.run()

    

def bias_correct(subject, settings):
    """
    Run N4 bias correction on the anatomical image.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    t1_in = path.join(settings['anat_out'], f'{subject}_T1w_std_crop.nii.gz')
    mask_in = path.join(settings['anat_out'], 'temp_brain_mask.nii.gz')
    # Set up the bias correction function
    bias_corr = ants.N4BiasFieldCorrection(dimension = 3,
                                           mask_image = mask_in,
                                           output_image = path.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                                           bias_image = path.join(settings['anat_out'], f'{subject}_T1w_biasfield.nii.gz'),
                                           num_threads = settings['num_threads'])
                                           
    # Ensure there are no negative values within the brain area
    in_img = nib.load(t1_in)
    mask_img = nib.load(mask_in).get_fdata()
    img_min = np.min(in_img.get_fdata()[mask_img == 1])
    if img_min < 0:
        # If nonnegative values are seen get rid of them and run the bias correction
        out_img = in_img.get_fdata() + (img_min*-1)
        out_img = nib.Nifti1Image(out_img, in_img.affine)
        out_img.to_filename(path.join(settings['anat_out'], 'temp_nonneg.nii.gz'))
        bias_corr.inputs.input_image=path.join(settings['anat_out'], 'temp_nonneg.nii.gz')
        bias_corr.run()
    else:
        bias_corr.inputs.input_image = t1_in
        bias_corr.run()

    
def align_to_template(subject, settings):
    """
    Aligns the anatomical image to the relevant standard-space template with ANTs.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    # Get path to template image through templateflow
    template_path = utils.check_template_file(settings['template_name'], settings['template_resolution'], desc=None, suffix='T1w')
    # Set up registration parameters
    ants_reg = ants.Registration(fixed_image = template_path,
                                 moving_image = path.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                                 output_transform_prefix = path.join(settings['anat_out'], f"{subject}_T1w_to_{settings['template_name']}_"),
                                 output_warped_image = path.join(settings['anat_out'], f"{subject}_T1w_{settings['template_name']}_res-{str(settings['template_resolution']).zfill(2)}.nii.gz"),
                                 dimension = 3,
                                 num_threads = settings['num_threads'],
                                 metric = settings['ants_reg_params']['metric'],
                                 metric_weight = settings['ants_reg_params']['metric_weight'],
                                 shrink_factors = settings['ants_reg_params']['shrink_factors'],
                                 smoothing_sigmas = settings['ants_reg_params']['smoothing_sigmas'],
                                 number_of_iterations = settings['ants_reg_params']['number_of_iterations'],
                                 winsorize_lower_quantile = settings['ants_reg_params']['winsorize_lower_quantile'],
                                 winsorize_upper_quantile = settings['ants_reg_params']['winsorize_upper_quantile'],
                                 transforms = settings['ants_reg_params']['transforms'],
                                 radius_or_number_of_bins = settings['ants_reg_params']['radius_or_number_of_bins'],
                                 transform_parameters = [(2.0,), (0.25, 3.0, 0.0)])
    ants_reg.run()


def brain_extract(subject, settings):
    """
    Skull strip the anatomical image and create a brain mask.

    Can be done with either BET or ANTs.

    If the required tissue probability templates aren't available for ANTs then it will switch to using BET.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    if settings['T1_brain_extract_type'] == 'ANTs':
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
                                                anatomical_image=path.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                                                brain_template=template_path,
                                                brain_probability_mask=brain_prob_path,
                                                num_threads=settings['num_threads'],
                                                out_prefix=path.join(settings['anat_out'], f'{subject}_'))
    
            ants_extract.run()

            # Rename files 
            move(path.join(settings['anat_out'], f'{subject}_BrainExtractionBrain.nii.gz'),
                 path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'))
            
            move(path.join(settings['anat_out'], f'{subject}_BrainExtractionMask.nii.gz'),
                 path.join(settings['anat_out'], f'{subject}_T1w_brain_mask.nii.gz'))
            
            return

        else:
            print('No tissue probability map available. Switing to BET for brain extraction.')

    # Set up BET extraction
    bet_extract = fsl.BET(in_file=path.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                          mask=True,
                          skull=True,
                          reduce_bias=False,
                          out_file=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'))
    
    bet_extract.run()
    
            
def tissue_segment(subject, settings):
    """
    Segment the anatomical image into different tissues (grey matter, white matter, and CSF).

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    fast = fsl.FAST(in_files=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'),
                    img_type=1,
                    no_bias=True,
                    no_pve=False,
                    probability_maps=False)
                    #out_basename=path.join(root_dir, 'derivatives', out_dir, subject, 'anat', f'{subject}_T1w_brain.nii.gz'))
    
    fast.run()
    
def run_anat_preprocess(subject, settings):
    """
    Run each of the anatomical preprocessing steps.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------
    None

    """
    # Reorient image to standard
    reorient_t1_to_standard(subject, settings)
    # Reduce field of view
    reduce_fov(subject, settings)
    # Biasfield correct image
    create_rough_mask(subject, settings)
    bias_correct(subject, settings)
    # Align to template
    align_to_template(subject, settings)
    # Brain extraction
    brain_extract(subject, settings)
    # Tissue segmentation
    tissue_segment(subject, settings)

