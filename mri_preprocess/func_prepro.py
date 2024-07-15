#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for preprocessing BOLD functional data.

"""

import json
import os
import subprocess
import nibabel as nib
import numpy as np
from nipype.interfaces import fsl, afni
from nipype.algorithms.confounds import NonSteadyStateDetector, ComputeDVARS, FramewiseDisplacement
from os import path
from shutil import copyfile, move

def tr_from_json(subject, settings, run_number):
    with open(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.json"), 'r') as json_file:
        temp = json.load(json_file)
    settings['bold_TR'] = temp['RepetitionTime']
    return settings


def reorient_bold_to_standard(subject, settings, run_number):
    in_file = path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.nii.gz")
    out_root = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_")
    # Calculate the transform between original orientation and standard orientation
    with open(out_root + 'orig2std.mat', 'w') as fd:
        subprocess.run(['fslreorient2std', in_file], stdout=fd)
    # Calculate the inverse of that transform
    subprocess.run(['convert_xfm',
                    '-inverse',
                    out_root + 'orig2std.mat',
                    '-omat',
                    out_root + 'std2orig.mat'])
    # Create new image
    subprocess.run(['fslreorient2std', in_file,
                    out_root + 'bold-preproc.nii.gz'])


def detect_nonsteady(subject, settings, run_number):
    steady = NonSteadyStateDetector(in_file=path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.nii.gz"))
    temp = steady.run()
    n_vols = str(temp.outputs)
    n_vols = n_vols.split('=')[-1]
    return int(n_vols)

def brain_extract_bold(subject, settings, run_number):
    bet = fsl.BET(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                  functional=True,
                  mask=True,
                  out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    bet.run()
    # Rename mask
    move(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_mask.nii.gz"),
         path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"))

def initiate_preprocessed_image(subject, settings, run_number):
    # Create a copy of the BOLD input image.
    copyfile(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.nii.gz"),
             path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    # Detect non-steady state volumes
    nonsteady_vols = detect_nonsteady(subject, settings, run_number)
    # Remove any non-steady volumes if required
    if settings['drop_nonsteady_vols']:
        in_img = nib.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
        out_img = nib.Nifti1Image(in_img.get_fdata()[:, :, :, nonsteady_vols:], in_img.affine)
        out_img.to_filename(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    # Reorient to standard
    reorient_bold_to_standard(subject, settings, run_number)
    # Brain extract
    brain_extract_bold(subject, settings, run_number)
    return nonsteady_vols

def nonsteady_reference(subject, settings, n_vols, run_number):

        in_img = nib.load(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.nii.gz"))
        if n_vols == 1:
            in_data = in_img.get_fdata()[:, :, :, 0]
            out_img = nib.Nifti1Image(in_data, in_img.affine)
        elif n_vols > 1:
            in_data = in_img.get_fdata()[:, :, :, 0:n_vols]
            out_img = nib.Nifti1Image(np.mean(in_data, axis=-1), in_img.affine)
        out_img.to_filename(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"))


def external_reference(subject, settings, run_number):
    copyfile(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_{settings['reference_file']}.nii.gz"),
             path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"))


def median_reference(subject, settings, run_number):
    # Do an initial alignment of the image volumes
    mcflirt = fsl.MCFLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                          dof=6,
                          ref_vol=0,
                          save_mats=False,
                          save_plots=False,
                          stats_imgs=False,
                          out_file=path.join(settings['func_out'], 'temp-bold.nii.gz'))
    mcflirt.run()
    # Calculate median
    med_image = fsl.MedianImage(in_file=path.join(settings['func_out'], 'temp-bold.nii.gz'),
                                dimension='T',
                                nan2zeros=True,
                                out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"))
    med_image.run()
    # Delete temporary file
    os.remove(path.join(settings['func_out'], 'temp-bold.nii.gz'))


def create_wmseg(subject, settings):
    if not path.isfile(path.join(settings['anat_out'], f'{subject}_wmseg.nii.gz')):
        # Threshold white matter probability at 50% and make binary mask
        thresh = fsl.Threshold(in_file=path.join(settings['anat_out'], f'{subject}_T1w_brain_pve_2.nii.gz'),
                      thresh=0.5,
                      out_file=path.join(settings['anat_out'], f'{subject}_wmseg.nii.gz'))
        thresh.run()
        binarise = fsl.UnaryMaths(in_file=path.join(settings['anat_out'], f'{subject}_wmseg.nii.gz'),
                                  operation='bin',
                                  out_file=path.join(settings['anat_out'], f'{subject}_wmseg.nii.gz'))
        binarise.run()
    if not path.isfile(path.join(settings['anat_out'], f'{subject}_wmedge.nii.gz')):
        subprocess.run(['fslmaths',
                        path.join(settings['anat_out'], f'{subject}_wmseg.nii.gz'),
                        '-edge',
                        '-bin',
                        '-mas',
                        path.join(settings['anat_out'], f'{subject}_wmseg.nii.gz'),
                        path.join(settings['anat_out'], f'{subject}_wm-edge.nii.gz')])


def align_to_anatomical(subject, settings, run_number):
    # Create the white matter segmenation needed for BBR
    create_wmseg(subject, settings)

    # Do initial alignment
    flirt = fsl.FLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                      reference=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'),
                      dof=6,
                      out_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat_init.mat"),
                      out_file=path.join(settings['func_out'], 'temp.nii.gz'))
    flirt.run()

    # Clean up unnecessary image
    os.remove(path.join(settings['func_out'], 'temp.nii.gz'))

    # Do BBR registration
    flirt = fsl.FLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                      reference=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'),
                      in_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat_init.mat"),
                      cost='bbr',
                      wm_seg=path.join(settings['anat_out'], f'{subject}_wmseg.nii.gz'),
                      dof=6,
                      schedule=os.environ['FSLDIR'] + '/etc/flirtsch/bbr.sch',
                      out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference_anat-space.nii.gz"),
                      out_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat.mat"))
    flirt.run()

    # Calculate inverse transform
    invert = fsl.ConvertXFM(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat.mat"),
                            invert_xfm=True,
                            out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_anat2bold.mat"))


def estimate_head_motion(subject, settings, run_number):
    mcflirt = fsl.MCFLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                          ref_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                          dof=6,
                          interpolation='nn',
                          save_mats=False,
                          save_plots=True,
                          save_rms=True,
                          stats_imgs=False,
                          out_file=path.join(settings['func_out'], 'temp.nii.gz'))
    mcflirt.run()

    # Tidy filenames
    move(path.join(settings['func_out'], "temp.nii.gz.par"),
                   path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement.par"))
    move(path.join(settings['func_out'], "temp.nii.gz_abs.rms"),
                   path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-abs-RMS.csv"))
    move(path.join(settings['func_out'], "temp.nii.gz_rel.rms"),
                   path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-rel-RMS.csv"))
    os.remove(path.join(settings['func_out'], "temp.nii.gz_rel_mean.rms"))
    os.remove(path.join(settings['func_out'], "temp.nii.gz_abs_mean.rms"))
    os.remove(path.join(settings['func_out'], "temp.nii.gz"))
    if path.isdir(path.join(settings['func_out'], "temp.nii.gz.mat")):
        os.rmdir(path.join(settings['func_out'], "temp.nii.gz.mat"))


    # Calculate head motion parameters
    # dvars = ComputeDVARS(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
    #                      in_mask=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
    #                      series_tr=settings['bold_TR'],
    #                      out_std=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-DVARS.csv"))
    # dvars.run()

    framewise = FramewiseDisplacement(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement.par"),
                                      parameter_source='FSL',
                                      series_tr=settings['bold_TR'],
                                      out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-FD.csv"))
    framewise.run()

def slicetime_correct(subject, settings, run_number):

    if path.isfile(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.json")):
        with open(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.json"), 'r') as json_file:
            temp = json.load(json_file)
        settings['bold_TR'] = temp['RepetitionTime']
        slice_times = temp['SliceTiming']
        if 'SliceEncodingDirection' in temp.keys():
            slice_encoding_direction = temp['SliceEncodingDirection']
        else:
            slice_encoding_direction = settings['slice_encoding_direction']
    else:
        slice_encoding_direction = settings['slice_encoding_direction']
        ### Need to finish this to work with data where the BIDs sidecar isn't available

    first, last = min(slice_times), max(slice_times)
    frac = settings['slice_time_ref']
    tzero = np.round(first + frac * (last - first), 3)

    tshift = afni.TShift(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                         tzero=tzero,
                         tr=str(settings['bold_TR']),
                         slice_timing=slice_times,
                         num_threads=settings['num_threads'],
                         slice_encoding_direction=slice_encoding_direction,
                         interp='Fourier',  # Use this???
                         out_file=path.join(settings['func_out'], "temp.nii.gz"))  # ANFI won't overwrite an existing file
    tshift.run()

    move(path.join(settings['func_out'], "temp.nii.gz"),
         path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    os.remove(path.join(os.getcwd(), 'slice_timing.1D'))


def volume_realign(subject, settings, run_number):
    mcflirt = fsl.MCFLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                          ref_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                          dof=6,
                          interpolation='sinc',
                          save_mats=False,
                          save_plots=False,
                          save_rms=False,
                          stats_imgs=False,
                          out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    mcflirt.run()


def apply_brain_mask(subject, settings, run_number):

    mask_img = fsl.ApplyMask(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                             mask_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                             out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))

    mask_img.run()


def run_func_preprocess(subject, settings, run_number):
    # Make run number into string
    run_number = str(run_number).zfill(2)

    # Ensure the TR value is set
    if not settings['bold_TR']:
        settings = tr_from_json(subject, settings, run_number)

    # Prepare the image for preprocessing
    nonsteady_vols = initiate_preprocessed_image(subject, settings, run_number)

    # Create the bold reference image
    if settings['bold_reference_type'] == 'nonsteady':
        nonsteady_reference(subject, settings, nonsteady_vols, run_number)
    elif settings['bold_reference_type'] == 'median':
        median_reference(subject, settings, run_number)
    elif settings['bold_reference_type'] == 'external':
        external_reference(subject, settings,  run_number)

    # Align reference to anatomical
    align_to_anatomical(subject, settings, run_number)

    # Estimate head motion
    estimate_head_motion(subject, settings, run_number)

    # Slice-time correct
    slicetime_correct(subject, settings, run_number)

    # Volume realignment
    volume_realign(subject, settings, run_number)

    # Zero values outside of brain
    apply_brain_mask(subject, settings, run_number)
