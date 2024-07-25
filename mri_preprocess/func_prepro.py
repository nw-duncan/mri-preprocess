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
    settings['number_nonsteady_vols'] = detect_nonsteady(subject, settings, run_number)
    # Initiate number of usable volumes
    in_img = nib.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    settings['number_usable_vols'] = in_img.shape[-1]
    # Remove any non-steady volumes if required
    if settings['drop_nonsteady_vols']:
        out_img = nib.Nifti1Image(in_img.get_fdata()[:, :, :, settings['number_nonsteady_vols']], in_img.affine)
        out_img.to_filename(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
        settings['number_usable_vols'] = settings['number_usable_vols'] - settings['number_nonsteady_vols']
    # Reorient to standard
    reorient_bold_to_standard(subject, settings, run_number)
    # Brain extract
    brain_extract_bold(subject, settings, run_number)
    return settings

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


def create_tissue_masks(subject, settings):
    for pve, tissue in enumerate(['csf', 'gm', 'wm']):
        if not path.isfile(path.join(settings['anat_out'], f'{subject}_{tissue}seg.nii.gz')):
            # Threshold white matter probability at 50% and make binary mask
            thresh = fsl.Threshold(in_file=path.join(settings['anat_out'], f'{subject}_T1w_brain_pve_{pve}.nii.gz'),
                          thresh=0.5,
                          out_file=path.join(settings['anat_out'], f'{subject}_{tissue}-mask.nii.gz'))
            thresh.run()
            binarise = fsl.UnaryMaths(in_file=path.join(settings['anat_out'], f'{subject}_{tissue}-mask.nii.gz'),
                                      operation='bin',
                                      out_file=path.join(settings['anat_out'], f'{subject}_{tissue}-mask.nii.gz'))
            binarise.run()
        if not path.isfile(path.join(settings['anat_out'], f'{subject}_{tissue}-edge.nii.gz')):
            subprocess.run(['fslmaths',
                            path.join(settings['anat_out'], f'{subject}_{tissue}-mask.nii.gz'),
                            '-edge',
                            '-bin',
                            '-mas',
                            path.join(settings['anat_out'], f'{subject}_{tissue}-mask.nii.gz'),
                            path.join(settings['anat_out'], f'{subject}_{tissue}-edge.nii.gz')])


def align_to_anatomical(subject, settings, run_number):
    # Create the white matter segmenation needed for BBR
    create_tissue_masks(subject, settings)

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
                      wm_seg=path.join(settings['anat_out'], f'{subject}_wm-mask.nii.gz'),
                      dof=6,
                      schedule=os.environ['FSLDIR'] + '/etc/flirtsch/bbr.sch',
                      out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference_anat-space.nii.gz"),
                      out_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat.mat"))
    flirt.run()

    # Calculate inverse transform
    invert = fsl.ConvertXFM(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat.mat"),
                            invert_xfm=True,
                            out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_anat2bold.mat"))
    invert.run()

def anat_to_func(subject, settings, run_number):
    # T1w image to functional space
    flirt = fsl.FLIRT(in_file=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'),
                      reference=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                      in_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_anat2bold.mat"),
                      apply_xfm=True,
                      interp='trilinear',
                      out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_T1w_brain.nii.gz"))
    flirt.run()

    # Tissue masks
    flirt.inputs.interp = 'nearestneighbour'
    for tissue in ['wm', 'csf', 'gm']:
        flirt.inputs.in_file = path.join(settings['anat_out'], f'{subject}_{tissue}-mask.nii.gz')
        flirt.inputs.out_file = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_{tissue}-mask.nii.gz")
        flirt.run()

    # Erode masks
    erode = fsl.ErodeImage(kernel_shape='boxv',
                           kernel_size=3,
                           nan2zeros=True)
    for tissue in ['wm', 'csf', 'gm']:
        erode.inputs.in_file = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_{tissue}-mask.nii.gz")
        erode.inputs.out_file = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_{tissue}-mask-erode.nii.gz")
        erode.run()


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

def detrend_data(subject, settings, run_number):
    # Load in functional data
    in_img = nib.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    mask_img = nib.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz")).get_fdata()
    tcs = in_img.get_fdata()[mask_img == 1]
    tcs_mean = np.nanmean(tcs, axis=-1, keepdims=True)

    # Generate design matrix
    regressors = np.ones((settings['number_usable_vols'], 1))
    for i in range(settings['detrend_degree']):
        polynomial_func = np.polynomial.Legendre.basis(i + 1)
        value_array = np.linspace(-1, 1, settings['number_usable_vols'])
        regressors = np.hstack((regressors, polynomial_func(value_array)[:, np.newaxis]))

    np.savetxt(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_poly-detrend.csv"), regressors, delimiter=',')

    # Remove from data
    betas = np.linalg.pinv(regressors).dot(tcs.T)
    datahat = regressors[:, 1:].dot(betas[1:, ...]).T
    tcs_clean = (tcs - datahat) + tcs_mean

    # Make new image
    out_img = np.zeros(in_img.shape)
    out_img[mask_img == 1] = tcs_clean
    out_img = nib.Nifti1Image(out_img, in_img.affine)
    out_img.to_filename(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))

    # Load in head motion parameters
    hm = np.loadtxt(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement.par"))
    hm_mean = np.mean(hm, axis=0, keepdims=True)

    # Remove trend from head motion
    betas = np.linalg.pinv(regressors).dot(hm)
    datahat = regressors[:, 1:].dot(betas[1:, ...])
    hm_clean = (hm - datahat) + hm_mean

    np.savetxt(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement_detrend.par"), hm_clean)


def smooth_data(subject, settings, run_number, melodic_smooth=False):

    # Calculate brightness threshold
    temp = subprocess.run(['fslstats',
                           path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                           '-k',
                           path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                           '-p',
                           '50'],
                          stdout=subprocess.PIPE)
    img_median = float(temp.stdout.decode())

    brightness_threshold = 0.75 * img_median

    # Do smoothing
    smooth = fsl.SUSAN(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                       dimension=3,
                       brightness_threshold=brightness_threshold,
                       fwhm=5.0,
                       smoothed_file=path.join(settings['func_out'], 'temp.nii.gz'))

    if melodic_smooth:
        smooth.inputs.fwhm=float(5),
        smooth.inputs.smoothed_file=path.join(settings['func_out'], 'temp.nii.gz')
    else:
        smooth.inputs.fwhm=settings['smoothing_fwhm']
        smooth.inputs.smoothed_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_smooth-{settings['smoothing_fwhm']}mm.nii.gz")

    smooth.run()

    # Clean up voxels outside brain
    mask_img = fsl.ApplyMask(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_smooth-{settings['smoothing_fwhm']}mm.nii.gz"),
                             mask_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                             out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_smooth-{settings['smoothing_fwhm']}mm.nii.gz"))
    mask_img.run()

def run_melodic_ica(subject, settings, run_number):
    #  Apply some smoothing to the input image
    smooth_data(subject, settings, run_number, melodic_smooth=True)

    # Run MELODIC
    melodic = fsl.MELODIC(in_files=path.join(settings['func_out'], 'temp.nii.gz'),
                          approach='tica',
                          bg_image=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                          mask=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                          tr_sec=settings['bold_TR'],
                          no_bet=True,
                          report=True,
                          out_dir=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}.ica"))
    melodic.run()

    # Clean up the temporary smoothed image
    os.remove(path.join(settings['func_out'], 'temp.nii.gz'))


def run_func_preprocess(subject, settings, run_number):
    # Make run number into string
    run_number = f'{run_number:02d}'

    # Ensure the TR value is set
    if not settings['bold_TR']:
        settings = tr_from_json(subject, settings, run_number)

    # Prepare the image for preprocessing
    settings = initiate_preprocessed_image(subject, settings, run_number)

    # Create the bold reference image
    if settings['bold_reference_type'] == 'nonsteady':
        nonsteady_reference(subject, settings, settings['number_nonsteady_vols'], run_number)
    elif settings['bold_reference_type'] == 'median':
        median_reference(subject, settings, run_number)
    elif settings['bold_reference_type'] == 'external':
        external_reference(subject, settings,  run_number)

    # Align reference to anatomical
    align_to_anatomical(subject, settings, run_number)
    anat_to_func(subject, settings, run_number)

    # Estimate head motion
    estimate_head_motion(subject, settings, run_number)

    # Slice-time correct
    slicetime_correct(subject, settings, run_number)

    # Volume realignment
    volume_realign(subject, settings, run_number)

    # Zero values outside of brain
    apply_brain_mask(subject, settings, run_number)

    # Detrend data - optional
    if settings['detrend_functional']:
        detrend_data(subject, settings, run_number)

    # Smooth data - optional
    if settings['smoothing_fwhm']:
        smooth_data(subject, settings, run_number)
