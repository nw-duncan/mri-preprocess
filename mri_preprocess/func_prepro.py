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
import pandas as pd
from nipype.interfaces import fsl, afni
from nipype.algorithms.confounds import NonSteadyStateDetector, FramewiseDisplacement
from os import path
from shutil import copyfile, move
from mri_preprocess.plotting import create_functional_report


def tr_from_json(subject, settings, run_number):
    """
    Updates the TR for the run based on the value from the sidecar json file (where available).

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    Updated settings object (dict)

    """
    with open(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.json"), 'r') as json_file:
        temp = json.load(json_file)
    settings['bold_TR'] = temp['RepetitionTime']
    return settings


def fmap_info_from_json(subject, settings):
    """
    Gets necessary information from the json sidecar file (where available) to calculate fieldmaps.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------

    """
    if settings['scanner_manufacturer'] is 'Siemens':
        with open(path.join(settings['fmap_in'], f"{subject}_phasediff.json"), 'r') as json_file:
            temp = json.load(json_file)
        fmap_info = {'te1': temp['EchoTime1'],
                     'te2': temp['EchoTime2'],
                     'dwell_time': temp['DwellTime']}
        fmap_info['te_diff'] = fmap_info['te1'] - fmap_info['te2']
        fmap_info['dwell_to_asym'] = fmap_info['dwell_time'] / fmap_info['te_diff']
        return fmap_info
    elif settings['scanner_manufacturer'] is 'GE':
        ## Need to get a GE fieldmap to complete this!!
        with open(path.join(settings['fmap_in'], f"{subject}_phasediff.json"), 'r') as json_file:
            temp = json.load(json_file)
        fmap_info = {}
        return fmap_info
    else:
        print('Data comes from a scanner type for which fieldmaps cannot be processed')
        return {'no_valid_fieldmap': None}


def reorient_bold_to_standard(subject, settings, run_number):
    """
    Rearanges the functional data so that it follows the standard orientation used by FSL.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    in_file = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz")
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
    """
    Identifies non-steady state volumes at the start of the functional data.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------

    The number of non-steady volumes (int)

    """
    steady = NonSteadyStateDetector(in_file=path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.nii.gz"))
    temp = steady.run()
    n_vols = str(temp.outputs)
    n_vols = n_vols.split('=')[-1]
    return int(n_vols)


def brain_extract_bold(subject, settings, run_number):
    """
    Runs brain extraction on functional data using BET.

    A brain mask is produced as well as the brain extracted data.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    bet = fsl.BET(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                  functional=True,
                  mask=True,
                  out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
    bet.run()
    # Rename mask
    move(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_mask.nii.gz"),
         path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"))


def calc_dvars(subject, settings, run_number):
    """
    Create time series showing volumes that are likely corrupted by artifacts.

    Run this on the data before it's slice time and head motion corrected.

    Based on Tom Nichols' approach and the original Power (2012) paper.

    https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/scripts/fsl/StandardizedDVARS.pdf

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------

    """
    # Standardise the image to an overall value of 1000
    median = fsl.MedianImage(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                             nan2zeros=True,
                             out_file=path.join(settings['func_out'], f'temp_{run_number}-median.nii.gz'))
    median.run()

    stats = fsl.ImageStats(in_file=path.join(settings['func_out'], f'temp_{run_number}-median.nii.gz'),
                           mask_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                           op_string='-M')
    img_stat = stats.run()
    scaling_factor = 1000/img_stat.outputs.out_stat

    bin_maths = fsl.BinaryMaths(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                                operation='mul',
                                operand_value=scaling_factor,
                                out_file=path.join(settings['func_out'], f'temp_{run_number}.nii.gz'))
    bin_maths.run()

    # Calculate robust SD
    pct = fsl.PercentileImage(in_file=path.join(settings['func_out'], f'temp_{run_number}.nii.gz'),
                              nan2zeros=True,
                              perc=25,
                              out_file=path.join(settings['func_out'], f'temp_{run_number}-lq.nii.gz'))
    pct.run()

    pct.inputs.in_file = path.join(settings['func_out'], f'temp_{run_number}.nii.gz')
    pct.inputs.perc = 75
    pct.inputs.out_file = path.join(settings['func_out'], f'temp_{run_number}-uq.nii.gz')
    pct.run()

    subprocess.run(['fslmaths',
                    path.join(settings['func_out'], f'temp_{run_number}-uq.nii.gz'),
                    '-sub',
                    path.join(settings['func_out'], f'temp_{run_number}-lq.nii.gz'),
                    '-div',
                    '1.349',
                    path.join(settings['func_out'], f'temp_{run_number}-SD.nii.gz')])

    # Calculate AR1
    ar1 = fsl.AR1Image(in_file=path.join(settings['func_out'], f'temp_{run_number}.nii.gz'),
                       nan2zeros=True,
                       out_file=path.join(settings['func_out'], f'temp_{run_number}-AR.nii.gz'))
    ar1.run()

    # Calculate predicted SD
    subprocess.run(['fslmaths',
                    path.join(settings['func_out'], f'temp_{run_number}-AR.nii.gz'),
                    '-mul', '-1',
                    '-add', '1',
                    '-mul', '2',
                    '-sqrt',
                    '-mul', path.join(settings['func_out'], f'temp_{run_number}-SD.nii.gz'),
                    path.join(settings['func_out'], f'temp_{run_number}-diffSDhat.nii.gz')])

    stats.inputs.in_file = path.join(settings['func_out'], f'temp_{run_number}-diffSDhat.nii.gz')
    img_stat = stats.run()
    pred_sd = img_stat.outputs.out_stat

    # Prepare images for temporal difference
    roi = fsl.ExtractROI(in_file=path.join(settings['func_out'], f'temp_{run_number}.nii.gz'),
                         t_min=0,
                         t_size=settings['number_usable_vols']-1,
                         roi_file=path.join(settings['func_out'], f'temp_{run_number}-0.nii.gz'))
    roi.run()

    roi.inputs.t_min = 1
    roi.inputs.t_size = settings['number_usable_vols']-1
    roi.inputs.roi_file = path.join(settings['func_out'], f'temp_{run_number}-1.nii.gz')
    roi.run()

    # Calculate DVARS
    subprocess.run(['fslmaths',
                    path.join(settings['func_out'], f'temp_{run_number}-0.nii.gz'),
                    '-sub',
                    path.join(settings['func_out'], f'temp_{run_number}-1.nii.gz'),
                    '-sqr',
                    path.join(settings['func_out'], f'temp_{run_number}-diffSq.nii.gz')])

    stats.inputs.in_file = path.join(settings['func_out'], f'temp_{run_number}-diffSq.nii.gz')
    stats.inputs.split_4d = True
    img_stat = stats.run()
    dvars = np.sqrt(img_stat.outputs.out_stat)

    # Calculate standardised DVARS
    subprocess.run(['fslmaths',
                    path.join(settings['func_out'], f'temp_{run_number}-0.nii.gz'),
                    '-sub',
                    path.join(settings['func_out'], f'temp_{run_number}-1.nii.gz'),
                    '-div',
                    path.join(settings['func_out'], f'temp_{run_number}-diffSDhat.nii.gz'),
                    '-sqr',
                    path.join(settings['func_out'], f'temp_{run_number}-diffSqStd.nii.gz')])

    stats.inputs.in_file = path.join(settings['func_out'], f'temp_{run_number}-diffSqStd.nii.gz')
    img_stat = stats.run()
    dvars_std = np.sqrt(img_stat.outputs.out_stat)/pred_sd

    # Save
    np.savetxt(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-DVARS.csv"),
               np.vstack((dvars, dvars_std)).T)

    # Clean up files
    for fname in ['', '-lq', '-uq', '-SD', '-AR', '-diffSDhat',
                  '-median', '-0', '-1', '-diffSq', '-diffSqStd']:
        os.remove(path.join(settings['func_out'], f'temp_{run_number}{fname}.nii.gz'))


def initiate_preprocessed_image(subject, settings, run_number):
    """
    Creates the output preprocessed functional image.

    Will remove any non-steady volumes from this if required. The settings object is updated with the number of such
    volumes and with the number of usable volumes.

    Image is reoriented to the FSL standard arrangement.

    Finally, brain extraction is applied to the data through BET.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    Updated settings object (dict)

    """
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
        out_img = nib.Nifti1Image(in_img.get_fdata()[:, :, :, settings['number_nonsteady_vols']:], in_img.affine)
        out_img.to_filename(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
        settings['number_usable_vols'] = settings['number_usable_vols'] - settings['number_nonsteady_vols']
    # Reorient to standard
    reorient_bold_to_standard(subject, settings, run_number)
    # Brain extract
    brain_extract_bold(subject, settings, run_number)
    # Calculate DVARS
    calc_dvars(subject, settings, run_number)
    return settings


def nonsteady_reference(subject, settings, n_vols, run_number):
    """
    Creates the functional data reference image by averaging the non-steady volumes.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    n_vols: int
            How many non-steady volumes there are.
    run_number: int
            Run number

    Returns
    -------
    None

    """
    in_img = nib.load(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.nii.gz"))
    if n_vols == 1:
        in_data = in_img.get_fdata()[:, :, :, 0]
        out_img = nib.Nifti1Image(in_data, in_img.affine)
    elif n_vols > 1:
        in_data = in_img.get_fdata()[:, :, :, 0:n_vols]
        out_img = nib.Nifti1Image(np.mean(in_data, axis=-1), in_img.affine)
    ref_file = path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz")
    out_img.to_filename(ref_file)
    subprocess.run(['fslreorient2std', ref_file, ref_file])


def external_reference(subject, settings, run_number):
    """
    Renames a specified functional reference image to fit with the naming convention of this tool.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    copyfile(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_{settings['reference_file']}.nii.gz"),
             path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"))


def median_reference(subject, settings, run_number):
    """
    Create a functional reference image by taking the temporal median of the data.

    First does volume realignment, then calculates the median.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
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


def align_to_anatomical(subject, settings, run_number):
    """
    Runs BBR alignment between functional and anatomical images in FLIRT.

    Also produces inverse transforms from anatomical to functional.


    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """

    # Do initial alignment
    flirt = fsl.FLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                      reference=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'),
                      dof=7,
                      out_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat_init.mat"),
                      out_file=path.join(settings['func_out'], f'temp_{run_number}.nii.gz'))
    flirt.run()

    # Clean up unnecessary image
    os.remove(path.join(settings['func_out'], f'temp_{run_number}.nii.gz'))

    # Do registration
    flirt = fsl.FLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                      reference=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'),
                      in_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat_init.mat"),
                      dof=7,
                      out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference_anat-space.nii.gz"),
                      out_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat.mat"))

    if settings['bold_to_anat_cost'] == 'bbr':
        flirt.inputs.cost = 'bbr'
        flirt.inputs.wm_seg = path.join(settings['anat_out'], f'{subject}_wm-mask.nii.gz')
        flirt.inputs.schedule = os.environ['FSLDIR'] + '/etc/flirtsch/bbr.sch'
    else:
        flirt.inputs.cost = settings['bold_to_anat_cost']

    flirt.run()

    # Calculate inverse transform
    invert = fsl.ConvertXFM(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold2anat.mat"),
                            invert_xfm=True,
                            out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_anat2bold.mat"))
    invert.run()


def anat_to_func(subject, settings, run_number):
    """
    Puts the brain extracted anatomical image into functional space. Useful as a background image for
    visualising data in functional space.

    Also puts tissue masks into functional space. Eroded versions of these are made with a three voxel cube kernel.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    # T1w image to functional space
    flirt = fsl.FLIRT(in_file=path.join(settings['anat_out'], f'{subject}_T1w_brain.nii.gz'),
                      reference=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                      in_matrix_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_anat2bold.mat"),
                      apply_xfm=True,
                      interp='trilinear',
                      out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_T1w-brain.nii.gz"))
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
    """
    Estimates head motion from volume realignment parameters. This is done on the original data that has not had slice
    time correction applied.

    DVARs and framewise displacement are calculated and saved as text files.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    mcflirt = fsl.MCFLIRT(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                          ref_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
                          dof=6,
                          interpolation='nn',
                          save_mats=False,
                          save_plots=True,
                          save_rms=True,
                          stats_imgs=False,
                          out_file=path.join(settings['func_out'], f'temp_{run_number}.nii.gz'))
    mcflirt.run()

    # Tidy filenames
    move(path.join(settings['func_out'], f"temp_{run_number}.nii.gz.par"),
                   path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement.par"))
    move(path.join(settings['func_out'], f"temp_{run_number}.nii.gz_abs.rms"),
                   path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-abs-RMS.csv"))
    move(path.join(settings['func_out'], f"temp_{run_number}.nii.gz_rel.rms"),
                   path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-rel-RMS.csv"))
    os.remove(path.join(settings['func_out'], f"temp_{run_number}.nii.gz_rel_mean.rms"))
    os.remove(path.join(settings['func_out'], f"temp_{run_number}.nii.gz_abs_mean.rms"))
    os.remove(path.join(settings['func_out'], f"temp_{run_number}.nii.gz"))
    if path.isdir(path.join(settings['func_out'], f"temp_{run_number}.nii.gz.mat")):
        os.rmdir(path.join(settings['func_out'], f"temp_{run_number}.nii.gz.mat"))

    framewise = FramewiseDisplacement(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement.par"),
                                      parameter_source='FSL',
                                      series_tr=settings['bold_TR'],
                                      out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-FD.csv"))
    framewise.run()


def slicetime_correct(subject, settings, run_number):
    """
    Run slice time correction with AFNI Tshift when a sidecar json is available.

    If no sidecar json is available, slicetime correction is run with FSL's slicetimer. TR and slice acquisition
    order must be set in the settings file.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    if path.isfile(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.json")):
        with open(path.join(settings['func_in'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold.json"), 'r') as json_file:
            temp = json.load(json_file)
        settings['bold_TR'] = temp['RepetitionTime']
        slice_times = temp['SliceTiming']
        if 'SliceEncodingDirection' in temp.keys():
            slice_encoding_direction = temp['SliceEncodingDirection']
        else:
            slice_encoding_direction = settings['slice_encoding_direction']

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
                             out_file=path.join(settings['func_out'], f"temp_{run_number}.nii.gz"))  # ANFI won't overwrite an existing file
        tshift.run()

        move(path.join(settings['func_out'], f"temp_{run_number}.nii.gz"),
             path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))
        os.remove(path.join(os.getcwd(), 'slice_timing.1D'))

    else:
        slicetime = fsl.SliceTimer(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                                   time_repetition=settings['bold_TR'],
                                   out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))

        if not settings['slice_acquisition_order']:
            print('Insufficient information to run slice-time correction. Continuing without this step')
            return
        # Set the slice acquisition order
        if settings['slice_acquisition_order'] == 'top-bottom':
            slicetime.inputs.index_dir = True
        elif settings['slice_acquisition_order'] == 'bottom-top':
            slicetime.inputs.index_dir = False
        elif settings['slice_acquisition_order'] == 'interleaved':
            slicetime.inputs.interleaved = True

        slicetime.run()


def volume_realign(subject, settings, run_number):
    """
    Runs volume realingment on slice time corrected data.

    Note that the head motion parameters are not calculated from this step.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
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
    """
    General function to apply the previously calculated brain mask to functional data.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    mask_img = fsl.ApplyMask(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                             mask_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                             out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"))

    mask_img.run()

def detrend_data(subject, settings, run_number):
    """
    Do polynomial detrending of data. The default polynomial degrees to do are up to the third. This can be changed by
    the user.

    Polynomial regressors are saved in the data features dataframe.

    Head motion parameters have the same detrending applied to them. This detrended version of these are stored in the
    features dataframe so that they match the functional data.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
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

    # Save polynomial regressors and detrended head motion
    tcs_df = pd.DataFrame()

    for ii in range(regressors.shape[1]):
        tcs_df[f'polynomial_deg_{ii}'] = regressors[:, ii]

    for ii, par in enumerate(['rotation_x', 'rotation_y', 'rotation_z', 'translation_x', 'translation_y', 'translation_z']):
        tcs_df[par] = hm_clean[:, ii]

    tcs_df.to_csv(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_features.csv"),
                  index=False)


def extract_confound_tcs(subject, settings, run_number):
    """
    Get mean time series, plus the first three eigenvariates, from the grey matter, white matter, and CSF.

    These are stored in the data feature dataframe.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    tcs_df = pd.read_csv(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_features.csv"))
    for tissue in ['gm', 'wm', 'csf']:
        # Mean time series
        temp = subprocess.run(['fslmeants',
                               '-i',
                               path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                               '-m',
                               path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_{tissue}-mask.nii.gz")],
                              stdout=subprocess.PIPE)
        tcs = temp.stdout.decode().split()
        tcs_df[f'{tissue}_mean'] = np.array(tcs).astype(float)

        # First three eigenvariates
        temp = subprocess.run(['fslmeants',
                               '-i',
                               path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
                               '-m',
                               path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_{tissue}-mask.nii.gz"),
                               '--eig',
                               '--order=3'],
                              stdout=subprocess.PIPE)
        tcs = temp.stdout.decode().split()
        tcs = np.reshape(tcs, (settings['number_usable_vols'], 3))
        for ii in range(3):
            tcs_df[f'{tissue}_eig_{ii+1}'] = np.array(tcs[:, ii]).astype(float)

    tcs_df.to_csv(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_features.csv"),
                  index=False)


def smooth_data(subject, settings, run_number, melodic_smooth=False):
    """
    Applies spatial smoothing to functional data.

    Smoothing kernel size is set by the user.

    In the special case where smoothing is being done prior to running MELODIC, a set kernel of 5mm is used and
    a temporary file is created.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number
    melodic_smooth: Bool
            Whether or not to apply spatial smoothing to the data before running MELODIC

    Returns
    -------
    None

    """
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
    if melodic_smooth:
        cmd = ['susan',
               path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
               str(brightness_threshold),
               str(5 / np.sqrt(8 * np.log(2))),
               str(3),
               str(0),
               str(0),
               path.join(settings['func_out'], f"temp_{run_number}.nii.gz")]

        subprocess.run(cmd)

    else:
        cmd = ['susan',
               path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz"),
               str(brightness_threshold),
               str(settings['smoothing_fwhm'] / np.sqrt(8 * np.log(2))),
               str(3),
               str(0),
               str(0),
               path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_smooth-{settings['smoothing_fwhm']}mm.nii.gz")]

        subprocess.run(cmd)

        # Clean up voxels outside brain
        mask_img = fsl.ApplyMask(in_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_smooth-{settings['smoothing_fwhm']}mm.nii.gz"),
                                 mask_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                                 out_file=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc_smooth-{settings['smoothing_fwhm']}mm.nii.gz"))
        mask_img.run()

def run_melodic_ica(subject, settings, run_number):
    """
    Runs MELODIC ICA on the data for FIX denoising.

    The data is smoothed before this is run, even if smoothing is not requested by the user.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
    #  Apply some smoothing to the input image
    smooth_data(subject, settings, run_number, melodic_smooth=True)

    # Run MELODIC
    melodic = fsl.MELODIC(in_files=path.join(settings['func_out'], f'temp_{run_number}.nii.gz'),
                          approach='tica',
                          bg_image=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_T1w-brain.nii.gz"),
                          mask=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_brain-mask.nii.gz"),
                          tr_sec=settings['bold_TR'],
                          no_bet=True,
                          report=True,
                          out_dir=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}.ica"))
    melodic.run()

    # Clean up the temporary smoothed image
    os.remove(path.join(settings['func_out'], f'temp_{run_number}.nii.gz'))


def run_func_preprocess(subject, settings, run_number):
    """
    Run through all steps of functional preprocessing.


    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    run_number: int
            Run number

    Returns
    -------
    None

    """
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
    if settings['run_slice_timing']:
        slicetime_correct(subject, settings, run_number)

    # Volume realignment
    volume_realign(subject, settings, run_number)

    # Zero values outside of brain
    apply_brain_mask(subject, settings, run_number)

    # Detrend data
    detrend_data(subject, settings, run_number)

    # Get standard confound time series
    extract_confound_tcs(subject, settings, run_number)

    # Smooth data - default is to not do
    if settings['smoothing_fwhm']:
        smooth_data(subject, settings, run_number)

    # Run MELODIC for FIX denoising - default is to not do
    if settings['run_melodic_ica']:
        run_melodic_ica(subject, settings, run_number)

    # Create QC image
    create_functional_report(subject, settings, run_number)

