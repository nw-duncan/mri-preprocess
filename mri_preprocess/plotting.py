#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper functions for preprocessing.

"""
import os
import numpy as np
import nibabel as ni
import matplotlib.pyplot as plt
import nilearn.plotting as niplt
from nipype.interfaces import ants
from scipy.stats import zscore
from mri_preprocess import utils
from os import path


def normalise_gm(subject, settings, template_path):
    """

    Put the GM edge map calculate in the preprocessing steps into template space.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object
    template_path: str
            Path to the template image to be used

    Returns
    -------

    """
    apply_xfm = ants.ApplyTransforms(input_image=path.join(settings['anat_out'], f'{subject}_gm-edge.nii.gz'),
                                     reference_image=template_path,
                                     transforms=[path.join(settings['anat_out'], f"{subject}_T1w_to_{settings['template_name']}_1Warp.nii.gz"),
                                                 path.join(settings['anat_out'], f"{subject}_T1w_to_{settings['template_name']}_0GenericAffine.mat")],
                                     interpolation='NearestNeighbor',
                                     output_image=path.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"))
    apply_xfm.run()


def combine_segmentation(subject, settings):
    """
    Combine the GM and the CSF edge images in preparation for superimposing on the T1 image.

    Parameters
    ----------
    subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------

    """
    img = ni.load(path.join(settings['anat_out'], f'{subject}_gm-edge.nii.gz'))
    aff = img.affine
    img = img.get_fdata()
    csf = ni.load(path.join(settings['anat_out'], f'{subject}_csf-edge.nii.gz')).get_fdata()
    img[csf == 1] = 2
    return ni.Nifti1Image(img, aff)


def create_anatomical_report(subject, settings):
    """
    Create a PNG image that shows the quality of the tissue segmentation and the alignment to the template space for
    the T1-weighted image.

    Parameters
    ----------
     subject: str
            Subject ID
    settings: dictionary
            Settings object

    Returns
    -------

    """
    # Set template
    template_path = utils.check_template_file(settings['template_name'], settings['template_resolution'], desc=None,
                                              suffix='T1w')
    # Put GM image into template space
    normalise_gm(subject, settings, template_path)
    # Make segmentation image
    segmentation_img = combine_segmentation(subject, settings)
    # Make image
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=(1, 1))
    # Show anatomical segmentation
    gs0 = gs[0, 0].subgridspec(3, 1, wspace=0.2, hspace=0)
    ax1, ax2, ax3 = gs0.subplots()
    niplt.plot_roi(segmentation_img,
                      bg_img=path.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                      display_mode='z', cut_coords=(20, 40, 50, 60, 70, 80, 90),
                      axes=ax1,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(segmentation_img,
                      bg_img=path.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                      display_mode='x', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax2,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(segmentation_img,
                      bg_img=path.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                      display_mode='y', cut_coords=(-40, -20, 0, 25, 45, 55, 65),
                      axes=ax3,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    # Show anatomical alignment to template
    gs1 = gs[1, 0].subgridspec(3, 1, wspace=0.2, hspace=0)
    ax4, ax5, ax6 = gs1.subplots()
    niplt.plot_roi(path.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"),
                      bg_img=template_path,
                      display_mode='z', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax4,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(path.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"),
                      bg_img=template_path,
                      display_mode='x', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax5,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(path.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"),
                      bg_img=template_path,
                      display_mode='y', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax6,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    # Save image
    fig.savefig(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], 'process_qc_images',
                         'anat', f'{subject}_anatomical-preprocessing.png'),
                 bbox_inches='tight', dpi=300)
    plt.close(fig)


def extract_timeseries(subject, settings, run_number):
    # Load data
    data = ni.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-preproc.nii.gz")).get_fdata()
    wm_mask = ni.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_wm-mask.nii.gz")).get_fdata()
    gm_mask = ni.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_gm-mask.nii.gz").get_fdata()
    csf_mask = ni.load(path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_csf-mask.nii.gz"))).get_fdata()
    # Get timeseries
    wm_tcs = data[wm_mask==1]
    gm_tcs = data[gm_mask==1]
    csf_tcs = data[csf_mask==1]
    # Remove voxels where there is no signal
    wm_tcs = wm_tcs[np.sum(wm_tcs, axis=1)!=0]
    gm_tcs = gm_tcs[np.sum(gm_tcs, axis=1)!=0]
    csf_tcs = csf_tcs[np.sum(csf_tcs, axis=1)!=0]
    # Zscore
    wm_tcs = zscore(wm_tcs, axis=1)
    gm_tcs = zscore(gm_tcs, axis=1)
    csf_tcs = zscore(csf_tcs, axis=1)
    # Combine for carpet plot
    all_tcs = np.vstack((wm_tcs, csf_tcs, gm_tcs))
    axis_indices = [0, wm_tcs.shape[0], wm_tcs.shape[0]+csf_tcs.shape[0], all_tcs.shape[0]]
    return all_tcs, axis_indices, np.mean(gm_tcs, axis=0)


def create_functional_report(subject, settings, run_number):
    # Get voxel time series
    all_tcs, y_ticks, gm_mean = extract_timeseries(subject, settings, run_number)
    # Make image
    fig = plt.figure(figsize=(8, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=(2, 0.3, 0.3, 0.3, 2.5))
    # Show anatomical segmentation
    gs0 = gs[0, 0].subgridspec(3, 1, wspace=0.2, hspace=0)
    ax1, ax2, ax3 = gs0.subplots()
    plotting.plot_roi(
        path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_gm-mask.nii.gz"),
        bg_img=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
        display_mode='z', cut_coords=(20, 40, 50, 60, 70, 80, 90),
        axes=ax1,
        draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    plotting.plot_roi(
        path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_gm-mask.nii.gz"),
        bg_img=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
        display_mode='x', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
        axes=ax2,
        draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    plotting.plot_roi(
        path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_gm-mask.nii.gz"),
        bg_img=path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_bold-reference.nii.gz"),
        display_mode='y', cut_coords=(-40, -20, 0, 25, 45, 55, 65),
        axes=ax3,
        draw_cross=False, annotate=False, black_bg=False, cmap='Paired')

    # Show GM global signal
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(gm_mean)
    ax4.set_xlim(0, all_tcs.shape[1])
    ax4.set_ylim(gm_mean.min() * 1.05, gm_mean.max() * 1.05)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])

    # Show DVARS
    ax5 = fig.add_subplot(gs[2, 0])
    dvars = np.loadtxt(
        path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-DVARS.csv"))[
            :, 1]
    ax5.plot(dvars)
    ax5.set_xlim((0, all_tcs.shape[1]))
    ax5.set_ylim(dvars.min() * 1.05, dvars.max() * 1.05)
    ax5.set_xticklabels([])
    ax5.set_ylabel('DVARS')

    # Show FD 
    ax6 = fig.add_subplot(gs[3, 0])
    fd = np.loadtxt(
        path.join(settings['func_out'], f"{subject}_task-{settings['task_name']}_run-{run_number}_movement-FD.csv"),
        skiprows=1)
    ax6.plot(fd)
    ax6.set_xlim((0, all_tcs.shape[1]))
    ax6.set_ylim(fd.min() * 1.05, fd.max() * 1.05)
    ax6.set_xticklabels([])
    ax6.set_ylabel('FD')
    ax6.set_title(f'mean FD = {np.mean(fd):0.2f}, max FD = {fd.max():0.2f}', fontsize=10)

    # Show carpetplot
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.imshow(all_tcs, cmap='gray', aspect='auto')
    ax7.set_yticks(y_ticks)
    ax7.set_yticklabels(['WM', 'CSF', '', 'GM'])
    ax7.set_xlabel('Volume')
    ax7.set_xlim((0, all_tcs.shape[1]))
    # Save image
    fig.savefig(path.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], 'process_qc_images',
                         'func', f'f"{subject}_task-{settings['task_name']}_run-{run_number}_functional-preprocessing.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
