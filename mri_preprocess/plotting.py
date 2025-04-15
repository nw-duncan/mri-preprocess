#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper functions for preprocessing.

"""
import os
import os.path as osp
import numpy as np
import nibabel as ni
import matplotlib.pyplot as plt
import nilearn.plotting as niplt
from nipype.interfaces import ants
from scipy.stats import zscore
from mri_preprocess import utils


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
    apply_xfm = ants.ApplyTransforms(input_image=osp.join(settings['anat_out'], f'{subject}_gm-edge.nii.gz'),
                                     reference_image=template_path,
                                     transforms=[osp.join(settings['anat_out'], f"{subject}_T1w_to_{settings['template_name']}_1Warp.nii.gz"),
                                                 osp.join(settings['anat_out'], f"{subject}_T1w_to_{settings['template_name']}_0GenericAffine.mat")],
                                     interpolation='NearestNeighbor',
                                     output_image=osp.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"))
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
    img = ni.load(osp.join(settings['anat_out'], f'{subject}_gm-edge.nii.gz'))
    aff = img.affine
    img = img.get_fdata()
    csf = ni.load(osp.join(settings['anat_out'], f'{subject}_csf-edge.nii.gz')).get_fdata()
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
                      bg_img=osp.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                      display_mode='z', cut_coords=(20, 40, 50, 60, 70, 80, 90),
                      axes=ax1,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(segmentation_img,
                      bg_img=osp.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                      display_mode='x', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax2,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(segmentation_img,
                      bg_img=osp.join(settings['anat_out'], f'{subject}_T1w_std_crop_biascorr.nii.gz'),
                      display_mode='y', cut_coords=(-40, -20, 0, 25, 45, 55, 65),
                      axes=ax3,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    # Show anatomical alignment to template
    gs1 = gs[1, 0].subgridspec(3, 1, wspace=0.2, hspace=0)
    ax4, ax5, ax6 = gs1.subplots()
    niplt.plot_roi(osp.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"),
                      bg_img=template_path,
                      display_mode='z', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax4,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(osp.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"),
                      bg_img=template_path,
                      display_mode='x', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax5,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    niplt.plot_roi(osp.join(settings['anat_out'], f"{subject}_gm-edge_{settings['template_name']}.nii.gz"),
                      bg_img=template_path,
                      display_mode='y', cut_coords=(-50, -40, -20, 0, 20, 40, 50),
                      axes=ax6,
                      draw_cross=False, annotate=False, black_bg=False, cmap='Paired')
    # Save image
    fig.savefig(osp.join(settings['root_dir'], 'derivatives', settings['output_dir_name'], 'process_qc_images',
                         'anat', f'{subject}_anatomical-preprocessing.png'),
                 bbox_inches='tight', dpi=300)
    plt.close(fig)
