#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:51:05 2024

@author: niall
"""

from mri_preprocess import utils, anat_prepro, func_prepro


def mri_preprocess(subject, settings_file):

    # Initiate preprocessing settings
    settings = utils.initiate_settings(settings_file)

    # Update settings with user defined ones
    settings = utils.update_settings(settings)

    # Check that the root directory follows BIDs style
    utils.check_bids_style(settings['root_dir'])

    # Do directory management stuff
    utils.create_output_dirs(subject, settings)
    
    
    ###########################
    # Anatomical preprocessing
    ###########################
    
    if setup_params['process_anat']:
        # Reorient image to standard
        anat_prepro.reorient_t1_to_standard(subject,
                                         setup_params['root_dir'],
                                         setup_params['output_dir'])
        # Reduce field of view
        anat_prepro.reduce_fov(subject,
                               setup_params['root_dir'],
                               setup_params['output_dir'])
        
        # Biasfield correct image
        anat_prepro.create_rough_mask(subject,
                                      setup_params['root_dir'],
                                      setup_params['output_dir'])
        
        anat_prepro.bias_correct(subject,
                                 setup_params['root_dir'],
                                 setup_params['output_dir'],
                                 setup_params['num_threads'])
        
        # Align to template        
        anat_prepro.align_to_template(subject,
                                      setup_params['root_dir'],
                                      setup_params['output_dir'],
                                      setup_params['template_name'],
                                      setup_params['template_resolution'],
                                      setup_params['num_threads'],
                                      setup_params['ants_reg_params'])
        
        # Brain extraction
        anat_prepro.brain_extract(subject,
                                  setup_params['root_dir'],
                                  setup_params['output_dir'],
                                  setup_params['T1_brain_extract_type'],
                                  setup_params['num_threads'])
        
        # Tissue segmentation
        anat_prepro.tissue_segment(subject,
                                   setup_params['root_dir'],
                                   setup_params['output_dir'])
    
    ######################################################
    # Functional preprocessing
    # Assumes anatomical preprocessing has been done
    ######################################################
    
    if setup_params['process_func']:
        
        if number_of_runs:
            for run_number in np.arange(1, setup_params['number_of_runs']+1):
                
                
        