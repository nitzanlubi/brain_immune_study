"""
nf_immune_data_preprocessing.py

This code performs all preprocessing of neural, mental and immunological data for analyses reported in the manuscript:
"Upregulation of Reward Mesolimbic Activity via fMRI-Neurofeedback Improves Vaccination Efficiency in Humans"
Its outputs are called for data analysis in "nf_immune_analyses.py".

The code is organized in the following sections:
    Import libraries
    Define paths and global variables
    1. fMRI data Preprocessing
    2. Immunological data preprocessing
    3. Mental data preprocessing

Author:
    Nitzan Lubianiker, PhD.
    nitsan.lubianiker@yale.edu

Date:
    2024-07-25
"""
#%% 1. Import Libraries
from statsmodels.graphics.factorplots import interaction_plot
import pingouin as pg
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from svd_class_joy import svd_msq
from sklearn.decomposition import PCA
from statsmodels.imputation.mice import MICE
from surprise.model_selection import cross_validate
from surprise import KNNWithMeans
from surprise import Dataset, Reader
from cycler import cycler
from nipype.interfaces.ants import ApplyTransforms
import re
import random
from nilearn import image
import glob
import subprocess
import os
import shutil
import json
import pandas as pd
import numpy as np
import csv
import gzip
import nibabel as nib
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statistics as stat
from scipy import stats
from scipy import special
import statsmodels.formula.api as smf
import statsmodels.api as sm

#%% Define paths and global variables:
main_dir = "/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/fmri/offline"
func_dir = "/derivatives/fmriprep/sub-*/ses-*/func"
phys_dir = "/sub-*/ses-*/phys"
deriv_dir = "/derivatives/fmriprep"
# ENTER the task names as they appear in the BIDS .nii boldfiles
task_names_list = ['fmrinfpractice', 'fmrimidcrrt']

# Task specific inputs:
task_name = task_names_list[0]  # CHOOSE task number for analyses
n_vols_task = '225' # 225 for fmrinfpractice, 186 for fmrimidcrrt

#%% 1. fMRI Preprocessing
# %% 1.1. spatial smoothing

def smooth_all_runs():
    # CHOOSE smoothing volume size. Typical size are twice the voxel size.
    smooth = 4 # twice the voxel size of 2mm
    for file in list(glob.glob(
            f"{main_dir}{func_dir}\\sub-*_ses-1_task-{task_name}*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")):
        print(file)
        smoothed = image.smooth_img(file, smooth)
        smoothed.to_filename(str(file)[:-7] + f"_desc-sm{str(smooth)}.nii.gz")


### smooth missing runs ###
    smooth = 4
    subs = ['1161']
    sess = ['4']
    runs = ['1']
    i_list = [0, 1, 2, 3]
    for i in i_list:
        for file in list(glob.glob(
                f"{main_dir}{func_dir}\\sub-{subs[i]}_ses-{sess[i]}_task-{task_name}_run-{runs[i]}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")):
            smoothed = image.smooth_img(file, smooth)
            smoothed.to_filename(
                str(file)[:-7] + f"_desc-sm{str(smooth)}.nii.gz")
############################
# %% 1.2. fix dirs and files names
# the next two functions are needed for compatability with the annoying TAPAS PhysIO filenames conventions that diverge from the SIMENS PRISMA output files.

def rename_phys_files():
    # before creating physiological regressors, change all the physio filenames to be similar the func data filenames
    # (all lowercase, no spaces)
    for f in glob.glob(f"{main_dir}{phys_dir}\\*_AcquisitionInfo_*.log"):
        split = f.split("AcquisitionInfo_")
        name = split[0] + "AcquisitionInfo_" + \
            split[-1].lower().replace("_", "")
        os.rename(f, name)

    # then, rename a copy of every ECG1.log to ECG.log
    for f in glob.glob(f"{main_dir}{phys_dir}\\*_ECG1.log"):
        name = f.replace("ECG1", "ECG")
        shutil.copy(f, name)

def add_runnum_to_files():
    # before creating physiological regressors, change all the physio filenames to be similar the func data filenames
    # (all lowercase, no spaces)
    for file in list(glob.glob(f"{main_dir}{func_dir}\\sub-*_ses-*_task-{task_name}_desc-confounds_regressors.tsv")):
        index = file.find('_desc')
        newname = file[:index] + '_run-1' + file[index:]

        os.rename(file, newname)

    # then, rename a copy of every ECG1.log to ECG.log
    for f in glob.glob(f"{main_dir}{phys_dir}\\*_ECG1.log"):
        name = f.replace("ECG1", "ECG")
        shutil.copy(f, name)


def change_runnum_in_files():
    # change file numbers when erasing error runs
    sub = 1135
    ses = 1
    old_runnum = 3
    new_runnum = 2
    for file in list(glob.glob(f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_run-{old_runnum}_*")):
        index = file.find(f"{old_runnum}_")
        newname = file[:index] + f"{new_runnum}_" + file[index+2:]

        os.rename(file, newname)

    # then, rename a copy of every ECG1.log to ECG.log
    for f in glob.glob(f"{main_dir}{phys_dir}\\*_ECG1.log"):
        name = f.replace("ECG1", "ECG")
        shutil.copy(f, name)

# %% 1.3. physiological noise extraction:

def create_physiological_regressors():
    # CHECK whether this regex matches the specific task you wish to capture (replace "practice" with a dissociating str...)
    pattern = re.compile(".*sub-(\d*)\\\\ses-(\d*).*?practice_(run-\d)?.*")
    jsons = glob.glob(f"{main_dir}{phys_dir}\\*.json")
    tr = 2
    model_type = 'basic'
    # if you wish to run a random sub-sample, create a random selection of n runs in order to plot physiology and loop on sample_jsons instead
    # sample_jsons=[]
    # for i in range(10):
    #     n=random.randint(0,len(jsons)-1)
    #     sample_jsons.append(jsons[n])

    # the process_phys_run_BIDS.m script assumes similar number of bold.nii runs of a certain task in a session as physio json files (if not, it should skip a session))
    for f in jsons:
        match = pattern.match(f)
        sub = match.groups()[0]
        ses = match.groups()[1]

        if match.groups()[2] is not None:
            task_name += "_" + match.groups()[2]
        mods = []
        with open(f) as fp:
            mods = json.load(fp)['columns']
        mod_card = ''
        mod_resp = ''
        if "ECG" in mods:
            mod_card = "ECG"
        elif "pulse" in mods:
            mod_card = "pulse"
        if "resp" in mods:
            mod_resp = 'resp'
        fun_str = (f"matlab -wait -nosplash -nodesktop -r \"process_phys_run_BIDS "
                   f"('sub-{sub}', 'ses-{ses}',"
                   f"'{task_name}', '{main_dir}', '{main_dir}{deriv_dir}', "
                   f"{tr}, '{mod_card}', '{mod_resp}', '{model_type}');exit\"")
        print(fun_str)
        p = subprocess.call(fun_str)


### for single sesssion ###
sub = 1196
ses = 3
main_dir = "H:\\projects\\joy\\analyses\\fmri\\offline"
deriv_dir = "H:\\projects\\joy\\analyses\\fmri\\offline\\derivatives\\fmriprep"
tr = 2
mod_card = 'ECG'
mod_resp = 'resp'
model_type = 'basic'
fun_str = (f"matlab -wait -nosplash -nodesktop -r \"process_phys_run_BIDS "
           f"('sub-{sub}', 'ses-{ses}',"
           f"'{task_name}', '{main_dir}', '{main_dir}{deriv_dir}', "
           f"{tr}, '{mod_card}', '{mod_resp}', '{model_type}');exit\"")

print(fun_str)
p = subprocess.call(fun_str)
###

def find_missing_physio():
    # after running "def create_physiological_regressors()" function, some physiological models may fail to be created due to raw physio files-boldfiles discrepency.
    # this function creates files with lists of such cases - of bolds without physiology, and physiology without bolds, to be inspected manually.

    # CHECK whether this regex matches the specific task you wish to capture (replace "practice" with the dissociating str...)
    pattern = re.compile(".*sub-(\d*)\\\\ses-(\d*).*?practice_(run-\d_)?")
    jsons = glob.glob(f"{main_dir}{phys_dir}\\*.json")
    boldfiles = glob.glob(
        '%s\\sub-*1[1-2][0-9][0-9]\\ses-[1-4]\\func\\*{task_name}*_space-MNI152NLin2009cAsym_desc-preproc_bold_desc-sm4.nii.gz' % (main_dir+deriv_dir))
    jsons_set = set()
    for f in jsons:
        match = pattern.match(f)
        sub = match.groups()[0]
        ses = match.groups()[1]
        if match.groups()[2] is not None:
            run = match.groups()[2][-2]
        else:
            run = 1
        jsons_set.add((int(sub), int(ses), int(run)))

    bold_set = set()
    for f in boldfiles:
        match = pattern.match(f)
        sub = match.groups()[0]
        ses = match.groups()[1]
        if match.groups()[2] is not None:
            run = match.groups()[2][-2]
        else:
            run = 1
        bold_set.add((int(sub), int(ses), int(run)))
    diff = bold_set-jsons_set
    diff2 = jsons_set-bold_set
    df = pd.DataFrame(diff)
    df2 = pd.DataFrame(diff2)
    df.to_csv('bolds_without_physiology_200622.csv')
    df2.to_csv('physiology_without_bolds_200622.csv')
    print(diff)

# %% 1.4. scrubbing regressors:

def create_scrubing_regressors():
    # Goes over all runs on dataset for a certain task that has fmriprep confound outputs
    # and creates movement outliers regressors according to some movement regressor from fmriprep outputs above a certain threshold, to use in  1st level noise model.
    ###################################################################################################
    # Inputs:
    #       mov_reg: str; the name of the movement regressor in question.
    #       threshold: int; threshold value above which TRs should be regressed out.
    #       task_name: str; name of task.
    #       n_vols_task: int; number of repetitions in task.
    #
    # output:
    #       outliers_regressors.tsv file in every ses folder for every run. This file could be inserted to the first level analysis as part of
    #       the noise regressors in the model.
    #       a QA list containing all runs that exceeded 20% of outliers, to remove from analysis.
    ###################################################################################################
    # FILL INPUTS:
    # the name of the regressor according to which volumes will be scrubbed. could be dvars or framewise displacement (more common)
    mov_reg = "framewise_displacement"
    threshold = 0.5  # given a framewise displacement option is chosen, this threshold value was set according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3895106/
    # setting a cutoff of 20% of volumes removed, above which runs would be discarded entirely.
    cutoff = 0.2
    #####
    n_vols_task_int = int(n_vols_task)
    thresh_str = str(threshold)
    remove_cutoff = n_vols_task_int*cutoff
    pattern = re.compile(".*sub-(\d*)_ses-(\d*).*_(run-(\d))?.*_desc.*tsv")
    # a list that will save runs info that have more than remove_cutoff outliers volumes
    bad_bold_list = []
    for file in list(glob.glob(
            f"{main_dir}{func_dir}\\sub-*_ses-*_task-{task_name}*_desc-confounds_regressors.tsv")):
        data = pd.read_csv(file, sep='\t')
        col_movreg = data[mov_reg]
        despike_mat = np.diag(col_movreg)
        despike_mat[0, 0] = 0.0
        mask = (despike_mat < threshold)
        idx = mask.all(axis=0)
        scrub_mat = despike_mat[:, ~idx]
        scrub_mat = 1*scrub_mat
        scrub_mat[scrub_mat != 0] = 1
        match = pattern.match(file)
        sub = match.groups()[0]
        ses = match.groups()[1]
        if 'run' in file:
            run = match.groups()[2]
            np.savetxt(f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_{run}_{mov_reg}{thresh_str}_outliers_regressors.tsv",
                       scrub_mat, delimiter='\t', fmt='%d')
        else:
            run = 'run-1'
            np.savetxt(f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_{mov_reg}{thresh_str}_outliers_regressors.tsv",
                       scrub_mat, delimiter='\t', fmt='%d')
        if len(scrub_mat[0]) > remove_cutoff:
            bad_bold_list.append([sub, ses, run])

    with open(f"{main_dir}{deriv_dir}\\QA\\{task_name}_runs_with_more_than_{str(remove_cutoff)[:-2]}_vol_{mov_reg}_outliers.txt", 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(bad_bold_list)

# %% 1.5. noise model creation:
def create_noise_model_for_first_level():
    # Goes over all runs on dataset for a certain task and creates a txt file with all chosen regressors for 1st level analysis
    ###################################################################################################
    # Local inputs:
    #
    #       fmriprep_regressors: list; names of regressors to add to the noise model from the "*desc-confounds_regressors.tsv" fmriprep  output file
    #       include_physio: bool; controls whether to add physiological regressors (TAPAS physIO outputs) or not
    #       include_outliers_reg: bool; controls whether to add scrubbing regressors created in def "create_scrubing_regressors()" function
    #
    # output:
    #       creates noise_model_regressors.txt file in every ses folder for every run of the task. This file will be uploaded as confound EVs for
    #       1st level analysis
    #
    ###################################################################################################
    # Inputs (FILL/CHOOSE):
    fmriprep_regressors = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'a_comp_cor_00',
                           'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05',
                           'csf', 'white_matter', 'framewise_displacement']

    # fmriprep_regressors_extended = ['trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
    # 'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
    # 'trans_z','trans_z_derivative1','trans_z_derivative1_power2','trans_z_power2',
    # 'rot_x','rot_x_derivative1','rot_x_derivative1_power2','rot_x_power2','rot_y',
    # 'rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2','rot_z',
    # 'rot_z_derivative1','rot_z_derivative1_power2','rot_z_power2','a_comp_cor_00'
    # 'a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05',
    # 'csf','white_matter','framewise_displacement']

    include_physio = False
    include_outliers_reg = True
    threshold = 0.5  # if include_outliers_reg = True, this input needs to correspond with mov_reg chosen threshold that was used in "def create_scrubing_regressors()"
    ###

    # CHECK whether this regex matches the specific task you wish to capture (replace "practice" with the dissociating str...)
    pattern = re.compile(".*sub-(\d*)_ses-(\d*).*_(run-(\d))?.*_desc.*tsv")
    for file in list(glob.glob(
            f"{main_dir}{func_dir}\\sub-*_ses-*_task-{task_name}*_desc-confounds_regressors.tsv")):
        data = pd.read_csv(file, sep='\t')
        noise_model = data[fmriprep_regressors]
        match = pattern.match(file)
        sub = match.groups()[0]
        ses = match.groups()[1]
        if 'run' in file:
            run = match.groups()[2]

            if include_outliers_reg:
                # the filename corresponds to def create_scrubing_regressors() outputs.
                outliers_reg_file = glob.glob(
                    (f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_{run}*{threshold}_outliers_regressors.tsv"))[0]

                with open(outliers_reg_file, 'r') as read_obj:
                    # read first character
                    one_char = read_obj.read(1)
                    # if one_char = \n then file is empty
                    if one_char != '\n':
                        outliers_regs = pd.read_csv(
                            outliers_reg_file, sep='\t', header=None)
                        # the prefix is hardcoded for framewise displacement. CHANGE if using another censoring regressor
                        outliers_regs = outliers_regs.add_prefix('FD_outlier_')
                        noise_model = pd.concat(
                            [noise_model, outliers_regs], axis=1)

            ### physiological regressors insertion ###
            if include_physio:
                physio_regs_file = glob.glob(
                    (f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_{run}_desc-physio_regressors_basic.txt"))[0]
                physio_regs_col_names = glob.glob(
                    (f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_{task_name}_{run}_physio_col_names_basic.csv"))[0]

                with open(physio_regs_col_names, newline='') as f:
                    reader = csv.reader(f)
                    physio_regs_col_names = list(reader)

                # without regressors names. could be uploaded from run-specific physio_col_names_basic.csv files
                physio_regs = pd.read_csv(
                    physio_regs_file, sep='\t', header=None)
                # CHECK whether there is a nan column in the 19th column (index 18)... and erase it if so:
                physio_regs.drop(18, inplace=True, axis=1)
                #
                noise_model = pd.concat([noise_model, physio_regs], axis=1)

            noise_model = np.nan_to_num(noise_model)
            print(f"sub-{sub}_ses-{ses}_{run}")
            np.savetxt(f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_{run}_noise_model_regressors.txt",
                       noise_model, delimiter='\t')  # without physio
            # np.savetxt(f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_{run}_noise_model_regressors_with_physio.txt", noise_model, delimiter='\t') # with physio

        else:
            if include_outliers_reg:
                # the filename corresponds to def create_scrubing_regressors() outputs.
                outliers_reg_file = glob.glob(
                    (f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}*{threshold}_outliers_regressors.tsv"))[0]
                with open(outliers_reg_file, 'r') as read_obj:
                    # read first character
                    one_char = read_obj.read(1)
                    # if one_char = \n then file is empty
                    if one_char != '\n':
                        outliers_regs = pd.read_csv(
                            outliers_reg_file, sep='\t', header=None)
                        # the prefix is hardcoded for framewise displacement. CHANGE if using another censoring regressor
                        outliers_regs = outliers_regs.add_prefix('FD_outlier_')
                        noise_model = pd.concat(
                            [noise_model, outliers_regs], axis=1)

            ### physiological regressors insertion ###
            if include_physio:
                physio_regs_file = glob.glob(
                    (f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_desc-physio_regressors_basic.txt"))[0]
                physio_regs_col_names = glob.glob(
                    (f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_{task_name}_physio_col_names_basic.csv"))[0]

                with open(physio_regs_col_names, newline='') as f:
                    reader = csv.reader(f)
                    physio_regs_col_names = list(reader)

                # without regressors names. could be uploaded from run-specific physio_col_names_basic.csv files
                physio_regs = pd.read_csv(
                    physio_regs_file, sep='\t', header=None)
                # CHECK whether there is a nan column in the 19th column (index 18)... and erase it if so:
                physio_regs.drop(18, inplace=True, axis=1)
                #
                noise_model = pd.concat([noise_model, physio_regs], axis=1)

            noise_model = np.nan_to_num(noise_model)
            print(f"sub-{sub}_ses-{ses}")
            np.savetxt(f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_noise_model_regressors.txt",
                       noise_model, delimiter='\t')  # without physio
            # np.savetxt(f"{main_dir}{deriv_dir}\\sub-{sub}\\ses-{ses}\\func\\sub-{sub}_ses-{ses}_task-{task_name}_noise_model_regressors_with_physio.txt", noise_model, delimiter='\t') # with physio
# %% 1.6. MID task behavioural protocol - BIDS events and EV for FSL

def create_BIDS_events_and_fsl_EVs_mid_task():
    import scipy.io as sio

    # copy the relevant output files from the raw data repository
    # a network path requires forward slashes...
    mid_output_folder = '//fmri-st1/nitzan$/Joy/OpenNFT_backup/experiment'
    mid_subs_folders = glob.glob(
        '%s/sub-*/ses-1/MID_CRRT/behav_protocol' % (mid_output_folder))

    mid_dest_folder = main_dir + '\\tasks_protocols\\task-fmrimidcrrt\\'
    pattern = re.compile(".*sub-(\d*)\\\\ses-(\d*)")
    for lib in mid_subs_folders:
        match = pattern.match(lib)
        sub = match.groups()[0]
        ses = match.groups()[1]

        dst_path = f"{mid_dest_folder}sub-{sub}\\ses-{ses}"
        shutil.copytree(lib, dst_path)

    # copy the no treatment folders that do not have a real-time folder
    # a network path requires forward slashes...
    mid_output_folder = '//fmri-st1/nitzan$/Joy/MID_output/Experiment/Experiment/no_treatment'
    mid_subs_folders = glob.glob(
        '%s/sub-[1][1][0-9][0-9]/' % (mid_output_folder))

    mid_dest_folder = main_dir + '\\tasks_protocols\\task-fmrimidcrrt\\'
    pattern = re.compile(".*sub-(\d*)")
    for lib in mid_subs_folders:
        match = pattern.match(lib)
        sub = match.groups()[0]
        ses = '1'
        dst_path = f"{mid_dest_folder}sub-{sub}\\ses-{ses}"
        os.makedirs(dst_path)
        shutil.copy2(f"{lib}MID_HC{sub}T1_run3_CRRT_error_pmod.mat", dst_path)
        shutil.copy2(f"{lib}MID_HC{sub}T1_run4_CRRT_error_pmod.mat", dst_path)

    # run on all subjects and runs and create ev files per run
    pattern = re.compile(".*sub-(\d*)\\\\ses-(\d*)")
    # reset the path names to copied raw folders:
    mid_subs_folders = glob.glob('%s/sub-*/ses-1' % (mid_dest_folder))
    # the condition names and order is similar for every subject and unreadable with scipy.io so here it is:
    cond_names = ['anticipation_noreward', 'anticipation_lowreward', 'anticipation_highreward', 'consumption_noreward',
                  'consumption_lowreward', 'consumption_highreward', 'target', 'error_anticipation', 'error_target', 'error_consumption']
    cond_names_pmod = ['consumption_lowreward_pmod',
                       'consumption_highreward_pmod']

    for lib in mid_subs_folders:
        match = pattern.match(lib)
        sub = match.groups()[0]
        ses = match.groups()[1]
        for n in [3, 4]:  # the run numbers of the online .mat outputs are 3 and 4 for run 1 and 2 respectively
            # load the data from the online matlab outputs
            mat_file = sio.loadmat(
                f"{lib}\\MID_HC{sub}T1_run{n}_CRRT_error_pmod.mat")
            onsets = mat_file["onsets"]
            durations = mat_file["durations"]

            # create output folder
            curr_out_dir = f"{main_dir}\\derivatives\\fsl\\task_events\\task-fmrimidcrrt\\sub-{sub}\\ses-{ses}\\run-{n-2}"
            if not os.path.exists(curr_out_dir):
                os.makedirs(curr_out_dir)
            # create FSL 3 column EV files per condition for level 1 analysis. without parametric modulation.
            # for i in range(0,len(cond_names)):
            #     curr_cond = np.concatenate((onsets[0,i].transpose(),durations[0,i].transpose()),axis=1)
            #     par_mod = np.ones((len(curr_cond),1))
            #     curr_cond = np.append(curr_cond, par_mod, axis=1)
            #     np.savetxt(f"{curr_out_dir}\\{cond_names[i]}.txt", curr_cond, delimiter='\t', fmt='%1.2f')

            # create FSL 3 column EV files for high and low consumption with parametric modulation:
            for i in range(4, len(cond_names_pmod)+4):
                if len(onsets[0, i].transpose()) == len(mat_file['pmod'][0, i][1][0][0].transpose()):
                    curr_cond = np.concatenate((onsets[0, i].transpose(), durations[0, i].transpose(
                    ), mat_file['pmod'][0, i][1][0][0].transpose()), axis=1)
                    np.savetxt(
                        f"{curr_out_dir}\\{cond_names_pmod[i-4]}.txt", curr_cond, delimiter='\t', fmt='%1.2f')
                    par_mod_demean = mat_file['pmod'][0, i][1][0][0].transpose(
                    ) - np.mean(mat_file['pmod'][0, i][1][0][0].transpose())
                    curr_cond_demean = np.concatenate(
                        (onsets[0, i].transpose(), durations[0, i].transpose(), par_mod_demean), axis=1)
                    np.savetxt(
                        f"{curr_out_dir}\\{cond_names_pmod[i-4]}_demean.txt", curr_cond_demean, delimiter='\t', fmt='%1.2f')
                else:
                    print(
                        f"sub-{sub} run {n-2} has a problem with pmod regressor lengths")

            # create a single events.tsv file that is BIDS compatible and copy it to the BIDS general directory - not realized yet!!!

# %% 1.7. fMRI GLM analysis - 1st and 2nd level analyses
#%% 1.7.1 level 1 fsfs creation:
def create_1st_level_fsfs():

    fsfdir = f"{main_dir}\\derivatives\\models\\fsfs\\level_1\\{task_name}"
    boldfiles = (glob.glob(
        f"{main_dir}{func_dir}\\sub-*_ses-*_task-{task_name}*_space-MNI152NLin2009cAsym_desc-preproc_bold_desc-sm4.nii.gz"))

    for file in list(boldfiles):
        # grab sub,ses and run
        if file.find('run') != -1:
            pattern = re.compile(".*sub-(\d*)\\\\ses-(\d*).*(run-\d)")
            match = pattern.match(file)
            subnum = match.groups()[0]
            sesnum = match.groups()[1]
            runnum = match.groups()[2]
            print(f"sub-{subnum}_ses-{sesnum}_{runnum}")

            if task_name == 'fmrimidcrrt':
                if os.path.exists(f"{main_dir}\\derivatives\\fsl\\task_events\\task-fmrimidcrrt\\sub-{subnum}\\ses-{sesnum}\\{runnum}\\error_anticipation.txt"):
                    # the filename corresponds to def create_scrubing_regressors() outputs.
                    error_ev = glob.glob(
                        (f"{main_dir}\\derivatives\\fsl\\task_events\\task-fmrimidcrrt\\sub-{subnum}\\ses-{sesnum}\\{runnum}\\error_anticipation.txt"))[0]
                    with open(error_ev, 'r') as read_obj:
                        # read first character
                        one_char = read_obj.read(1)
                        # if one_char = '' then ev is empty and there are no error trials in this run, so a different fsf template is required
                        if one_char != '':  # there are error trials
                            replacements = {'SUBNUM': subnum, 'SESNUM': sesnum,
                                            'N_VOLS': n_vols_task, 'TASKNAME': task_name, 'RUNNUM': runnum}
                            with open(f"{fsfdir}\\template_{task_name}_pmod.fsf") as infile:
                                with open(f"{fsfdir}\\design_1st_level_task-{task_name}_sub-{subnum}_ses-{sesnum}_{runnum}.fsf", 'w') as outfile:
                                    for line in infile:
                                        for src, target in replacements.items():
                                            line = line.replace(src, target)
                                        outfile.write(line)
                        else:  # there are no error trials
                            print('the above run had no error trials!')
                            replacements = {'SUBNUM': subnum, 'SESNUM': sesnum,
                                            'N_VOLS': n_vols_task, 'TASKNAME': task_name, 'RUNNUM': runnum}
                            with open(f"{fsfdir}\\template_{task_name}_pmod_no_errors.fsf") as infile:
                                with open(f"{fsfdir}\\design_1st_level_task-{task_name}_sub-{subnum}_ses-{sesnum}_{runnum}.fsf", 'w') as outfile:
                                    for line in infile:
                                        for src, target in replacements.items():
                                            line = line.replace(src, target)
                                        outfile.write(line)
            else:  # for fmrinfpractice task
                replacements = {'SUBNUM': subnum, 'SESNUM': sesnum,
                                'N_VOLS': n_vols_task, 'TASKNAME': task_name, 'RUNNUM': runnum}
                with open(f"{fsfdir}\\template_{task_name}_PPI.fsf") as infile:
                    with open(f"{fsfdir}\\design_1st_level_task-{task_name}_sub-{subnum}_ses-{sesnum}_{runnum}_PPI.fsf", 'w') as outfile:
                        for line in infile:
                            for src, target in replacements.items():
                                line = line.replace(src, target)
                            outfile.write(line)

        else:  # for bolds which have only one run and thus fmriprep did not indicate a run number in their names.
            pattern = re.compile(".*sub-(\d*)\\\\ses-(\d*).*")
            match = pattern.match(file)
            subnum = match.groups()[0]
            sesnum = match.groups()[1]
            runnum = 'run-1'
            print(
                f"sub-{subnum}_ses-{sesnum}_{runnum} - had no run in file name!")
            replacements = {'SUBNUM': subnum, 'SESNUM': sesnum,
                            'N_VOLS': n_vols_task, 'TASKNAME': task_name, 'RUNNUM': runnum}
            with open(f"{fsfdir}\\template_{task_name}_wo_runnum.fsf") as infile:
                with open(f"{fsfdir}\\design_1st_level_task-{task_name}_sub-{subnum}_ses-{sesnum}_{runnum}.fsf", 'w') as outfile:
                    for line in infile:
                        for src, target in replacements.items():
                            line = line.replace(src, target)
                        outfile.write(line)

# run 1st level GLM - FSL batch script:

    # Run 1st level GLM:
# This is a Python script that is intended to run through wsl linux machine in order to conduct an FSL 1st level analysis (hence the wierd '/mnt/H...' paths)
# Calls fsl feat on all fsf design files of a certain task (after they were produced, see above...)

# SAVE this script on a seperate .py file, and run through linux comצand prompt with FSL installed.


# Inputs:
main_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives'
fsfsdir = '/models/fsfs/level_1'  # location of the fsf folders
task_name = 'fmrinfpractice'  # name of task
###

# # Get *all* the fsfs:
# fsffiles = glob.glob('%s%s/%s/design_1st_level_task-%s_sub-*_ses-*_run-*.fsf'%(main_dir,fsfsdir,task_name,task_name))

# for fsf in list(fsffiles):
#     os.system('feat %s'%(fsf))

# if running parralel analyses on multiple subjects, get a subset of the fsfs. for instance:
fsffiles_subset = glob.glob(
    '%s%s/%s/design_1st_level_task-%s_sub-11[2-3]*_ses-*_run-*.fsf' % (main_dir, fsfsdir, task_name, task_name))
for fsf in list(fsffiles_subset):
    os.system('feat %s' % (fsf))

#%%  1.7.2. QA level 1
def create_QA_html_for_1st_level_analyses():
    # Creates one html file with the design and coefficient matrices of all runs, for inspection.
    outfile = f"{main_dir}\\derivatives\\QA\\{task_name}_level_1_QA.html"
    feat_folders = glob.glob(
        f"{main_dir}{func_dir}\\{task_name}_run-[1-4].feat")

    f = open(outfile, 'w')
    for file in list(feat_folders):
        f.write("<p>===================================================================================================")
        f.write("<h1>%s</h1>" % (file))
        f.write('<br/>')
        f.write("<IMG SRC=\"%s/design.png\">" % (file))
        f.write('<br/>')
        f.write("<IMG SRC=\"%s/design_cov.png\" >" % (file))
    f.close()

#%%  1.7.3. prepare for level 2:

def create_dummy_reg_folders_for_FSL_2nd_level():
    # When using FSL for higher level analyses on fmriPrep preprocessed boldfiles that are already on standard space, adjustments are needed.
    # This is because FSL assumes 1st level analysis was conducted in native space, and in 2nd level expects registration preprocessing files in the level 1 feat folders.
    # The following function creates what is needed in order to workaround this issue (not registering the already registered data) following the guidelines of https://mumfordbrainstats.tumblr.com/post/166054797696/feat-registration-workaround

    # EDIT the suffix name of the folders
    feat_folders = glob.glob(
        f"{main_dir}{func_dir}\\{task_name}_run-[1-4].feat")
    template_reg_folder = "H:\\projects\\joy\\analyses\\fmri\\offline\\derivatives\\models\\reg"

    for feat in list(feat_folders):
        # copy the template reg folder
        dst_path = feat + '\\reg'
        # for level 1 that did NOT run with registration preprocessing:
        if not os.path.exists(dst_path):
            shutil.copytree(template_reg_folder, dst_path)
            # copy the specific run's mean_func.nii.gz into the folder and name it standard.nii.gz
            shutil.copy(f"{feat}\\mean_func.nii.gz",
                        f"{dst_path}\\standard.nii.gz")
        else:  # for level 1 that did run with registration preprocing step
            shutil.copytree(dst_path, dst_path+'_original')
            shutil.rmtree(dst_path)           # removes all the subdirectories!
            shutil.copytree(template_reg_folder, dst_path)
            shutil.copy(f"{feat}\\mean_func.nii.gz",
                        f"{dst_path}\\standard.nii.gz")


def create_2nd_level_fsfs():
    # 2nd level = subject level. create an fsf file according to number of runs
    fsfdir = f"{main_dir}\\derivatives\\models\\fsfs\\level_2\\{task_name}_ses_averaged_runs"
    task_sessions = glob.glob(
        f"{main_dir}\\derivatives\\fmriprep\\sub-*\\ses-*\\func")
    ses_runnum_list = []
    # CHECK whether this regex matches the specific nums of sub and ses.
    pattern = re.compile(".*sub-(\d*)\\\\ses-(\d*)")

    for session in task_sessions:
        task_runs = list(
            glob.glob(f"{session}\\{task_name}_run-[1-4].feat"))
        match = pattern.match(session)
        subnum = match.groups()[0]
        sesnum = match.groups()[1]
        if len(task_runs) == 3:
            fsf_template = f"template_{task_name}_level_2_3runs.fsf"
            ses_runnum_list += [[f"sub-{subnum}", f"ses-{sesnum}", 3]]
        elif len(task_runs) == 2:
            fsf_template = f"template_{task_name}_level_2_2runs.fsf"
            ses_runnum_list += [[f"sub-{subnum}", f"ses-{sesnum}", 2]]
        elif len(task_runs) == 4:
            fsf_template = f"template_{task_name}_level_2_4runs.fsf"
            ses_runnum_list += [[f"sub-{subnum}", f"ses-{sesnum}", 4]]
        elif len(task_runs) == 1:
            fsf_template = f"template_{task_name}_level_2_1run.fsf"
            ses_runnum_list += [[f"sub-{subnum}", f"ses-{sesnum}", 1]]
        else:
            continue
        print(f"sub-{subnum}_ses-{sesnum}")
        replacements = {'SUBNUM': subnum, 'SESNUM': sesnum}
        with open(f"{fsfdir}\\{fsf_template}") as infile:
            with open(f"{fsfdir}\\design_2nd_level_task-{task_name}_sub-{subnum}_ses-{sesnum}.fsf", 'w') as outfile:
                for line in infile:
                    for src, target in replacements.items():
                        line = line.replace(src, target)
                    outfile.write(line)

#%% 1.7.4. Run 2nd level GLM:
# This is a Python script that is intended to run through wsl linux machine in order to conduct an FSL 2nd level analysis (hence the wierd '/mnt/H...' paths)
# Calls fsl feat on all fsf design files of a certain task (after they were produced, see above...)

# SAVE this script on a seperate .py file, and run through linux command prompt with FSL installed.

# Inputs:
main_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives'
fsfsdir = '/models/fsfs/level_2'
task_name = 'fmrinfpractice'
###

# get all the level 2 fsfs files
fsffiles = glob.glob('%s%s/%s_ses_averaged_runs/design_2nd_level_task-%s_sub-*_ses-*.fsf' % (main_dir, fsfsdir,
                     task_name, task_name))  # name of specific fsfs lib is hard coded (see suffix 'ses_averaged_runs')

for fsf in list(fsffiles):
    os.system("feat %s" % (fsf))

# 2nd level FSL batch script

# Run 2nd level GLM:
# This is a Python script that is intended to run through wsl linux machine in order to conduct an FSL 2nd level analysis (hence the wierd '/mnt/h...' paths)
# Calls fsl feat on all fsf design files of a certain task (after they were produced, see above...)

# SAVE this script on a seperate .py file, and run through linux command prompt with FSL installed.


# Inputs:
main_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives'
fsfsdir = '/models/fsfs/level_2'
task_name = 'fmrinfpractice'
###

# get all the level 2 fsfs files
fsffiles = glob.glob('%s%s/%s_ses_averaged_runs/design_2nd_level_task-%s_sub-*_ses-*.fsf' % (main_dir, fsfsdir,
                     task_name, task_name))  # name of specific fsfs lib is hard coded (see suffix 'ses_averaged_runs')

for fsf in list(fsffiles):
    os.system("feat %s" % (fsf))


#%% 1.7.5. Randomization for group statistics
def randomize_group_coding():
    # =============================================================================
    # This function randomizes coding for group allocation.
    # Input: the online subject allocation file. The original coding in it is:
    #    1 - no treatment group (n=17)
    #    2 - TEST: reward mesolimbic network group (n=34)
    # CONTROL: Randomized ROIs subgroups (n=34):
    #    3 - Arithmetic processing
    #    4 - Auditory imagery
    #    5 - Motor imagery
    #    6 - Spatial navigation
    #
    # Output: a blinded csv file with coding of NF (2-6) subjects to groups: either test=1 and cont = 2, or vice versa (test = 2 and cont = 1)
    #
    # =============================================================================
    from random import randrange
    lib_path = 'H:\projects\joy\general\subjects_allocation'
    alloc_file = pd.read_excel(
        f"{lib_path}\\online_group_allocation_all_subs_with_subgroups.xlsx")
    # drop the no treatment group
    alloc_file_fmri = alloc_file[alloc_file['group'] != 1]
    alloc_file_fmri_w_subgroups = alloc_file_fmri[:]

    # randomize the coding for the test group (either 1 or 2)
    test_group_code = randrange(1, 3, 1)
    # for fmri analyses:
    if test_group_code == 1:
        alloc_file_fmri['group'] = alloc_file_fmri['group'].replace(
            [2], [1])  # change the mesolimbic group to be coded as 1
        alloc_file_fmri['group'] = alloc_file_fmri['group'].replace(
            [3, 4, 5, 6], [2, 2, 2, 2])  # change the randomized ROI subgroups to be coded as 2
    elif test_group_code == 2:
        # change the randomized ROI subgroups to be coded as 1 (ml reward is already 2)
        alloc_file_fmri['group'] = alloc_file_fmri['group'].replace([3, 4, 5, 6], [
                                                                    1, 1, 1, 1])

    alloc_file_fmri.to_csv(
        f"{lib_path}\\group_allocation_blinded_fmri.csv", index=False)

    alloc_file_fmri_w_subgroups.rename(
        columns={'group': 'subgroups'}, inplace=True)
    alloc_file_fmri_w_subgroups['group'] = alloc_file_fmri['group']
    alloc_file_fmri_w_subgroups.to_csv(
        f"{lib_path}\\group_allocation_fmri_w_subgroups.csv", index=False)

    # for immunological analyses (with no treatment group allocation) - no treatment remains, and changed to 3. 1/2 are test control according to randomization
    alloc_file['group'] = alloc_file['group'].replace(
        [1], [0])  # temporarily changed
    if test_group_code == 1:
        alloc_file['group'] = alloc_file['group'].replace(
            [2], [1])  # change the mesolimbic group to be coded as 1
        # change the randomized ROI subgroups to be coded as 2
        alloc_file['group'] = alloc_file['group'].replace(
            [3, 4, 5, 6], [2, 2, 2, 2])
    elif test_group_code == 2:
        # change the randomized ROI subgroups to be coded as 1 (ml reward is already 2)
        alloc_file['group'] = alloc_file['group'].replace(
            [3, 4, 5, 6], [1, 1, 1, 1])

    alloc_file['group'] = alloc_file['group'].replace(
        [0], [3])  # no treatment code

    alloc_file.to_csv(
        f"{lib_path}\\group_allocation_blinded_for_immunological_analyses.csv")

#%% 1.8. Create design matrix for FSL fmri group statistics

def create_design_matrix_for_fsl_fmri_group_statistics():
    # =============================================================================
    # This function creates a design matrix for fmri group statistics according to FSL FEAT requirements and specific models
    # Input: the randomize_group_coding() alloc_file_fmri saved csv file with the randomized groups (test/control) coding
    #
    # Output: a tsv file that contains the evs for group statistics ordered by subjects (N=68), aligned with fsl FEAT requirements
    #         and according to each specific model (i.e. a different file for each model...)
    #         this tsv file is intended to be manually copied into the 'full model setup' paste option when defining the evs.
    #
    # =============================================================================
    # 1. load the alloc_file_fmri saved csv file
    lib_path = 'H:\projects\joy\general\subjects_allocation'

    # model: fmrinfpractice session 4, for all subs, test, control and test vs control contrast maps
    alloc_file_fmri = pd.read_csv(
        f"{lib_path}\\group_allocation_blinded_fmri.csv", header=None)
    # 2. drop the headlines and subnum column
    alloc_file_fmri.drop(labels=0, inplace=True)
    alloc_file_fmri.reset_index(drop=True, inplace=True)
    alloc_file_fmri = alloc_file_fmri.drop(0, axis=1)
    # 3. add the ev's according to specific model requirements

    conditions = [
        alloc_file_fmri[1] == '1',
        alloc_file_fmri[1] == '2',
    ]
    outputs_ev1 = [1, 0]
    outputs_ev2 = [0, 1]

    alloc_file_fmri[2] = pd.Series(np.select(conditions, outputs_ev1))
    alloc_file_fmri[3] = pd.Series(np.select(conditions, outputs_ev2))
    design_ev_output = alloc_file_fmri.columns[2, 3]
    design_ev_output = alloc_file_fmri[[2, 3]].copy()

    design_ev_output.to_csv(
        f"{main_dir}\\derivatives\\models\\group_statistics\\fmrinfpractice\\whole_brain\\fmrinfpractice_ses4_test_vs_control_design_mat_evs.tsv", sep="\t", header=None, index=False)

    design_ev_output.to_csv(
        f"{main_dir}\\derivatives\\models\\group_statistics\\fmrinfpractice_roi_analysis.csv", sep=",", header=None, index=False)


# check how many ses files for group analysis
ses_4_dirs = glob.glob(
    f"{main_dir}\\derivatives\\fmriprep\\sub-*\\ses-4\\func\\{task_name}_lev2_averaged_runs.gfeat")

# mask creation (check if needed)

# create a functional mask for each individual 2nd level analysis and replace the original mask (which is not brain-only).
# This is a Python script that is intended to execute fsl commands through wsl linux OS (hence the wierd '/mnt/H...' paths)
#
# operations - for each level 2 (session-wise) gfeat task folder:
# 1) resample anatomical mask to individual funcitonal space (both are already in MNI space, as s the data, via fmriprep)
# 2) mask the original 2nd level mask within the cope folders with the resampled anatomical mask (after renaming the old one as mask_orig.nii.gz, not to lose anything)

# SAVE this script on a seperate .py file, make it executable and run through linux command prompt with FSL installed.

# Inputs:
subs_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep'
task_name = 'fmrinfpractice'  # name of task
###

# catch all individual anatomical MNI masks
subs_anat_masks = glob.glob(
    '%s/sub-*/anat/sub-1[1-2][0-9][0-9]_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz' % (subs_dir))

for ind_anat_mask in list(subs_anat_masks):
    index = ind_anat_mask.find('anat')
    # catch all task level 2 gfeat folders (pathname semi hardcoded, change accordingly) of subject
    sub_ses_paths = glob.glob(
        '%s/ses-*/func/%s_lev2_averaged_runs.gfeat' % (ind_anat_mask[:index], task_name))
    if len(sub_ses_paths) > 0:  # if there exist such folders do the following:
        print(ind_anat_mask)
        for ses in sub_ses_paths:  # go through all sessions
            if os.path.exists('%s/mask.nii.gz' % (ses)):
                # resample the anatomical mask to current session's functional dimensions
                os.system('flirt -in %s -ref %s/mask.nii.gz -out %s/mask_RS.nii.gz -applyxfm ' %
                          (ind_anat_mask, ses, ses))
                # catch cope dirs (that will be gathered for group analyses)
                copes_dirs = glob.glob('%s/cope*.feat' % (ses))
                for cope in copes_dirs:  # go through all cope dirs
                    # rename the original mask that FSL's feat created to mask_orig.nii.gz
                    os.rename(cope + '/mask.nii.gz',
                              cope + '/mask_orig.nii.gz')
                    # mask this original mask with resapled anatomical mask, and name it mask.nii.gz for group analyses
                    os.system(
                        'fslmaths %s/mask_orig.nii.gz -mas %s/mask_RS.nii.gz %s/mask.nii.gz' % (cope, ses, cope))
            else:
                print('no functional mask detected in' + ses)

#%% 2. Immunologiccal data preprocessing

# TP5 in the data refers to a time-point relevant only for cytokines analysis (is not used below)
# In the paper it is no mentioned, thus TP6, 7 and 8 here correspond to 5,6 and 7 in the paper. 

hbv_data = pd.read_excel(
    r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_raw.xlsx')

hbv_data[hbv_data[['TP1', 'TP4', 'TP5', 'TP6', 'TP7', 'TP8']] < 1] = 1 # if below 1 than equal to 1 (needed for conversion to logarithmic scale)
hbv_data[['TP1', 'TP4', 'TP5', 'TP6', 'TP7', 'TP8']] = np.log10(
    hbv_data[['TP1', 'TP4', 'TP5', 'TP6', 'TP7', 'TP8']])  # log 10
hbv_data['baseline'] = np.mean(hbv_data[['TP1', 'TP4']], axis=1)
# additive log difference and mean postvac values (TP6 and 7 change).
hbv_data['TP6 change'] = hbv_data['TP6'] - hbv_data['baseline']
hbv_data['TP7 change'] = hbv_data['TP7'] - hbv_data['baseline']
hbv_data['mean HBV postvac'] = np.mean(
    hbv_data[['TP6 change', 'TP7 change']], axis=1)
hbv_data['TP8 change'] = hbv_data['TP8'] - hbv_data['baseline']

hbv_data.to_excel(r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx', index=False)

#%% 3. Mental Strategies preprocessing
#%% 3.1. Mental Strategies Qualtrics Output Preprocessing
# import data
df = pd.read_excel(r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies/msq_qualtrics_output/MSQNF_joy_071122.xlsx')

# make the Q row as column names
df.columns = df.iloc[0]

# now remove the Q row
df = df.iloc[2:].reset_index(drop=True)

# temorary- organize session num column
df['Q2'] = df['Q2'].str.replace(';', '')
df['Q2'] = df['Q2'].str.replace('?', '')


# convert range of cycle num to list of cycle nums
def f(x):
    result = []
    for part in x.split(','):
        if '-' in part:
            a, b = part.split('-')
            a, b = int(a), int(b)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return result


df['Q3'] = df['Q3'].apply(lambda x: f(x))
df['Q3'] = df['Q3'].apply(lambda x: str(x)[1:-1])

# create rows for each cycle
df['Q3'] = df['Q3'].str.split(', ')
df = df.explode('Q3')
df = df.reset_index(drop=True)

# convert to int
df['Q3'] = df['Q3'].astype('int')
df['Q2'] = df['Q2'].astype('int')

######################### organize sensory exteroception #########################
# if didn't use (Q5=2) or don't remember - write 0 in Q6
df.loc[(df.Q5 == 2) | (df.Q5 == 3), 'Q6'] = 0
df['Q6'] = df['Q6'].fillna(0)

# sensoey exetorception categories
sensory_ext = df['Q6'].str.join(sep='*').str.get_dummies(sep='*')
sensory_ext.drop(columns=sensory_ext.columns[0], axis=1, inplace=True)
sensory_ext = sensory_ext.reset_index(drop=True)
# change columns name:
sensory_ext_name = {'1': 'vision_exteroception',
                    '2': 'auditory_exteroception',
                    '3': 'taste_exteroception',
                    '4': 'smell_exteroception',
                    '5': 'tactile_exteroception'}

sensory_ext.rename(columns=sensory_ext_name, inplace=True)

# remove Q5, Q6
df = df.drop(['Q5', 'Q6'], axis=1)
# add sensory exteroception columns
df = df.join(sensory_ext)

######################### organize sensory interoception #########################
# if didn't use (Q10=2) - write 0 in Q11
df.loc[(df.Q10 == 2) | (df.Q10 == 3), 'Q11'] = 0
df['Q11'] = df['Q11'].fillna(0)
# sensoey exetorception categories
sensory_inter = pd.get_dummies(df['Q11'], drop_first=True, dtype=int)
# change columns names
sensory_inter_name = {1: 'pulse',
                      2: 'breathing',
                      3: 'viceral_sensations',
                      4: 'muscle_sensation',
                      5: 'interoception_other'}
sensory_inter.rename(columns=sensory_inter_name, inplace=True)

# remove Q10, Q11
df = df.drop(['Q10', 'Q11'], axis=1)
# add sensory exteroception columns
df = df.join(sensory_inter)

######################### organize sensory imagery #########################
# if didn't use (Q7=2) - write 0 in Q8
df.loc[(df.Q7 == 2) | (df.Q7 == 3), 'Q8'] = 0
df['Q8'] = df['Q8'].fillna(0)

# sensoey imagination categories
sensory_img = df['Q8'].str.join(sep='*').str.get_dummies(sep='*')
sensory_img.drop(columns=sensory_img.columns[0], axis=1, inplace=True)

# change columns name:
sensory_img_name = {'1': 'vision_imagery',
                    '2': 'auditory_imagery',
                    '3': 'taste_imagery',
                    '4': 'smell_imagery',
                    '5': 'tactile_imagery'}

sensory_img.rename(columns=sensory_img_name, inplace=True)

# remove Q7, Q8
df = df.drop(['Q7', 'Q8'], axis=1)
# add sensory exteroception columns
df = df.join(sensory_img)

######################### organize emotions #########################
# categories for emotion
emotions = df['Q12'].str.get_dummies(sep=',')
emotions = emotions[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                     '12', '15', '16']]  # change order of columns
# change columns name
emotion_name = {'1': 'happiness',
                '2': 'love',
                '3': 'calmness',
                '4': 'sadness',
                '5': 'anger',
                '6': 'fear',
                '7': 'distress',
                '8': 'worry',
                '9': 'frustration',
                '10': 'tension',
                '11': 'other_emotion',
                '12': 'no_emotion',
                '15': 'pleasure',
                '16': 'disgust'}

emotions.rename(columns=emotion_name, inplace=True)
emotions.drop(columns=['no_emotion'], axis=1, inplace=True)

# remove Q12 and text
df = df.drop(['Q12', 'Q12_11_TEXT'], axis=1)
# add sensory exteroception columns
df = df.join(emotions)

######################### organize motivatio n#########################
# get categories
motivation = pd.get_dummies(df['Q28'], dtype=int)

# change columns names
motivation_name = {1: 'positive_expectation',
                   2: 'negative_expectation',
                   4: 'neutral_motivation',
                   7: 'dont_remember'}
motivation.rename(columns=motivation_name, inplace=True)
# neutral case is instantiated in the other columns
motivation.drop(columns=['neutral_motivation'], axis=1, inplace=True)
# instantiate 'dont_remember' as nan by changing other columns to nan
motivation.loc[motivation['dont_remember'] == 1, 'positive_expectation'] = np.nan
motivation.loc[motivation['dont_remember'] == 1, 'negative_expectation'] = np.nan

# insert nan to 'positive_expectation' and 'negative_expectation' columns where df['Q28'] == nan
motivation.loc[df['Q28'].isnull(), 'positive_expectation'] = np.nan
motivation.loc[df['Q28'].isnull(), 'negative_expectation'] = np.nan

# drop 'dont_remember' column
motivation.drop(columns=['dont_remember'], axis=1, inplace=True)

# remove Q12 and text
df = df.drop(['Q28'], axis=1)
# add motivation columns
df = df.join(motivation)

######################### change columns names #########################
change_col_name = {'Q1': 'sub_num',
                   'Q2': 'session_num',
                   'Q3': 'cycle_num',
                   'Q4': 'strategy',
                   'Q9': 'motor_imagery',
                   'Q13': 'memory',
                   'Q14': 'imagination',
                   'Q26': 'lingual',
                   'Q29': 'navigation',
                   'Q15': 'rhythmic',
                   'Q16': 'arithmetic',
                   'Q17': 'interface_engaged_detached',
                   'Q18': 'involving_one_or_many_strategies',
                   'Q19': 'arousal',
                   'Q20': 'valence',
                   'Q21': 'intercation_with_other_people',
                   'Q22': 'agency'}

df.rename(columns=change_col_name, inplace=True)

######################### organize arousal labeling #########################
df['arousal'] = df['arousal'].replace(1, 0)  # high arousal
df['arousal'] = df['arousal'].replace(2, np.nan)  # neutral
df['arousal'] = df['arousal'].replace(3, 1)  # low arousal
df['arousal'] = df['arousal'].replace(5, np.nan)  # dont remember

######################### organize valence labeling #########################
df['positive_valence'] = 0
df['negative_valence'] = 0
df['neutral_valence'] = 0

df.loc[(df['valence'] == 1) | (df['valence'] == 4), 'positive_valence'] = 1
df.loc[df['valence'] == 2, 'neutral_valence'] = 1
df.loc[(df['valence'] == 3) | (df['valence'] == 4), 'negative_valence'] = 1

df = df.drop(['valence'], axis=1)

######################### organize num strategies labeling #########################
df['involving_one_or_many_strategies'] = df['involving_one_or_many_strategies'].replace(
    1, 0)
df['involving_one_or_many_strategies'] = df['involving_one_or_many_strategies'].replace(
    2, 1)
df['involving_one_or_many_strategies'] = df['involving_one_or_many_strategies'].replace(
    3, np.nan)

######################### organize other labeling #########################

# 1 = Yes, 2 = no, 3 = don't remember
same_label_cols = ['motor_imagery', 'rhythmic',
                   'arithmetic', 'interface_engaged_detached',
                   'intercation_with_other_people', 'memory', 'imagination',
                   'agency']

for col in same_label_cols:
    df[col] = df[col].replace(2, 0)
    df[col] = df[col].replace(3, np.nan)


# 1 = Yes, 2 = no, 4 = don't remember
same_label_cols = ['lingual', 'navigation']

for col in same_label_cols:
    df[col] = df[col].replace(2, 0)
    df[col] = df[col].replace(4, np.nan)

#########################add identifying columns#########################
df['study'] = 2
df['neural_target'] = 'TBD'
df['interface_code'] = 6

#########################add meta_categories#########################
# create list of meta categories columns
sensory_exteroception = ['vision_exteroception', 'auditory_exteroception',
                         'smell_exteroception', 'taste_exteroception',
                         'tactile_exteroception']

somatic_sensations = ['pulse', 'breathing', 'viceral_sensations', 'muscle_sensation',
                      'interoception_other']

affect = ['happiness', 'love', 'calmness', 'sadness', 'anger', 'fear', 'distress',
          'worry', 'frustration', 'tension', 'other_emotion', 'pleasure',
          'disgust', 'positive_expectation', 'negative_expectation']

imagery_exteroception = ['vision_imagery', 'auditory_imagery', 'smell_imagery',
                         'taste_imagery', 'tactile_imagery']

current_senations = sensory_exteroception + somatic_sensations + affect

episodic_semantic = imagery_exteroception + ['memory', 'imagination',
                                             'motor_imagery', 'lingual',
                                             'arithmetic',
                                             'intercation_with_other_people',
                                             'navigation']

manner = ['rhythmic', 'interface_engaged_detached',
          'involving_one_or_many_strategies']

content = current_senations + episodic_semantic

# function that adds column by the conditions


def add_meta_category(df, col_list, meta_category):
    # create list of conds
    conds = []
    for col_name in col_list:
        cond = (df[col_name] == 1)
        conds.append(cond)

    # add column
    values = [1] * len(col_list)
    df[meta_category] = np.select(conds, values, default=0)


# add columns
add_meta_category(df, sensory_exteroception, 'Sensory_Exteroception')
add_meta_category(df, somatic_sensations, 'Somatic_Sensations')
add_meta_category(df, affect, 'Affect')
add_meta_category(df, imagery_exteroception, 'Imagery_Exteroception')
add_meta_category(df, manner,  'Manner')
add_meta_category(df, current_senations, 'Current_Sensations')
add_meta_category(df, episodic_semantic, 'Episodic_Semantic')
add_meta_category(df, content, 'Content')

# add psychological dimension meta category
psychological_dimension_conds = [df['positive_valence'].notnull(),
                                 df['negative_valence'].notnull(),
                                 df['neutral_valence'].notnull(),
                                 df['arousal'].notnull(),
                                 df['agency'] == 1]
psychological_dimension_vals = [1, 1, 1, 1, 1]

df['Psychological_Dimensions'] = np.select(
    psychological_dimension_conds, psychological_dimension_vals, default=0)

#########################change columns order#########################
df = df[['study', 'neural_target', 'interface_code', 'sub_num', 'session_num',
         'cycle_num', 'strategy', 'vision_exteroception', 'auditory_exteroception',
         'taste_exteroception', 'smell_exteroception', 'tactile_exteroception',
         'pulse', 'breathing', 'viceral_sensations', 'muscle_sensation',
         'interoception_other', 'vision_imagery', 'auditory_imagery',
         'taste_imagery', 'smell_imagery', 'tactile_imagery', 'motor_imagery',
         'navigation', 'arithmetic',  'lingual', 'memory', 'imagination',
         'intercation_with_other_people', 'rhythmic', 'interface_engaged_detached',
         'involving_one_or_many_strategies', 'agency', 'positive_valence', 'negative_valence',
         'neutral_valence',  'arousal', 'positive_expectation', 'negative_expectation',
         'happiness', 'love', 'calmness', 'pleasure', 'sadness', 'anger', 'fear',
         'distress', 'worry', 'frustration', 'tension', 'disgust',
         'other_emotion', 'Sensory_Exteroception', 'Somatic_Sensations', 'Affect',
         'Imagery_Exteroception', 'Manner', 'Current_Sensations', 'Episodic_Semantic',
         'Content', 'Psychological_Dimensions']]

#########################save to excel#########################
df.to_excel(r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies/msq_coded.xlsx', index=False)

#%% 3.2 combine NF report choice data with MSQ coding for mental strategies analysis

online1 = pd.read_csv(
    r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies/online_strategies_info_1122-1138.csv')
online2 = pd.read_csv(
    r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies/online_strategies_info.csv')
msq_coded = pd.read_excel(
    r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies/msq_coded.xlsx')

online = pd.concat([online1, online2])
online.drop(['strategy_id'], axis=1, inplace=True)
online.drop_duplicates(inplace=True)
online['strategy_label'] = online['strategy_label'].astype(str)

msq_coded['sub_num'] = msq_coded['sub_num'].astype(int)

# combine to one df
df = pd.merge(online, msq_coded, how='outer', on=[
              'sub_num', 'session_num', 'cycle_num'])

# if we have strategy but no label, create label
df['strategy_label'] = df['strategy_label'].fillna(df['strategy'])

# get only strats with strategy labels + strategy description
df_u = df[(df['strategy'].notna())]


# get strat id for combination of sub, sess, label, desc
df_u['ident'] = list(zip(df_u['sub_num'], df_u['session_num'],
                     df_u['strategy_label'], df_u['strategy']))

s = df_u[['ident']].stack()
s[:] = s.factorize()[0] + 1
df_u = s.unstack(1).add_suffix('_id').join(df_u)

df_u = df_u[['ident_id', 'sub_num', 'session_num',
             'cycle_num', 'strategy_label', 'strategy']]
df_u['ident_id'] = df_u['ident_id'].astype(int)


# return strat id to non-unique df
df1 = df.merge(df_u, on=['sub_num', 'session_num',
               'cycle_num', 'strategy_label', 'strategy'], how='outer')

# change column order
df1 = df1[['study', 'neural_target', 'interface_code', 'sub_num', 'session_num',
           'cycle_num', 'strategy_label', 'strategy', 'ident_id', 'vision_exteroception', 'auditory_exteroception',
           'taste_exteroception', 'smell_exteroception', 'tactile_exteroception',
           'pulse', 'breathing', 'viceral_sensations', 'muscle_sensation',
           'interoception_other', 'vision_imagery', 'auditory_imagery',
           'taste_imagery', 'smell_imagery', 'tactile_imagery', 'motor_imagery',
           'navigation', 'arithmetic',  'lingual', 'memory', 'imagination',
           'intercation_with_other_people', 'rhythmic', 'interface_engaged_detached',
           'involving_one_or_many_strategies', 'agency', 'positive_valence', 'negative_valence',
           'neutral_valence',  'arousal', 'positive_expectation', 'negative_expectation',
           'happiness', 'love', 'calmness', 'pleasure', 'sadness', 'anger', 'fear',
           'distress', 'worry', 'frustration', 'tension', 'disgust',
           'other_emotion', 'Sensory_Exteroception', 'Somatic_Sensations', 'Affect',
           'Imagery_Exteroception', 'Manner', 'Current_Sensations', 'Episodic_Semantic',
           'Content', 'Psychological_Dimensions']]


# copy strat id to all strategies from same session with same strategy label
# create df for relevent columns in order to drop nan values
#label_id = new[['strategie_label', 'ident_id']]
def copy_strat_id_per_sub(df, sub):
    cur_df = df[(df['sub_num'] == sub)]
    cur_df['ident_id'] = cur_df.groupby(['strategy_label'])['ident_id'].ffill()
    #cur_df['ident_id2'] = cur_df['ident_id'].fillna(cur_df['strategie_label'].map(label_id))

    return cur_df


subs = df1['sub_num'].unique()
lst_of_df = [copy_strat_id_per_sub(df1, sub) for sub in subs]

df2 = pd.concat(lst_of_df).drop_duplicates()
ident_id_col = df2['ident_id']

df_no_id = df2[df2['ident_id'].isna()]


# copy labeling of strategies with same ident_id
df2 = df2.groupby('ident_id').ffill()
df2['ident_id'] = ident_id_col

# change column order for df2
df2 = df2[['study', 'neural_target', 'interface_code', 'sub_num', 'session_num',
           'cycle_num', 'strategy_label', 'strategy', 'ident_id', 'vision_exteroception', 'auditory_exteroception',
           'taste_exteroception', 'smell_exteroception', 'tactile_exteroception',
           'pulse', 'breathing', 'viceral_sensations', 'muscle_sensation',
           'interoception_other', 'vision_imagery', 'auditory_imagery',
           'taste_imagery', 'smell_imagery', 'tactile_imagery', 'motor_imagery',
           'navigation', 'arithmetic',  'lingual', 'memory', 'imagination',
           'intercation_with_other_people', 'rhythmic', 'interface_engaged_detached',
           'involving_one_or_many_strategies', 'agency', 'positive_valence', 'negative_valence',
           'neutral_valence',  'arousal', 'positive_expectation', 'negative_expectation',
           'happiness', 'love', 'calmness', 'pleasure', 'sadness', 'anger', 'fear',
           'distress', 'worry', 'frustration', 'tension', 'disgust',
           'other_emotion', 'Sensory_Exteroception', 'Somatic_Sensations', 'Affect',
           'Imagery_Exteroception', 'Manner', 'Current_Sensations', 'Episodic_Semantic',
           'Content', 'Psychological_Dimensions']]


# remove nan
df2 = df2[df2['study'].notna()]
# save to excel
df2.to_excel(r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies/fmrinfpractice_msq_data.xlsx', index=False)


path = '/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies'
msq_data = pd.read_excel(f"{path}/fmrinfpractice_msq_data.xlsx")

# add neural targets
group_alloc = pd.read_excel('/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/general/subjects_allocation/online_group_allocation_all_subs_with_subgroups.xlsx')
for ind in msq_data.index:
    sub_ind = group_alloc.index[group_alloc['sub_num']== msq_data['sub_num'][ind]].tolist()[0]
    msq_data['neural_target'][ind] = group_alloc['group'][sub_ind]

# add run number

def run_label_func(val):
    if val <= 5:
        return 1
    elif val > 5 and val <= 10:
        return 2
    elif val > 10 and val <= 15:
        return 3
    elif val > 15 and val <= 20:
        return 4
    else:
        return np.nan

# apply the function to create a new column 'col2'
msq_data['run_num'] = msq_data['cycle_num'].apply(run_label_func)

cols = msq_data.columns.tolist()
cols_reorder = cols[:5] + [cols[-1]] + cols[5:-1]
msq_data = msq_data[cols_reorder]

msq_data.to_excel(r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies/fmrinfpractice_msq_data.xlsx', index=False)
