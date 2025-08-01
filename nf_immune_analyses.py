"""
nf_immune_analyses.py

This code generates all the statistical analyses reported in the manuscript:
"Upregulation of Reward Mesolimbic Activity via fMRI-Neurofeedback Improves Vaccination Efficiency in Humans"
Data preprocessing and generation of data files are performed in the code "nf_immune_data_preprocessing.py".

The code is organized in the following sections:
    1. Import libraries
    2. Define paths and global variables
    3. Main Analyses
        - 3.1. NF regulation effects
        - 3.2. Brain-Immune associations
        - 3.3. Mental Strategies analysis
    4. Demographics and clinical baseline data
    5. Supplementary analyses (reported in Supplementary Information)

Author:
    Nitzan Lubianiker, PhD.
    nitsan.lubianiker@yale.edu

Date:
    2024-07-25
"""

#################################################################################
# 1. Import Libraries
##################################################################################
import pandas as pd
import numpy as np
import statistics as stat
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from math import pi
import math
from scipy import special
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
import statsmodels.api as sm
import re

#%%#################################################################################
# 2. Define paths and global variables
####################################################################################
#%%
out_path = '/Users/nitzanlubi/My Drive/Lab/brain_immune_paper/figures/figure2'
main_dir = "/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/fmri/offline"

#%%#################################################################################
# 3. Paper Analyses
####################################################################################
# %% 3.1. NF regulation effects
# %% 3.1.2. Extract ROI measures from FSL's Featquery and organize for analysis

#  Create CSF-excluded (via individual fmriprep csf masks) ROIs on MNI space - per subject
# this is to be run on python on linux wsl with ANTs and FSL installed.

subs_rois_folder = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs'
general_rois = glob.glob(f"{subs_rois_folder}/general_offline_ROIs/mid_*")
# catch all subjects (this will be used for extracting data from NF practice as well as mid and rest, so including all subs)
nf_subs_list = glob.glob(
    f"/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-*/anat")
# for each subject, take the CSF probabilistic mask in anat folder, exclude the general ML ROIs and save them in subs_rois_folder

for sub in nf_subs_list:
    ind_sub = sub.find('sub-') + 4
    subnum = sub[ind_sub:ind_sub + 4]
    csf_prob_mask = f"{sub}/sub-{subnum}_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz"
    sub_bold_ref = f"/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-{subnum}/ses-1/func/sub-{subnum}_ses-1_task-{task_name}_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz"
    csf_RS_mask = f"{sub}/sub-{subnum}_space-MNI152NLin2009cAsym_label-CSF_probseg_RS2.nii.gz"
    at = ApplyTransforms()
    at.inputs.input_image = csf_prob_mask
    at.inputs.reference_image = sub_bold_ref
    at.inputs.transforms = 'identity'
    at.inputs.output_image = csf_RS_mask
    res = at.run()

    # threshold to 0.5 and inverse-binarize ROI using fslmaths
    csf_thr_binv = f"{sub}/sub-{subnum}_space-MNI152NLin2009cAsym_label-CSF_probseg_RS2_thr05_binv.nii.gz"
    os.system(f"fslmaths {csf_RS_mask} -thr 0.5 -binv {csf_thr_binv}")

    # mask offline ML ROIs and save in sub folder
    curr_sub_dir = f"{subs_rois_folder}/sub-{subnum}"
    os.makedirs(curr_sub_dir)
    ind = 1
    for roi in general_rois:
        os.system(
            f"fslmaths {roi} -mas {csf_thr_binv} {curr_sub_dir}/ml_roi_{ind}")
        ind = ind + 1

#  *before 2* - Create dummy standard2example_func.mat file in all .feat\reg folders, for the Featquery to work.
#               This is similar to the Mumford workaround for the group level analysis, that is needed since the data was preprocessed and registered
#               via fmriprep and not FSL. The file created here is an identity matrix that makes sure the ROI (already in the MNI functional space
#               of the data) will not be moved by Featquery!

feat_folders = glob.glob(
    f"{main_dir}{func_dir}\\{task_name}_run-[1-4]_pmod_demeaned.feat")
for feat in list(feat_folders):
    # copy the specific run's mean_func.nii.gz into the folder and name it standard.nii.gz
    shutil.copy(f"{feat}\\reg\\example_func2standard.mat",
                f"{feat}\\reg\\standard2example_func.mat")


###  2) Extract contrasts beta values of rest, regulate, feedback and reg>rest, in PSC, from each run (Featquery) ### ADD choice stat1 ###

# running Featquery for all runs (on linux)
main_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-*/ses-*/func'
task_name = 'fmrinfpractice'
roi = 'ml_roi_1'  # change according to required roi name
feat_folders = glob.glob(f"{main_dir}/{task_name}_run-[1-4].feat")
# feat_1122_ses2 = glob.glob(f"/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-1122/ses-2/func/{task_name}_run-[1-4].feat")
pattern = re.compile(".*sub-(\d*).ses-(\d*).*(run-\d)")
for feat in feat_folders:
    match = pattern.match(feat)
    subnum = match.groups()[0]
    sesnum = match.groups()[1]
    runnum = match.groups()[2]
    os.system(
        f"/home/nitzan/fsl/bin/featquery 1 /mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-{subnum}/ses-{sesnum}/func/{task_name}_{runnum}.feat 4  stats/cope2 stats/cope3 stats/cope4 stats/cope5 featquery_{roi} -p -w /mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs/sub-{subnum}/{roi}.nii.gz")
    print(f"sub-{subnum} ses-{sesnum} {runnum}")

##############

# SET parameters for statistical analysis:

contrast = 'regVSrest'  # regVSrest / feedback
# reg>rest = 3; feedback = 2; regulate = 1; rest = 0.
contrast_num = 3

# CHOOSE which rois are averaged:

# VTA only
rois_name = 'VTA'
rois_list = ['ml_roi_3']
roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}/roi_analyses/offline_VTA"

# OR bilateral nucleus accumbens
rois_name = 'bilateral_nac'
rois_list = ['ml_roi_1', 'ml_roi_2']
roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}/roi_analyses/offline_bilateral_Nac"

#############

subs_list = glob.glob(
    f"{main_dir}\\derivatives\\fmriprep\\sub-*\\ses-4\\func\\{task_name}_run-1.feat")
group_roi_anal = pd.DataFrame()

for sub in subs_list:
    pattern = re.compile(".*sub-(\d*)")
    match = pattern.match(sub)
    subnum = match.groups()[0]
    sub_feat_folders = glob.glob(
        f"{main_dir}\\derivatives\\fmriprep\\sub-{subnum}\\ses-*\\func\\{task_name}_run-[1-4].feat")
    ml_sub_data = pd.DataFrame()
    for feat in sub_feat_folders:
        pattern_sub = re.compile(".*(ses-\d*).*(run-\d)")
        match_sub = pattern_sub.match(feat)
        ses = match_sub.groups()[0]
        run = match_sub.groups()[1]
        roi_data = pd.DataFrame()
        for roi in rois_list:
            roi_anal_report = f"{feat}\\featquery_{roi}\\report.txt"
            tmp = pd.read_csv(roi_anal_report, sep=' ', header=None)
            tmp = tmp[5]
            roi_data = pd.concat([roi_data, tmp], axis=1)

        ml_sub_data = pd.concat([ml_sub_data, pd.DataFrame(
            {f"{ses}_{run}": roi_data.mean(axis=1)})], axis=1)

    ml_sub_data.index = ['rest', 'regulate', 'feedback', 'reg>rest']
    ml_sub_data.to_csv(
        f"{roi_anal_dir}\\subjects_results\\sub-{subnum}_{task_name}_roi_analysis_{rois_name}_psc.csv")

    # choose contrast to extract according to contrast_num var. defined above.
    a = ml_sub_data.iloc[contrast_num]

    a = a.rename(f"sub-{subnum}")
    group_roi_anal = pd.concat([group_roi_anal, a], axis=1)

# prepare the data for R mixed effects analysis
group_roi_anal_t = group_roi_anal.T
cols = group_roi_anal_t.columns.tolist()
cols_reorder = cols[:5] + cols[-3:-2] + cols[5:8] + [cols[12]] + cols[8:11]
group_roi_anal_t = group_roi_anal_t[cols_reorder]
lib_path = 'H:\\projects\\joy\\general\\subjects_allocation'
alloc_file = pd.read_csv(f"{lib_path}\\group_allocation_blinded_fmri.csv")
b = alloc_file['group']
group_roi_anal_t.insert(0, 'group', b.to_frame())
group_roi_anal_t.insert(0, 'sub_id', range(1, 69))

# save csv with original session and run numbers per sub
# index=True for real subnums for correlations
group_roi_anal_t.to_csv(
    f"{roi_anal_dir}\\group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums.csv", index=True)


# erase nans and indent values to left:
out = (pd.DataFrame(group_roi_anal_t.apply(sorted, key=pd.isna, axis=1).to_list(),
                    index=group_roi_anal_t.index, columns=group_roi_anal_t.columns)
       .fillna('')
       )
out = out.iloc[:, 0:13]
out.rename(columns={'ses-1_run-1': 'T1', 'ses-1_run-2': 'T2', 'ses-2_run-1': 'T3', 'ses-2_run-2': 'T4', 'ses-2_run-3': 'T5', 'ses-2_run-4': 'T6', 'ses-3_run-1': 'T7',
                    'ses-3_run-2': 'T8', 'ses-3_run-3': 'T9', 'ses-3_run-4': 'T10', 'ses-4_run-1': 'T11', 'ses-4_run-2': 'T12', }, inplace=True)
# save csv
out.to_csv(f"{roi_anal_dir}\\group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums_indented.csv",
           index=True)  # index=True for real subnums for correlations


### to create indented data without session 1:
group_roi_anal_t = pd.read_csv(f"{roi_anal_dir}/group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums.csv")

# delete columns of session 1:
group_roi_anal_t = group_roi_anal_t.drop(['ses-1_run-1','ses-1_run-2'],axis=1)   

# erase nans and indent values to left:
out = (pd.DataFrame(group_roi_anal_t.apply(sorted, key=pd.isna, axis=1).to_list(),
                    index=group_roi_anal_t.index, columns=group_roi_anal_t.columns)
       .fillna('')
       )
out.rename(columns={'ses-2_run-1': 'T1', 'ses-2_run-2': 'T2', 'ses-2_run-3': 'T3', 'ses-2_run-4': 'T4', 'ses-3_run-1': 'T5',
                    'ses-3_run-2': 'T6', 'ses-3_run-3': 'T7', 'ses-3_run-4': 'T8', 'ses-4_run-1': 'T9', 'ses-4_run-2': 'T10','ses-4_run-3': 'T11' }, inplace=True)
# save csv
out.to_csv(f"{roi_anal_dir}/group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums_indented_no_ses1.csv",
           index=False)

# create NF activity dataframe per session per subject
NF_act = pd.read_csv(
    f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{rois_name}/group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums.csv")
NF_act_ses = pd.DataFrame(np.zeros([len(NF_act), 4]), columns=[
                          'ses-1', 'ses-2', 'ses-3', 'ses-4'])
for i in NF_act_ses.index:
    for c in NF_act_ses:
        ses_cols = [col for col in NF_act.columns if c in col]
        NF_act_ses.loc[i, c] = np.mean(NF_act.loc[i, ses_cols])

# add subnums and save to csv?
NF_act_ses['sub_num'] = NF_act['Unnamed: 0'].astype(str)
NF_act_ses['group'] = NF_act['group']
cols = NF_act_ses.columns.tolist()
cols = cols[-2:] + cols[:-2]
NF_act_ses = NF_act_ses[cols]

NF_act_ses.to_csv(f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{rois_name}/group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums_sessions.csv", index=False)

#%% 3.1.3. Statistical analyses and figures:

# %% 3.1.3.1 Reward ML ROI Analysis

# Prepare data for R analysis:
#######################

### SET PARAMETERS: ###
rois_name = 'rand_rois'  # VTA/bilateral_nac/rand_rois/ml_reward
contrast = 'regVSrest'  # regVSrest/Feedback (feedback condition is inspected for ses-4 activity rather than activity slopes)
#######################

roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}/roi_analyses/offline_{rois_name}"
###

NF_act = pd.read_csv(
    f"{roi_anal_dir}/group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums.csv")

TP_names = NF_act.columns.to_list()
TP_names = [e for e in TP_names if e not in (
    'group', 'subgroup','sub_num', 'Unnamed: 0', 'sub_id')]
###
if rois_name in ['bilateral_nac', 'VTA']:
    NF_act = pd.melt(NF_act, id_vars=[
                    'Unnamed: 0', 'sub_id', 'group'], value_vars=TP_names, var_name='time')
    NF_act[['session', 'run','run_serial']] = np.nan
else:  # rand_rois
    NF_act = pd.melt(NF_act, id_vars=['sub_num','sub_id', 'subgroup'], value_vars=TP_names, var_name='time')
    NF_act[['session', 'run','run_serial']] = np.nan

pattern = re.compile("ses-(\d)_run-(\d)")

for ind in NF_act.index:
    match = pattern.match(NF_act.loc[ind, 'time'])
    ses = match.groups()[0]
    run = match.groups()[1]
    NF_act.loc[ind, 'session'] = ses
    NF_act.loc[ind,'run'] = run
    ### make a time variable with serial run numbers going up 
    if ses == '1':
        NF_act.loc[ind, 'run_serial'] = run
    elif ses == '2':
        NF_act.loc[ind, 'run_serial'] = str(int(run) + 2)
    elif ses == '3':
        NF_act.loc[ind, 'run_serial'] = str(int(run) + 6)
    elif ses == '4':
        NF_act.loc[ind, 'run_serial'] = str(int(run) + 10)

NF_act['session'] = NF_act['session'].astype(int)
NF_act['run'] = NF_act['run'].astype(int)
NF_act['run_serial'] = NF_act['run_serial'].astype(int)

NF_act.rename(columns={'value': f"{rois_name}_act",
              'Unnamed: 0': 'sub_num'}, inplace=True)
cols = NF_act.columns.to_list()
cols_reorder = cols[:3] + cols[-3:-1] + cols[3:5] + [cols[-1]]
NF_act = NF_act[cols_reorder]

NF_act.to_csv(f"{roi_anal_dir}/group_roi_analysis_{rois_name}_{contrast}_psc_long.csv", index=False)

### From here on, move to R script "nf_immune_ROI_analysis.R" for the mixed effects analysis. ###

# Examine t-test simple effects for session 4
reward_ml = NF_act_plot[NF_act_plot['group']==2]['ses-4'].reset_index(drop=True)
rand_roi = NF_act_plot[NF_act_plot['group']==1]['ses-4'].reset_index(drop=True)

tvalue, pvalue = stats.ttest_ind(reward_ml, rand_roi, equal_var=False)
print(tvalue, pvalue)

# %% 3.1.3.2. Figure 2a: Neural Regulation effects per group across sessions
main_dir = "/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/fmri/offline"
rois = [ 'VTA', 'bilateral_Nac','rand_rois','ml_reward','inflam_r_ins']  # 'rand_rois' for rand_ROI NF subgroups plotting, 'ml_reward' for control group plotting
contrast = 'regVSrest' 
plot_subgroups = False  # True for rand_rois subgroups plotting, False for control group plotting
out_path = '/Users/nitzanlubi/My Drive/Lab/brain_immune_paper/figures/figure2'
for i in range(len(rois)):
    roi_name= rois[i]
        
    data_dir = f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{roi_name}"
    NF_act_plot = pd.read_csv(f"{data_dir}/group_roi_analysis_{roi_name}_{contrast}_psc_w_subnums_sessions.csv")

    # remove session 1 (is not included in the analysis):
    NF_act_plot.drop(["ses-1"], axis=1, inplace=True)

    TP_names = NF_act_plot.columns.to_list()
    if roi_name == 'rand_rois':
        # remove the rand_rois subgroups columns:
        TP_names = [e for e in TP_names if e not in ('subgroup', 'sub_num')]
        if plot_subgroups:
            # for rand_ROI NF subgroups plotting: add subgroup label:
            NF_act_plot.loc[NF_act_plot.loc[:, 'subgroup'] == 3, 'group label'] = 'arithmetic NF'
            NF_act_plot.loc[NF_act_plot.loc[:, 'subgroup'] == 4, 'group label'] = 'auditory imagery NF'
            NF_act_plot.loc[NF_act_plot.loc[:, 'subgroup'] == 5, 'group label'] = 'motor imagery NF'
            NF_act_plot.loc[NF_act_plot.loc[:, 'subgroup'] == 6, 'group label'] = 'spatial navigation NF'
        else: # plot everyone from the control group together
            NF_act_plot.loc[NF_act_plot.loc[:, 'subgroup'].isin([3,4,5,6]), 'group label'] = 'Control group'

    else:
        TP_names = [e for e in TP_names if e not in ('group', 'sub_num')]
        ##########
        NF_act_plot.loc[NF_act_plot.loc[:, 'group'] == 1, 'group label'] = 'Control group'
        NF_act_plot.loc[NF_act_plot.loc[:, 'group'] == 2, 'group label'] = 'Experimental group'

    dd = pd.melt(NF_act_plot, id_vars=['group label'],
                value_vars=TP_names, var_name='time')

    # title:
    titles = ['VTA', 'Nac', 'Control ROIs','Reward ML', 'Insula']
    title = titles[i]

    # plot_reg(dd, out_path, title)
    import matplotlib.lines as mlines

    def plot_reg(dd, out_path, title):
        # Define line styles for each group
        line_styles = ['-', '-']
        # Set up the plot
        plt.figure(figsize=(4, 3))
        if roi_name == 'rand_rois':
            if plot_subgroups:
                # Define a custom color palette for rand_rois subgroups
                custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            else:
                # Define a custom color palette for control group
                custom_palette = ['grey']
        
        else: # VTA and bilateral Nac
            custom_palette = ['black','grey']
        # Iterate over each group and plot the line
        # for i, group_label in enumerate(dd["group label"].unique()):
        #     group_data = dd[dd["group label"] == group_label]
        #     line_style = line_styles[i % len(line_styles)]  # Cycle through line styles

        sns.pointplot(x="time",
                    y="value",
                    # linestyles=line_style,
                    hue="group label",
                    errorbar='se',
                    dodge=True if dd["group label"].nunique() > 1 else None,
                    data=dd,
                    palette=custom_palette)
        # Change the line width using plt.plott
        for line in plt.gca().lines:
            line.set_linewidth(1.5)  # Set the line width to 2 (adjust as needed)

        plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1),fontsize=9)
        # Create custom legend
        # legend_elements = [mlines.Line2D([0], [0], color='blue', lw=2, linestyle=style) for style in line_styles]
        # plt.legend(handles=legend_elements, labels=list(dd["group label"].unique()), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=9)
        # axes labels
        plt.xlabel("Time", size=9)
        plt.xticks(size=9)
        plt.ylabel("BOLD%(Regulate>Watch)",size=9)
        # add title
        plt.title(f"{title} Regulation Effects", loc="center")
        plt.savefig(rf"{out_path}/NF_groups_{roi_name}_{contrast}_sessions_lineplots.jpg",
                    format='jpg', bbox_inches='tight',dpi=150)

    plot_reg(dd,out_path,title)

#%% 3.2. Brain-Immune associations (Figure 2b) and HBV postvac change group differences.

### SET PARAMETERS: ###
ml_rois = ['bilateral_nac','VTA', 'rand_rois', 'inflam_r_ins']
contrast = 'regVSrest'  # regVSrest / feedback
neural_marker = 'slopes'  # slopes / ses-4 activity (for feedback condition)
#######################
cor_dir_path = r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/brain_behav_immune_corr/nf_immune'
paper_plot_path = r'/Users/nitzanlubi/My Drive/Lab/brain_immune_paper/plotting'

### hbv antibodies:
hbv_log = pd.read_excel(r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')
hbv_log = hbv_log[hbv_log['group'] != 3]  # drop the no NF group
hbv_log.reset_index(inplace=True)

# excluding non responders:
hbv_log = hbv_log[hbv_log['mean HBV postvac'] > 0.99].reset_index(drop=True)

for roi in ml_rois:
    
    # load nf data:
    roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{roi}"
    ml_activity_data = pd.read_csv(f"{roi_anal_dir}/group_roi_analysis_{roi}_{contrast}_psc_w_subnums_indented.csv")

    cor_test_data = pd.DataFrame()
    cor_test_data[['sub_num', 'group', 'mean HBV postvac','TP8 change']] = hbv_log[['subnum', 'group', 'mean HBV postvac', 'TP8 change']]

    ml_activity_data = ml_activity_data.rename(columns={'Unnamed: 0': 'sub_num'})

    ml_activity_data['sub_num'] = ml_activity_data['sub_num'].replace(
        {'sub-': ''}, regex=True)
    ml_activity_data['sub_num'] = ml_activity_data['sub_num'].astype(int)
    
    df_tmp = ml_activity_data.iloc[:, 3:]
    if neural_marker == 'slopes':

        # calculate activity slopes per subject
        for ind in list(range(0, len(df_tmp))):

            results = stats.linregress(list(
                range(1, len(df_tmp.iloc[ind, :].dropna())+1)), df_tmp.iloc[ind, :].dropna())
            ml_activity_data.loc[ind, f"{roi}_{contrast} slopes"] = results.slope
    elif neural_marker == 'ses-4 activity':
        # calculate ses 4 activity per subject, which is the mean of the last three runs
        for ind in list(range(0, len(df_tmp))):
            np.mean(df_tmp.iloc[ind, :].dropna().to_frame().iloc[-3:, :]).to_frame().iloc[0, 0]
            ml_activity_data.loc[ind, f"{roi}_{contrast} ses-4 activity"] = np.mean(
                df_tmp.iloc[ind, :].dropna().to_frame().iloc[-3:, :]).to_frame().iloc[0, 0]

    # insert activity markers to cor test data dataframe
    ml_activity_data = ml_activity_data[[
        'sub_num', f"{roi}_{contrast} {neural_marker}"]]

    cor_test_data = cor_test_data.merge(ml_activity_data, on='sub_num')

    # calculate correlations between mean postvac and ml activity markers via regression analysis:
    reg_brain_immune = stats.linregress(cor_test_data['mean HBV postvac'], cor_test_data[f"{roi}_{contrast} {neural_marker}"])
    print(f"{roi}:","r=", reg_brain_immune.rvalue, "p=", reg_brain_immune.pvalue)

    # calculate correlations between TP8 change and ml activity markers via regression analysis:
    # cor_test_data_TP8 = cor_test_data.dropna(subset=['TP8 change'])
    # reg_brain_immune_TP8 = stats.linregress(cor_test_data_TP8['TP8 change'], cor_test_data_TP8[f"{roi}_{contrast} {neural_marker}"])
    # print(f"{roi}:","r=", reg_brain_immune_TP8.rvalue, "p=", reg_brain_immune_TP8.pvalue)

    # plot the correlations:
    # plotting correlations:
    plt.figure(figsize=(4,3))
    sns.regplot(x=f"{roi}_{contrast} {neural_marker}",
                y="mean HBV postvac", data=cor_test_data, color='black', scatter=False)
    sns.scatterplot(x=f"{roi}_{contrast} {neural_marker}",
                    y="mean HBV postvac", hue="group", palette={1: 'grey', 2: 'black'}, data=cor_test_data, s=65, legend=False)
    plt.xlabel(f"")
    plt.xticks(fontsize=10)
    plt.ylabel("")
    plt.yticks(fontsize=10)
    # plt.title(f"{rois_name} {contrast} {neural_marker} vs. HBVab postvac change", fontsize=12)
    plt.savefig(rf"{out_path}/corr_{roi}_{contrast}_{neural_marker}_HBVab_wo_nonresponders.png",
                format='png',transparent=True, dpi=200)
    plt.show()

# Anova group differences in HBV antibodies postvac change

hbv_log = pd.read_excel(
    r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')

# plot the distribution of values via histograms:
plt.figure(figsize=(8, 6))
plt.hist(hbv_log[hbv_log['group'] == 1]['mean HBV postvac'],
         bins=20, color='red', label='control NF', alpha=0.5)
plt.hist(hbv_log[hbv_log['group'] == 2]['mean HBV postvac'],
         bins=20, color='blue', label='reward ML NF', alpha=0.5)
plt.hist(hbv_log[hbv_log['group'] == 3]['mean HBV postvac'],
         bins=20, color='grey', label='no NF', alpha=0.5)
plt.legend()

# for TP 8 change per group
data = pd.DataFrame({"group 1": hbv_log[hbv_log['group'] == 1]['TP8 change'].dropna(),
                     "group 2": hbv_log[hbv_log['group'] == 2]['TP8 change'].dropna(),
                     "group 3": hbv_log[hbv_log['group'] == 3]['TP8 change'].dropna()})
# for mean postvac change
data = pd.DataFrame({"group 1": hbv_log[hbv_log['group'] == 1]['mean HBV postvac'].dropna(),
                     "group 2": hbv_log[hbv_log['group'] == 2]['mean HBV postvac'].dropna(),
                     "group 3": hbv_log[hbv_log['group'] == 3]['mean HBV postvac'].dropna()})

data = data.reset_index(drop=True)

# drop non responders for mean postvac change
data[data.iloc[:, :] < 0.99] = np.nan

# anova
fvalue, pvalue = stats.f_oneway(data['group 1'].dropna(),data['group 2'].dropna(), data['group 3'].dropna())

# t-test comparisons between NF groups
hbv_log = pd.read_excel(
    r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')


data = pd.DataFrame({"control": hbv_log[hbv_log['group'] == 1]['mean HBV postvac'].dropna(),
                     "test": hbv_log[hbv_log['group'] == 2]['mean HBV postvac'].dropna()})

# drop non responders for mean postvac change
data[data.iloc[:, :] < 0.99] = np.nan

data = data.reset_index(drop=True)

# t-test
ttest = pg.ttest(data['control'], data['test'], paired=False) # paired=False for independent samples
print(ttest)

#%% 3.3. Mental Strategies analysis

# %% 3.3.1. Mental Strategies mixed effects regression analysis and plotting (Figure 3b)

#  This code tests the link between positive expectation and ROI regulation for VTA, Nac and randomized ROIs.

# import libraries
import os
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene
from scipy.stats import ttest_ind
import numpy as np
from pymer4.models import Lmer
from statsmodels.stats.multitest import multipletests

# Path to data
path_msq = '/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/mental_strategies'
path_nf = '/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/fmri/offline/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/denoised_psc/denoised_2-3TR' # get either denoised or non-denoised data
out_path = '/Users/nitzanlubi/My Drive/Lab/brain_immune_paper/analysis/msq'

# Task specific inputs:
task_name = 'fmrinfpractice'

# reward_ftrs = ['positive_valence', 'happiness', 'love', 'pleasure']
reward_ftrs = ['positive_expectation'] # only positive expectation

# dict for legend labels for reward features
reward_ftr_labels = {
    'positive_valence': 'Positive Valence',
    'happiness': 'Happiness',
    'love': 'Love',
    'pleasure': 'Pleasure',
    'positive_expectation': 'Positive Expectation'
}

plot = True

for roi in ['VTA','bilateralNac','rand_rois']:
    for reward_ftr in reward_ftrs:
        print(f"Running analysis for {roi} and {reward_ftr}")
        msq_data = pd.read_excel(f"{path_msq}/fmrinfpractice_msq_data.xlsx")
        nf_act_trials = pd.read_csv(f"{path_nf}/reg-rest_diff_psc_denoised_2-3TR_delay_{roi}.csv")

        # insert subjective performance quality measures into msq data.
        df_perf_quality = pd.read_excel(f"{path_msq}/fmrinfpractice_msq_subj_performance_report.xlsx")
        # add subj_performance_quality to msq_data. merge on sub_num, session_num, run_num and cycle_num
        msq_data = pd.merge(msq_data, df_perf_quality, on=[
                            'sub_num', 'session_num', 'run_num', 'cycle_num', 'strategy'])

        # deal with duplicates (subject 1135 has duplicate trials) every row that has the same sub_num, session_num, run_num and cycle_num should have only one copy of it (leave the first one)
        msq_data = msq_data.drop_duplicates(
            subset=['sub_num', 'session_num', 'run_num', 'cycle_num'], keep='first')
        msq_data = msq_data.reset_index(drop=True)

        # insert nf activity scores into datasets:
        msq_data['trial_id'] = msq_data['sub_num'].astype(str) + '_' + msq_data['session_num'].astype(
            str) + '_' + msq_data['run_num'].astype(str) + '_' + msq_data['cycle_num'].astype(str)
        nf_act_trials['trial_id'] = nf_act_trials['sub_num'].astype(str) + '_' + nf_act_trials['session_num'].astype(int).astype(
            str) + '_' + nf_act_trials['run_num'].astype(int).astype(str) + '_' + nf_act_trials['cycle_num_ses'].astype(int).astype(str)
        nf_act_trials = nf_act_trials[['trial_id', 'TR1', 'TR2', 'TR3', 'TR4', 'TR5', 'TR6', 'TR7', 'TR8', 'TR9', 'TR10', 'TR11', 'TR12', 'TR13', 'TR14', 'TR15', 'TR16', 'TR17', 'TR18', 'TR19', 'TR20','mean_psc_20tr']]

        # add mean of each quadrant to the nf_act_trials dataframe
        nf_act_trials['Early'] = nf_act_trials[['TR1', 'TR2', 'TR3', 'TR4', 'TR5']].mean(axis=1)
        nf_act_trials['Sustained'] = nf_act_trials[['TR6', 'TR7', 'TR8', 'TR9', 'TR10','TR11', 'TR12', 'TR13', 'TR14', 'TR15','TR16', 'TR17', 'TR18', 'TR19', 'TR20']].mean(axis=1)
        # nf_act_trials['Sustained'] = nf_act_trials[['TR11', 'TR12', 'TR13', 'TR14', 'TR15','TR16', 'TR17', 'TR18', 'TR19', 'TR20']].mean(axis=1)

        msq_data = pd.merge(msq_data, nf_act_trials, on='trial_id')
        msq_data.drop(['trial_id'], axis=1, inplace=True)
        cols = msq_data.columns.to_list()
        cols_reorder = cols[:7] + cols[-25:] + cols[7:-25]
        msq_data = msq_data[cols_reorder]

        # filtering based on subjective performance quality:
        # msq_data = msq_data[msq_data['subj_performance_quality'] > 2].reset_index(drop=True)

        # take only the relevant columns for the analysis
        cols_to_keep = ['sub_num', 'session_num', 'cycle_num', reward_ftr, 'Early','Sustained','mean_psc_20tr']
        msq_data = msq_data[cols_to_keep]
        # exclude rows with Nan values in reward feature column
        msq_data = msq_data[~msq_data[reward_ftr].isna()].reset_index(drop=True)
        # Include sessions 2, 3, and 4
        msq_data = msq_data[msq_data['session_num'].isin([2, 3, 4])].reset_index(drop=True)

        # Change to long format
        df = msq_data.melt(id_vars=['sub_num', 'session_num', 'cycle_num', reward_ftr],
                        value_vars=['Early', 'Sustained'],
                        var_name='quartile', value_name='activity')

        # Make quartile a categorical variable with specific order
        df['quartile'] = pd.Categorical(df['quartile'], categories=['Early', 'Sustained'], ordered=True)
        # Map session numbers to categorical labels
        df['session_label'] = df['session_num'].map({2: 'Session 2', 3: 'Session 3', 4: 'Session 4'})

        # Set 'pre' as the reference level
        df['session_label'] = pd.Categorical(df['session_label'], categories=['Session 2', 'Session 3', 'Session 4'])

        # Mixed linear model
        # model = smf.mixedlm(f"activity ~ session_label * quartile * {reward_ftr}", data=df, groups=df["sub_num"])
        # result = model.fit()
        # print(result.summary())

        # Convert sub_num to string (categorical IDs are better handled as strings in pymer4)
        df['sub_num'] = df['sub_num'].astype(str)

        # Run the model with pymer4
        # model = Lmer(f"activity ~ session_label * quartile * {reward_ftr} + (1|sub_num)", data=df)
        # alterrnative model with cycle_num as a fixed effect
        model = Lmer(f"activity ~ session_label * quartile * {reward_ftr} + cycle_num + (1|sub_num)", data=df)
        # alternative model with sub_num as a random slope
        # model = Lmer(f"activity ~ session_label * quartile * {reward_ftr} + cycle_num + (1 + session_label|sub_num)", data=df)

        results = model.fit()
        print(f"{roi} MLE:")
        print(results)
        # save the results to a csv file
        results_df = model.coefs
        # round all results to 3 decimal places
        results_df = results_df.round(3)
        results_df.to_csv(os.path.join(out_path, f"{roi}_{reward_ftr}_results_QBinary.csv"), index=True)
        if plot:
            # Run t-tests and collect results
            results_ttest = []
            for session in ['Session 2', 'Session 3', 'Session 4']:
                for q in ['Early', 'Sustained']:
                    subset = df[(df['session_label'] == session) & (df['quartile'] == q)]
                    a = subset[subset[reward_ftr] == 0]['activity']
                    b = subset[subset[reward_ftr] == 1]['activity']
                    tmp, p_var_test = levene(a, b)
                    if p_var_test < 0.05:
                        stat, p = ttest_ind(a, b, equal_var=False)
                        welch_dof = lambda a, b: ((np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b))**2) / (
                            (np.var(a, ddof=1)**2)/((len(a)**2)*(len(a)-1)) + (np.var(b, ddof=1)**2)/((len(b)**2)*(len(b)-1))
                        )
                        dof = welch_dof(a, b)
                    else:
                        stat, p = ttest_ind(a, b, equal_var=True)
                        dof = len(a) + len(b) - 2

                    cohen_d = (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
                    results_ttest.append({'session_label': session, 'quartile': q, 'T stat': stat, 'pval': p, 'DoF': dof, 'Cohens D': cohen_d})

            results_ttest_df = pd.DataFrame(results_ttest)
            results_ttest_df['pval_FDRcorrected'] = multipletests(results_ttest_df['pval'], method='fdr_bh')[1]
            results_ttest_df = results_ttest_df.round(3)
            results_ttest_df.to_csv(os.path.join(out_path, f"{roi}_{reward_ftr}_QBinary_ttest_results.csv"), index=False)

            # Convert to DataFrame and map p-values to stars
            pvals_df = pd.DataFrame(results_ttest)
            def p_to_star(p):
                if p < 0.001:
                    return '***'
                elif p < 0.01:
                    return '**'
                elif p < 0.05:
                    return '*'
                elif p < 0.1:
                    return '~'
                else:
                    return ''
            pvals_df['star'] = pvals_df['pval'].apply(p_to_star)

            summary_df = (
                df.groupby(["session_label", "quartile", reward_ftr])
                .agg(
                    mean=("activity", "mean"),
                    sem=("activity", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                )
                .reset_index()
            )

            def plot_activity_by_session_quartile(df, reward_ftr, roi, pvals_df, summary_df):
                sns.set(style="whitegrid", font_scale=1.2)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
                session_to_ax = {'Session 2': axes[0], 'Session 3': axes[1], 'Session 4': axes[2]}

                for session in ['Session 2', 'Session 3', 'Session 4']:
                    ax = session_to_ax[session]
                    session_data = df[df['session_label'] == session]
                    for pe_val, color, label in zip([0, 1], ['red', 'green'], ['No', 'Yes']):
                        pe_data = session_data[session_data[reward_ftr] == pe_val]
                        means = []
                        sems = []
                        for q in ['Early', 'Sustained']:
                            q_data = pe_data[pe_data['quartile'] == q]['activity']
                            means.append(q_data.mean())
                            sems.append(q_data.std() / np.sqrt(len(q_data)))
                        x = range(2)
                        ax.errorbar(
                            [i + (0.1 if pe_val == 0 else -0.1) for i in x],
                            means, yerr=sems,
                            color=color, marker='o',
                            label=f"{reward_ftr_labels[reward_ftr]}: {label}"
                        )

                    ax.set_xticks(range(2))
                    ax.set_xticklabels(['Early', 'Sustained'])
                    # ax.set_xlabel('Quartile')
                    ax.set_title(f"{session.capitalize()}")
                fig.text(0.01, 0.5, f"{roi} Activity", va='center', rotation='vertical', fontsize=14)
                handles, labels = axes[0].get_legend_handles_labels()
                # a x axis label for the whole figure
                fig.text(0.5, 0.04, "Temporal Component of Activity", ha='center', fontsize=16)
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=2)
                plt.suptitle(f"{roi} Activity Across Time and {reward_ftr_labels[reward_ftr]} Use", fontsize=18)
                plt.tight_layout(rect=[0.02, 0.1, 0.98, 0.95])
                plt.savefig(os.path.join(f"{out_path}/plots", f"{roi}_{reward_ftr}_QBinary_activity_by_session_quartile.png"))
                plt.show()

            def plot_by_expectation_panel(df, reward_ftr, roi):
                import seaborn as sns
                import matplotlib.pyplot as plt

                sns.set(style="whitegrid", font_scale=1.2)
                custom_palette = {
                    "Session 2": "lightgrey",
                    "Session 3": "grey",
                    "Session 4": "black"
                }

                g = sns.FacetGrid(
                    df,
                    col=reward_ftr,
                    hue='session_label',
                    palette=custom_palette,
                    height=5,
                    aspect=1
                )

                g.map_dataframe(
                    sns.pointplot,
                    x='quartile', y='activity',
                    errorbar='se', dodge=True, markers='o', linestyles='-'
                )

                g.add_legend(title='Session')
                g.set_axis_labels("Quartile", f"{roi} Activity")
                new_titles = {0: "Without Positive Expectation", 1: "With Positive Expectation"}
                for ax, col_val in zip(g.axes.flat, g.col_names):
                    ax.set_title(new_titles.get(col_val, str(col_val)))

                plt.suptitle(f"{roi} Activity Across Quartiles by Positive Expectation and Session", fontsize=16, y=1.05)
                plt.tight_layout()
                plt.savefig(os.path.join(f"{out_path}/plots", f"{roi}_{reward_ftr}_QBinary_activity_by_expectation.png"))
                plt.show()

            def plot_difference_by_session(df, reward_ftr, roi):
                diff_df = (
                    df.groupby(['session_label', 'quartile', reward_ftr])
                    .agg(mean_activity=('activity', 'mean'))
                    .reset_index()
                    .pivot_table(index=['session_label', 'quartile'], columns=reward_ftr, values='mean_activity')
                    .reset_index()
                )
                diff_df['difference'] = diff_df[1] - diff_df[0]

                sns.set(style="whitegrid", font_scale=1.2)
                plt.figure(figsize=(7, 4))
                sns.pointplot(
                    data=diff_df,
                    x='quartile', y='difference', hue='session_label',
                    markers='o', linestyles='-', palette=['lightgrey', 'grey', 'black']
                )
                plt.axhline(0, color='gray', linestyle='--')
                # plt.xlabel("Temporal Component of Activity", fontsize=18)
                plt.ylabel(f"Difference in {roi} Activity\nPositive Exp Yes-No", fontsize=14)
                plt.yticks(fontsize=16)
                plt.xticks(fontsize=16)
                # place the legend outside the plot
                plt.legend(title='Session', loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

                # plt.title(f"Positive Expectation Effect on {roi} Upregulation", fontsize=20)
                plt.tight_layout()
                plt.savefig(os.path.join(f"{out_path}/plots", f"{roi}_{reward_ftr}_QBinary_activity_dif_interaction.png"))
                plt.show()

            # Call all plots
            plot_activity_by_session_quartile(df, reward_ftr, roi, pvals_df, summary_df)
            # plot_by_expectation_panel(df, reward_ftr, roi)
            plot_difference_by_session(df, reward_ftr, roi)




#%%#################################################################################
# 4. Demographics
####################################################################################
#%% Extract demographics and create summary table
df = pd.read_excel(r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/subs_demographics.xlsx')
group_data = pd.read_excel(r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')
group_data = group_data[['subnum','group']]
df['group'] = np.nan
for ind in df.index:
    if df.loc[ind,'sub'] in group_data['subnum'].to_list():
        df.loc[ind,'group'] = group_data.loc[group_data['subnum']==df.loc[ind,'sub'],'group'].reset_index(drop=True)[0]
    else:
        print('shut')

df.loc[df['sub']==1192,'group'] = 3

# df_NF = df[df['group']!=3].reset_index(drop=True)

# read questionnaires data
df_quest = pd.read_csv(r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/questionnaires/clustered_data_raw.csv')
# add to df the data in df_quest based on 'sub' column
df = pd.merge(df,df_quest,on='sub',how='left')

col_names = df.columns.to_list()

col_names.remove('group')

# create the mean and std for each variable in the df, fro each group (group 1,2,3)
df_mean = df.groupby('group').mean().reset_index()
df_std = df.groupby('group').std().reset_index()

# run an anova for each variable in the df across the groups
anova_df = pd.DataFrame()
for col in col_names[2:]:
    aov = pg.anova(data=df, dv=col, between='group', detailed=True)
    # add a column with the variable name
    aov['variable'] = col
    anova_df = pd.concat([anova_df,aov],axis=0)

# leave only rows with Source='group' in the anova_df
anova_df = anova_df[anova_df['Source']=='group'].reset_index(drop=True)

# build the df_summary table. for each variable (rows), add the following columns: variable name (column 1), a mean and std as a string (mean(std)) for each group (columns 2,3 and 4, with names: 'group1', 'group2', group3')
df_summary = pd.DataFrame()
for col in col_names[2:]:
    mean1 = str(round(df_mean.loc[0,col],2)) + ' (' + str(round(df_std.loc[0,col],2)) + ')'
    mean2 = str(round(df_mean.loc[1,col],2)) + ' (' + str(round(df_std.loc[1,col],2)) + ')'
    mean3 = str(round(df_mean.loc[2,col],2)) + ' (' + str(round(df_std.loc[2,col],2)) + ')'
    df_summary = pd.concat([df_summary,pd.DataFrame({'variable':[col],'group1':[mean1],'group2':[mean2],'group3':[mean3]})],axis=0)
# regenerate indices to df_summary
df_summary = df_summary.reset_index(drop=True)

# now add the p-values from the anova_df to the df_summary as a new column
df_summary['p-value'] = np.nan
for ind in df_summary.index:
    var = df_summary.loc[ind,'variable']
    df_summary.loc[ind,'p-value'] = anova_df.loc[anova_df['variable']==var,'p-unc'].reset_index(drop=True)[0]

# calculate gender proportion in df (0 is male, 1 is female ) in each group and add to df_summary
# count how many females (gender=1) in each group
gender = df.groupby('group')['gender'].value_counts().unstack(fill_value=0)

# add a row to df_summary, with variable name = 'Gender (F/M)', and for each group column enter the number of females and males in the format 'F/M' string
# add an empty row to df_summary with variable column value of 'Gender (F/M)'
df_summary = pd.concat([df_summary,pd.DataFrame({'variable':'Gender (F/M)'},index=[0])],axis=0).reset_index(drop=True)
# enter the number of female and male in each group in the format 'F/M' to the df_summary from gender
for i in range(1,4):
    # create the string to be entered to the df_summary
    str_ = f"({gender.loc[i,1]}" + '/' + f"{gender.loc[i,0]})"
    df_summary.loc[df_summary['variable']=='Gender (F/M)',f'group{i}'] = str_

# add a row with the variable name 'N' to df_summary, and for each group column enter the number of subjects in the group based on df
df_summary = pd.concat([df_summary,pd.DataFrame({'variable':'N'},index=[0])],axis=0).reset_index(drop=True)
for i in range(1,4):
    df_summary.loc[df_summary['variable']=='N',f'group{i}'] = len(df[df['group']==i])

# reorder rows in df_summary to have the 'N' row first, then the gender row, and then the rest of the variables
df_summary = df_summary.reindex([12,11] + list(range(0,len(df_summary)-2))).reset_index(drop=True)

# change group column names to group names: '1' to 'Control NF', '2' to 'Reward-ML NF', '3' to 'no NF', and move the 'Reward-ML NF' column to the second place, after varibale name
df_summary.columns = ['variable','Control NF','Reward-ML NF','no NF','p-value']
df_summary = df_summary[['variable','Reward-ML NF','Control NF','no NF','p-value']]

# save to excel
df_summary.to_excel(r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/questionnaires/demographics_summary.xlsx', index=False)

#%%#################################################################################
# 5. Supplementary Results
####################################################################################
# %%  5.1. Monetary Incentive Delay and HBV antibodies:

### SET PARAMETERS: ###
rois_name = 'ml_reward'  # VTA/bilateral_nac/l_vmpfc
contrast_mid = 'anticipate'  # anticipate(for VTA/Nac)/consumption (for l_vmpfc)

#######################

cor_dir_path = r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/analyses/brain_behav_immune_corr/nf_mid'

roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/fmrimidcrrt/roi_analyses/offline_{rois_name}"
mid_activity_data = pd.read_csv(f"{roi_anal_dir}/group_roi_analysis_{rois_name}_{contrast_mid}_psc_w_subnums.csv")
mid_activity_data.rename(columns={'Unnamed: 0': 'sub_num'}, inplace=True)

### for hbv antibodies:
hbv_log = pd.read_excel(r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')
hbv_log.rename(columns={'subnum': 'sub_num'}, inplace=True)

cor_test_data = pd.DataFrame()
cor_test_data[['sub_num', f"{rois_name}_{contrast_mid}"]] = mid_activity_data[['sub_num', f"{rois_name}_{contrast_mid}"]]

a = hbv_log.loc[hbv_log['sub_num'].isin(cor_test_data['sub_num'])].reset_index()
cor_test_data = cor_test_data.merge(a, on='sub_num')

# cor_test_data = cor_test_data.merge(a, on='sub_num').dropna()

# calculate correlations between mean postvac and ml activity markers via regression analysis:
reg_MID_immune = stats.pearsonr(cor_test_data['mean HBV postvac'], cor_test_data[f"{rois_name}_{contrast_mid}"])

# without non responders

x = cor_test_data[cor_test_data['mean HBV postvac'] > 0.99]['mean HBV postvac']
y = cor_test_data[cor_test_data['mean HBV postvac'] > 0.99][f"{rois_name}_{contrast_mid}"]

reg_MID_immune_wo_nonres = stats.pearsonr(x, y)

# %% 5.2. EEfRT and HBV antibodies:
data = pd.read_excel(f"/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/analyses/brain_behav_immune_corr/brain-immune-eefrt/effrt_brain_immune.xlsx")
data = data [['sub','all%/any$','all%/high$']]
data.rename(columns={'sub': 'sub_num'}, inplace=True)

### for hbv antibodies:
hbv_log = pd.read_excel(r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')
hbv_log.rename(columns={'subnum': 'sub_num'}, inplace=True)
hbv_log = hbv_log[['sub_num', 'group', 'mean HBV postvac']]
# exclude non responders
hbv_log = hbv_log[hbv_log['mean HBV postvac'] > 0.99]
cor_test_data = pd.DataFrame()
# merge data
cor_test_data = data.merge(hbv_log, on='sub_num')

# correlate mean HBV postvac and any/all eefrt
reg_eefrt_hbv = stats.pearsonr(cor_test_data['all%/any$'], cor_test_data['mean HBV postvac'])
reg_eefrt_hbv_high = stats.pearsonr(cor_test_data['all%/high$'], cor_test_data['mean HBV postvac'])

# import NF slope effects in VTA
rois_name = 'VTA'
contrast = 'regVSrest'
neural_marker = 'slopes'
roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{rois_name}"
ml_activity_data = pd.read_csv(
    f"{roi_anal_dir}/group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums_indented.csv")
ml_activity_data = ml_activity_data.rename(columns={'Unnamed: 0': 'sub_num'})
ml_activity_data['sub_num'] = ml_activity_data['sub_num'].replace(
    {'sub-': ''}, regex=True)
ml_activity_data['sub_num'] = ml_activity_data['sub_num'].astype(int)

df_tmp = ml_activity_data.iloc[:, 3:]
# calculate activity slopes per subject
for ind in list(range(0, len(df_tmp))):
    results = stats.linregress(list(
        range(1, len(df_tmp.iloc[ind, :].dropna())+1)), df_tmp.iloc[ind, :].dropna())
    ml_activity_data.loc[ind, f"{rois_name}_{contrast} slopes"] = results.slope

# keep only the slopes and sub_num
ml_activity_data = ml_activity_data[['sub_num', f"{rois_name}_{contrast} slopes"]]

# import eefrt data again and merge with activity markers to cor test data dataframe 
data = pd.read_excel(f"/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/analyses/brain_behav_immune_corr/brain-immune-eefrt/effrt_brain_immune.xlsx")
data = data [['sub','all%/any$','all%/high$']]
data.rename(columns={'sub': 'sub_num'}, inplace=True)

cor_test_data = data.merge(ml_activity_data, on='sub_num')

# correlate NF slopes and any/all eefrt
reg_eefrt_nf = stats.pearsonr(cor_test_data['all%/any$'], cor_test_data[f"{rois_name}_{contrast} slopes"])
reg_eefrt_nf_high = stats.pearsonr(cor_test_data['all%/high$'], cor_test_data[f"{rois_name}_{contrast} slopes"])

#%% 5.3. K-Means clustering of questionnaires data and HBV antibodies
#%% 5.3.1. run k-means clustering on questionnaires data

path = '/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/questionnaires'

@dataclass
class Quest:
    name: str
    subscales: List[str]
    score_func: Callable
    naming_func: Callable = lambda x: x


def NEO5(org: str) -> str:
    temp = org.split("_")
    return "_".join(temp[:-1])


def two_to_zero_plus_rev(org: pd.Series, name: str) -> pd.Series:
    if "rev" in name:
        res = org.replace(1, 0)
        res = res.replace(2, 1)

    else:
        res = org.replace(2, 0)
    return res


def rev5(org: pd.Series, name: str) -> pd.Series:
    if "rev" in name:
        res = 5 - org + 1
    else:
        res = org
    return res


def preproc_df(all_data: pd.DataFrame, quests: List[Quest]) -> pd.DataFrame:
    res_df = pd.DataFrame()
    res_df['sub'] = all_data['sub']
    for col in all_data.columns:
        for q in quests:
            for scale in q.subscales:
                if scale in col:
                    res_df[col] = q.score_func(all_data[col], col)
                    break
    res_df.dropna(inplace=True)
    res_df.reset_index(inplace=True, drop=True)

    return res_df


def subscales_df(org_data: pd.DataFrame, quests: List[Quest]) -> pd.DataFrame:
    scores = [{t: 0 for t in ['sub'] + list(itertools.chain.from_iterable([q.subscales for q in quests]))} for _ in
              range(len(org_data))]
    for i, r in org_data.iterrows():
        scores[i]['sub'] = r['sub']
        for c, v in r.items():
            if c == "sub":
                continue
            real_name = c.split(".")[0].split("(")[0]
            if real_name[-1].isdigit():
                real_name = real_name[:-1]
            for q in quests:
                if q.name in real_name:
                    real_name = q.naming_func(real_name)
                    break
            scores[i][real_name] += v
    res = pd.DataFrame.from_records(scores)
    res.set_index('sub', inplace=True)
    return res


def normalize(org: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame()
    for c in org.columns:
        if c == "sub":
            continue
        c_mean = org[c].mean()
        c_std = org[c].std()
        res[c] = (org[c] - c_mean) / c_std
    return res


def two_step_Kmeans(data: pd.DataFrame):
    kmeans = KMeans(n_clusters=2).fit(data)
    data['cluster'] = kmeans.labels_

    for c in data.columns:
        if c == 'cluster':
            continue
        p = ttest_ind(data[data['cluster'] == 0][c], data[data['cluster'] == 1][c]).pvalue * (len(data.columns) - 1)
        print(f"{c}: {p:.5f}")
        if p > 0.05:
            data.drop(c, axis=1, inplace=True)
    kmeans = KMeans(n_clusters=2).fit(data)
    data['cluster'] = kmeans.labels_
    print(kmeans.cluster_centers_)
    return data

# altered to take imputed dataset
if __name__ == '__main__':
    quests = [Quest('TPQ', ['TPQ_NS', 'TPQ_HA', 'TPQ_RD'], two_to_zero_plus_rev),
              Quest('NEO', ['NEO-FFI_N', 'NEO-FFI_O', 'NEO-FFI_A', 'NEO-FFI_C', 'NEO-FFI_E'], rev5, NEO5),
              Quest('spsrq', ['spsrq_reward', 'spsrq_punishment'], two_to_zero_plus_rev)]
    raw = pd.read_csv(f"{path}/joy_qs_imputed.csv")
    ses_data = raw[raw['ses'] == 1]
    assert (len(ses_data['sub'].unique()) == len(ses_data))
    data = preproc_df(ses_data, quests)
    raw_scores = subscales_df(data, quests)
    for c in raw_scores.columns:
        raw_scores[c].plot(kind='hist', edgecolor='black', title=c)
        plt.show()
    # normalized_scores = normalize(raw_scores)
    # normalized_scores = two_step_Kmeans(normalized_scores)
    # normalized_scores.to_csv(f"{path}/clustered_data_imputed.csv")
    raw_scores.to_csv(f"{path}/clustered_data_raw.csv")

# complete session 1 missing NOE FFI coded in session 9:
quest_data_raw = pd.read_csv(f"{path}/joy_questionnaires_coded.csv")
quest_data = pd.read_csv(f"{path}/joy_qs_240423.csv")

quest_data_s19 = quest_data[(quest_data['ses']==1) | (quest_data['ses']==9)]

spsrq_cols = [col for col in quest_data_s19.columns if 'spsrq' in col]
neo_cols = [col for col in quest_data_s19.columns if 'NEO-FFI' in col]
tpq_cols = [col for col in quest_data_s19.columns if 'TPQ' in col]

clustering_data = quest_data_s19[['sub','ses'] + spsrq_cols + neo_cols + tpq_cols].reset_index(drop=True)

subs_s9 = clustering_data[clustering_data['ses']==9]['sub'].to_list()

for sub in subs_s9:
    ind1 = clustering_data[(clustering_data['sub']==sub) & (clustering_data['ses']==1)].index[0]
    ind2 = clustering_data[(clustering_data['sub']==sub) & (clustering_data['ses']==9)].index[0]
    
    clustering_data.iloc[ind1] = clustering_data.iloc[ind1].fillna(clustering_data.iloc[ind2])


clustering_data = clustering_data.drop(clustering_data[clustering_data['ses']==9].index).reset_index(drop=True)

# throw out subjects that have more than one missing value (nans)
clustering_data.dropna(thresh=clustering_data.shape[1]-2, inplace=True)
clustering_data.reset_index(drop=True,inplace=True)
# Find the indices of NaN values in the dataframe
miss_vals_ind = np.where(clustering_data.isna())

# impute cases where only one answer is missing questionnaires...
col_means = clustering_data.mean()
clustering_data_imputed = clustering_data.fillna(col_means)
clustering_data_imputed.to_csv(f"{path}/joy_qs_imputed.csv",index=False)

# # examine clusters 
# data_with_clusters = pd.read_csv(f"{path}/clustered_data_250423.csv")


# check clusters across subscales

    # plot subscales
for sub_scale in normalized_scores.columns[:-1]:
    plt.figure(figsize=(6.5,4))
    sns.distplot(normalized_scores[normalized_scores['cluster']==1][sub_scale],bins=20, color="red", label="Avoiders")
    sns.distplot(normalized_scores[normalized_scores['cluster']==0][sub_scale],bins=20, color="green", label="Approchers")
    plt.xlabel(f"{sub_scale} score", size=12)
    plt.ylabel("probability density", size=11)
    plt.title(f"{sub_scale} per cluster", size=12)
    plt.legend(loc='upper left')
    plt.savefig(rf"{path}/plotting/{sub_scale}_approchers_avoiders_clustering.jpg",
                format='jpg',dpi=150)

app_av_clusters = pd.read_csv(f"{path}/clustered_data_imputed.csv")
# 1 is avoiders, 0 is approachers

def assign_labels(cluster):
    if cluster == 1:
        return 'avoiders'
    elif cluster == 0:
        return 'approachers'

app_av_clusters['cluster_label'] = np.nan
app_av_clusters['cluster_label'] = app_av_clusters['cluster'].apply(lambda x: assign_labels(x))
app_av_clusters.to_csv(f"{path}/clustered_data_imputed_labeled.csv",index=False)

###plotting it as distributions:

# reorder sub scales  
# Subset the dataframe to only include rows where the 'cluster' column is 1
subset = app_av_clusters[app_av_clusters['cluster'] == 1]

# Calculate the mean values for each column in the subset
means = subset.iloc[:, 1:-1].mean()

# Sort the means in descending order
sorted_means = means.sort_values(ascending=False).index.to_list()

# Reorder the columns in the original dataframe using the sorted mean
cols_reorder = [app_av_clusters.columns[0]] + sorted_means + [app_av_clusters.columns[-1]] 
app_av_clusters = app_av_clusters[cols_reorder]

import matplotlib.patches as mpatches

# Mapping of variable names to custom labels
labels = {'spsrq_punishment': 'Punishment Sensitivity',
          'TPQ_HA': 'Harm Avoidance',
          'NEO-FFI_N': 'Neuroticism',
          'NEO-FFI_E': 'Extraversion',
          'TPQ_RD': 'Reward Dependence',
          'NEO-FFI_A': 'Agreeableness',
          'NEO-FFI_C': 'Conscientiousness'}

# Set the style for the plots
sns.set(style='white', rc={'axes.facecolor':'white', 'figure.facecolor':'white'},font_scale=1.2)

# Create the figure and subplots
fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(10, 30))

# Set a fixed x-axis limit for all subplots
x_limit = abs(app_av_clusters.iloc[:, 1:-1]).max().max()
for ax in axs:
    ax.set_xlim([-x_limit, x_limit])
    ax.axvline(x=0, color='black', linestyle='-')

# Loop over the columns and create the ridgeplots
for i, col in enumerate(app_av_clusters.columns[1:-1]):
    # Create the ridgeplot
    sns.kdeplot(data=app_av_clusters, x=col, hue='cluster', palette=['limegreen', 'red'], alpha=0.7, fill=True, ax=axs[i])
    
    axs[i].set_xlabel('Normalized Scores', fontsize=13)
    axs[i].set_ylabel(labels.get(col, col), fontsize=18, rotation=0, ha='right', va='center')

    # Add a dashed line for the mean value of each cluster
    for j in range(2):
        mean_val = app_av_clusters[app_av_clusters['cluster']==j][col].median()
        axs[i].axvline(x=mean_val, color='grey', linestyle='--')

    # Remove the individual subplot legends
    axs[i].legend_ = None

# Create a custom legend with the correct colors and labels
limegreen_patch = mpatches.Patch(color='limegreen', label='Approach Tendencies')
red_patch = mpatches.Patch(color='red', label='Avoidance Tendencies')
legend_handles = [limegreen_patch, red_patch]
legend_labels = ['Approach Tendencies', 'Avoidance Tendencies']

fig.legend(handles=legend_handles, labels=legend_labels, title='K-means Clusters',
           loc='upper right', bbox_to_anchor=(0.5, 1.04),fontsize=16)

plt.tight_layout()
plt.savefig(rf"{path}/plotting/approach_avoidance_tendencies_clustering_dists.jpg",
            format='jpg',dpi=200,bbox_inches='tight')

#%% 5.3.2. run analysis for examine association between motivaitonal clustering and HBV antibodies
path = '/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/questionnaires'

app_av_clusters = pd.read_csv(f"{path}/clustered_data_imputed_labeled.csv")
app_av_clusters = app_av_clusters[['sub', 'cluster', 'cluster_label']]
app_av_clusters.columns = ['sub_num', 'cluster', 'cluster_label']

### hbv postvac
hbv_log = pd.read_excel(
    r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')
hbv_log = hbv_log[['subnum', 'group', 'mean HBV postvac', 'TP8 change']]
hbv_log.columns = ['sub_num', 'group', 'mean HBV postvac', 'TP8 change']

# test hbv antibodies effects:
cor_test_data = pd.DataFrame()
cor_test_data[['sub_num', 'group', 'mean HBV postvac']] = hbv_log[[
    'sub_num', 'group', 'mean HBV postvac']].dropna().reset_index(drop=True)
cor_test_data = cor_test_data.merge(app_av_clusters, on='sub_num')

approchers = cor_test_data[cor_test_data['cluster'] == 0]['mean HBV postvac']
avoiders = cor_test_data[cor_test_data['cluster'] == 1]['mean HBV postvac']

# test effects by clustering
tvalue, pvalue = stats.ttest_ind(approchers, avoiders, equal_var=False)
print(tvalue, pvalue, 'N avoiders=',len(avoiders),'N approchers=',len(approchers))

#%% 5.4. gPPI Analysis
#%% 5.4.1. Extract PPI Functional Connectivity between VTA and ROIs 

# 1. Create the physiological regressor - mean time series of the seed region (VTA)

    # Extract mean time series:
# This is a Python script that is intended to run through wsl linux machine
# uses fslmeants to extract the mean time series per run of the VTA of each subject

# SAVE this script on a seperate .py file, and run through linux comצand prompt with FSL installed.


# Inputs:
main_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives'
task_name = 'fmrinfpractice'  # name of task
###

# Get all the filtered func data files from which to extract the ROI activity ts:
fil_func_files = glob.glob(
    '%s/fmriprep/sub-*/ses-*/func/%s_run-[1-4].feat/filtered_func_data.nii.gz' % (main_dir, task_name))
pattern = re.compile(".*sub-(\d*)/ses-(\d*).*run-(\d)")
for file in list(fil_func_files):
    match = pattern.match(file)
    subnum = match.groups()[0]
    sesnum = match.groups()[1]
    runnum = match.groups()[2]
    # ml_roi_3 is the VTA as selected offline per subject.
    roi = '%s/ROIs/offline_ROIs/sub-%s/ml_roi_3.nii.gz' % (main_dir, subnum)
    os.system(
        f"fslmeants -i {file} -o {main_dir}/fmriprep/sub-{subnum}/ses-{sesnum}/func/{task_name}_run-{runnum}.feat/VTA_meants.txt -m {roi}")


# 2. Create fsf files based on the template fsf design file, containing three contrasts - PPI for regulate VS rest, regulate only, and feedback conditions

# realized with def create_1st_level_fsfs(), lines 476.
# template fsf file for 1st level analysis : H:\projects\joy\analyses\fmri\offline\derivatives\models\fsfs\level_1\fmrinfpractice\PPI\template_fmrinfpractice_PPI.fsf

# 3. Run 1st level analysis batch

    # Run 1st level GLM:
# This is a Python script that is intended to run through wsl linux machine in order to conduct an FSL 1st level analysis
# Calls fsl feat on all fsf design files of a certain task (after they were produced, see above...)

# SAVE this script on a seperate .py file, and run through linux command prompt with FSL installed.


# Inputs:
main_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives'
fsfsdir = '/models/fsfs/level_1/fmrinfpractice/PPI'  # location of the fsf folders

###

# # Get *all* the fsfs:
# fsffiles = glob.glob('%s%s/%s/design_1st_level_task-%s_sub-*_ses-*_run-*.fsf'%(main_dir,fsfsdir,task_name,task_name))

# for fsf in list(fsffiles):
#     os.system('feat %s'%(fsf))

# if running parralel analyses on multiple subjects, get a subset of the fsfs. for instance:
fsffiles_subset = glob.glob(
    '%s%s/design_1st_level_task-%s_sub-11[2-3]*_ses-*_run-*_PPI.fsf' % (main_dir, fsfsdir, task_name))
for fsf in list(fsffiles_subset):
    os.system('feat %s' % (fsf))
#%% 5.4.2. Run ROI analysis on PPI results
# 1. create ROIs: bilateral nac (exists), bilateral amygdala and left vmpfc.

# fslmaths commands following Andrew Jahn tutorial:http://andysbrainblog.blogspot.com/2013/04/fsl-tutorial-creating-rois-from.html

# # create a point roi:
# fslmaths /mnt/h/projects/joy/analyses/fmri/offline/derivatives/templates/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz -mul 0 -add 1 -roi 59 1 63 1 31 1 0 1 /mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs/general_offline_ROIs/rois_for_PPI/creation/r_amyg_point.nii.gz -odt float

# # create a sphere from the point:
# fslmaths /mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs/general_offline_ROIs/rois_for_PPI/creation/r_amyg_point.nii.gz -kernel sphere 5 -fmean /mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs/general_offline_ROIs/rois_for_PPI/creation/r_amyg_sphere_5mm.nii.gz -odt float

# # binarize:
# fslmaths /mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs/general_offline_ROIs/rois_for_PPI/creation/r_amyg_sphere_5mm.nii.gz -bin /mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs/general_offline_ROIs/rois_for_PPI/creation/r_amyg_sphere_5mm.nii.gz_bin.nii.gz

#  2) Create CSF-excluded (via individual fmriprep csf masks) ROIs on MNI space - per subject. for vmpfc - only copy the general roi without excluding csf (it is on the medial plane)
# this is to be run on python on linux wsl with FSL installed.

subs_rois_folder = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs'

# get roi:
general_roi = f"{subs_rois_folder}/general_offline_ROIs/rois_for_PPI/r_amyg_sphere_5mm_bin.nii.gz"
# catch all subjects (this will be used for extracting data from NF practice as well as mid and rest, so including all subs)
nf_subs_list = glob.glob(
    "/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-*/anat")

# for each subject, take the CSF probabilistic mask in anat folder, exclude the general ML ROIs and save them in subs_rois_folder
for sub in nf_subs_list:
    ind_sub = sub.find('sub-') + 4
    subnum = sub[ind_sub:ind_sub + 4]
    # csf inverse-binary csf mask
    csf_thr_binv = f"{sub}/sub-{subnum}_space-MNI152NLin2009cAsym_label-CSF_probseg_RS2_thr05_binv.nii.gz"
    # mask offline ML ROIs and save in sub folder
    curr_sub_dir = f"{subs_rois_folder}/sub-{subnum}"
    os.system(
        f"fslmaths {general_roi} -mas {csf_thr_binv} {curr_sub_dir}/r_amyg_roi")

# 3) for each ROIS: extract contrasts PPI beta values of reg>rest (cope 1), reg only (cope 2) and feedback (cope 3), in PSC, from each run (Featquery)

# running Featquery for all runs (on linux)
main_dir = '/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-*/ses-*/func'
task_name = 'fmrinfpractice'
roi = 'r_amyg_roi'  # change according to required roi name
feat_folders = glob.glob(f"{main_dir}/{task_name}_run-[1-4]_PPI.feat")
# feat_1122_ses2 = glob.glob(f"/mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-1122/ses-2/func/{task_name}_run-[1-4].feat")
pattern = re.compile(".*sub-(\d*).ses-(\d*).*(run-\d)")
for feat in feat_folders:
    match = pattern.match(feat)
    subnum = match.groups()[0]
    sesnum = match.groups()[1]
    runnum = match.groups()[2]
    os.system(
        f"/home/nitzan/fsl/bin/featquery 1 /mnt/h/projects/joy/analyses/fmri/offline/derivatives/fmriprep/sub-{subnum}/ses-{sesnum}/func/{task_name}_{runnum}_PPI.feat 3  stats/cope1 stats/cope2 stats/cope3 featquery_{roi} -p -w /mnt/h/projects/joy/analyses/fmri/offline/derivatives/ROIs/offline_ROIs/sub-{subnum}/{roi}.nii.gz")
    print(f"sub-{subnum} ses-{sesnum} {runnum}")


# 4) Extract values for group and correlational analyses of PPI activity in ROIs across subjects and sessions.

##############

# SET parameters for statistical analysis:

contrast = 'regVSrest'  # regVSrest / regulate / feedback

# CHOOSE which rois are averaged:

# all mesolimbic reward network:
# ml_roi_1 (l_nac)/ml_roi_2 (r_nac)/r_amyg_roi/l_amyg_roi/l_vmpfc_roi
roi_name = 'l_vmpfc_roi'
roi_anal_dir = f"{main_dir}\\derivatives\\models\\group_statistics\\{task_name}_PPI\\roi_analyses\\{roi_name}"

subs_list = glob.glob(
    f"{main_dir}\\derivatives\\fmriprep\\sub-*\\ses-3\\func\\{task_name}_run-1_PPI.feat")
group_roi_anal = pd.DataFrame()
for sub in subs_list:
    pattern = re.compile(".*sub-(\d*)")
    match = pattern.match(sub)
    subnum = match.groups()[0]
    sub_feat_folders = glob.glob(
        f"{main_dir}\\derivatives\\fmriprep\\sub-{subnum}\\ses-*\\func\\{task_name}_run-[1-4]_PPI.feat")
    ml_sub_data = pd.DataFrame()
    for feat in sub_feat_folders:
        pattern_sub = re.compile(".*(ses-\d*).*(run-\d)")
        match_sub = pattern_sub.match(feat)
        ses = match_sub.groups()[0]
        run = match_sub.groups()[1]
        roi_data = pd.DataFrame()

        roi_anal_report = f"{feat}\\featquery_{roi_name}\\report.txt"
        tmp = pd.read_csv(roi_anal_report, sep=' ', header=None)
        tmp = tmp[5]

        ml_sub_data = pd.concat(
            [ml_sub_data, pd.DataFrame({f"{ses}_{run}": tmp})], axis=1)

    ml_sub_data.index = ['regVSrest', 'regulate', 'feedback']
    # ml_sub_data.to_csv(f"{roi_anal_dir}\\subjects_results\\sub-{subnum}_{task_name}_PPI_roi_analysis_{roi_name}_psc.csv")

    # choose contrast to extract according to contrast var. defined above.
    a = ml_sub_data.loc[contrast, :]

    a = a.rename(f"sub-{subnum}")
    a = pd.DataFrame(a.values, index=a.index, columns=[f"sub-{subnum}"])
    group_roi_anal = pd.concat([group_roi_anal, a], axis=1)


# prepare the data for R mixed effects analysis
group_roi_anal_t = group_roi_anal.T
cols = group_roi_anal_t.columns.tolist()
cols_reorder = cols[:5] + [cols[11]] + cols[5:8] + [cols[12]] + cols[8:11]
group_roi_anal_t = group_roi_anal_t[cols_reorder]
lib_path = 'H:\\projects\\joy\\general\\subjects_allocation'
alloc_file = pd.read_csv(f"{lib_path}\\group_allocation_blinded_fmri.csv")
b = alloc_file['group']
group_roi_anal_t.insert(0, 'group', b.to_frame())
# group_roi_anal_t.insert(0,'sub_id',range(1,69))

# save csv with original session and run numbers per sub
# index=True for real subnums for correlations
group_roi_anal_t.to_csv(
    f"{roi_anal_dir}\\group_roi_analysis_PPI_{roi_name}_{contrast}_psc_w_subnums.csv", index=True)

# erase nans and indent values to left:
out = (pd.DataFrame(group_roi_anal_t.apply(sorted, key=pd.isna, axis=1).to_list(),
                    index=group_roi_anal_t.index, columns=group_roi_anal_t.columns)
       .fillna('')
       )
out = out.iloc[:, 0:12]
out.rename(columns={'ses-1_run-1': 'T1', 'ses-1_run-2': 'T2', 'ses-2_run-1': 'T3', 'ses-2_run-2': 'T4', 'ses-2_run-3': 'T5', 'ses-2_run-4': 'T6', 'ses-3_run-1': 'T7',
                    'ses-3_run-2': 'T8', 'ses-3_run-3': 'T9', 'ses-3_run-4': 'T10', 'ses-4_run-1': 'T11', }, inplace=True)
# save csv
out.to_csv(f"{roi_anal_dir}\\group_roi_analysis_PPI_{roi_name}_{contrast}_psc_w_subnums_indented.csv",
           index=True)  # index=True for real subnums for correlations


# create NF activity dataframe per session per subject
NF_act = pd.read_csv(
    f"{roi_anal_dir}\\group_roi_analysis_PPI_{roi_name}_{contrast}_psc_w_subnums.csv")
NF_act_ses = pd.DataFrame(np.zeros([len(NF_act), 4]), columns=[
                          'ses-1', 'ses-2', 'ses-3', 'ses-4'])
for i in NF_act_ses.index:
    for c in NF_act_ses:
        ses_cols = [col for col in NF_act.columns if c in col]
        NF_act_ses.loc[i, c] = np.mean(NF_act.loc[i, ses_cols])

# add subnums and save to csv?
NF_act_ses['sub_num'] = NF_act['Unnamed: 0'].astype(str)
NF_act_ses['group'] = NF_act['group']
cols = NF_act_ses.columns.tolist()
cols = cols[-2:] + cols[:-2]
NF_act_ses = NF_act_ses[cols]
# subject 1178 did not have a 4th session. For him ses 4 is an iteration of ses 3, thus erase session 4 for him (this should be corrected in the original NF_act creation above!!!)
# NF_act_ses.loc[NF_act_ses.index[NF_act_ses['sub_num']=='sub-1178'],'ses-4'] = np.nan

NF_act_ses.to_csv(
    f"{roi_anal_dir}\\group_roi_analysis_PPI_{roi_name}_{contrast}_psc_w_subnums_sessions.csv", index=False)

#%% before running analysis, load the data for ml_roi_1 and ml_roi_2, which are the bilateral Nac ROIs, and average them together. then save them together in a designated folder
# load the data for ml_roi_1 and ml_roi_2, which are the bilateral Nac ROIs, and average them together.
new_nac_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}_PPI/roi_analyses/bilateral_nac"
for contrast in ['regVSrest', 'regulate', 'feedback']:
    # for sessions data:
    roi1_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}_PPI/roi_analyses/ml_roi_1"
    roi2_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}_PPI/roi_analyses/ml_roi_2"
    nf_ml_roi1 = pd.read_csv(
        f"{roi1_dir}/group_roi_analysis_PPI_ml_roi_1_{contrast}_psc_w_subnums_sessions.csv")
    nf_ml_roi2 = pd.read_csv(
        f"{roi2_dir}/group_roi_analysis_PPI_ml_roi_2_{contrast}_psc_w_subnums_sessions.csv")
    # average the two rois together:
    nf_ml_roi1['ses-1'] = (nf_ml_roi1['ses-1'] + nf_ml_roi2['ses-1']) / 2
    nf_ml_roi1['ses-2'] = (nf_ml_roi1['ses-2'] + nf_ml_roi2['ses-2']) / 2
    nf_ml_roi1['ses-3'] = (nf_ml_roi1['ses-3'] + nf_ml_roi2['ses-3']) / 2
    nf_ml_roi1['ses-4'] = (nf_ml_roi1['ses-4'] + nf_ml_roi2['ses-4']) / 2

    # save the averaged data to a new csv file:
    nf_ml_roi1.to_csv(
        f"{new_nac_dir}/group_roi_analysis_PPI_bilateral_nac_{contrast}_psc_w_subnums_sessions.csv", index=False)
    
    # for runs data:
    roi1_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}_PPI/roi_analyses/ml_roi_1"
    roi2_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}_PPI/roi_analyses/ml_roi_2"
    nf_ml_roi1 = pd.read_csv(
        f"{roi1_dir}/group_roi_analysis_PPI_ml_roi_1_{contrast}_psc_w_subnums_indented.csv")
    nf_ml_roi2 = pd.read_csv(
        f"{roi2_dir}/group_roi_analysis_PPI_ml_roi_2_{contrast}_psc_w_subnums_indented.csv")
    # average the two rois together:
    nf_ml_roi1['T1'] = (nf_ml_roi1['T1'] + nf_ml_roi2['T1']) / 2
    nf_ml_roi1['T2'] = (nf_ml_roi1['T2'] + nf_ml_roi2['T2']) / 2
    nf_ml_roi1['T3'] = (nf_ml_roi1['T3'] + nf_ml_roi2['T3']) / 2
    nf_ml_roi1['T4'] = (nf_ml_roi1['T4'] + nf_ml_roi2['T4']) / 2
    nf_ml_roi1['T5'] = (nf_ml_roi1['T5'] + nf_ml_roi2['T5']) / 2
    nf_ml_roi1['T6'] = (nf_ml_roi1['T6'] + nf_ml_roi2['T6']) / 2
    nf_ml_roi1['T7'] = (nf_ml_roi1['T7'] + nf_ml_roi2['T7']) / 2 
    nf_ml_roi1['T8'] = (nf_ml_roi1['T8'] + nf_ml_roi2['T8']) / 2
    nf_ml_roi1['T9'] = (nf_ml_roi1['T9'] + nf_ml_roi2['T9']) / 2
    nf_ml_roi1['T10'] = (nf_ml_roi1['T10'] + nf_ml_roi2['T10']) / 2
    nf_ml_roi1['T11'] = (nf_ml_roi1['T11'] + nf_ml_roi2['T11']) / 2
    # save the averaged data to a new csv file:
    nf_ml_roi1.to_csv(
        f"{new_nac_dir}/group_roi_analysis_PPI_bilateral_nac_{contrast}_psc_w_subnums_indented.csv", index=True)  # index=True for real subnums for correlations
#%% 5.4.3. Plot Effects of FC during NF training across group and time points.
# set a loop to print the results for each roi in the mesolimbic reward network:
rois_list = ['bilateral_nac', 'r_amyg_roi', 'l_amyg_roi', 'l_vmpfc_roi']
rois_names = ['bilateral_nac', 'r_amyg', 'l_amyg', 'l_vmpfc']
for roi, roi_name in zip(rois_list, rois_names):
    contrast = 'regVSrest'  # regVSrest / regulate / feedback

#######################

    roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}_PPI/roi_analyses/{roi}"
    ###

    # for sessions data:
    NF_act_plot = pd.read_csv(
        f"{roi_anal_dir}/group_roi_analysis_PPI_{roi}_{contrast}_psc_w_subnums_sessions.csv")

    NF_act_plot.drop(["ses-1"], axis=1, inplace=True)
    TP_names = NF_act_plot.columns
    TP_names = [e for e in TP_names if e not in ('group', 'sub_num')]
    # remove session 1 for ploting purposes:

    NF_act_plot.loc[NF_act_plot.loc[:, 'group'] == 1, 'group label'] = 'control NF'
    NF_act_plot.loc[NF_act_plot.loc[:, 'group']
                    == 2, 'group label'] = 'reward ML NF'

    dd = pd.melt(NF_act_plot, id_vars=['group label'],
                value_vars=TP_names, var_name='time')

    out_path = f"{main_dir}/derivatives/models/group_statistics/{task_name}_PPI/roi_analyses/{roi}/plots"
    # create output path if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    title = f"{roi_name} PPI with VTA"
    # plot_reg(dd, out_path, title)
    import matplotlib.lines as mlines

    def plot_reg(dd, out_path, title):
        # Define line styles for each group
        line_styles = ['-', '-']
        # Set up the plot
        plt.figure(figsize=(4, 3))
        if roi_name == 'rand_rois':
            if plot_subgroups:
                # Define a custom color palette for rand_rois subgroups
                custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            else:
                # Define a custom color palette for control group
                custom_palette = ['grey']
        
        else: # VTA and bilateral Nac
            custom_palette = ['black','grey']
        # Iterate over each group and plot the line
        # for i, group_label in enumerate(dd["group label"].unique()):
        #     group_data = dd[dd["group label"] == group_label]
        #     line_style = line_styles[i % len(line_styles)]  # Cycle through line styles

        sns.pointplot(x="time",
                    y="value",
                    # linestyles=line_style,
                    hue="group label",
                    errorbar='se',
                    dodge=True if dd["group label"].nunique() > 1 else None,
                    data=dd,
                    palette=custom_palette)
        # Change the line width using plt.plott
        for line in plt.gca().lines:
            line.set_linewidth(1.5)  # Set the line width to 2 (adjust as needed)

        # plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1),fontsize=9)

        # remove legend
        plt.legend([],[], frameon=False)  # remove legend

        # Create custom legend
        # legend_elements = [mlines.Line2D([0], [0], color='blue', lw=2, linestyle=style) for style in line_styles]
        # plt.legend(handles=legend_elements, labels=list(dd["group label"].unique()), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=9)
        # axes labels
        plt.xlabel("Time", size=9)
        plt.xticks(size=9)
        plt.ylabel(f"BOLD%({contrast})",size=9)
        # add title
        plt.title(f"{title} Regulation Effects", loc="center")
        plt.savefig(rf"{out_path}/NF_groups_{roi_name}_{contrast}_sessions_lineplots.jpg",
                    format='jpg', bbox_inches='tight',dpi=150)

    plot_reg(dd,out_path,title)

#%% 5.4.4. Correlate PPI changes with HBV antibodies

### SET PARAMETERS: ###
rois_list = ['bilateral_nac', 'r_amyg_roi', 'l_amyg_roi', 'l_vmpfc_roi']
rois_names = ['bilateral_nac', 'r_amyg', 'l_amyg', 'l_vmpfc']
contrast = 'regVSrest'  # regVSrest / feedback
neural_marker = 'slopes'  # slopes / ses-4 activity (for feedback condition)
#######################
cor_dir_path = r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/brain_behav_immune_corr/nf_immune'
paper_plot_path = r'/Users/nitzanlubi/My Drive/Lab/brain_immune_paper/plotting'

### hbv antibodies:
hbv_log = pd.read_excel(r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')
hbv_log = hbv_log[hbv_log['group'] != 3]  # drop the no NF group
hbv_log.reset_index(inplace=True)

# excluding non responders:
hbv_log = hbv_log[hbv_log['mean HBV postvac'] > 0.99].reset_index(drop=True)

for roi, roi_name in zip(rois_list, rois_names):
    
    # load nf data:
    roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice_PPI/roi_analyses/{roi}"
    if neural_marker == 'slopes':
        ml_activity_data = pd.read_csv(f"{roi_anal_dir}/group_roi_analysis_PPI_{roi}_{contrast}_psc_w_subnums_indented.csv")
    elif neural_marker == 'ses-4 activity':
        ml_activity_data = pd.read_csv(f"{roi_anal_dir}/group_roi_analysis_PPI_{roi}_{contrast}_psc_w_subnums_sessions.csv")

    cor_test_data = pd.DataFrame()
    cor_test_data[['sub_num', 'group', 'mean HBV postvac']] = hbv_log[['subnum', 'group', 'mean HBV postvac']]

    ml_activity_data = ml_activity_data.rename(columns={'Unnamed: 0': 'sub_num'})

    ml_activity_data['sub_num'] = ml_activity_data['sub_num'].replace(
        {'sub-': ''}, regex=True)
    ml_activity_data['sub_num'] = ml_activity_data['sub_num'].astype(int)
    
    df_tmp = ml_activity_data.iloc[:, 3:]
    if neural_marker == 'slopes':

        # calculate activity slopes per subject
        for ind in list(range(0, len(df_tmp))):

            results = stats.linregress(list(
                range(1, len(df_tmp.iloc[ind, :].dropna())+1)), df_tmp.iloc[ind, :].dropna())
            ml_activity_data.loc[ind, f"{roi}_{contrast} slopes"] = results.slope
    elif neural_marker == 'ses-4 activity':
        # calculate ses 4 activity per subject, which is the mean of the last three runs
        for ind in list(range(0, len(df_tmp))):
            ml_activity_data.loc[ind, f"{roi}_{contrast} ses-4 activity"] = df_tmp.loc[ind,'ses-4']

    # insert activity markers to cor test data dataframe
    ml_activity_data = ml_activity_data[[
        'sub_num', f"{roi}_{contrast} {neural_marker}"]]

    cor_test_data = cor_test_data.merge(ml_activity_data, on='sub_num')

    # drop lines with nans in the activity markers:
    cor_test_data = cor_test_data.dropna(
        subset=[f"{roi}_{contrast} {neural_marker}"])
    cor_test_data = cor_test_data.reset_index(drop=True)
    # calculate correlations between mean postvac and ml activity markers via regression analysis:
    reg_brain_immune = stats.linregress(cor_test_data['mean HBV postvac'], cor_test_data[f"{roi}_{contrast} {neural_marker}"])
    print(f"{roi_name}:","r=", reg_brain_immune.rvalue, "p=", reg_brain_immune.pvalue)

    # calculate correlations between TP8 change and ml activity markers via regression analysis:
    # cor_test_data_TP8 = cor_test_data.dropna(subset=['TP8 change'])
    # reg_brain_immune_TP8 = stats.linregress(cor_test_data_TP8['TP8 change'], cor_test_data_TP8[f"{roi}_{contrast} {neural_marker}"])
    # print(f"{roi}:","r=", reg_brain_immune_TP8.rvalue, "p=", reg_brain_immune_TP8.pvalue)

    # plot the correlations:
    # plotting correlations:
    plt.figure(figsize=(4,3))
    sns.regplot(x=f"{roi}_{contrast} {neural_marker}",
                y="mean HBV postvac", data=cor_test_data, color='black', scatter=False)
    sns.scatterplot(x=f"{roi}_{contrast} {neural_marker}",
                    y="mean HBV postvac", hue="group", palette={1: 'grey', 2: 'black'}, data=cor_test_data, s=65, legend=False)
    plt.xlabel(f"VTA and {roi_name} functional connectivity", fontsize=10)
    plt.ylabel("HBVab postvac change", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # add correlation and p value to the plot:
    plt.text(0.05, 0.9, f"r={reg_brain_immune.rvalue:.2f}, p={reg_brain_immune.pvalue:.3f}", transform=plt.gca().transAxes)
    plt.title(f"VTA <-> {roi_name} FC and immune effects", fontsize=11)
    # plt.savefig(rf"{out_path}/corr_{roi}_{contrast}_{neural_marker}_HBVab_wo_nonresponders.png",
    #             format='png',transparent=True, dpi=200)
    plt.show()

# try correlating for each group separately:    
for roi, roi_name in zip(rois_list, rois_names):

    # load nf data:
    roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice_PPI/roi_analyses/{roi}"
    if neural_marker == 'slopes':
        ml_activity_data = pd.read_csv(f"{roi_anal_dir}/group_roi_analysis_PPI_{roi}_{contrast}_psc_w_subnums_indented.csv")
    elif neural_marker == 'ses-4 activity':
        ml_activity_data = pd.read_csv(f"{roi_anal_dir}/group_roi_analysis_PPI_{roi}_{contrast}_psc_w_subnums_sessions.csv")

    cor_test_data = pd.DataFrame()
    cor_test_data[['sub_num', 'group', 'mean HBV postvac']] = hbv_log[['subnum', 'group', 'mean HBV postvac']]

    ml_activity_data = ml_activity_data.rename(columns={'Unnamed: 0': 'sub_num'})

    ml_activity_data['sub_num'] = ml_activity_data['sub_num'].replace(
        {'sub-': ''}, regex=True)
    ml_activity_data['sub_num'] = ml_activity_data['sub_num'].astype(int)

    # keep group information
    df_tmp = ml_activity_data.iloc[:, 2:] # group will be the first column
    if neural_marker == 'slopes':
        # calculate activity slopes per subject
        for ind in list(range(0, len(df_tmp))):

            results = stats.linregress(list(
                range(1, len(df_tmp.iloc[ind, :].dropna())+1)), df_tmp.iloc[ind, :].dropna())
            ml_activity_data.loc[ind, f"{roi}_{contrast} slopes"] = results.slope
    elif neural_marker == 'ses-4 activity':
        # calculate ses 4 activity per subject, which is the mean of the last three runs
        for ind in list(range(0, len(df_tmp))):
            ml_activity_data.loc[ind, f"{roi}_{contrast} ses-4 activity"] = df_tmp.loc[ind,'ses-4']
    # insert activity markers to cor test data dataframe
    ml_activity_data = ml_activity_data[[
        'sub_num', 'group', f"{roi}_{contrast} {neural_marker}"]]
    cor_test_data = cor_test_data.merge(ml_activity_data, on=['sub_num', 'group'])
    # drop lines with nans in the activity markers:
    cor_test_data = cor_test_data.dropna(
        subset=[f"{roi}_{contrast} {neural_marker}"])
    cor_test_data = cor_test_data.reset_index(drop=True)
    # calculate correlations between mean postvac and ml activity markers for each group separately via regression analysis:
    print(f"Correlations for {roi_name} {contrast} {neural_marker} PPI vs. HBVab postvac change:")
    for group in cor_test_data['group'].unique():
        group_data = cor_test_data[cor_test_data['group'] == group]
        if not group_data.empty:
            results = stats.linregress(group_data[f"{roi}_{contrast} {neural_marker}"],
                                        group_data['mean HBV postvac'])
            print(f"Group {group}: r={results.rvalue}, p={results.pvalue}")
            # plot the correlations:
            plt.figure(figsize=(4,3))
            sns.regplot(x=f"{roi}_{contrast} {neural_marker}",
                        y="mean HBV postvac", data=group_data, color='black', scatter=False)
            sns.scatterplot(x=f"{roi}_{contrast} {neural_marker}",
                            y="mean HBV postvac", hue="group", palette={1: 'grey', 2: 'black'}, data=group_data, s=65, legend=False)
            plt.xlabel(f"FC between VTA and {roi_name}", fontsize=10)
            plt.ylabel("HBVab postvac change", fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            # add correlation and p value to the plot:
            plt.text(0.05, 0.9, f"r={results.rvalue:.2f}, p={results.pvalue:.3f}", transform=plt.gca().transAxes)
            plt.title(f"{roi_name} {contrast} {neural_marker} PPI vs. HBVab postvac change - Group {group}", fontsize=11)
            # plt.savefig(rf"{cor_dir_path}/corr_{roi}_{contrast}_{neural_marker}_HBVab_group{group}.png",
            #             format='png', transparent=True, dpi=200)
            plt.show()
# %% 5.5. Positive Expectation --> Immune outcome correlations
# load the positive expectation data:

#import data
path = r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/mental_strategies'
out_path = r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/behav/task-fmrinfpractice/mental_strategies'
msq_data = pd.read_excel(f"{path}/fmrinfpractice_msq_data.xlsx")

# filter the data to keep sub_num, session_num, and the positive expectation column:
pos_exp_data = msq_data[['sub_num', 'session_num', 'positive_expectation']]

# calculate th frequency of positive expectation use per subject.
pos_exp_freq = pos_exp_data.groupby('sub_num')['positive_expectation'].mean().reset_index()
pos_exp_freq.rename(columns={'positive_expectation': 'pos_exp_freq'}, inplace=True)

### hbv antibodies:
hbv_log = pd.read_excel(r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/Immunology/HBV_antibodies/HBV_antibodies_log_change.xlsx')
hbv_log = hbv_log[hbv_log['group'] != 3]  # drop the no NF group
hbv_log.reset_index(inplace=True)

# excluding non responders:
hbv_log = hbv_log[hbv_log['mean HBV postvac'] > 0.99].reset_index(drop=True)

# change column name 'subnum' to 'sub_num' for merging:
hbv_log.rename(columns={'subnum': 'sub_num'}, inplace=True)
# merge the positive expectation frequency data with the hbv log data:
cor_test_data = hbv_log.merge(pos_exp_freq, on='sub_num')

# drop lines with nans in the activity markers:
cor_test_data = cor_test_data.dropna(subset=['pos_exp_freq'])
cor_test_data = cor_test_data.reset_index(drop=True)\

# calculate pearson correlation:
results = stats.linregress(cor_test_data['pos_exp_freq'], cor_test_data['mean HBV postvac'])
print("Positive Expectation Frequency vs. HBVab postvac change:")
print(f"r={results.rvalue}, p={results.pvalue}")
# plot the correlation:
plt.figure(figsize=(4,3))
sns.regplot(x='pos_exp_freq', y='mean HBV postvac', data=cor_test_data, color='black', scatter=False)
sns.scatterplot(x='pos_exp_freq', y='mean HBV postvac', hue='group', palette={1: 'grey', 
2: 'black'}, data=cor_test_data, s=65, legend=False)
plt.xlabel("Positive Expectation Frequency (mean)", fontsize=10)
plt.ylabel("HBVab postvac change", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# add correlation and p value to the plot:
plt.text(0.05, 0.9, f"r={results.rvalue:.2f}, p={results.pvalue:.3f}", transform=plt.gca().transAxes)
plt.title("Mean Positive Expectation use vs. HBVab change", fontsize=11)
# plt.savefig(rf"{out_path}/corr_pos_exp_freq_HBVab.png",
#             format='png', transparent=True, dpi=200)
plt.show()

# check whether the change in positive expectation frequency is correlated with the change in HBV antibodies:
pos_exp_freq_session = pos_exp_data.groupby(['sub_num', 'session_num'])['positive_expectation'].mean().reset_index()
pos_exp_freq_session_pivot = pos_exp_freq_session.pivot(index='sub_num', columns='session_num', values='positive_expectation').reset_index()
pos_exp_freq_session_pivot['change'] = pos_exp_freq_session_pivot[4] - pos_exp_freq_session_pivot[2]
cor_test_data = hbv_log.merge(pos_exp_freq_session_pivot, on='sub_num')
cor_test_data = cor_test_data.dropna(subset=['change'])
cor_test_data = cor_test_data.reset_index(drop=True)

# calculate pearson correlation:
results = stats.linregress(cor_test_data['change'], cor_test_data['mean HBV postvac'])
print("Change in Positive Expectation Frequency vs. HBVab postvac change:")
print(f"r={results.rvalue}, p={results.pvalue}")

# plot the correlation:
plt.figure(figsize=(4,3))
sns.regplot(x='change', y='mean HBV postvac', data=cor_test_data, color='black', scatter=False)
sns.scatterplot(x='change', y='mean HBV postvac', hue='group', palette={1: 'grey', 
2: 'black'}, data=cor_test_data, s=65, legend=False)
plt.xlabel("Positive Expectation Frequency (Session 4- Session 2)", fontsize=10)
plt.ylabel("HBVab postvac change", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# add correlation and p value to the plot:
plt.text(0.05, 0.9, f"r={results.rvalue:.2f}, p={results.pvalue:.3f}", transform=plt.gca().transAxes)
plt.title("Change in Positive Expectation use vs. HBVab change", fontsize=11)
# plt.savefig(rf"{out_path}/corr_pos_exp_freq_change_HBVab.png",
#             format='png', transparent=True, dpi=200)
plt.tight_layout()
plt.show()