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
rois_name = 'bilateral_nac'  # VTA/bilateral_nac
contrast = 'regVSrest'  # regVSrest/Feedback (feedback condition is inspected for ses-4 activity rather than activity slopes)
#######################

roi_anal_dir = f"{main_dir}/derivatives/models/group_statistics/{task_name}/roi_analyses/offline_{rois_name}"
###

NF_act = pd.read_csv(
    f"{roi_anal_dir}/group_roi_analysis_{rois_name}_{contrast}_psc_w_subnums.csv")

TP_names = NF_act.columns.to_list()
TP_names = [e for e in TP_names if e not in (
    'group', 'sub_num', 'Unnamed: 0', 'sub_id')]
###

NF_act = pd.melt(NF_act, id_vars=[
                 'Unnamed: 0', 'sub_id', 'group'], value_vars=TP_names, var_name='time')
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
rois = ['ml_reward', 'VTA', 'bilateral_Nac']
contrast = 'regVSrest' 

for i in range(len(rois)):
    roi_name= rois[i]
        
    data_dir = f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{roi_name}"
    NF_act_plot = pd.read_csv(f"{data_dir}/group_roi_analysis_{roi_name}_{contrast}_psc_w_subnums_sessions.csv")

    # remove session 1 (is not included in the analysis):
    NF_act_plot.drop(["ses-1"], axis=1, inplace=True)

    TP_names = NF_act_plot.columns.to_list()
    TP_names = [e for e in TP_names if e not in (
        'group', 'sub_num')]
    ##########

    NF_act_plot.loc[NF_act_plot.loc[:, 'group'] == 1, 'group label'] = 'Control group'
    NF_act_plot.loc[NF_act_plot.loc[:, 'group'] == 2, 'group label'] = 'Experimental group'

    dd = pd.melt(NF_act_plot, id_vars=['group label'],
                value_vars=TP_names, var_name='time')

    # title:
    titles = ['Reward ML', 'VTA', 'Nac']
    title = titles[i]

    # plot_reg(dd, out_path, title)
    import matplotlib.lines as mlines

    def plot_reg(dd, out_path, title):
        # Define line styles for each group
        line_styles = ['-', '-']
        # Set up the plot
        plt.figure(figsize=(4, 3))
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
                    dodge=True,
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
ml_rois = ['bilateral_nac','VTA', 'rand_rois']
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
    # reg_brain_immune = stats.linregress(cor_test_data['mean HBV postvac'], cor_test_data[f"{roi}_{contrast} {neural_marker}"])
    # print(f"{roi}:","r=", reg_brain_immune.rvalue, "p=", reg_brain_immune.pvalue)

    # calculate correlations between TP8 change and ml activity markers via regression analysis:
    cor_test_data_TP8 = cor_test_data.dropna(subset=['TP8 change'])
    reg_brain_immune_TP8 = stats.linregress(cor_test_data_TP8['TP8 change'], cor_test_data_TP8[f"{roi}_{contrast} {neural_marker}"])
    print(f"{roi}:","r=", reg_brain_immune_TP8.rvalue, "p=", reg_brain_immune_TP8.pvalue)

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

# %% 3.3.1. Mental Strategies chi square tests 

#import data
path = r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/data/mental_strategies'
out_path = r'/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/analyses/behav/task-fmrinfpractice/mental_strategies'
data = pd.read_excel(f"{path}/fmrinfpractice_msq_data.xlsx")
main_dir = "/Users/nitzanlubi/Google Drive/Lab/Joy/joy_italy/analyses/fmri/offline"

roi = 'VTA' # VTA/bilateral_nac/rand_rois

# for session 4 only
data = data[data['session_num']==4].reset_index(drop=True)

def df_valid_features(df, features):
    valid = features.copy()
    for feat in features:
        prop = list(df[feat].value_counts(normalize=True))
        if len(prop) > 1:
            if prop[0] < 0.1 or prop[1] < 0.1:
                valid.remove(feat)
        if len(prop) == 1:
            valid.remove(feat)
    
    return valid

# arrange data for chi square tests

#list of all features
features = list(data)[10:55]

#create df only with valid features and neural target column
features_valid = df_valid_features(data, features)
df = data[['sub_num','neural_target','session_num','run_num','cycle_num']+ features_valid].reset_index(drop=True)

group_cri = 'ml_act' # ml_act / test_cont
#for NF groups comparison

if roi == 'rand_rois':
    df = df[df['neural_target']!=2].reset_index(drop=True)
# for ml activity grouping
if group_cri == 'ml_act':
    # create subjects grouping based on session 4 ml activity level, above or below median:
    ml_activity = pd.read_csv(f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{roi}/group_roi_analysis_{roi}_regVSrest_psc_w_subnums_sessions.csv")
    
    # subect 1178 had only 2 sessions, that in activity measures are sesion 2 and 3. 
    # however, in mental strategies these are 3 and 4. So ses-3 here is his last pre-vaccination session. for the purpose of grouping
    # session 3 is the one that matters and is tagged in metnal stratgeies as session 4. 
    ml_activity.loc[ml_activity['sub_num']=='sub-1178','ses-4'] = ml_activity[ml_activity['sub_num']=='sub-1178']['ses-3'] 

    act_med = stat.median(ml_activity["ses-4"])
    for ind in ml_activity.index:
        if ml_activity.loc[ind, "ses-4"] > act_med:
            ml_activity.loc[ind, "high_low_ml"] = 2
        else:
            ml_activity.loc[ind, "high_low_ml"] = 1

    ml_activity["sub_num"] = ml_activity["sub_num"].str.replace("sub-", "")
    ml_activity["sub_num"] = ml_activity["sub_num"].astype(int)
    df["ses-4_ml_grouping"] = 0
    for ind in df.index:
        sub_ind = ml_activity.index[ml_activity['sub_num'] == df['sub_num'][ind]].tolist()[0]
        df.loc[ind,'ses-4_ml_grouping'] = ml_activity['high_low_ml'][sub_ind]
    cols = df.columns.tolist()
    cols_reorder = cols[:2] + [cols[-1]] + cols[2:-1]
    df = df[cols_reorder]
elif group_cri == 'test_cont':
    df.insert(df.columns.get_loc('neural_target') + 1, 'group', df['neural_target'].apply(lambda x: 2 if x == 2 else 1))

# chi square tests
def chi_square_test(df, group_col, features):
    results = {}
    for f in features:
        tab = pd.crosstab(df[group_col], df[f])
        
        # CONDUCT CHI SQUARE STATISTICAL TEST
        chi2, p, dof, expected = chi2_contingency(tab)
        
        results[f] = [chi2, p, dof, expected]
        
    results_df = pd.DataFrame.from_dict(results, orient = 'index', columns = ["Chi-squared", 'p_value', 'Degrees_of_freedom', 'Expected_frequencies'])
        
    return results_df    

#correct for multiple comparisons
def fdr_test(results_df: pd.DataFrame(), valid_features): 
    #import packages
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
   

    #dict of each features and its pval (uncorrected)
    pval_dict = results_df['p_value'].T.to_dict()
       
    #calculate new p-values after fdr correction
    fdr = multipletests(list(pval_dict.values()), alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
    corrected_p = fdr[1]
    
    #arrange corrected pval written next to each features
    corrected_p_dict = {}
    for i in range(len(pval_dict.values())):
        corrected_p_dict[valid_features[i]] = corrected_p[i]

    
    return pd.DataFrame.from_dict(corrected_p_dict, orient='index', columns= ['corrected_pval'])


def save_results(df1, df2, path, file_name:str):
    #import packages
    import pandas as pd
    
    #save file
    full_path = path+ f"/{file_name}.xlsx"
    with pd.ExcelWriter(full_path) as writer:  
            df1.to_excel(writer, sheet_name='chi_square_results')
            df2.to_excel(writer, sheet_name='fdr_results')

# run chi square tests
results_chi = chi_square_test(df, 'ses-4_ml_grouping', features_valid)
fdr_chi = fdr_test(results_chi, features_valid)
save_results(results_chi, fdr_chi, f"{out_path}/feature_frequency", f"chi_square_freq_ml_{roi}")

#%% 3.3.2 Figure 2c: 

# 3.3.2.1 Radarplots for mental strategies
path = r'/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/data/mental_strategies'
data = pd.read_excel(f"{path}/fmrinfpractice_msq_data.xlsx")
data.rename(columns={'intercation_with_other_people': 'social','interface_engaged_detached':'interface engagement','positive_valence':'positive\nvalence'}, inplace=True)
msq_cols = list(data.columns.values)

# for session 4 only?
data = data[data['session_num']==4].reset_index(drop=True)

# criterion for sparse features to discard
freq_cri = 0.10
# take only the features data, without subjects data and meta categories
df = data.iloc[:,10:55]

cols = list(df.columns.values)
valid_cols = list(cols)

# drop features with lower than threshold frequency
for col in cols:
    if df[col].sum()/df[col].notnull().sum() < freq_cri:
        valid_cols.remove(col)


print(f"removed features: {set(cols) - set(valid_cols)}")

# Affective features (excluding invalid features):
affect_features = ['happiness', 'love', 'calmness', 'pleasure','social','positive_expectation','positive\nvalence', 'neutral_valence',]

def plot_mental_ftrs(data,ftrs,group_cri,nf_target):

    if group_cri == 'nf_act':
        
        # create subjects grouping based on session 4 ml activity level, above or below median:

        nf_activity = pd.read_csv(f"{main_dir}/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_{nf_target}/group_roi_analysis_{nf_target}_regVSrest_psc_w_subnums_sessions.csv")

        # subect 1178 had only 2 sessions, that in activity measures are sesion 2 and 3. 
        # however, in mental strategies these are 3 and 4. So ses-3 here is his last pre-vaccination session. for the purpose of grouping
        # session 3 is the one that matters and is tagged in metnal stratgeies as session 4. 
        nf_activity.loc[nf_activity['sub_num']=='sub-1178','ses-4'] = nf_activity[nf_activity['sub_num']=='sub-1178']['ses-3'] 

        if nf_target in ['ml_reward','VTA','bilateral_Nac']:
            act_med = stat.median(nf_activity["ses-4"])
            for ind in nf_activity.index:
                if nf_activity.loc[ind, "ses-4"] > act_med:
                    nf_activity.loc[ind, "high_low_nf"] = 2
                else:
                    nf_activity.loc[ind, "high_low_nf"] = 1
        elif nf_target == 'rand_rois':
            # compute grouping per subgroup (as the different targets have different statistics)
            for sg in nf_activity['subgroup'].unique():
                act_med = stat.median(nf_activity[nf_activity['subgroup']==sg]['ses-4'])
                for ind in nf_activity[nf_activity['subgroup']==sg].index:
                    if nf_activity.loc[ind, "ses-4"] > act_med:
                        nf_activity.loc[ind, "high_low_nf"] = 2
                    else:
                        nf_activity.loc[ind, "high_low_nf"] = 1
        
        

        nf_activity["sub_num"] = nf_activity["sub_num"].str.replace("sub-", "")
        nf_activity["sub_num"] = nf_activity["sub_num"].astype(int)
        data["ses-4_nf_grouping"] = 0
        for ind in data.index:
            if data.loc[ind,'sub_num'] in nf_activity['sub_num'].tolist():
                sub_ind = nf_activity.index[nf_activity['sub_num'] == data['sub_num'][ind]].tolist()[0]
                data['ses-4_nf_grouping'][ind] = nf_activity['high_low_nf'][sub_ind]
        cols = data.columns.tolist()
        cols_reorder = cols[:2] + [cols[-1]] + cols[2:-1]
        data = data[cols_reorder]

    # organize data for plotting frequencies of msq_features use:

    def organize_data_for_plotting(df, features: list, group_cri,nf_target):
        # create df of specific meta category
        # important to add neural target column to seperate groups
        meta_category = df[features]

        if group_cri in ['test_cont', 'subgroups']:
            meta_category = meta_category.join(df['neural_target'])

            if group_cri == 'test_cont':
                # split data by neural target
                mesolimbic_reward = meta_category[meta_category['neural_target'] == 2]
                rand_ROI_subgroups = meta_category[(meta_category['neural_target'] == 3) | (meta_category['neural_target'] == 4) | (
                    meta_category['neural_target'] == 5) | (meta_category['neural_target'] == 6)]

        elif group_cri == 'nf_act':
            # add activity grouping:
            meta_category = meta_category.join(df['ses-4_nf_grouping'])
            # split data by ml activity grouping
            high_ml_activity = meta_category[meta_category['ses-4_nf_grouping'] == 2]
            low_ml_activity = meta_category[meta_category['ses-4_nf_grouping'] == 1]

        # create value count for each neural target
        def create_value_count(df, grouping: str):
            df_count = df.apply(pd.Series.value_counts)
            df_count = df_count.transpose()
            df_count.reset_index(inplace=True)
            df_count.rename(columns={'index': 'Feature', 0: 'No', 1: 'Yes'}, inplace=True)
            df_count['total_count'] = df_count['Yes'] + df_count['No']
            df_count['yes_proportion'] = df_count['Yes'] / df_count['total_count']
            df_count['no_proportion'] = df_count['No'] / df_count['total_count']
            df_count['grouping'] = grouping
            df_count = df_count.drop(df_count[df_count['Feature'] == 'neural_target'].index)
            if group_cri == 'nf_act':
                df_count = df_count.drop(df_count[df_count['Feature'] == 'ses-4_nf_grouping'].index)
            return df_count

        if group_cri == 'test_cont':
            # count for neural target grouping:
            ml_reward_count = create_value_count(mesolimbic_reward, 'reward ml NF')
            rand_ROI_count = create_value_count(rand_ROI_subgroups, 'rand ROI NF')
            # concat value count df's
            total_count = pd.concat([ml_reward_count, rand_ROI_count], ignore_index=True)
            total_count.drop(columns=[2, 3, 4, 5, 6], inplace=True)
        if group_cri == 'nf_act':
            # count for activity grouping:
            high_ml_activity_count = create_value_count(high_ml_activity, f"high {nf_target}\nactivity")
            low_ml_activity_count = create_value_count(low_ml_activity, f"low {nf_target}\nactivity")
            # concat value count df's
            total_count = pd.concat([high_ml_activity_count, low_ml_activity_count], ignore_index=True)
            total_count.drop(columns=[2], inplace=True)

        # set for current sepration of valence...
        # if ('Valence' in features) or ('Arousal' in features):
        #     total_count.rename(columns = {'Yes': 'Low/Negative', 'No': 'High/Positive'}, inplace = True)

        return total_count

    affect_count = organize_data_for_plotting(data, affect_features, group_cri,nf_target)
    return affect_count

def create_spyder_plot(df, features_for_plot, group_cri, nf_target, title, out_path):
    
    if group_cri == 'test_cont':
        groups_labels = ['reward ml NF','rand ROI NF']
        color_codes = ['blue','dimgrey']
    elif group_cri == 'nf_act':
        groups_labels = [f"high {nf_target}\nactivity",f"low {nf_target}\nactivity"]
        if nf_target == 'ml_reward':
            color_codes = ['blue','lightskyblue']
        elif nf_target == 'VTA':
            color_codes = ['blue','lightskyblue']
        elif nf_target == 'bilateral_Nac':
            color_codes = ['blueviolet','pink']
        elif nf_target == 'rand_rois':
            color_codes = ['darkgreen','limegreen']

    #    color_codes = ['purple','darkorange']
    # Get the data for the two groups
    df_exp = df[df['grouping'] == groups_labels[0]]
    df_cont = df[df['grouping'] == groups_labels[1]]


    # Select the features for the plot
    df_exp = df_exp[df_exp['Feature'].isin(features_for_plot)]
    df_cont = df_cont[df_cont['Feature'].isin(features_for_plot)]

    # Create the background of the plot
    categories = list(df_exp['Feature'])
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linewidth(0.5)
    ax.spines['polar'].set_zorder(0)  # Set the z-order of the grid lines to 0
    ax.xaxis.grid(linestyle='-', linewidth=0.5, color='gray', alpha=0.5, zorder=1)
    ax.yaxis.grid(linestyle='-', linewidth=0.5, color='gray', alpha=0.5, zorder=1)
    ax.set_xticks(angles[:-1])
    
    # Replace '_' with ' ' in x-axis labels
    ax.set_xticklabels([category.replace('_', ' ') for category in categories], fontsize=13, zorder=10)
    
    ax.set_rlabel_position(0)
    y_ticks = [i/100 for i in range(0, 110, 20)]
    plt.yticks(y_ticks, [str(x) for x in y_ticks], color="black", size=10)
    plt.ylim(0, 1)

    # Plot the data
    values_exp = df_exp['yes_proportion'].values.flatten().tolist()
    values_exp += values_exp[:1]
    ax.plot(angles, values_exp, color=color_codes[0], linewidth=1.5,
            linestyle='solid', label=groups_labels[0])
    ax.fill(angles, values_exp, 'b', alpha=0.1)

    values_cont = df_cont['yes_proportion'].values.flatten().tolist()
    values_cont += values_cont[:1]
    ax.plot(angles, values_cont, color=color_codes[1], linewidth=1.5,
            linestyle='solid', label=groups_labels[1])
    ax.fill(angles, values_cont, 'b', alpha=0.1)

    # Set the z-order of the y-axis labels to 10
    ax.yaxis.set_tick_params(labelcolor='black', zorder=10)

    # # Add the legend
    # plt.legend(loc='lower left', prop={'size': 10}, bbox_to_anchor=(-0.25, -0.25), borderaxespad=0)
    # Add the legend
    # plt.legend(loc='lower left', prop={'size': 10}, bbox_to_anchor=(-0.25, -0.25), borderaxespad=0, bbox_transform=ax.transAxes, ncol=2)


    # Add the title
    # plt.title(title)

    # Set the background to be transparent
    plt.gca().set_facecolor('none')
    # Save the plot
    plt.savefig(out_path, bbox_inches='tight', dpi=300, transparent=True)

# extract affect features for VTA, Nac and Rand. Networks

group_cri = 'nf_act'
nf_target = 'bilateral_Nac' # 'rand_rois'/ 'VTA'/ 'bilateral_Nac'
affect_count = plot_mental_ftrs(data,affect_features,group_cri,nf_target)
create_spyder_plot(affect_count, affect_features, group_cri, nf_target,
                   'Affect Features',f"{out_path}/affect_{group_cri}_{nf_target}_ses4.png")

# 3.3.2.2. Barplots for positive expectation across targets:

# change rows order based on nf_target and then nf_grouping:
ftr_freq = ftr_freq.sort_values(by=['nf_target','nf_grouping']).reset_index(drop=True) 


def plot_pos_exp(ftr_freq, ftr, out_path):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Define your color mapping
    color_dict = {'vta': {'low': 'lightskyblue', 'high': 'blue'}, 
                  'bilateral_nac': {'low': 'violet', 'high': 'blueviolet'}, 
                  'rand_rois': {'low': 'limegreen', 'high': 'darkgreen'}}

    # Define the order of the 'nf_target' and 'nf_grouping' values
    nf_target_order = ['vta', 'bilateral_nac', 'rand_rois']
    nf_grouping_order = ['low', 'high']

    # Create a new figure
    fig, ax = plt.subplots()

    # Iterate over the 'nf_target' and 'nf_grouping' values and create the bars manually
    for i, nf_target in enumerate(nf_target_order):
        for j, nf_grouping in enumerate(nf_grouping_order):
            data = ftr_freq[(ftr_freq['nf_target'] == nf_target) & (ftr_freq['nf_grouping'] == nf_grouping)]
            ax.bar(i + j * 0.4, data[f"{ftr}_frequency"], width=0.4, color=color_dict[nf_target][nf_grouping])

    ax.set(xlabel='neural target', ylabel=f"{ftr} frequency")

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="NF modulation\nlevels")
    # set y axis label:
    ax.set_ylabel(f"Frequency of {ftr}\n in mental strategies", fontsize=15)
    # set yticks
    ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5], fontsize=14)
    # remove x grid lines:
    ax.xaxis.grid(False)
    # remove x axis label:
    ax.set_xlabel("")
    ax.set_xticks([i + 0.2 for i in range(len(nf_target_order))])  # Set the x-tick locations
    # set x axis ticks:
    ax.set_xticklabels(['VTA','Nac','Control Networks'], fontsize=15)
    # plt.savefig(f"{out_path}/{ftr}_frequency_across_targets_SUBS.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
plot_pos_exp(ftr_freq, ftr, out_path)

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

#%% 5.4. Examine logistic regression model for positive expectation frequency

import statsmodels.api as sm

# Assuming data is your pandas DataFrame containing the variables
# 'positive expectation', 'vta_grouping', and 'bilateral_nac_grouping'
df_test = data[['positive_expectation', 'vta_grouping', 'bilateral_nac_grouping']]
# Add a constant (intercept) to the independent variables
df_test['const'] = 1

df_test[['vta_grouping', 'bilateral_nac_grouping']] = df_test[['vta_grouping', 'bilateral_nac_grouping']] - 1
# Define the independent variables (features)
X = df_test[['const', 'vta_grouping', 'bilateral_nac_grouping']]

# Define the dependent variable (target)
y = df_test['positive_expectation']

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Print the summary statistics
print(result.summary())

# Now try having all three variables in the model
# Assuming data is your pandas DataFrame containing the variables
# 'positive expectation', 'vta_grouping', and 'bilateral_nac_grouping'
df_test = data[['positive_expectation', 'vta_grouping', 'bilateral_nac_grouping','rand_rois_grouping']]
# Add a constant (intercept) to the independent variables
df_test['const'] = 1

# exclude lines with rand_rois_grouping=0 and reset index
df_test = df_test[df_test['rand_rois_grouping']!=0].reset_index(drop=True)

df_test[['vta_grouping', 'bilateral_nac_grouping','rand_rois_grouping']] = df_test[['vta_grouping', 'bilateral_nac_grouping','rand_rois_grouping']] - 1

# Define the independent variables (features)
X = df_test[['const', 'vta_grouping', 'rand_rois_grouping']]
# Define the dependent variable (target)
y = df_test['positive_expectation']

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Print the summary statistics
print(result.summary())