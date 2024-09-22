
# nf_immune_ROI_analysis.R

# This script runs mixed linear effects regression analyses to estimate group neurofeedback reglation effects, as reported in the manuscript:
# "Upregulation of Reward Mesolimbic Activity via fMRI-Neurofeedback Improves Vaccination Efficiency in Humans"

# Author:
#    Nitzan Lubianiker, PhD.
#    nitsan.lubianiker@yale.edu

# Date:
#    2024-07-25

# import packages:
library(lme4)
library(tidyr)
library(readr)
library(dplyr)

# define ROI variable:
roi = 'VTA' # VTA or bilateral_nac

# import data:
if (roi == 'VTA') {
  nf_act <- read_csv(file = '/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/fmri/offline/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_VTA/group_roi_analysis_VTA_regVSrest_psc_long.csv')
} else if (roi == 'NAC') {
  nf_act <- read_csv(file = '/Users/nitzanlubi/My Drive/Lab/Joy/joy_italy/analyses/fmri/offline/derivatives/models/group_statistics/fmrinfpractice/roi_analyses/offline_bilateral_nac/group_roi_analysis_bilateral_nac_regVSrest_psc_long.csv')
}

# Remove 1st session due to extensive partial data 
nf_act<-nf_act[!(nf_act$session==1 ),]

# to test group specific effects, exclude the other group:
nf_act<-nf_act[!(nf_act$group==1 ),] # 1 is the control group, 2 is the experimental group.

# define variables:
ID <- nf_act[, 2]
group <- nf_act[, 3]
session <- nf_act[, 4]; run <- nf_act[, 5]; time <- nf_act[, 8]; roi_act <- nf_act[, 7]

df<-data.frame(ID,group,session,run,roi_act) # for full model
df<-data.frame(ID,session,run,roi_act) # for group-specific model

# define factors
df$sub_id<-as.factor(df$sub_id)
df$group<-factor(df$group,levels=c(1,2),labels=c("rand ROI NF", "reward ml NF")) # don't run this line when testing a specific group
df$session<-as.factor(df$session)
df$run<-as.numeric(df$run)

# for the full model:
if (roi == 'VTA') {
  lme_model <- lmer(VTA_act ~ group*session + session/run + (1|sub_id), data=df,REML=T)
} else if (roi == 'NAC') {
  lme_model <- lmer(bilateral_nac_act ~ group*session + session/run + (1|sub_id), data=df,REML=T)
}

# for group specific modelL
if (roi == 'VTA') {
  lme_model <- lmer(VTA_act ~ session + session/run + (1|sub_id), data=df,REML=T)
} else if (roi == 'NAC') {
  lme_model <- lmer(bilateral_nac_act ~ session + session/run + (1|sub_id), data=df,REML=T)
}

anova(lme_model)