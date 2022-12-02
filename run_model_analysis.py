#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cv_master import Grid2DKernel, SatelliteImageData
from cv_master.cv import cv_classifer, nested_cv_classifer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from matplotlib import cm, colors
import argparse

from evalaute import *
from plot import *

sns.set_style('whitegrid')
np.random.seed(521)

# python run_model_analysis.py --train_set 2,3 --test_set 1 --fp 'tr_23_te_1'
# python run_model_analysis.py --train_set 1,2 --test_set 3 --fp 'tr_12_te_3'


parser = argparse.ArgumentParser()

parser.add_argument('-i', '--train_set', required=True, 
    help="train set image no",type=str)
parser.add_argument('-j', '--test_set', required=True, 
    help="Test set image", type=str)
parser.add_argument('-k', '--fp', required=True, 
    help="Filepath_name")


args = parser.parse_args()

train_set_image_ids = [int(item) for item in args.train_set.split(',')]
test_set_image_ids = [int(item) for item in args.test_set.split(',')]
images_fp = args.fp


#### Run everything again for feature importances ####
# Import Full Data

COLUMNES = ['y', 'x', 'label', 'ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']
train_columns = ['ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']

imagem1 = pd.read_csv('cv_master/data/imagem1.txt',delim_whitespace=True, header=None)
imagem2 = pd.read_csv('cv_master/data/imagem2.txt',delim_whitespace=True, header=None)
imagem3 = pd.read_csv('cv_master/data/imagem3.txt',delim_whitespace=True, header=None)

imagem1.columns = COLUMNES
imagem2.columns = COLUMNES
imagem3.columns = COLUMNES

imagem1['Image_no'] = 1
imagem2['Image_no'] = 2
imagem3['Image_no'] = 3

image_set = pd.concat([imagem1, imagem2, imagem3],0)
remap_labels = {1:1,  # Cloud
                -1:0, # Not Cloud
                0:2} # Unlabelled


image_set['label'] = image_set['label'].apply(lambda x: remap_labels[x])
image_set = image_set[image_set['label']!=2]


#### Test scheme 1 (CV + Fixed test set) ####
kernel = Grid2DKernel(55, 55)
image_test1 = SatelliteImageData(kernel, images=['imagem1.txt', 'imagem2.txt'])

# Load test set
COLUMNES = ['y', 'x', 'label', 'ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']
train_columns = ['ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']

X_train_1 = image_set[image_set['Image_no'].isin(train_set_image_ids)][train_columns]
y_train_1 = image_set[image_set['Image_no'].isin(train_set_image_ids)][['label']]

X_test_1 = image_set[image_set['Image_no'].isin(test_set_image_ids)][train_columns]
y_test_1 = image_set[image_set['Image_no'].isin(test_set_image_ids)][['label']]


#### Test scheme 2 (Nested CV + Testing)####
image_test2 = SatelliteImageData(kernel, images=['imagem1.txt', 'imagem2.txt', 'imagem3.txt'])

# Define Metrics
metric_list = [accuracy_score, balanced_accuracy_score]



model_dict = {'LR': LogisticRegression(penalty='l2', max_iter=1000),
              'RF': RandomForestClassifier(),
              'CART': DecisionTreeClassifier(min_samples_split=10),
              'Adaboost': AdaBoostClassifier(),
              'Baseline': DummyClassifier(strategy='constant',constant=0)}

result_store = {}

for mod_name, model in model_dict.items():

    scheme1_results = eval_model_test_scheme1(model,
                                 dataset1=(image_test1, (X_train_1, y_train_1), (X_test_1, y_test_1)),
                                 metric_list=metric_list)

    scheme2_results = eval_model_test_scheme2(model,
                                 dataset2=image_test2,
                                 metric_list=metric_list)
    
    result_store[mod_name] = [scheme1_results, scheme2_results]
    

# Model analysis

features = ['ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']


dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train_1[features], y_train_1)

stump_pred = dt_stump.predict(X_test_1[features])
print(np.mean(stump_pred == y_test_1.values.reshape(-1)))
print(balanced_accuracy_score(stump_pred, y_test_1.values.reshape(-1)))


adaboost_analysis = AdaBoostClassifier(n_estimators=50,learning_rate=0.1)
adaboost_analysis.fit(X_train_1[features], y_train_1)

ada_pred = adaboost_analysis.predict(X_test_1[features])
print(np.mean(ada_pred == y_test_1.values.reshape(-1)))
print(balanced_accuracy_score(ada_pred, y_test_1.values.reshape(-1)))



log_reg_analysis = LogisticRegression(penalty='l2', max_iter=1000)
log_reg_analysis.fit(X_train_1[features], y_train_1)

lr_pred = log_reg_analysis.predict(X_test_1[features])
np.mean(lr_pred == y_test_1.values.reshape(-1))


n_est = adaboost_analysis.n_estimators



ada_train_err = np.zeros((n_est,))
for i, y_pred_train in enumerate(adaboost_analysis.staged_predict(X_train_1[features])):
    ada_train_err[i] = 1 - accuracy_score(y_pred_train, y_train_1.values)

ada_test_err = np.zeros((n_est,))
for i, y_pred_test in enumerate(adaboost_analysis.staged_predict(X_test_1[features])):
    ada_test_err[i] = 1 - accuracy_score(y_pred_test, y_test_1.values)


# In[72]:


fig, ax = plt.subplots(1,2, figsize=(10,4))

ax[0].plot(range(n_est),ada_train_err,label='Train Error')
ax[1].plot(range(n_est),ada_test_err,color='red', label='Test Error')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('Number Weak Learners')
ax[1].set_xlabel('Number Weak Learners')

ax[0].set_ylabel('Train Error Rate')
ax[1].set_ylabel('Test Error Rate')
plt.savefig(f'statics/ada_iterations_{images_fp}.png')


# ## Generic Decision Surface Analysis

# Calculate Probability on image 3 diverging color scheme

COLUMNES = ['y', 'x', 'label', 'ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']
train_columns = ['ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']

imagem1 = pd.read_csv('cv_master/data/imagem1.txt',delim_whitespace=True, header=None)
imagem2 = pd.read_csv('cv_master/data/imagem2.txt',delim_whitespace=True, header=None)
imagem3 = pd.read_csv('cv_master/data/imagem3.txt',delim_whitespace=True, header=None)

imagem1.columns = COLUMNES
imagem2.columns = COLUMNES
imagem3.columns = COLUMNES

imagem1['Image_no'] = 1
imagem2['Image_no'] = 2
imagem3['Image_no'] = 3

image_set = pd.concat([imagem1, imagem2, imagem3],0)
remap_labels = {1:1,
                -1:0,
                0:2}


image_set['label_remaped'] = image_set['label'].apply(lambda x: remap_labels[x])
#image_set = image_set[image_set['label']!=2]


# Train Preds
train_im = image_set[image_set['Image_no'].isin(train_set_image_ids)]
train_im_features = train_im[features]
prob_preds = adaboost_analysis.predict_proba(train_im_features)[:,1]
train_im['Predicted_Probs'] = prob_preds


# Test Preds
test_im = image_set[image_set['Image_no'].isin(test_set_image_ids)]
test_im_features = image_set[image_set['Image_no'].isin(test_set_image_ids)][features]
prob_preds = adaboost_analysis.predict_proba(test_im_features)[:,1]
test_im['Predicted_Probs'] = prob_preds



image = test_im.copy()

fig, ax = plt.subplots(1,2,figsize=(5*2, 4))

fake_row = image.iloc[-1,:].copy()
fake_row['Predicted_Probs']=1
fake_row['x']=+0.01
image = pd.concat([image, pd.DataFrame(fake_row).T])
image.reset_index(inplace=True)

fake_row = image.iloc[-1,:].copy()
fake_row['Predicted_Probs']=0
fake_row['y']=+0.01
image = pd.concat([image, pd.DataFrame(fake_row).T])
image.reset_index(inplace=True)

# universial divergent color map
cmap = sns.color_palette("vlag", as_cmap=True)
plot_heatmap(get_heatmap_data(image, 'label'), label='Test Image', cmap=cmap, ax=ax[0])
plot_heatmap(get_heatmap_data(image, 'Predicted_Probs'), label='Test Image Predicted Probs', cmap=cmap, ax=ax[1])
plt.savefig(f'statics/ada_test_prob_{images_fp}.png')



image = train_im[train_im['Image_no']==train_set_image_ids[0]].copy()

fig, ax = plt.subplots(1,2,figsize=(5*2, 4))

fake_row = image.iloc[-1,:].copy()
fake_row['Predicted_Probs']=1
fake_row['x']=+0.01
image = pd.concat([image, pd.DataFrame(fake_row).T])
image.reset_index(inplace=True)

fake_row = image.iloc[-1,:].copy()
fake_row['Predicted_Probs']=0
fake_row['y']=+0.01
image = pd.concat([image, pd.DataFrame(fake_row).T])
image.reset_index(inplace=True)

# universial divergent color map
cmap = sns.color_palette("vlag", as_cmap=True)
plot_heatmap(get_heatmap_data(image, 'label'), label='Train Image', cmap=cmap, ax=ax[0])
plot_heatmap(get_heatmap_data(image, 'Predicted_Probs'), label='Train Image Predicted Probs', cmap=cmap, ax=ax[1])
plt.savefig(f'statics/ada_train_prob1_{images_fp}.png')


image = train_im[train_im['Image_no']==train_set_image_ids[1]].copy()

fig, ax = plt.subplots(1,2,figsize=(5*2, 4))

fake_row = image.iloc[-1,:].copy()
fake_row['Predicted_Probs']=1
fake_row['x']=+0.01
image = pd.concat([image, pd.DataFrame(fake_row).T])
image.reset_index(inplace=True)

fake_row = image.iloc[-1,:].copy()
fake_row['Predicted_Probs']=0
fake_row['y']=+0.01
image = pd.concat([image, pd.DataFrame(fake_row).T])
image.reset_index(inplace=True)

# universial divergent color map
cmap = sns.color_palette("vlag", as_cmap=True)
plot_heatmap(get_heatmap_data(image, 'label'), label='Train Image', cmap=cmap, ax=ax[0])
plot_heatmap(get_heatmap_data(image, 'Predicted_Probs'), label='Train Image Predicted Probs', cmap=cmap, ax=ax[1])
plt.savefig(f'statics/ada_train_prob2_{images_fp}.png')


# # Feature Values

# # Generic Surface Analysis
log_reg_analysis = LogisticRegression(penalty='l2', max_iter=1000)
log_reg_analysis.fit(X_train_1[features], y_train_1)



train_im = train_im[train_im['label_remaped']!=2]

pos_samps = train_im[train_im['label_remaped']==1].copy().sample(2500)
neg_samps = train_im[train_im['label_remaped']==0].copy().sample(3000)

projected_points = pd.concat([pos_samps, neg_samps])

pca_samples = train_im.sample(4000)
pca = PCA(n_components=2)
pca = KernelPCA(n_components=2,
                kernel='linear',degree=2,
                gamma=0.00000001,
                fit_inverse_transform=True,
                alpha=0.1)
#tsne = TSNE(n_components=2)
# Standardize
pca.fit(pca_samples[features])
out = pca.transform(projected_points[features])

#inverse_transform



plt.scatter(out[:,0],
            out[:,1],
            color=cmap(projected_points['label']),
            alpha=0.5,
           marker='x')


# In[97]:


x_min = -120
x_max = 350
y_min = -120
y_max = 155

x_range = np.linspace(x_min, x_max, 300)
y_range = np.linspace(y_min, y_max,300)
low_dim_cands = np.array([[x0, y0] for x0 in x_range for y0 in y_range])

high_dim_rep = pca.inverse_transform(low_dim_cands)

# Pass high dim rep into classifier
low_dim_cands_labels = log_reg_analysis.predict_proba(high_dim_rep)
#low_dim_cands_labels = adaboost_analysis.predict_proba(high_dim_rep)

fig, ax = plt.subplots(figsize=(10,8))


cmap_labels = {0: 'powderblue',
               1: 'mistyrose'}

#low_dim_cands_colors =  pd.Series(low_dim_cands_labels).apply(lambda x: cmap_labels[x])

im_plot = ax.scatter(low_dim_cands[:,0],
            low_dim_cands[:,1],
            color=cmap(low_dim_cands_labels[:,1]),
            alpha=0.1)


ax.scatter(out[:,0],
            out[:,1],
            color=cmap(projected_points['label']),
            alpha=1,
           marker='x',s=2)

ax.set_ylim([y_min, y_max])
ax.set_xlim([x_min, x_max])

from matplotlib import cm, colors

fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)
plt.savefig(f'statics/log_reg_decision_surface_{images_fp}.png')



x_min = -120
x_max = 350
y_min = -120
y_max = 155

x_range = np.linspace(x_min, x_max, 300)
y_range = np.linspace(y_min, y_max,300)
low_dim_cands = np.array([[x0, y0] for x0 in x_range for y0 in y_range])

high_dim_rep = pca.inverse_transform(low_dim_cands)

# Pass high dim rep into classifier
#low_dim_cands_labels = log_reg_analysis.predict_proba(high_dim_rep)
low_dim_cands_labels = adaboost_analysis.predict_proba(high_dim_rep)

fig, ax =plt.subplots(figsize=(10,8))


cmap_labels = {0: 'powderblue',
               1: 'mistyrose'}

#low_dim_cands_colors =  pd.Series(low_dim_cands_labels).apply(lambda x: cmap_labels[x])

"""
ax.scatter(low_dim_cands[:,0],
            low_dim_cands[:,1],
            color=low_dim_cands_colors,
            alpha=0.1)
"""
im_plot = ax.scatter(low_dim_cands[:,0],
            low_dim_cands[:,1],
            color=cmap(low_dim_cands_labels[:,1]),
            alpha=0.1)


ax.scatter(out[:,0],
            out[:,1],
            color=cmap(projected_points['label']),
            alpha=1,
           marker='x',s=2)

ax.set_ylim([y_min, y_max])
ax.set_xlim([x_min, x_max])


fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)

plt.savefig(f'statics/PCA_decision_surface_{images_fp}.png')




#### Analyze feature importance ####
test_feaure_importance = pd.DataFrame([adaboost_analysis.feature_names_in_,
            adaboost_analysis.feature_importances_]).T

test_feaure_importance.columns = ['Feature Name','Importance']



cv_importances = [clf_k[-1].feature_importances_ for clf_k in result_store['Adaboost'][0]['scheme_1_raw']]

cv_impt = [pd.DataFrame([adaboost_analysis.feature_names_in_,
            impt]).T for impt in cv_importances]



mean_cv_importance = (1/len(cv_importances)) * sum(cv_importances)


fig, ax = plt.subplots(figsize=(7,7))
x_vals = list(range(1, 1+len(features)))

for impt in cv_importances:
    
    ax.plot(x_vals, impt,
            color='lightskyblue',
            lw=0.1,
            alpha=0.3,marker='*')

ax.plot(x_vals, impt,
        color='lightskyblue',
        lw=0.1,
        alpha=0.3,marker='*',label='Feature Importance Indiv Folds')
ax.plot(x_vals, test_feaure_importance['Importance'],
        color='red',lw=0.5,alpha=1,marker='*', label='Feature Importance Full Train (Test Scheme 1)')
ax.plot(x_vals, mean_cv_importance,
        color='blue',lw=0.5,alpha=1,marker='*', label='Mean Feature Importance Cross Validation (Test Scheme 1)')
    
ax.set_xticklabels([""]+features)
ax.set_xlabel("Feature Name")
ax.set_ylabel("CV Feature Importance")
ax.legend()
plt.savefig(f'statics/Feature_importance_test_scheme1_{images_fp}.png')


### Histogram analysis ###
ada_pred = adaboost_analysis.predict(X_test_1[features])
print(np.mean(ada_pred == y_test_1.values.reshape(-1)))
print(balanced_accuracy_score(ada_pred, y_test_1.values.reshape(-1)))

X_test_1_analysis = X_test_1.copy()
X_test_1_analysis['y_true'] = y_test_1.values.reshape(-1)
X_test_1_analysis['y_pred'] = ada_pred

X_test_1_analysis['Misclassified'] = X_test_1_analysis['y_pred'] != X_test_1_analysis['y_true'] 

true_neg = X_test_1_analysis[X_test_1_analysis['y_true']==0]
true_pos = X_test_1_analysis[X_test_1_analysis['y_true']==1]

print(true_neg['Misclassified'].value_counts())
print(true_pos['Misclassified'].value_counts())

subsampled_true_neg = pd.concat([true_neg[true_neg['Misclassified']==False].sample(10000),
                      true_neg[true_neg['Misclassified']==True]])

subsampled_true_pos = pd.concat([true_pos[true_pos['Misclassified']==False].sample(2000),
                      true_pos[true_pos['Misclassified']==True]])

subsampled_true_neg['Misclassified'].value_counts()
subsampled_true_pos['Misclassified'].value_counts()


fig,ax = plt.subplots(1,3,figsize=(30,10))

sns.set(font_scale=2.5)
sns.set_style('whitegrid')


sns.histplot(subsampled_true_neg, x="ndai", hue="Misclassified", ax=ax[0],stat='probability')
sns.histplot(subsampled_true_neg, x="sd", hue="Misclassified", ax=ax[1],stat='probability')
sns.histplot(subsampled_true_neg, x="corr", hue="Misclassified", ax=ax[2],stat='probability')

plt.tight_layout()
plt.savefig(f'statics/feature_compare_true_neg_{images_fp}.png')


fig,ax = plt.subplots(1,3,figsize=(30,10))

sns.set(font_scale=2.5)
sns.set_style('whitegrid')


sns.histplot(subsampled_true_pos, x="ndai", hue="Misclassified", ax=ax[0],stat='probability')
sns.histplot(subsampled_true_pos, x="sd", hue="Misclassified", ax=ax[1],stat='probability')
sns.histplot(subsampled_true_pos, x="corr", hue="Misclassified", ax=ax[2],stat='probability')

plt.tight_layout()
plt.savefig(f'statics/feature_compare_true_pos_{images_fp}.png')

