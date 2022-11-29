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


from evalaute import *
from plot import *

sns.set_style('whitegrid')
np.random.seed(521)



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

X_train_1 = image_set[image_set['Image_no'].isin([2,3])][train_columns]
y_train_1 = image_set[image_set['Image_no'].isin([2,3])][['label']]

X_test_1 = image_set[image_set['Image_no'].isin([1])][train_columns]
y_test_1 = image_set[image_set['Image_no'].isin([1])][['label']]


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
    


# ### Scheme 1 Results
model_namelist = ['LR','RF','CART','Adaboost','Baseline']



scheme1_results = pd.DataFrame.from_dict({'Test Acc': [result_store[x][0]['Test_acc'] for x in model_namelist],
                       'Test Balanced Acc': [result_store[x][0]['Test_bal_acc'] for x in model_namelist],
                       'CV Acc': [result_store[x][0]['CV_acc'] for x in model_namelist],
                       'CV Balanced ACC':[result_store[x][0]['CV_bal_acc'] for x in model_namelist]})
scheme1_results.index = model_namelist
scheme1_results.to_csv('statics/scheme1_results.csv')


# Indiv Fold Accuracy
model_namelist_for_plot = ['LR', 'RF', 'CART', 'Adaboost']

# Plot Fold Errors for Test Scheme 1
coords = [(0,0),(0,1),(1,0),(1,1)]
fig,ax = plt.subplots(2,2,figsize=(10,10))

for co,mod in zip(coords,model_namelist_for_plot):
    
    fold_ids = list(range(len(result_store[mod][0]['scheme_1_raw'])))

    acc_scores = [result_store[mod][0]['scheme_1_raw'][f][0][0]                  for f in fold_ids]

    bal_acc_scores = [result_store[mod][0]['scheme_1_raw'][f][0][1]                  for f in fold_ids]

    ax[co[0],co[1]].plot(fold_ids, acc_scores, marker='*', color='blue', label='Accuracy Across Folds')
    ax[co[0],co[1]].plot(fold_ids, bal_acc_scores, marker='*',color='green', label='Balanced Accuracy Across Folds')
    ax[co[0],co[1]].legend()
    ax[co[0],co[1]].title.set_text(f'Scheme 1 CV Per Fold Error Rate: {mod}')

plt.savefig('statics/test_Scheme_1_Fold_error.png')


# ### Scheme 2 Results
scheme2_results = pd.DataFrame.from_dict({'Test Acc': [result_store[x][1]['Test_acc'] for x in model_namelist],
                       'Test Balanced Acc': [result_store[x][1]['Test_bal_acc'] for x in model_namelist],
                       'CV Acc': [result_store[x][1]['CV_acc'] for x in model_namelist],
                       'CV Balanced ACC':[result_store[x][1]['CV_bal_acc'] for x in model_namelist]})
scheme2_results.index = model_namelist
scheme2_results.to_csv('statics/scheme2_results.csv')



# Plot Fold Errors Test Scheme 2
coords = [(0,0),(0,1),(1,0),(1,1)]
fig,ax = plt.subplots(2,2,figsize=(10,10))

for co,mod in zip(coords,model_namelist_for_plot):
    
    fold_ids = list(range(len(result_store[mod][1]['scheme_2_raw'])))

    acc_scores = [result_store[mod][1]['scheme_2_raw'][f][0][0]                  for f in fold_ids]

    bal_acc_scores = [result_store[mod][1]['scheme_2_raw'][f][0][1]                  for f in fold_ids]

    ax[co[0],co[1]].plot(fold_ids, acc_scores, marker='*', color='blue', label='Accuracy Across Folds')
    ax[co[0],co[1]].plot(fold_ids, bal_acc_scores, marker='*',color='green', label='Balanced Accuracy Across Folds')
    ax[co[0],co[1]].legend()
    ax[co[0],co[1]].title.set_text(f'Scheme 2 CV Per Fold Error Rate: {mod}')

plt.savefig('statics/test_Scheme2_fold_error.png')


## Plot ROC Curves

# Test Scheme 1
chosen_fpr = 0.3



import warnings
warnings.filterwarnings("ignore")




fig,ax = plt.subplots(figsize=(7,7))

plot_CV_ROC(result_store['LR'][0]['scheme_1_raw'],
            'bisque',
            'darkorange',
            'LR',
            ax=ax,
            chosen_fpr=chosen_fpr)

plot_CV_ROC(result_store['RF'][0]['scheme_1_raw'],
            'mistyrose',
            'red',
            'RF',
            ax=ax,
            chosen_fpr=chosen_fpr)


plot_CV_ROC(result_store['CART'][0]['scheme_1_raw'],
            'honeydew',
            'green',
            'CART',
            ax=ax,
            chosen_fpr=chosen_fpr)


plot_CV_ROC(result_store['Adaboost'][0]['scheme_1_raw'],
            'lavender',
            'mediumblue',
            'Ada',
            ax=ax,
            chosen_fpr=chosen_fpr)

ax.legend()
ax.title.set_text('Test Scheme 1: Cross Validated Folds and Mean ROC')
plt.savefig('statics/test_Scheme1_cv_roc.png')


# Test ROC Scheme 1
test_scores = [result_store[mod_name][0]['test_scores'] for mod_name in model_namelist]

fpr_tpr_col = []
roc_col = []

for ts in test_scores:

    fpr, tpr, _ = roc_curve(y_test_1.values,ts[:,1])
    roc_auc = auc(fpr, tpr)
    
    fpr_tpr_col.append((fpr, tpr))
    roc_col.append(roc_auc)
    
model_names = ['LR', 'RF','CART', 'Ada']
model_colors = ['darkorange','red','green','mediumblue']


fig, ax = plt.subplots(figsize=(7,7))

lw = 2

mean_fpr = np.linspace(0, 1, 100)
# Add in the chosen FPR level
mean_fpr = np.insert(mean_fpr, find_change_point(mean_fpr < chosen_fpr), chosen_fpr)

for fpr_tpr, roc, mod_name, mod_col in zip(fpr_tpr_col,
                                           roc_col,
                                           model_names,
                                           model_colors):
    
    interp_tpr = np.interp(mean_fpr, fpr_tpr[0], fpr_tpr[1])
    # Find associated mean tpr value
    tpr_index = np.where(mean_fpr == chosen_fpr)[0][0]
    tpr_val_chosen = interp_tpr[tpr_index]
    
    ax.plot(
        mean_fpr,
        interp_tpr,
        color=mod_col,
        lw=lw,
        label="ROC curve %s (area = %0.2f) (TPR = %0.2f)" % (mod_name,roc,tpr_val_chosen),
    )
    
    # Plot tpr val

    
    ax.plot([chosen_fpr,chosen_fpr],
            [0,tpr_val_chosen], alpha=0.5, color='black', ls='-')

    ax.plot([0,chosen_fpr],
            [tpr_val_chosen,tpr_val_chosen], alpha=0.5, color=mod_col, ls='--')
    
    
ax.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.title.set_text("Test Scheme 1: Test Set ROC")
ax.legend(loc="lower right")
plt.show()
plt.savefig('statics/test_Scheme1_test_roc.png')


# ## ROC Test Scheme 2
fig,ax = plt.subplots(figsize=(7,7))

plot_CV_ROC(result_store['LR'][1]['scheme_2_raw'],
            'bisque',
            'darkorange',
            'LR',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=False)

plot_CV_ROC(result_store['RF'][1]['scheme_2_raw'],
            'mistyrose',
            'red',
            'RF',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=False)


plot_CV_ROC(result_store['CART'][1]['scheme_2_raw'],
            'honeydew',
            'green',
            'CART',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=False)


plot_CV_ROC(result_store['Adaboost'][1]['scheme_2_raw'],
            'lavender',
            'mediumblue',
            'Ada',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=False)

ax.legend()
ax.title.set_text('Test Scheme 2: Cross Validated Folds and Mean ROC')
plt.savefig('statics/test_Scheme2_cv_roc.png')


# In[62]:


fig,ax = plt.subplots(figsize=(7,7))

plot_CV_ROC(result_store['LR'][1]['scheme_2_raw'],
            'bisque',
            'darkorange',
            'LR',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=True)

plot_CV_ROC(result_store['RF'][1]['scheme_2_raw'],
            'mistyrose',
            'red',
            'RF',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=True)


plot_CV_ROC(result_store['CART'][1]['scheme_2_raw'],
            'honeydew',
            'green',
            'CART',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=True)


plot_CV_ROC(result_store['Adaboost'][1]['scheme_2_raw'],
            'lavender',
            'mediumblue',
            'Ada',
            ax=ax,
            chosen_fpr=chosen_fpr,
            nested_test=True)

ax.legend()
ax.title.set_text('Test Scheme 2: Test Set ROC')
plt.savefig('statics/test_Scheme2_test_roc.png')


# # Model Analysis
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
image_set = image_set[image_set['label_remaped']!=2]


# Analyse Error vs Number of Trees (Can it further decrease)

X_train_1 = image_set[image_set['Image_no'].isin([2,3])][train_columns]
y_train_1 = image_set[image_set['Image_no'].isin([2,3])][['label_remaped']]

X_test_1 = image_set[image_set['Image_no'].isin([1])][train_columns]
y_test_1 = image_set[image_set['Image_no'].isin([1])][['label_remaped']]


features = ['ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']


# In[67]:


dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train_1[features], y_train_1)

stump_pred = dt_stump.predict(X_test_1[features])
print(np.mean(stump_pred == y_test_1.values.reshape(-1)))
print(balanced_accuracy_score(stump_pred, y_test_1.values.reshape(-1)))


# In[68]:


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
plt.savefig('statics/ada_iterations.png')


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
train_im = image_set[image_set['Image_no']!=1]
train_im_features = train_im[features]
prob_preds = adaboost_analysis.predict_proba(train_im_features)[:,1]
train_im['Predicted_Probs'] = prob_preds


# Test Preds
test_im = image_set[image_set['Image_no']==1]
test_im_features = image_set[image_set['Image_no']==1][features]
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
plt.savefig('statics/ada_test_prob.png')


# In[78]:


image = train_im[train_im['Image_no']==2].copy()

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
plt.savefig('statics/ada_train_prob1.png')


# In[79]:


image = train_im[train_im['Image_no']==3].copy()

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
plt.savefig('statics/ada_train_prob2.png')


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


# In[84]:


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

from matplotlib import cm, colors

fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)
plt.savefig('statics/log_reg_decision_surface.png')

"""
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im_plot, cax=cax)

plt.show()
"""


# In[98]:


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

plt.savefig('statics/PCA_decision_surface.png')

"""
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im_plot, cax=cax)

plt.show()
"""


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
plt.show()
plt.savefig('statics/Feature_importance_test_scheme1.png')

