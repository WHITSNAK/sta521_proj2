import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

def wrap_index(i, ncol):
    return i//ncol, i%ncol

def get_heatmap_data(df, col):
    return df[['y', 'x', col]].set_index(['y', 'x']).unstack()

def plot_heatmap(data, label, cmap, ax, **kws):
    sns.heatmap(data, cmap=cmap, ax=ax, **kws)
    ax.tick_params(
        left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(label)

def find_change_point(arr):
    """
    arr: boolean array
    """
    for i in range(len(arr)-1):
        if arr[i] != arr[i+1]:
            return i

def plot_CV_ROC(cv_scores, cv_curve_color, mean_curve_color, mod_name, ax, chosen_fpr=0.5, nested_test=False):
    lw=2
    
    mean_fpr = np.linspace(0, 1, 100)
    # Add in the chosen FPR level
    mean_fpr = np.insert(mean_fpr, find_change_point(mean_fpr < chosen_fpr), chosen_fpr)
    
    fpr_tpr_col = []
    fpr_tpr_interp_col = []
    roc_col = []
    tprs_interp = []
    for n in range(len(cv_scores)):
        
        if nested_test:
            fold_prob = cv_scores[n][4][1]
            fold_true = cv_scores[n][5][1].values
        else:
            fold_prob = cv_scores[n][1][1]
            fold_true = cv_scores[n][2][1].values      

        fpr, tpr, _ = roc_curve(fold_true,fold_prob[:,1])

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0

        roc_auc = auc(fpr, tpr)

        fpr_tpr_col.append((fpr, tpr))
        fpr_tpr_interp_col.append((mean_fpr, interp_tpr))
        roc_col.append(roc_auc)
        
    # Plot CV Folds
    for n in range(len(cv_scores)):

        fpr_tpr = fpr_tpr_interp_col[n]

        ax.plot(
            fpr_tpr[0],
            fpr_tpr[1],
            color=cv_curve_color,
            lw=lw, alpha=0.5,
        )
    
    # Plot Means
    mean_tpr = np.nanmean([x[1] for x in fpr_tpr_interp_col],0)
    # Find associated mean tpr value
    tpr_index = np.where(mean_fpr == chosen_fpr)[0][0]
    tpr_val_chosen = mean_tpr[tpr_index]
    
    ax.plot(
        mean_fpr,
        mean_tpr,
        color=mean_curve_color,
        lw=lw,
        label="ROC curve %s (area = %0.3f) (TPR = %0.3f)" % (mod_name,np.nanmean(roc_col), tpr_val_chosen)
    )

    
    # Plot tpr val
    ax.plot([chosen_fpr,chosen_fpr],
            [0,tpr_val_chosen], alpha=0.5, color='black', ls='-')

    ax.plot([0,chosen_fpr],
            [tpr_val_chosen,tpr_val_chosen], alpha=0.5, color=mean_curve_color, ls='--')


    ax.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel(f"True Positive Rate")
