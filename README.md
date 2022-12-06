# STA 521 Project 2
Instructions for reproducing

## Step 0: Clone this repository
```bash
git clone git@github.com:WHITSNAK/sta521_proj2.git
```
## Step 1: Create a Conda Environment 
You will need to install anaconda for this: https://docs.anaconda.com/anaconda/install/index.html
Also, you will need the following packages:
- matplotlib
- seaborn
- pandas
- numpy
- scitkit-Learn
- joblib
- tqdm

## Step 2: Generate Plots Found From Section I to V
Please see the jupyter notebooks for each respective sections.
- Section II - Exploration
- Section III - Preperatio & CV
- Section IV  & V- Modeling & Diagnostic

## Alternative: Generate Plots in bulks Found From Section I to VI
You can simply run the entire file.
```python
python run_model_analysis.py --train_set 2,3 --test_set 1 --fp 'tr_23_te_1'
python run_model_analysis.py --train_set 1,2 --test_set 3 --fp 'tr_12_te_3'
```

## The CV_Master Package
Please see `naive_model_cv.ipynb` for a quick tutorial on using the CV_Master package to run the specialized CV functions we used in the report.