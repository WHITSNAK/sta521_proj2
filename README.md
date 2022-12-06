# STA 521 Project 2
Instructions for reproducing

## Step 0: Clone this repository
```bash
git clone git@github.com:WHITSNAK/sta521_proj2.git
```

## Step 1: Create a Conda Environment 
You will need to install anaconda for this: https://docs.anaconda.com/anaconda/install/index.html
```bash
conda env create -f sta_521_env.yml
```

## Step 2: Generate Plots Found From Section I to III
```python
```

## Step 3: Generate Plots Found From Section IV to VI
```python
python run_model_analysis.py --train_set 2,3 --test_set 1 --fp 'tr_23_te_1'
python run_model_analysis.py --train_set 1,2 --test_set 3 --fp 'tr_12_te_3'
```