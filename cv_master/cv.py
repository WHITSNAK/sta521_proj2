import pandas as pd

from typing import Callable, List
from joblib import Parallel, delayed
from cv_master.satellite_data import SatelliteImageData
from cv_master.utils import iter_by_chunk

def train_validate_classifer(
    clf: Callable, X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series, metric: Callable
) -> float:
    """The main evalaution function called by each worker"""
    clf.fit(X_train, y_train)
    score = metric(y_val, clf.predict(X_val))
    return score

def train_validate_classifer_v2(
    clf: Callable, X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series, metric_list: List[Callable]) -> float:
    """The main evalaution function called by each worker"""
    
    clf.fit(X_train, y_train)
    val_preds = clf.predict(X_val)
    
    val_probs = clf.predict_proba(X_val)
    
    score = []
    for metric in metric_list:
        score.append(metric(y_val, val_preds))
    return score, (val_preds,val_probs), (X_val, y_val)

def train_validate_test_classifer_v2(
    clf: Callable, X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    metric_list: List[Callable]) -> float:
    """The main evalaution function called by each worker"""
    
    clf.fit(X_train, y_train)
    
    # Val preds
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)
    
    # Test Preds
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    
    score_val = []
    score_test = []
    for metric in metric_list:
        score_val.append(metric(y_val, val_preds))
        score_test.append(metric(y_test, test_preds))
    return score_val, (val_preds, val_probs), (X_val, y_val), score_test, (test_preds,test_probs), (X_test, y_test)


def cv_classifer(clf: Callable, image_data: SatelliteImageData, metric: Callable, n_jobs=8,verbose=False, **kws) -> List[float]:
    """
    Cross Valide a Classfier using the Satallite Image Data
    """
    columns = ['ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']
    n_jobs = 8
    scores = []

    # get train validate set by chunk for parallelization
    for chunks in iter_by_chunk(image_data.iter_train_validate(), chunksize=n_jobs):
        # creating delayed objects
        delays = []
        for train_p, val_p in chunks:
            train = image_data.get_data_patches(train_p)
            val = image_data.get_data_patch(val_p)
            
            # Remove unlabelled data (coded as 2)
            train = train[train['label'] != 2]
            val = val[val['label'] != 2]
            
            X_train, y_train = train[columns], train.label
            X_val, y_val = val[columns], val.label
            
            if not verbose:
                delays.append(delayed(train_validate_classifer)(clf, X_train, y_train, X_val, y_val, metric))
            else:
                delays.append(delayed(train_validate_classifer_v2)(clf, X_train, y_train, X_val, y_val, metric))

        # process batch
        results = Parallel(n_jobs=n_jobs)(delays)

        # store scores
        scores.extend(results)
    
    return scores


def nested_cv_classifer(clf: Callable, image_data: SatelliteImageData, metric: Callable, n_jobs=8, **kws) -> List[float]:
    """
    Cross Valide a Classfier using the Satallite Image Data
    """
    columns = ['ndai', 'sd', 'corr', 'ra_df', 'ra_cf', 'ra_bf', 'ra_af', 'ra_an']
    n_jobs = 8
    scores = []

    # get train validate set by chunk for parallelization
    for chunks in iter_by_chunk(image_data.iter_train_validate_test(), chunksize=n_jobs):
        # creating delayed objects
        delays = []
        for train_p, val_p, test_p in chunks:
            
            train = image_data.get_data_patches(train_p)
            val = image_data.get_data_patch(val_p)
            test = image_data.get_data_patch(test_p)
            
            # Remove unlabelled data (coded as 2)
            train = train[train['label'] != 2]
            val = val[val['label'] != 2]
            test = test[test['label'] != 2]
            
            X_train, y_train = train[columns], train.label
            X_val, y_val = val[columns], val.label
            X_test, y_test = test[columns], test.label

            delays.append(delayed(train_validate_test_classifer_v2)(clf, X_train, y_train,
                                                                    X_val, y_val,
                                                                    X_test, y_test,
                                                                    metric))

        # process batch
        results = Parallel(n_jobs=n_jobs)(delays)

        # store scores
        scores.extend(results)
    
    return scores