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

def cv_classifer(clf: Callable, image_data: SatelliteImageData, metric: Callable, n_jobs=8, **kws) -> List[float]:
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
            X_train, y_train = train[columns], train.label
            X_val, y_val = val[columns], val.label

            delays.append(delayed(train_validate_classifer)(clf, X_train, y_train, X_val, y_val, metric))

        # process batch
        results = Parallel(n_jobs=n_jobs)(delays)

        # store scores
        scores.extend(results)
    
    return scores
