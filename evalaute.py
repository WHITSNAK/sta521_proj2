import numpy as np
from cv_master.cv import cv_classifer, nested_cv_classifer
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def eval_model_test_scheme1(model, dataset1, metric_list):
    """
    model: Callable
    dataset1: (image_data, (X_train, y_tain), (X_test, y_test))
    dataset2: (image_data)
    metric_list: List(metrics)
    """
    
    #### Test scheme 1 ####
    cv_scores = cv_classifer(model, dataset1[0], metric_list, n_jobs=8, verbose=True)

    acc = [x[0][0] for x in cv_scores]
    balanced_acc = [x[0][1] for x in cv_scores]  

    scheme1_acc = np.mean(acc)
    scheme1_bal_acc = np.mean(balanced_acc)

    # Retrain on Train set 
    model.fit(dataset1[1][0], dataset1[1][1])

    # Test on Test Set
    test_preds = model.predict(dataset1[2][0])
    test_scores = model.predict_proba(dataset1[2][0])

    test_acc = accuracy_score(test_preds, dataset1[2][1].values)
    test_bal_acc = balanced_accuracy_score(test_preds, dataset1[2][1].values)
    
    test_scheme_1_info = {'Test_acc': test_acc,
                          'Test_bal_acc': test_bal_acc,
                          'CV_acc': scheme1_acc,
                          'CV_bal_acc': scheme1_bal_acc,
                          'scheme_1_raw': cv_scores,
                          'val_folds_acc': acc,
                          'val_folds_acc': balanced_acc,
                          'test_preds':test_preds,
                          'test_scores':test_scores}
    
    
    return test_scheme_1_info
    
def eval_model_test_scheme2(model, dataset2, metric_list):
    """
    model: Callable
    dataset2: (image_data)
    metric_list: List(metrics)
    """
    
    #### Test scheme 2 ####
    cv_scores = nested_cv_classifer(model, dataset2, metric_list, n_jobs=8, verbose=True)

    # Val scores
    acc_val = [x[0][0] for x in cv_scores]
    balanced_acc_val = [x[0][1] for x in cv_scores]  

    scheme2_acc_val = np.mean(acc_val)
    scheme2_bal_acc_val = np.mean(balanced_acc_val)
    
    # Test Scores
    acc_test = [x[3][0] for x in cv_scores]
    balanced_acc_test = [x[3][1] for x in cv_scores]  

    scheme2_acc_test = np.mean(acc_test)
    scheme2_bal_acc_test = np.mean(balanced_acc_test)

    
    test_scheme_2_info = {'Test_acc': scheme2_acc_test,
                          'Test_bal_acc': scheme2_bal_acc_test,
                          'CV_acc': scheme2_acc_val,
                          'CV_bal_acc': scheme2_bal_acc_val,
                          'scheme_2_raw': cv_scores,
                          'val_folds_acc': acc_val,
                          'val_folds_acc': balanced_acc_val,
                          'test_folds_acc': acc_test,
                          'test_folds_acc': balanced_acc_test}
    
    
    return test_scheme_2_info
