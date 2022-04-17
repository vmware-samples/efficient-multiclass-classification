# Copyright 2019-2022 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3

import itertools

from duet_classifier import DuetClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

def dictListsToListDicts(dictOfLists): 
    listOfDicts = [dict(zip(dictOfLists, t)) for t in itertools.product(*dictOfLists.values())]
    return listOfDicts

if __name__ == "__main__":
    
    # load toy dataset
    X, y = load_iris(return_X_y=True)

    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  
    
    # duet instance
    clf = DuetClassifier()
    
    # coarse-grained rf parameter grid
    cg_rf_params = {
    
      'n_estimators': [20],
      'max_samples': [None, 0.25],
     
    }
    
    # fine-grained xgboost parameter grid
    fg_xgb_params = {
    
      'n_estimators': [1000],
      'max_depth': [3, 8],
      'learning_rate': [0.05, 0.01]
                 
    }
    
    # total parameter combinations to try
    parameters = {
    
      'duet_fg_train_dataset_fraction': [0.25, 0.5],
      'duet_fg_test_confidence': [0.9, 0.99],
         
      'cg_rf_params': dictListsToListDicts(cg_rf_params),
      'fg_xgb_params': dictListsToListDicts(fg_xgb_params)
      
     }
        
    # run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring='f1_macro', cv=5, refit='f1_macro')
    grid_obj = grid_obj.fit(X_train, y_train)

    # Print the score of each configuration
    print("\nGrid scores:\n")
    means = grid_obj.cv_results_['mean_test_score']
    stds = grid_obj.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_obj.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    # Print the best classifier parameters and its score
    print("\nThe best grid-search score is {}\nAchieved by the following parameters set:".format(grid_obj.best_score_))
    print(grid_obj.best_params_)

    # Print classification report for the test dataset
    print("\nDetailed classification report (for the test dataset):\n")
    best_clf = grid_obj.best_estimator_
    y_true, y_pred = y_test, best_clf.predict(X_test)
    print(classification_report(y_true, y_pred))
