# Copyright 2019-2022 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3

from duet_classifier import DuetClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":
    
    # load toy dataset
    X, y = load_iris(return_X_y=True)

    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # default configuration
    duet = DuetClassifier()
    
    # fit
    duet.fit(X_train, y_train)
    
    # predict
    y_predicted = duet.predict(X_test)
    
    # print classification report
    print(classification_report(y_test, y_predicted, digits=5))
