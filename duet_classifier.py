# Copyright 2019-2022 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3

import time

import numpy as np
import xgboost as xgb

import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

###############################################################################
###############################################################################
class DuetClassifier(BaseEstimator, ClassifierMixin):
    """
    A Duet classifier.

    The two main building blocks of Duet are two classifiers. 

    The first classifier is a small Coarse-Grained (cg) model - Random Forest
    (RF) that is trained using the entire training dataset and which we use 
    to compute the labeled data predictability*. 

    The second classifier is a Fine-Grained (fg) model - XGBoost that is
    trained using only a predictability-driven fraction of the training 
    dataset. During classification, all data instances are classified by 
    the RF, and only the hard data instances (i.e., cases for which the RF 
    is not sufficiently confident) are forwarded to the XGBoost for
    reclassification.

    *Predictability of an instance is defined to be the distance (l2 norm) 
    among the distribution vector (i.e., predict_proba) by the RF and the 
    'perfect' one (i.e., 1 in the correct class and 0 at all others).
    
    For more information read: Efficient_Multi-Class_Classification_with_Duet.pdf
    
    Parameters
    ----------

    duet_cg_train_using_feature_subset : list or None, optional (default=None)
        List of columns to use (when None, all columns are used).

    duet_fg_train_sample_weight_balance : boolean, optional (default=False)
        Use weights when training the fg (XGBoost) model. The original class weights
        of the dataset are preserved in the predictability-driven fraction
        dataset that is used for training the fg (XGBoost) model (i.e., the total weight
        of each class sums up to the original total class weight of the dataset)

    duet_fg_extend_data_with_cg_distribution : boolean, optional (default=False)
        When True, the dataset for the training/classification of the fg (XGBoost) model
        is extended with the class distribution vector by the cg (RF) classifier.
        That is, the class distribution vector is added as additional features
        of the dataset used for the training of the fg (XGBoost) model.
 
    duet_fg_train_data_filter_type : string, optional (default='l2')
        The distance metric that is used to compute the predictability of the data instances.

    duet_fg_train_dataset_fraction : float, optional (default=0.25)
        A value in (0,1]. Indicated the data fraction that is used for the
        training of the fg (XGBoost) model. If duet_subsample_only is set to True,
        indicates the data fraction for dataset sub-sampling.

    duet_fg_test_confidence : float, optional (default=0.95)
        A value in [0,1]. Indicates the data confidence (i.e., 
        top-1 probability in the distribution vector) above which the instance
        is not passed to the fg (XGBoost) classifier for classification (i.e., classified
        only by the cg (RF) classifier). Used only with duet_fg_test=False.
        
    duet_verbose : boolean, optional (default=False)
        Verbose printing for debug. Prints warnings and the fraction of the data 
        that is used for the training and classification by the fg (XGBoost) classifier.

    duet_random_np_seed : int, optional (default=42)
        Random seed for the numpy package used in Duet.

    cg_rf_params : dict or None, optional (default={'max_leaf_nodes': 1000})
        Parameters for the cf (RF) classifier.
        The default max_leaf_nodes parameter is used to avoid any over-fitting
        by the cg (RF) model.

    fg_xgb_params : dict or None, optional (default=None)
        Parameters for the fg XGBoost classifier.
                       
    duet_subsample_only : boolean, optional (default=False)
        When true, use duet for sub-sampling of the dataset. Fit returns the sub-sampled dataset
        (X',y'), according to the duet_fg_train_dataset_fraction value.
               
    duet_fg_test : boolean, optional (default=False)
        If true, all data is classified only by the fg (XGBoost) classifier.
    
    Attributes
    ----------

    classes_ : array of shape (n_classes,) classes labels. 

    cg_clf_ : RandomForestClassifier
        The cg (Random Forest) classifier.
        
    fg_clf_ : xgboost
        The fg (XGBoost) classifier.
    
    fg_clf_fitted_ : Boolean
        Remembers if fit was called for the fg model within Duet.
    
    fit_time_, predict_time_ : float
        Temporary. Used for debug measurements

        
    Example program
    ---------------

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)   
    
    cg_rf_params = {
            
                  'n_estimators': 20,
                  'max_leaf_nodes': 100,
                    
            }
    
    fg_xgb_params = {
            
                  'n_estimators': 1000,
                  'max_depth': 8,
                  'learning_rate': 0.01,

            }
                    
    parameters = {
                  
                  'duet_fg_train_dataset_fraction': 0.1,
                  'duet_fg_test_confidence': 0.99,
                                   
                  'cg_rf_params': cg_rf_params,
                  'fg_xgb_params': fg_xgb_params          

                 }
    
    duet = DuetClassifier()
    duet.set_params(**parameters)       
    duet.fit(X_train, y_train)   

    y_predicted = duet.predict(X_test)
    print(classification_report(y_test, y_predicted, digits=5))


    Notes
    -----
    
    The parameters controlling the size of the dataset for the fg training
    (duet_fg_train_dataset_fraction) and the cg confidence level
    (duet_fg_test_confidence) are advised to be specifically tuned for each dataset
    (e.g., by grid-search).
    
    The parameters for the RF and XGBoost classifiers should also be tuned.
    Using the parameters that work well for the monolithic models is a good
    start.         
    """

    ###########################################################################
    ###########################################################################

    def __init__(self,

                 ### duet parameters
                 duet_cg_train_using_feature_subset=None,
                 duet_fg_train_sample_weight_balance=False,
                 duet_fg_extend_data_with_cg_distribution=False,
                 duet_fg_train_data_filter_type='l2',
                 duet_fg_train_dataset_fraction=0.25,
                 duet_fg_test_confidence=0.95,
                 duet_verbose=False,
                 duet_random_np_seed=42,
                 
                  ### arguments for internal classifiers
                  cg_rf_params={'max_leaf_nodes': 1000},
                  fg_xgb_params=None,
                  
                  ### use duet for dataset subsampling?
                  duet_subsample_only=False,
                  
                  ### test using only fg model?
                  duet_fg_test=False
                  
                 ):
        
        ### duet parameters
        self.duet_cg_train_using_feature_subset = duet_cg_train_using_feature_subset
        self.duet_fg_train_sample_weight_balance = duet_fg_train_sample_weight_balance
        self.duet_fg_extend_data_with_cg_distribution = duet_fg_extend_data_with_cg_distribution
        self.duet_fg_train_data_filter_type = duet_fg_train_data_filter_type
        self.duet_fg_train_dataset_fraction = duet_fg_train_dataset_fraction
        self.duet_fg_test_confidence = duet_fg_test_confidence
        self.duet_verbose = duet_verbose
        self.duet_random_np_seed = duet_random_np_seed
        
        ### kwards for internal classifiers
        self.cg_rf_params = cg_rf_params
        self.fg_xgb_params = fg_xgb_params
        
        ### use duet as filter only
        self.duet_subsample_only = duet_subsample_only
        
        ### test using only fg model?
        self.duet_fg_test = duet_fg_test
        
    ###########################################################################
    ###########################################################################

    def verify_duet_parameters(self, X, y):

        ### categorial parameters are verified in code

        if self.duet_fg_train_sample_weight_balance not in [True, False]:
            raise Exception("Illegal duet_fg_train_sample_weight_balance value. Should be in [True, False]")

        if self.duet_fg_extend_data_with_cg_distribution not in [True, False]:
            raise Exception("Illegal duet_fg_extend_data_with_cg_distribution value. Should be in [True, False]")

        if self.duet_fg_train_dataset_fraction < 0 or self.duet_fg_train_dataset_fraction > 1:
            raise Exception("Illegal duet_fg_train_dataset_fraction value. Should be in [0, 1]")

        if self.duet_fg_test_confidence < 0 or self.duet_fg_test_confidence > 1:
            raise Exception("Illegal duet_fg_test_confidence value. Should be in [0, 1]")

        if self.duet_verbose not in [True, False]:
            raise Exception("Illegal duet_verbose value. Should be in [True, False]")
        
        if self.duet_cg_train_using_feature_subset is not None:

            ### empty is not allowed
            if not len(self.duet_cg_train_using_feature_subset):
                raise Exception("Illegal duet_cg_train_using_feature_subset (err1): {}\nShould be None or specify unique columns".format(self.duet_cg_train_using_feature_subset))
                
            ### duplicates are not allowed
            if len(self.duet_cg_train_using_feature_subset) != len(set(self.duet_cg_train_using_feature_subset)):
                raise Exception("Illegal duet_cg_train_using_feature_subset (err2): {}\nShould be None or specify unique columns".format(self.duet_cg_train_using_feature_subset))
                
            ### translate column names (if X is a dataframe) to indices
            if isinstance(X, pd.DataFrame): 
                if all(elem in X.columns for elem in self.duet_cg_train_using_feature_subset):
                    self.duet_cg_train_using_feature_subset = [X.columns.get_loc(i) for i in self.duet_cg_train_using_feature_subset]
            
            ### verify legal column values                
            if not set(self.duet_cg_train_using_feature_subset).issubset(set(range(X.shape[1]))):
                raise Exception("Illegal duet_cg_train_using_feature_subset (err3): {}\nShould be None or specify unique columns".format(self.duet_cg_train_using_feature_subset))
            
    ###########################################################################
    ###########################################################################

    def fit(self, X, y):
        """
        Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples.

        y : array-like, shape=(n_samples,)
            The input sample labels.
            
        Returns
        -------
        self : object
        """
        
        ### set numpy seed
        np.random.seed(self.duet_random_np_seed)

        ### duet measurments
        self.fit_time_ = {}
        self.predict_time_ = {}
        
        if self.duet_subsample_only == True:
            Xc, yc = X, y
            
        ### input verification - required by scikit         
        X, y = check_X_y(X, y)

        ### duet parameters input checks
        self.verify_duet_parameters(X, y)

        ### store the classes seen during fit - required by scikit
        self.classes_ = unique_labels(y)

        ### init coarse-grained (cg) - random forest
        self.cg_clf_ = RandomForestClassifier()
        
        if self.cg_rf_params is None: 
            print("\nWarning: no kwards for the coarse-grained model.\n")
        else:
            self.cg_clf_.set_params(**self.cg_rf_params)
                   
        start = time.time()
        
        ### train cg - then, classify training data by cg and obtain classification distribution
        if self.duet_cg_train_using_feature_subset == None:
            self.cg_clf_.fit(X, y)
            cg_train_dataset_classifications_distribution = self.cg_clf_.predict_proba(X)
        else:
            ### train cg by features subset specified by self.duet_cg_train_using_feature_subset
            self.cg_clf_.fit(X[:, self.duet_cg_train_using_feature_subset], y)
            cg_train_dataset_classifications_distribution = self.cg_clf_.predict_proba(X[:, self.duet_cg_train_using_feature_subset])
            
        end = time.time()
        self.fit_time_['cg'] = end-start

        ### filter data
        train_filters = {
                'l2': self.l2_filter
        }

        if self.duet_fg_train_data_filter_type not in train_filters:
            raise Exception("\nUnknown filter type: {}\n".format(self.duet_fg_train_data_filter_type))
        
        filtered_data = train_filters[self.duet_fg_train_data_filter_type](cg_train_dataset_classifications_distribution, y)
        
        if self.duet_subsample_only == True:
            
            return Xc[filtered_data], yc[filtered_data]

        ### train the fine-grained (fg)?
        if np.sum(filtered_data) > 0:

            ### useful stat
            if self.duet_verbose:
                print("\nTraining a fine-grained classifier (XGBoost) with {}[%] of the data\n".format(100*np.sum(filtered_data)/len(filtered_data)))

            ### extend X_train with cg confidence for the fg training?
            if self.duet_fg_extend_data_with_cg_distribution:
                X = np.concatenate((X, cg_train_dataset_classifications_distribution), axis=1)

            ### init fg - xgboost
            self.fg_clf_ = xgb.XGBClassifier()
            
            if self.fg_xgb_params is None:
                ### useful stat
                if self.duet_verbose:
                    print("\nWarning: no kwards for the fine-grained model.\n")
            else:
                self.fg_clf_.set_params(**self.fg_xgb_params)                
            
            start = time.time() 
            
            ### train fg + balance
            if self.duet_fg_train_sample_weight_balance:
               
                filtered_instances_per_class = np.bincount(y[filtered_data]).astype('float')
                filtered_class_weights = np.max(filtered_instances_per_class) * np.reciprocal(filtered_instances_per_class, where=(filtered_instances_per_class>0))                        
                filtered_sample_weights = np.take(filtered_class_weights, y[filtered_data])

                self.fg_clf_.fit(X[filtered_data], y[filtered_data], sample_weight=filtered_sample_weights)

            ### train fg + no balance
            else:

                self.fg_clf_.fit(X[filtered_data], y[filtered_data])

            ### set the fg_clf as fitted
            self.fg_clf_fitted_ = True
            
            end = time.time()
            self.fit_time_['fg'] = end-start

        else:

            ### useful stat
            if self.duet_verbose:
                print("\nWarning: no training data for the fine-grained model.\n")

            ### set fg_clf as non-fitted
            self.fg_clf_fitted_ = False
        
        ### a call to fit should return the classifier - required by scikit
        return self

    ###########################################################################
    ###########################################################################

    def predict_basic(self, X, proba=False, return_filter=False):
        """
        Predict labels for X rows.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples.

        proba : Boolean 
            If True, proba is returned.
        
        return_filter : if True, returns a boolean array of size (n_samples,)
                        indicating if the sample was classified by the cg
                        model (False) or the fg model (True).
            
        Returns
        -------
        y : nparray of class labels or class distributions 
            for X, shape=(n_samples,) or shape=(n_samples, n_classes).
        
        or (y, filter).
        
        """
        
        ### is used for subsampling?
        if self.duet_subsample_only == True:
            
            raise Exception("\n Cannot predict since: self.duet_subsample_only == True\n")
        
        ### set numpy seed
        np.random.seed(self.duet_random_np_seed)

        ### check is that fit had been called - required by scikit
        check_is_fitted(self)

        ### input verification - required by scikit
        X_test = check_array(X)

        ### no fg model
        if not self.fg_clf_fitted_:
            
            ### useful stat
            if self.duet_verbose:
                print("\nWarning: no fine-grained model. Predict only based on the coarse-grained model\n")

            if self.duet_cg_train_using_feature_subset == None:

                if proba:
                    pp = self.cg_clf_.predict_proba(X_test)
                    if return_filter:
                        return (pp, np.zeros(len(pp), dtype=bool))
                    else:
                        return pp
                else:
                    p = self.cg_clf_.predict(X_test)
                    if return_filter:
                        return (p, np.zeros(len(p), dtype=bool))
                    else:
                        return p

            else:

                if proba:
                    pp = self.cg_clf_.predict_proba(X_test[:, self.duet_cg_train_using_feature_subset])
                    if return_filter:
                        return (pp, np.zeros(len(pp), dtype=bool))
                    else:
                        return pp
                else:
                    p = self.cg_clf_.predict(X_test[:, self.duet_cg_train_using_feature_subset])
                    if return_filter:
                        return (p, np.zeros(len(p), dtype=bool))
                    else:
                        return p

        ### fg model exists
        else:

            start = time.time()
            
            if self.duet_cg_train_using_feature_subset == None:

                ### cg classifications distribution
                classifications_distribution = self.cg_clf_.predict_proba(X_test)

            else:

                ### cg classifications distribution
                classifications_distribution = self.cg_clf_.predict_proba(X_test[:, self.duet_cg_train_using_feature_subset])
            
            ### cg classification
            classifications = self.classes_.take(np.argmax(classifications_distribution, axis=1), axis=0)  
            
            '''
            inconsistency = self.cg_clf_.predict(X_test) != classifications
            if any(inconsistency):
                raise Exception("\nPredict error: predict_proba inconsistency\n")
            '''
            
            end = time.time()
            self.predict_time_['cg'] = end-start
            
            ### calculate classification confidence level
            classification_confidence = np.amax(classifications_distribution, axis=1)
            
            if self.duet_fg_test == True:
                
                ### all
                filtered_data = [True]*len(X_test)
            
            else:
                
                ### low confidence only
                filtered_data = classification_confidence <= self.duet_fg_test_confidence

            ### predict by the fg model?
            if np.sum(filtered_data) > 0:

                ### useful stat
                if self.duet_verbose:
                    print("\nPredict {}[%] of the data by the fine-grained model\n".format(100*np.sum(filtered_data)/len(filtered_data)))

                ### extend X_test with cg confidence for the fg prediction?
                if self.duet_fg_extend_data_with_cg_distribution:
                    X_test = np.concatenate((X_test, classifications_distribution), axis=1)
                
                '''
                inconsistency = self.classes_.take(np.argmax(classifications_distribution, axis=1), axis=0) != classifications
                if any(inconsistency):
                    raise Exception("\nPredict error: predict_proba inconsistency\n")
                '''
                
            else:

                print("\nWarning: no test data for prediction by the fine-grained model\n")

            if proba:
                
                ### fg classifications distribution?
                if np.sum(filtered_data) > 0:
                    classifications_distribution[filtered_data] = self.compile_predict_proba(classifications_distribution[filtered_data], self.fg_clf_.predict_proba(X_test[filtered_data]))
                
                if return_filter:
                    return (classifications_distribution, filtered_data)
                else:
                    return classifications_distribution
                
            else:
                
                ### fg classifications
                start = time.time()
                classifications[filtered_data] = self.fg_clf_.predict(X_test[filtered_data])
                end = time.time()
                self.predict_time_['fg'] = end-start
                
                if return_filter:
                    return (classifications, filtered_data)
                else:
                    return classifications

    ###########################################################################
    ########################################################################### 

    def predict(self, X, return_filter=False):

        return self.predict_basic(X, False, return_filter)

    def predict_proba(self, X, return_filter=False):

        return self.predict_basic(X, True, return_filter)

    def compile_predict_proba(self, cg_predict_proba, fg_predict_proba):

        return fg_predict_proba

    ###########################################################################
    ########################################################################### 
    
    def l2_filter(self, cg_train_dataset_classifications_distribution, cg_train_dataset_labels):
    
        predictability = []
                
        for distribution, label in zip(cg_train_dataset_classifications_distribution, cg_train_dataset_labels):
            
            vec = np.zeros(len(distribution))
            vec[np.where(self.classes_ == label)[0][0]] = 1
    
            predictability.append(1 - 0.5*np.linalg.norm(np.subtract(distribution, vec), 2))
        
        return self.fg_train_data_filter(np.asarray(predictability), cg_train_dataset_labels)

    ###########################################################################
    ###########################################################################
    
    def fg_train_data_filter(self, predictability, cg_train_dataset_labels):
    
        ### number of total instances
        num_total_instances = len(predictability)
                
        ### number of fg instances
        num_fg_instances = int(self.duet_fg_train_dataset_fraction*num_total_instances)

        ### per-class low-predictability instances upper limit        
        num_fg_class_instances = max(int((0.5*num_fg_instances)/len(self.classes_)), 1)
        
        ### init to false array
        indices = np.zeros(num_total_instances, dtype=bool)

        ### balanced sampling from each class
        for cls in self.classes_:
            
            ### class indices
            cls_indices = np.flatnonzero(cg_train_dataset_labels == cls)
            
            indices = np.logical_or(indices, self.fg_train_data_filter_h(predictability, cls_indices, num_fg_class_instances))
                    
        ### global balanced sampling
        indices_to_quota = np.flatnonzero(np.logical_not(indices))
        indices_to_quota_len = num_fg_instances - sum(indices)
        
        indices = np.logical_or(indices, self.fg_train_data_filter_h(predictability, indices_to_quota, indices_to_quota_len))
        
        '''
        import matplotlib.pyplot as plt
        n, bins, patches = plt.hist([predictability[indices], predictability], 10, stacked=False, log=True)
        plt.show()
        '''
        
        ### return the selected instances marked as 'True'
        return indices
        
    ###########################################################################
    ###########################################################################
     
    def fg_train_data_filter_h(self, predictability, relevant_indices, num_instances):

        ### init to false array
        indices = np.zeros(len(predictability), dtype=bool)
        
        ### calculate the bin number of each instance - instance_predictability_bins
        predictability_bins = np.linspace(min(predictability[relevant_indices]), max(predictability[relevant_indices]), 10)           
        instance_predictability_bins = np.digitize(predictability[relevant_indices], predictability_bins, right=True)
        
        ### calculate the size of each bin
        bin_count = np.bincount(instance_predictability_bins).astype('float')
        
        ### bin-search for instance count upper bound to take from each bin
        l = 0
        r = num_instances
        
        while l <= r: 
            
            k = l + (r - l)/2 
            
            current_sum = sum([min(k,i) for i in bin_count])
        
            if current_sum == num_instances: 
                l = r + 1
          
            elif current_sum < num_instances: 
                l = k + 1
          
            else: 
                r = k - 1
                   
        ### now k is the number of instances we take from each bin
        for b in np.unique(instance_predictability_bins):
            
            ### skip empty bins
            if bin_count[b] > 0:
                
                ### take entire bin
                if bin_count[b] <= k:
                                    
                    np.put(indices, relevant_indices[np.flatnonzero(instance_predictability_bins == b)], np.ones(int(bin_count[b]), dtype=bool))
                
                ### sample from bin
                else:
                                
                    sample_from = np.flatnonzero(instance_predictability_bins == b)
                    sampled = np.random.choice(sample_from, int(k), replace=False)
                                       
                    np.put(indices, relevant_indices[sampled], np.ones(int(k), dtype=bool))  
        
        return indices

    ###########################################################################
    ###########################################################################
