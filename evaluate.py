# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 19:01:20 2021

@author: baron015
"""


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# importing useful functions
import utils as fun


def nn_accuracy_estimation(model, dataset, threshold=0.5):
    """
    Function to generate the change raster from the model
    args: the ground truth rasters as a dictionary (per year), with dem, rad and
          labels ; the model and its arguments (parameters of the model)
    outputs the codes, the binary change maps (gt) and the classes
          
    """
    
    # loading lists to store the results
    y = []
    y_hat = []
    
    model.eval()
    
    ## generating predictions
    for sample in dataset:
        
        pred = model(fun.torch_raster(sample[0][None,:,:,:]))
    
        y_hat.append(fun.numpy_raster(pred))
        
        y.append(sample[1])
        
    # putting into a single matrix
    y_hat = np.stack(y_hat, axis=0).flatten()
    y = np.stack(y, axis=0).flatten()
    # applying threshold on prediction
    y_hat = np.where(y_hat > threshold, 1, 0)
    
    # printing  results
    conf_mat = confusion_matrix(y, y_hat)
    class_report = classification_report(y, y_hat)
    
    
    return conf_mat, class_report


def svm_accuracy_estimation(train, test, cv=False):
    """
    performance on a svm
    This function builds the datasets up
    """
    
    ## creating train and test
    nb_bands, dims = train[0][0].shape[0], train[0][0].shape[1]
    x_train = np.concatenate([samp[0].reshape((nb_bands, dims**2)).T for samp in train])
    x_test = np.concatenate([samp[0].reshape((nb_bands, dims**2)).T for samp in test])
    y_test = np.concatenate([samp[1] for samp in test])
    y_train =  np.concatenate([samp[1] for samp in train])
    

    # loading the model
    svclassifier = svm.SVC(class_weight="balanced")
    
    # training the model
    svclassifier.fit(x_train, y_train)
    
    # predicting the labels
    pred_label = svclassifier.predict(x_test)
    
    # printing  results
    conf_mat = confusion_matrix(y_test, pred_label)
    class_report = classification_report(y_test, pred_label)
    
    # performing a cross validation (optional)
    if cv:
        # prepare the cross-validation procedure
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        
        # performing a k fold validation
        scores_cv = cross_val_score(svclassifier, x_test, y_test,
                                cv=cv, scoring='f1_macro')
    else:
        scores_cv=None
    
    return conf_mat, class_report, scores_cv


def rf_accuracy_estimation(data: np.array, labels: np.array, perform_cross_validation=False):

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(x_train, y_train)
    prediction_label = classifier.predict(x_test)

    accuracy_metrics_report(y_test, prediction_label)

    if perform_cross_validation:
        cross_validation_report(classifier, x_test, y_test)


def cross_validation_report(model, x_test: np.array, y_test: np.array):

    cross_validation_parameters = KFold(n_splits=10, random_state=1, shuffle=True)
    scores_cv = cross_val_score(model, x_test, y_test,
                                cv=cross_validation_parameters, scoring='f1_macro')

    print("Score cross-validation is:")
    print(scores_cv)


def accuracy_metrics_report(y_labels: np.array, y_prediction: np.array):

    conf_mat = confusion_matrix(y_labels, y_prediction)
    class_report = classification_report(y_labels, y_prediction)

    print("confusion matrix")
    print(conf_mat)
    print("\nclassification report")
    print(class_report)