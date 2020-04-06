#!/usr/bin/env python
# coding: utf-8

# In[4]:


# timeit

# Student Name : Michael Abramson
# Cohort       : 3

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

#must install catboost with pip
#pip install catboost --no-cache-dir
import pandas as pd # data science essentials
from sklearn.model_selection import train_test_split # train/test split
import numpy as np #tools for log transformation
from catboost import CatBoostClassifier #best gradient regression package
from contextlib import contextmanager #first package for output suppression
import sys, os #second package for output suppression
from sklearn.tree import export_graphviz             # exports graphics
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
import pydotplus                                     # interprets dot objects
from sklearn.metrics import roc_auc_score, roc_curve #roc auc score

################################################################################
# Load Data
################################################################################

file = 'Apprentice_Chef_Dataset.xlsx' #locating dataset

original_df = pd.read_excel(file) #reading dataset into a dataframe

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

#setting outliers
original_df = original_df.loc[original_df['UNIQUE_MEALS_PURCH']<9.5]
original_df = original_df.loc[original_df['EARLY_DELIVERIES']<5] 
 
# preparing response variable data
chef_target = original_df.loc[:, 'CROSS_SELL_SUCCESS']

#preparing explanatory variable data
chef_data   = original_df.drop(['CROSS_SELL_SUCCESS',
                            'NAME',
                           'EMAIL',
                           'FIRST_NAME',
                           'FAMILY_NAME',
                            'MOBILE_NUMBER'],
                           axis=1)

################################################################################
# Train/Test Split
################################################################################

# preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            chef_data,
            chef_target,
            test_size = 0.25,
            random_state = 222,
            stratify=chef_target)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

#function for suppressing unnecessary iteration output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# INSTANTIATING a model object with chosen hyperparamaters
cat_model = CatBoostClassifier(learning_rate=.0000003,iterations=4000,depth =5,  l2_leaf_reg=10,thread_count=4,
                             border_count=50, random_strength=.8, grow_policy='Depthwise', min_data_in_leaf=1)

#d=6,7,8
with suppress_stdout():
    # FITTING the training data
    cat_fit = cat_model.fit(X_train, y_train,
                           use_best_model=True,
                          eval_set= (X_test, y_test))


    # PREDICTING on new data
    cat_pred = cat_model.predict(X_test)

print(f"""AUC:{(roc_auc_score(y_test,cat_pred).round(4))}""")

# saving scoring data for future use
cat_train_score = cat_model.score(X_train, y_train).round(4)
cat_test_score  = cat_model.score(X_test, y_test).round(4)

################################################################################
# Final Model Score (score)
################################################################################

#printing train and test score
print('Training Score:', cat_model.score(X_train, y_train).round(4))
print('Testing Score:',  cat_model.score(X_test, y_test).round(4))

# saving scoring data for future use
cat_train_score = cat_model.score(X_train, y_train).round(4)
cat_test_score  = cat_model.score(X_test, y_test).round(4)
cat_auc_score = roc_auc_score(y_test,cat_pred).round(4)
test_score = cat_auc_score

