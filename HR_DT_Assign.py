# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:00:36 2024

@author: urmii
"""
'''4.	In the recruitment domain, HR faces the challenge of predicting if 
the candidate is faking their salary or not. For example, a candidate claims 
to have 5 years of experience and earns 70,000 per month working as a regional manager. 
The candidate expects more money than his previous CTC. We need a way to verify their claims 
(is 70,000 a month working as a regional manager with an experience of 5 years a genuine claim or 
 does he/she make less than that?) Build a Decision Tree and Random Forest model with monthly income 
as the target variable. 

Problem Understanding:
The problem entails using candidate attributes such as years of experience, 
job role (e.g., regional manager), and potentially other factors to predict 
the candidate's monthly income. By analyzing these attributes, HR professionals 
can determine whether a candidate's salary claim aligns with industry standards and 
their claimed level of experience and job role.

Maximize:
    1. Accuracy of salary prediction
    
Minimize:
    1. FP and FN
    2. decisions or negotiations
'''
import pandas as pd
import numpy as np

df=pd.read_csv("c:/10-ML/decisionTree/HR_DT.csv")
df
df.columns
df.shape
df.size
df.head()

#Checking for nan values
df.isnull().sum()

#Cheking dtype of each attribute
df.dtypes

# Separate features (X) and target variable (y)
X = df.drop(' monthly income of employee', axis=1)  # Features
y = df[' monthly income of employee']  # Target variable
X
y

#Check unique attributes
df['no of Years of Experience of employee'].unique()

#Importing train & test split
from sklearn.model_selection import train_test_split
     

#Spliting to train,test
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
     

#Checking shape of train,test
x_train.shape,x_test.shape,y_train.shape,y_test.shape
     
#Importing DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
     
#tree_1 as DecisionTreeClassifier
tree_1 = DecisionTreeClassifier(criterion='gini',max_depth=None,max_features=17)
     
#Fitting
tree_1.fit(x_train,y_train)
     
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=17, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
#Checking max_depth
tree_1.tree_.max_depth
     
#Checking important features
imp_feature = tree_1.tree_.compute_feature_importances()
     
#Plotting important features
plt.figure(figsize=(20,10))
pd.Series(imp_feature,index=xd.columns).sort_values().plot(kind='barh')
plt.show()
     
#Checking score of train and test
tree_1.score(x_train,y_train),tree_1.score(x_test,y_test)