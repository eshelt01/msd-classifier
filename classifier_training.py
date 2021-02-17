# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:12:19 2021

@author: Erin S
"""

# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import joblib

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier, plot_importance
sns.set_theme()

# Data
data_df = pd.read_pickle('C:/Users/Ed/Documents/particle-tracking/data/train_labelled.pkl').reset_index(drop = True)
train_df, test_df = train_test_split(data_df, test_size=0.3, random_state=1)

features = ['D','alpha','max_dist','max_disp','d_ratio']

def split_data(data_df, features):
    
    X = data_df[features]
    y = data_df['class']
    
    return X, y

# Save separate scaling pipeline for prediction

scaler_pipeline = Pipeline([('minmax_scaler', MinMaxScaler())])
scaler_pipeline.fit(data_df[features])
joblib.dump(scaler_pipeline, './models/pipeline.pkl')

# Classifier Grid Search
X_train, y_train = split_data(data_df, features)

pipeline = Pipeline([('minmax_scaler', MinMaxScaler()),('estimator', LogisticRegression())])

param_grid= [{'estimator': [LogisticRegression(solver='liblinear')],
                 'estimator__C': [0.01, 0.1, 1.0, 10.0],
                 'estimator__penalty': ['l1','l2']},
                {'estimator' : [svm.SVC()],
                 'estimator__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                 'estimator__kernel' : ['linear', 'poly', 'rbf']},
                {'estimator' : [RandomForestClassifier(random_state=0)],
                 'estimator__max_depth' : [1,2,3,4,5],
                 'estimator__criterion' :['gini', 'entropy']},
                {'estimator': [XGBClassifier(random_state=0, use_label_encoder =False, verbosity=0)],
                'estimator__max_depth' : [2, 4, 6, 8, 10],
                'estimator__n_estimators': [10, 100, 500],
                'estimator__min_child_weight' : [ 1, 3, 5, 7],
                'estimator__gamma' : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]}]

clf_grid = GridSearchCV(pipeline, param_grid=param_grid, cv= 5, scoring = 'roc_auc')
clf_grid.fit(X_train, y_train)
best_clf = clf_grid.best_estimator_ # Best classifier

# Output results
print('\n Best estimator: ', best_clf)
print('Max mean training score: \n', np.max(clf_grid.cv_results_['mean_test_score']), sep = '\n')
print('\n Best estimator params: ')
pprint.pprint(best_clf.get_params())


#  Test Output
X_test, y_test = split_data(test_df, features)

# Scale train and test data
X_test_scaled = pd.DataFrame(scaler_pipeline.transform(X_test), columns = features)
X_train_scaled = pd.DataFrame(scaler_pipeline.transform(X_train), columns = features)

# Re-fit Classifier
xgb_model = XGBClassifier(max_depth = 2, n_estimators = 10, min_child_weight = 1, gamma = 0.0, random_state=0, use_label_encoder=False, verbosity=0,)
xgb_model.fit(X_train_scaled,y_train)
y_test_pred = xgb_model.predict(X_test_scaled) # evaluate on test set

# Overall Accuracy and Precision
print('Accuracy: ',accuracy_score(y_test, y_test_pred))
print('Precision:',precision_score(y_test, y_test_pred))

# Confusion matrix 
confusion_matrix = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()

# Classification report
print('\n', classification_report(y_true=y_test, y_pred=y_test_pred))

# Feature Importance
plot_importance(xgb_model)
plt.show()

# Save Final Model
xgb_model.save_model('C:/Users/Ed/Documents/particle-tracking/models/xgb_model.json')
