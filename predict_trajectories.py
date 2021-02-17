# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:44:41 2021

@author: Erin S
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import xgboost as xgb

sns.set_theme()

# Final Model for prediction
bst = xgb.XGBClassifier()  # init model
bst.load_model('./models/xgb_model.json')

# Load Data for prediction
predict_df = pd.read_pickle('C:/Users/Ed/Documents/particle-tracking/data/predict.pkl')

# Transform data with appropriate pipeline
pipeline = joblib.load('./models/pipeline.pkl')
features  = ['D','alpha','max_dist','max_disp','d_ratio']

transformed_data = pipeline.transform(predict_df[features])
X_predict = pd.DataFrame(transformed_data, columns = features)
predict_df['predicted_class'] = bst.predict(X_predict)

def plot_msd_curve_class(df, predicted_class):
    
    plt.figure(figsize=(10,7))
    
    for row in df.itertuples():
        if row[7]==predicted_class:
            msd = [x[1] for x in row[1]]
            plt.plot(msd)
    
    if predicted_class==1:
        plt.title('Directed Motion')
        
    elif predicted_class == 0:
        plt.title('Confined Motion')
    
    plt.ylabel('Mean Square Displacement')
    plt.xlabel('Time interval (s)')   
    plt.show()
    return

plot_msd_curve_class(predict_df,1)

plot_msd_curve_class(predict_df,0)