# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:24:50 2021

@author: Erin S
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance,TimeSeriesResampler

# Cluster training data to create labels
X_train = pd.read_pickle('C:/Users/Ed/Documents/particle-tracking/data/train.pkl').reset_index(drop = True)

# Dynamic Time Warping K -Means Clustering

# # Extract MSD curves
# msd_series_dba = []

# msd_series_dba = X_train['msd'].to_list()
# formatted_time_series_dba = to_time_series_dataset(msd_series_dba)
# X_train_dba = TimeSeriesScalerMeanVariance().fit_transform(formatted_time_series_dba)

# seed = 0
# np.random.seed(seed)
# sz = X_train_dba.shape[1]

# # DBA-k-means
# print("DBA k-means")
# dba_km = TimeSeriesKMeans(n_clusters=2,
#                           n_init=1,
#                           metric="dtw",
#                           verbose=True,
#                           max_iter_barycenter=10,
#                           random_state=seed)
# y_pred_dba = dba_km.fit_predict(X_train_dba)

# for yi in range(2):
#     plt.subplot(3, 3, 4 + yi)
#     for xx in X_train_dba[y_pred_dba == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("DBA $k$-means")

# dba_km.to_pickle('C:/Users/Ed/Documents/particle-tracking/models/DBA_KM_model.pkl')

# Euclidean K-Means Clustering

msd_series_euc = []
indices =[]
for row in X_train.itertuples():  # Euclidean k-means requires time series all have equal length
    msd = [x[1] for x in row[1]]
    if len(msd)==799:
        msd_series_euc.append(msd)
        indices.append(row.Index)
        
formatted_time_series_euc = to_time_series_dataset(msd_series_euc)
X_train_euc = TimeSeriesScalerMeanVariance().fit_transform(formatted_time_series_euc)

seed = 0
np.random.seed(seed)
sz = X_train_euc.shape[1]

print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=2, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train_euc)

plt.figure()
for yi in range(2):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train_euc[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

#km.to_pickle('C:/Users/Ed/Documents/particle-tracking/models/KM_model.pkl')


# Label data and save
X_train_labelled = X_train.loc[indices]
X_train_labelled['class'] = y_pred
X_train_labelled.to_pickle('C:/Users/Ed/Documents/particle-tracking/data/train_labelled.pkl')

# Plot those classified as directed motion:
for row in X_train_labelled.itertuples():
    if row[7]==1:
        xdata = [x[0] for x in row[1]]
        ydata = [y[1] for y in row[1]]
        plt.plot(xdata,ydata)
plt.title('Directed Motion')    
plt.show()

# Plot those classified as confined motion:
for row in X_train_labelled.itertuples():
    if row[7]==0:
        xdata = [x[0] for x in row[1]]
        ydata = [y[1] for y in row[1]]
        plt.plot(xdata,ydata)
plt.title('Confined Motion')     
plt.show()