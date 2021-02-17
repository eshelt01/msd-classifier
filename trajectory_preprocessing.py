# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:41:46 2021

@author: Erin S
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split

traj = pd.read_csv('./data/particle_trajectories.csv',header=None, names = ['xdata','ydata','orientation','time','particle_num'])

# Reindex particle numbers
particle_count = 1

for row in traj.itertuples():
    
    traj['particle_num'].loc[row.Index] = particle_count
    
    if row.Index < 621500 and traj['time'].loc[row.Index+1] < row[4]:  
        particle_count+=1
        print(row[4])
        
traj.to_pickle('particle_trajectories_reindexed.pkl')

def calculate_msd(xdata,ydata):
    """ Calculate the mean-square displacement curve for a single (x,y) trajectory """ 
    
    # Number of time intervals - use 1 order of magnitude less than max interval.
    num_intervals = round(len(xdata-1))
    if num_intervals > 800:
        num_intervals = 800  
    particle_msd  = []
    particle_msd_sig = []

    for tau in range(1,num_intervals):
            
            dx = xdata[:-tau] - xdata[(tau):] 
            dy = ydata[:-tau] - ydata[(tau):]
            drSquared = np.square(dx) + np.square(dy)
            msd = np.mean(drSquared)
            particle_msd.append((tau,msd))    
            particle_msd_sig.append(np.std(drSquared)/np.sqrt(len(drSquared)))  # Uncertainty 
    
    return particle_msd, particle_msd_sig


def anomalous_diffusion(t, D_alpha, alpha):
    """ Anomalous Diffusion Equation """
    
    # 2*(dim)*D t^alpha
    return 2 * 2 * D_alpha * t**alpha


def fit_diffusion_features(particle_msd):
    """ Fit diffusion equation, calculate features """
    
    xdata = [x[0] for x in particle_msd]
    ydata = [x[1] for x in particle_msd]

    popt, pcov = curve_fit(anomalous_diffusion, xdata, ydata)
    D_alpha = popt[0]
    alpha = popt[1]
    max_dist = np.max([ydata[1:]-ydata[0]])
    max_disp = np.abs(ydata[-1]-ydata[0])
    d_ratio = max_disp/max_dist
    
    return D_alpha, alpha, max_dist, max_disp, d_ratio


# Create dataframe with features and plot
particle_nums = traj['particle_num'].unique()
traj_msd = pd.DataFrame(columns = ['msd','D','alpha','max_dist','max_disp','d_ratio'])
f, axes = plt.subplots()

for particle in particle_nums:
    
    xdata = np.array(traj['xdata'][traj['particle_num']==particle])
    ydata = np.array(traj['ydata'][traj['particle_num']==particle])
    particle_msd, particle_msd_sig = calculate_msd(xdata,ydata)
    D, alpha, max_dist, max_disp, d_ratio = fit_diffusion_features(particle_msd)
    
    # Remove stationary particles
    if np.mean(particle_msd)>100: # 100 pixels over 1000 s
        traj_msd.loc[particle] = [particle_msd, D, alpha, max_dist, max_disp, d_ratio]
        plt.plot(*zip(*particle_msd))

plt.show()

# Split into sets and save
train_df, predict_df = train_test_split(traj_msd, test_size=0.3, random_state=2)
predict_df.to_pickle('C:/Users/Ed/Documents/particle-tracking/data/predict.pkl')

