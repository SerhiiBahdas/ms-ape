# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 18:50:42 2021

@author: Admin

Script to draw predicitons of all five models vs target value. Predictions built for file reach_13_23_2.csv
"""

# import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting svg font to none in matplotlib to avoid text transformation to curves
plt.rcParams['svg.fonttype'] = 'none'

# load predictions for all models
df_rmse = pd.read_json(r'./Data rmse per file ver1/fig7_df_rmse.json') 

outliers = dict(marker='o')
sns.boxplot(data=df_rmse, orient='v', flierprops=outliers)
plt.ylim((0, 0.9))

plt.savefig('Fig7 rmse per file ver1.svg', format='svg')
plt.savefig('Fig7 rmse per file ver1.pdf', format='pdf')
plt.savefig('Fig7 rmse per file ver1.png', format='png')