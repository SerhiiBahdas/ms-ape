# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 18:50:42 2021

@author: Admin

Script to draw predicitons of all five models vs target value. Predictions built for file reach_13_23_2.csv
"""

# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# setting svg font to none in matplotlib to avoid text transformation to curves
plt.rcParams['svg.fonttype'] = 'none'

# load list of columns which need to be visualized
f = open(r'./Data Highest MSE ver1/bf_columns.json')
columns = json.load(f)

# load target values
target_arr = pd.read_json(r'./Data Highest MSE ver1/bf_target.json')
target_arr = target_arr.to_numpy()

# load predictions for all models
output_arr_rnn_1_115 = pd.read_json(r'./Data Highest MSE ver1/bf_output_arr_rnn_1_115.json')
output_arr_rnn_1_115 = output_arr_rnn_1_115.to_numpy()

output_arr_gru_1_69 = pd.read_json(r'./Data Highest MSE ver1/bf_output_arr_gru_1_69.json')
output_arr_gru_1_69 = output_arr_gru_1_69.to_numpy()

output_arr_lstm_3_69 = pd.read_json(r'./Data Highest MSE ver1/bf_output_arr_lstm_3_69.json')
output_arr_lstm_3_69 = output_arr_lstm_3_69.to_numpy()

output_arr_gru_1_115 = pd.read_json(r'./Data Highest MSE ver1/bf_output_arr_gru_1_115.json')
output_arr_gru_1_115 = output_arr_gru_1_115.to_numpy()

output_arr_lstm_1_115 = pd.read_json(r'./Data Highest MSE ver1/bf_output_arr_lstm_1_115.json')
output_arr_lstm_1_115 = output_arr_lstm_1_115.to_numpy()

# create array of labels and array of x values to show it as seconds
xlabels = np.arange(0, 0.6, 0.1)
x = np.arange(0, 0.5, 0.001)

# manually collecting array of y labels for each feature separetly into a dictionary
ylabels = {}
ylabels['tor_ra_wr_s_p'] = np.arange(-0.25, 0.06, 0.1)
ylabels['tor_ra_wr_e_f'] = np.arange(-1, 0.9, 0.6)
ylabels['tor_ra_cmc1_ad_ab'] = np.arange(-0.04, 0.09, 0.03)
ylabels['tor_ra_cmc1_f_e'] = np.arange(-0.11, 0.06, 0.04)
ylabels['tor_ra_mcp1_f_e'] = np.arange(-0.01, 0.027, 0.009)
ylabels['tor_ra_ip1_f_e'] = np.arange(-0.0015, 0.0026, 0.001)
ylabels['tor_ra_mcp2_e_f'] = np.arange(-0.04, 0.05, 0.02)
ylabels['tor_ra_pip2_e_f'] = np.arange(-0.006, 0.011, 0.004)
ylabels['tor_ra_dip2_e_f'] = np.arange(-0.0015, 0.0026, 0.001)
ylabels['tor_ra_mcp3_e_f'] = np.arange(-0.04, 0.05, 0.02)
ylabels['tor_ra_pip3_e_f'] = np.arange(-0.006, 0.013, 0.006)
ylabels['tor_ra_dip3_e_f'] = np.arange(-0.0015, 0.0026, 0.001)
ylabels['tor_ra_mcp4_e_f'] = np.arange(-0.03, 0.04, 0.015)
ylabels['tor_ra_pip4_e_f'] = np.arange(-0.004, 0.009, 0.003)
ylabels['tor_ra_dip4_e_f'] = np.arange(-0.001, 0.003, 0.001)
ylabels['tor_ra_mcp5_e_f'] = np.arange(-0.02, 0.017, 0.009)
ylabels['tor_ra_pip5_e_f'] = np.arange(-0.003, 0.006, 0.002)
ylabels['tor_ra_dip5_e_f'] = np.arange(-0.0007, 0.0012, 0.0006)
ylabels['tor_ra_sh_ab_ad'] = np.arange(-10, 21, 10)
ylabels['tor_ra_sh_e_f'] = np.arange(4, 17, 3)
ylabels['tor_ra_sh_rot'] = np.arange(0, 33, 8)
ylabels['tor_ra_el_e_f'] = np.arange(0, 33, 8)
ylabels['tor_ra_wr_ad_ab'] = np.arange(-0.8, 0.21, 0.25)  

# manually collecting array of y limits for each feature separetly into a dictionary
ylims = {}
ylims['tor_ra_wr_s_p'] = (-0.25, 0.05)
ylims['tor_ra_wr_e_f'] = (-1, 0.8)
ylims['tor_ra_cmc1_ad_ab'] = (-0.04, 0.08)
ylims['tor_ra_cmc1_f_e'] = (-0.11, 0.05)
ylims['tor_ra_mcp1_f_e'] = (-0.01, 0.026)
ylims['tor_ra_ip1_f_e'] = (-0.0015, 0.0025)
ylims['tor_ra_mcp2_e_f'] = (-0.04, 0.04)
ylims['tor_ra_pip2_e_f'] = (-0.006, 0.01)
ylims['tor_ra_dip2_e_f'] = (-0.0015, 0.0025)
ylims['tor_ra_mcp3_e_f'] = (-0.04, 0.04)
ylims['tor_ra_pip3_e_f'] = (-0.006, 0.012)
ylims['tor_ra_dip3_e_f'] = (-0.0015, 0.0025)
ylims['tor_ra_mcp4_e_f'] = (-0.03, 0.03)
ylims['tor_ra_pip4_e_f'] = (-0.004, 0.008)
ylims['tor_ra_dip4_e_f'] = (-0.001, 0.002)
ylims['tor_ra_mcp5_e_f'] = (-0.02, 0.016)
ylims['tor_ra_pip5_e_f'] = (-0.003, 0.005)
ylims['tor_ra_dip5_e_f'] = (-0.0007, 0.0011)
ylims['tor_ra_sh_ab_ad'] = (-10, 20)
ylims['tor_ra_sh_e_f'] = (4, 16)
ylims['tor_ra_sh_rot'] = (0, 32)
ylims['tor_ra_el_e_f'] = (0, 32)
ylims['tor_ra_wr_ad_ab'] = (-0.8, 0.2) 

# ploting all 23 features into single plot with subplot feature
plt.figure(figsize=(25, 40))
n = 0
for idx, column in enumerate(columns):

  n += 1
  plt.subplot(8, 3, n)
  plt.plot(x, target_arr[:,idx], label='True') # target values
  plt.plot(x, output_arr_rnn_1_115[:,idx], label='RNN_1_115') # RNN_1_115 model prediction
  plt.plot(x, output_arr_gru_1_69[:,idx], label='GRU_1_69') # GRU_1_69 model prediction
  plt.plot(x, output_arr_lstm_3_69[:,idx], label='LSTM_3_69') # LSTM_3_69 model prediction
  plt.plot(x, output_arr_gru_1_115[:,idx], label='GRU_1_115') # GRU_1_115 model prediction
  plt.plot(x, output_arr_lstm_1_115[:,idx], label='LSTM_1_115') # LSTM_1_115 model prediction
  plt.title(column) # set title as feature name
  plt.xlabel('Timestep, s') # label axis
  plt.ylabel('Torque, N $\cdot$ m')
  plt.xticks(xlabels) # set x ticks values for all subplots
  plt.yticks(ylabels[column]) # set y ticks for each subplot individually from prepared dictionary
  plt.ylim(ylims[column]) # set y limits for each subplot individually from prepared dictionary
  plt.legend()
  plt.margins(x=0.01, y=0.01) # remove margins for plots

plt.savefig('Fig2 Highest MSE ver1.svg', format='svg')
plt.savefig('Fig2 Highest MSE ver1.pdf', format='pdf')
plt.savefig('Fig2 Highest MSE ver1.png', format='png')