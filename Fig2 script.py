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
f = open(r'./Data Lowest MSE ver1/gf_columns.json')
columns = json.load(f)

# load target values
target_arr = pd.read_json(r'./Data Lowest MSE ver1/gf_target.json')
target_arr = target_arr.to_numpy()

# load predictions for all models
output_arr_rnn_1_115 = pd.read_json(r'./Data Lowest MSE ver1/gf_output_arr_rnn_1_115.json')
output_arr_rnn_1_115 = output_arr_rnn_1_115.to_numpy()

output_arr_gru_1_69 = pd.read_json(r'./Data Lowest MSE ver1/gf_output_arr_gru_1_69.json')
output_arr_gru_1_69 = output_arr_gru_1_69.to_numpy()

output_arr_lstm_3_69 = pd.read_json(r'./Data Lowest MSE ver1/gf_output_arr_lstm_3_69.json')
output_arr_lstm_3_69 = output_arr_lstm_3_69.to_numpy()

output_arr_gru_1_115 = pd.read_json(r'./Data Lowest MSE ver1/gf_output_arr_gru_1_115.json')
output_arr_gru_1_115 = output_arr_gru_1_115.to_numpy()

output_arr_lstm_1_115 = pd.read_json(r'./Data Lowest MSE ver1/gf_output_arr_lstm_1_115.json')
output_arr_lstm_1_115 = output_arr_lstm_1_115.to_numpy()

# create array of labels and array of x values to show it as seconds
xlabels = np.arange(0, 2.5, 0.5)
x = np.arange(0, 2, 0.001)

# manually collecting array of y labels for each feature separatly into a dictionary
ylabels = {}
ylabels['tor_ra_wr_s_p'] = np.arange(-0.085, -0.075, 0.001)
ylabels['tor_ra_wr_e_f'] = np.arange(0.01, 0.02, 0.002)
ylabels['tor_ra_cmc1_ad_ab'] = np.arange(0.028, 0.029, 0.0002)
ylabels['tor_ra_cmc1_f_e'] = np.arange(-0.0028, -0.0012, 0.0005)
ylabels['tor_ra_mcp1_f_e'] = np.arange(0.0088, 0.0091, 0.0001)
ylabels['tor_ra_ip1_f_e'] = np.arange(0.00088, 0.00103, 0.00004)
ylabels['tor_ra_mcp2_e_f'] = np.arange(0.0028, 0.0045, 0.0004)
ylabels['tor_ra_pip2_e_f'] = np.arange(0.001, 0.0016, 0.0002)
ylabels['tor_ra_dip2_e_f'] = np.arange(0.00008, 0.00013, 0.00002)
ylabels['tor_ra_mcp3_e_f'] = np.arange(0.003, 0.0046, 0.0005)
ylabels['tor_ra_pip3_e_f'] = np.arange(0.001, 0.0016, 0.0002)
ylabels['tor_ra_dip3_e_f'] = np.arange(0.00007, 0.00012, 0.00002)
ylabels['tor_ra_mcp4_e_f'] = np.arange(0.002, 0.0031, 0.0005)
ylabels['tor_ra_pip4_e_f'] = np.arange(0.0007, 0.00105, 0.0001)
ylabels['tor_ra_dip4_e_f'] = np.arange(0.00005, 0.000081, 0.00001)
ylabels['tor_ra_mcp5_e_f'] = np.arange(0.0013, 0.00191, 0.0002)
ylabels['tor_ra_pip5_e_f'] = np.arange(0.00045, 0.000651, 0.0001)
ylabels['tor_ra_dip5_e_f'] = np.arange(0.000035, 0.000051, 0.000005)
ylabels['tor_ra_sh_ab_ad'] = np.arange(0.92, 1.09, 0.04)
ylabels['tor_ra_sh_e_f'] = np.arange(3, 6.1, 1)
ylabels['tor_ra_sh_rot'] = np.arange(0.6, 1.1, 0.2)
ylabels['tor_ra_el_e_f'] = np.arange(5.8, 6.71, 0.3)
ylabels['tor_ra_wr_ad_ab'] = np.arange(-0.294, -0.287, 0.002)  

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
  plt.legend()
  plt.margins(x=0.01, y=0.01) # remove margins for plots

plt.savefig('Fig1 Lowest MSE ver1.svg', format='svg')
plt.savefig('Fig1 Lowest MSE ver1.pdf', format='pdf')
plt.savefig('Fig1 Lowest MSE ver1.png', format='png')