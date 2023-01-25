# -*- coding: utf-8 -*-
"""Denoising transformer Inference (NASA).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pOPheGyloUncMI6CIspye9_ov2eKP5cg

# Denoising Transformer (Inference)

Reference:

D. Chen, W. Hong and X. Zhou, "Transformer Network for Remaining Useful Life Prediction of Lithium-Ion Batteries," in IEEE Access, vol. 10, pp. 19621-19628, 2022.

Github: 
https://github.com/XiuzeZhou/RUL

**To use this notebook**: upload the trained model and test tensors (if using Google Colab) or make sure the trained model and test tensors are in the same directory as the Jupyter notebook file.

## Installing necessary packages not in Colab instance
"""

#!pip install transformers
#!unzip checkpoints_Transformer_NASA.zip


"""## Importing packages for metric measurement"""

# Setting up argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p','--print',action='store_true',
                    help='Print to 16x2 LCD Display')
args = parser.parse_args()

# For timing
from timeit import default_timer as timer
import torch

def print_inference_time(start: float,
                     end: float,
                     device: torch.device = None,
                     print_time = True):
  """
  Prints difference between start and end time
  """

  total_time = end - start
  if print_time:
    print(f"Inference time on {device}: {total_time:.3f} seconds")
  
  return total_time

# For memory tracing
import tracemalloc

def memory_stats(snapshot, key_type='lineno', print_mem=True):
  '''
  Compute memory usage from a tracemalloc snapshot.
  Results are in KiB (Kibibytes).
  '''
  snapshot = snapshot.filter_traces((
      tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
      tracemalloc.Filter(False, "<unknown>"),
  ))
  usage_stat = snapshot.statistics(key_type)

  total = sum(stat.size for stat in usage_stat)
  total = total / 1024

  if print_mem:
    print(f"Allocated memory: {total:.2f} KiB")

  return total

"""## Start memory profiling"""

tracemalloc.start()

"""## Start of Inference

### Import packages
"""

import numpy as np
import random
import math
#import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

device = "cuda" if torch.cuda.is_available() else "cpu"

rul_preds_list = {}

"""## Function declaration

### Denoising Transformer Architecture
"""

class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.01):
        super(Autoencoder, self).__init__()
        self.input_size, self.hidden_dim, self.noise_level = input_size, hidden_dim, noise_level
        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_size)
        
    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1
    
    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
        return corrupted_x
    
    def decoder(self, x):
        h2 = self.fc2(x)
        return h2
    
    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=16):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class Net(nn.Module):
    def __init__(self, feature_size=16, hidden_dim=32, num_layers=1, nhead=8, dropout=0.0, noise_level=0.01):
        super(Net, self).__init__()
        self.auto_hidden = int(feature_size/2)
        input_size = self.auto_hidden 
        self.pos = PositionalEncoding(d_model=input_size, max_len=input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(input_size, 1)
        self.autoencoder = Autoencoder(input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level)
 
    def forward(self, x): 
        batch_size, feature_num, feature_size  = x.shape 
        encode, decode = self.autoencoder(x.reshape(batch_size, -1))# batch_size*seq_len
        out = encode.reshape(batch_size, -1, self.auto_hidden)
        out = self.pos(out)
        out = out.reshape(1, batch_size, -1) # (1, batch_size, feature_size)
        out = self.cell(out)  
        out = out.reshape(batch_size, -1) # (batch_size, hidden_dim)
        out = self.linear(out)            # out shape: (batch_size, 1)
        
        return out, decode

"""## Main Code for inference"""

# setup seed
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Parameters
Rated_Capacity = {}
Rated_Capacity['B0005'] = 2.0143
Rated_Capacity['B0006'] = 1.9857
Rated_Capacity['B0007'] = 2.0143
Rated_Capacity['B0018'] = 2.0143

window_size = 16
feature_size = window_size
dropout = 0.0
EPOCH = 1000
nhead = 8
hidden_dim = 32
num_layers = 1
lr = 0.001    # learning rate
weight_decay = 1e-5
noise_level = 0.0
alpha = 2e-3

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
model = {}
X_test, y_test = {}, {}
test_sequence, test_labels = {}, {}

# Instantiate model
for leave_out in Battery_list:
  model[leave_out] = Net(feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout,
                    noise_level=noise_level)
  model[leave_out] = model[leave_out].to(device)

  # Load the checkpoint (state dictionary) from file
  model[leave_out].load_state_dict(torch.load(f'checkpoints/transformer_NASA_{leave_out}.pth', map_location=device)) 

# Load test_data_tensor
X_test, y_test, test_sequence, test_labels = torch.load("test_tensors_Transformer_NASA.pt", map_location=device)

"""### One-step Ahead"""

def accuracy(y_test: torch.tensor, y_pred: torch.tensor):
  error = torch.abs(y_pred-y_test)/y_test
  acc = 1 - error
  return float(acc)

# End memory snapshot for setup
memsnap_end_setup = tracemalloc.take_snapshot()
memuse_end_setup = memory_stats(memsnap_end_setup, print_mem=False)
# End capture memory stats before inference
tracemalloc.stop()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

plt.figure(figsize=(40,7))
plt.suptitle(f'Transformer (NASA) One-Step Ahead',fontsize=15, weight='bold')

OSP_Transformer_NASA = {}

for leave_out, idx in zip(Battery_list, range(len(Battery_list))):
  model[leave_out].eval()
  acc = 0
  pred = []

  # Start timer and tracemalloc
  start_time = timer()
  tracemalloc.start()

  # Do the predictions
  with torch.inference_mode():
    for test_tensor, label in zip(X_test[leave_out], y_test[leave_out]):
      preds, _ = model[leave_out]((test_tensor/Rated_Capacity[leave_out]).reshape(-1,1,window_size))
      pred.append(preds)
      acc += accuracy(preds*Rated_Capacity[leave_out], label)

  # End timer and capture memory stats after inference
  end_time = timer()
  memsnap_post_onestep = tracemalloc.take_snapshot()
  tracemalloc.stop()

  # Convert predictions from list to tensor
  pred = torch.cat(pred) * Rated_Capacity[leave_out]

  # Computing memory usage
  memuse_post_onestep = memory_stats(memsnap_post_onestep, print_mem=False)
  memuse_total = memuse_post_onestep + memuse_end_setup

  # Computing RUL Error
  Threshold = 0.7 * Rated_Capacity[leave_out]
  idx_true = (y_test[leave_out]<Threshold).nonzero().squeeze() # search idx less than threshold
  RUL_true = idx_true[0][0]                         # first entry is true RUL

  idx_pred = (pred<Threshold).nonzero().squeeze()  # search idx less than threshold
  RUL_pred = idx_pred[0][0]                        # first entry is pred RUL
  RUL_error = RUL_true - RUL_pred                  # if positive value, earlier than true RUL; 
                                                  # negative value, later than true RUL
  RUL_relative_error = RUL_error / RUL_true

  # Computing metrics
  acc = acc/len(y_test[leave_out]) * 100
  mae = mean_absolute_error(y_test[leave_out].cpu(),pred.cpu())
  rmse = mean_squared_error(y_test[leave_out].cpu(),pred.cpu())

  # Printing metrics
  print(f"Denoising Transformer (window_size={window_size}, nhead={nhead})\nOne-step ahead Prediction")
  print("="*60)
  total_time = print_inference_time(start_time, end_time, device)
  print(f"Mem alloc for inference: {memuse_post_onestep} KiB \t Total including init: {memuse_total:.3f}KiB")
  print(f"Accuracy on {leave_out}: {acc:4f}% | MAE: {mae:.5f} | RMSE: {rmse:.5f}")
  print(f"RUL Actual: {RUL_true} | RUL Predicted: {RUL_pred} | RUL Error: {RUL_error}\n{'-'*64}")

  # Plot 
  plt.subplot(1,4,idx+1)
  plt.plot(y_test[leave_out].cpu(),'k--',label="Ground Truth")
  plt.plot(pred.cpu(),'r-',label="Prediction")
  plt.title(f"Prediction for {leave_out}")
  plt.legend()
  plt.xlabel('Number of Discharge Cycles')
  plt.ylabel('Capacity (in Ah)')
  plt.grid()

  OSP_Transformer_NASA[leave_out] = {
      "model_name": leave_out,
      "mem_usage": memuse_total,
      "exec_time": total_time,
      "acc": acc,
      "mae": mae,
      "rmse": rmse,
      "rul_error": int(RUL_error),
      "RUL_relative_error": float(RUL_relative_error)
  }

plt.savefig('One-step_Transformer_NASA.pdf')

import pandas as pd
# Convert to PD DataFrame
OSP_results = pd.DataFrame([OSP_Transformer_NASA['B0005'],
                            OSP_Transformer_NASA['B0006'],
                            OSP_Transformer_NASA['B0007'],
                            OSP_Transformer_NASA['B0018']
                            ])
# Saving results
OSP_results.to_pickle("OSP_Transformer_NASA.pkl")

"""### Multi-Step Ahead"""

def plot_predictions(train_cap, train_cyc, label_cap, label_cyc, predictions, title):
  plt.title(title)
  plt.plot(train_cyc, train_cap, 'k-', label='Input data')
  plt.plot(label_cyc, label_cap, 'k:', label='Ground Truth')
  plt.plot(label_cyc, predictions, 'b-', label='Predicted')
  plt.legend()
  plt.xlabel('Number of Discharge Cycles')
  plt.ylabel('Capacity (in Ah)')
  plt.grid()

plt.figure(figsize=(40,7))
plt.suptitle(f'Transformer (NASA) Multi-Step Ahead', fontsize=16, weight='bold')

MSP_Transformer_NASA = {}

for leave_out, idx in zip(Battery_list, range(len(Battery_list)) ):
  # access dictionary contents
  eval_cap, eval_cyc = test_sequence[leave_out]
  test_cap, test_cyc = test_labels[leave_out]

  # convert lists to tensors
  eval_cap, eval_cyc = eval_cap, eval_cyc
  test_cap, test_cyc = np.array(test_cap), np.array(test_cyc)

  # create point list which contains predictions
  preds = []
  # sequence which contains prediction inputs (last window_size entries of array)
  seq = eval_cap[-window_size:]

  # Begin timer and capture memory stats before inference
  start_time = timer()
  tracemalloc.start()

  model[leave_out].eval()
  with torch.inference_mode():
    for j in range(len(test_cap)):
      pred, _ = model[leave_out](torch.tensor(seq, dtype=torch.float32, device=device).reshape(-1,1,window_size)/Rated_Capacity[leave_out])
      seq = seq[1:]
      seq.append(np.float64(pred * Rated_Capacity[leave_out]))
      preds.append(np.float64(pred * Rated_Capacity[leave_out]))

  # End timer and capture memory stats after inference
  end_time = timer()
  memsnap_post_multistep = tracemalloc.take_snapshot()
  tracemalloc.stop()

  # copy preds to dictionary
  preds = np.array(preds)

  plt.subplot(1,4,idx+1)
  plot_predictions(train_cap = eval_cap,
                    train_cyc = eval_cyc,
                    label_cap = test_cap,
                    label_cyc = test_cyc,
                    predictions = preds,
                    title = f"Prediction for Battery {leave_out}")

  # Compute memory usage
  memuse_post_multistep = memory_stats(memsnap_post_multistep, print_mem=False)
  memuse_total = memuse_post_multistep + memuse_end_setup

  # Computing RUL Error
  Threshold = 0.7 * Rated_Capacity[leave_out]
  idx_true = (test_cap<Threshold).nonzero()        # search idx less than threshold
  RUL_true = idx_true[0][0]                        # first entry is true RUL

  idx_preds = (preds<Threshold).nonzero()          # search idx less than threshold
  RUL_preds = idx_preds[0][0]                      # first entry is pred RUL
  RUL_error = RUL_true - RUL_preds                 # if positive value, early RUL; 
                                                  # negative value, late RUL
  RUL_relative_error = RUL_error / RUL_true

  # Computing metrics
  error = np.abs(preds - test_cap)/test_cap
  acc = np.ones_like(error) - error
  acc = 100 * np.sum(acc)/len(test_cap)
  mae = mean_absolute_error(test_cap,preds)
  rmse = sqrt(mean_squared_error(test_cap,preds))

  # Printing metrics
  print(f"Denoising Transformer (window_size={window_size}, nhead={nhead})\nMulti-step Ahead Prediction")
  print("="*60)

  total_time = print_inference_time(start_time, end_time, device)
  print(f"Memory alloc for inference: {memuse_post_multistep:.3f} KiB \t Including init: {memuse_total:.3f} KiB")
  print(f"Accuracy for Battery {leave_out}: {acc:.4f}% | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
  print(f"RUL True: {RUL_true} | RUL Predicted: {RUL_preds} | RUL Error: {RUL_error}\n{'-'*60}")
  
  rul_preds_list[leave_out] = RUL_preds

  MSP_Transformer_NASA[leave_out] = {
      "model_name": leave_out,
      "mem_usage": memuse_total,
      "exec_time": total_time,
      "acc": acc,
      "mae": mae,
      "rmse": rmse,
      "rul_error": int(RUL_error),
      "RUL_relative_error": float(RUL_relative_error)
  }

plt.savefig('Multi-step_Transformer_NASA.pdf')

# Convert to PD dataframe
MSP_results = pd.DataFrame([MSP_Transformer_NASA['B0005'],
                            MSP_Transformer_NASA['B0006'],
                            MSP_Transformer_NASA['B0007'],
                            MSP_Transformer_NASA['B0018']
                            ])
# Save as pkl
MSP_results.to_pickle("MSP_Transformer_NASA.pkl")

# Print to 16x2 LCD
if args.print:
    # LCD Stuff
    from Adafruit_CharLCD import Adafruit_CharLCD
    from time import sleep

    # Initialize LCD and specify pins
    lcd = Adafruit_CharLCD(rs=26, en=19,
                           d4=13, d5=6, d6=5, d7=21,
                           cols=16, lines=2)
    # Clear the LCD
    lcd.clear()
    for leave_out in Battery_list:
        # Display message
        lcd.clear()
        lcd.message(f'Cycles left for\n{leave_out}: {rul_preds_list[leave_out]}')
        sleep(5)