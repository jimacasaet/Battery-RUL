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

# Start tracemalloc
tracemalloc.start()

# Whole python code for inference
# Import all modules
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch import optim
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from pathlib import Path

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = F.relu(x + hidden_state)
        out = self.h2o(hidden_state)
        return out, hidden_state

    def init_zero_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Metrics
# Compute for accuracy
def accuracy(y_test, y_pred):
  error = np.abs(y_pred-y_test)/y_test
  acc = np.ones_like(error) - error
  acc = np.sum(acc)/len(y_pred)
  return float(acc)

# function for making predictions
def test(loadedmodel, X_test, Y_test):
    loss_fn = nn.HuberLoss()
    loadedmodel.eval()
    with torch.inference_mode():
        hidden = loadedmodel.init_zero_hidden()
        hidden = hidden.to(device)
        test_out = []
        for i in range(X_test.size()[0]):
            # 1. Forward Pass
            test_output, hidden = loadedmodel(X_test[i], hidden)
            test_out.append(test_output)
                
        y_pred = torch.stack(test_out).to(device)
        # 2. Calculate Loss
        test_loss = loss_fn(y_pred.squeeze(-1), Y_test)
    return test_loss, Y_test, y_pred

# End memory snapshot for setup
memsnap_end_setup = tracemalloc.take_snapshot()
memuse_end_setup = memory_stats(memsnap_end_setup)
# End capture memory stats before inference
tracemalloc.stop()

OSP_RNN_NASA = {}
rul_preds_list = {}
plt.figure(figsize=(40,7))
plt.suptitle(f'RNN (NASA) One-step Ahead Prediction', size=15, weight='bold')
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']

for batt, idx in zip(Battery_list, range(len(Battery_list))):
    # timer
    start_time = timer()
    tracemalloc.start()
    torch.manual_seed(42)

    # Setting up device-agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Rated Capacity
    Rated_Capacity = 2.0

    if(batt=='B0005'):
        model_save_path = 'models/RNN_NASAvB5.pth'
        input_path = "tensors/B0005/B0005_input.pt"
        target_path = "tensors/B0005/B0005_target.pt"
    elif(batt=='B0006'):
        model_save_path = 'models/RNN_NASAvB6.pth'
        input_path = "tensors/B0006/B0006_input.pt"
        target_path = "tensors/B0006/B0006_target.pt"
    elif(batt=='B0007'):
        model_save_path = 'models/RNN_NASAvB7.pth'
        input_path = "tensors/B0007/B0007_input.pt"
        target_path = "tensors/B0007/B0007_target.pt"
    elif(batt=='B0018'): 
        model_save_path = 'models/RNN_NASAvB18.pth'
        input_path = "tensors/B0018/B0018_input.pt"
        target_path = "tensors/B0018/B0018_target.pt"

    X_test = torch.load(input_path)
    Y_test = torch.load(target_path)

    # model hyperparameters
    input_size = 31
    hidden_size = 10
    output_size = 1
    # load saved model
    # Instantiate a new instance of our model (this will be instantiated with random weights
    loadedmodel = RNNModel(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=output_size)

    # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
    loadedmodel.load_state_dict(torch.load(f=model_save_path))
    loadedmodel.to(device)

    test_loss, Y_test, y_pred = test(loadedmodel=loadedmodel,
                                        X_test=X_test,
                                        Y_test=Y_test)

    acc = accuracy(Y_test.cpu().detach().numpy(), y_pred.squeeze(-1).cpu().detach().numpy())
    acc = acc * 100
    mae = mean_absolute_error(y_pred.squeeze(-1).cpu().detach().numpy(), Y_test.cpu().detach().numpy())
    rmse = sqrt(mean_squared_error(y_pred.squeeze(-1).cpu().detach().numpy(), Y_test.cpu().detach().numpy()))

    # Computing RUL Error
    Threshold = 0.75 * Rated_Capacity
    idx_true = (Y_test<Threshold).nonzero().squeeze()
    RUL_true = idx_true[0][0]

    if(torch.Tensor(Y_test[int(RUL_true + 1)])>Threshold):
        RUL_true = idx_true[1][0]

    idx_pred = (y_pred<Threshold).nonzero().squeeze()  # search idx less than threshold
    RUL_pred = idx_pred[0][0]                        # first entry is pred RUL
    RUL_error = RUL_true - RUL_pred
    rul_preds_list[batt] = RUL_pred
    # if positive value, earlier than true RUL; 
    # negative value, later than true RUL
    print(f"RNN (NASA)\nTest on {batt}")
    print(f"Test Loss: {test_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | Accuracy: {acc:.2f}")
    print(f"True RUL: {int(RUL_true)} | Prediction RUL: {int(RUL_pred)} | RUL Error: {int(RUL_error)}")

    end_time = timer()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    total_time = print_inference_time(start_time, end_time, device)
    memuse_total = memory_stats(snapshot) + memuse_end_setup
    
    # Plot
    y_pred = y_pred.cpu().detach().numpy().squeeze()
    Y_test = Y_test.cpu().detach().numpy()
    y = list(range(1, len(y_pred)+1))
    plt.subplot(1,4,idx+1)
    plt.plot(y, Y_test, 'k-', label='Ground Truth')
    plt.plot(y, y_pred, 'r', label='Predicted')
    plt.xlabel('Discharge cycles')
    plt.ylabel('Capacity (Ah)')
    plt.title(f'One-step Ahead for {batt}')
    plt.legend()
    plt.grid()

    OSP_RNN_NASA[batt] = {
        "model_name": batt,
        "mem_usage": memuse_total,
        "exec_time": total_time,
        "acc": acc,
        "mae": mae,
        "rmse": rmse,
        "rul_error": int(RUL_error),
        "RUL_relative_error": float(-RUL_error / RUL_true)
    }

    print('-'*64)

plt.savefig('One-step_RNN_NASA.pdf')

# Convert to PD DataFrame
OSP_results = pd.DataFrame([OSP_RNN_NASA['B0005'],
                            OSP_RNN_NASA['B0006'],
                            OSP_RNN_NASA['B0007'],
                            OSP_RNN_NASA['B0018']
                            ])
# Saving results
OSP_results.to_pickle("OSP_RNN_NASA.pkl")
print(OSP_results)

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