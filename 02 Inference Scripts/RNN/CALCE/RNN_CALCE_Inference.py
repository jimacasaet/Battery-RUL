# Setting up argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p','--print',action='store_true',
                    help='Print to 16x2 LCD Display')
args = parser.parse_args()

# For timing
from timeit import default_timer as timer
import torch

# Print Inference Time
def print_inference_time(start: float,
                         end: float,
                         device: torch.device = None):
    
    """
    Prints difference between start and end time
    """

    total_time = end - start
    print(f"Inference time on {device}: {total_time:.3f} seconds")

    return total_time

# For memory tracing
import tracemalloc

# Print Memory Usage
def memory_stats(snapshot, key_type='lineno'):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    usage_stat = snapshot.statistics(key_type)

    total = sum(stat.size for stat in usage_stat)

    print("Allocated memory: %.1f KiB" % (total / 1024))

    return (total / 1024)

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
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):   
        
        hidden = self.rnn(x.unsqueeze(0))[0]
        output = self.linear(hidden)
        return output

# Compute for accuracy
def accuracy(y_test, y_pred):
    error = np.abs(y_pred-y_test)/y_test
    acc = np.ones_like(error) - error
    acc = np.sum(acc)/len(y_pred)
    return float(acc)

def test(loaded_model, X_test, Y_test):
    loss_fn = nn.HuberLoss()
    loaded_model.eval()
    with torch.inference_mode():
        test_out = []
        for i in range(X_test.size()[0]):
            # 1. Forward Pass
            test_output = loaded_model(X_test[i])
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

OSP_RNN_CALCE = {}
rul_preds_list = {}
plt.figure(figsize=(40,7))
plt.suptitle(f'RNN (CALCE) One-step Ahead Prediction', size=15, weight='bold')
Battery_list = ['CS2_35','CS2_36','CS2_37','CS2_38']

for batt, idx in zip(Battery_list, range(len(Battery_list))):

    # timer
    start_time = timer()
    tracemalloc.start()
    torch.manual_seed(42)

    # Setting up device-agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Rated Capacity
    Rated_Capacity = 1.1
    
    if(batt=='CS2_35'):
        model_save_path = 'models/RNNModelvB35.pth'
        input_path = "tensors/CS2_35/CS2_35_input.pt"
        target_path = "tensors/CS2_35/CS2_35_target.pt"
    elif(batt=='CS2_36'):
        model_save_path = 'models/RNNModelvB36.pth'
        input_path = "tensors/CS2_36/CS2_36_input.pt"
        target_path = "tensors/CS2_36/CS2_36_target.pt"
    elif(batt=='CS2_37'):
        model_save_path = 'models/RNNModelvB37.pth'
        input_path = "tensors/CS2_37/CS2_37_input.pt"
        target_path = "tensors/CS2_37/CS2_37_target.pt"
    elif(batt=='CS2_38'):
        model_save_path = 'models/RNNModelvB38.pth'
        input_path = "tensors/CS2_38/CS2_38_input.pt"
        target_path = "tensors/CS2_38/CS2_38_target.pt"

    X_test = torch.load(input_path)
    Y_test = torch.load(target_path)

    input_size = 22
    hidden_size = 64
    output_size = 1

    # load saved model
    # Instantiate a new instance of our model (this will be instantiated with random weights
    loadedmodel = RNNModel(input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size)

    # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
    loadedmodel.load_state_dict(torch.load(f=model_save_path, map_location=torch.device(device)))

    # get predictions
    test_loss, Y_test, y_pred = test(loaded_model=loadedmodel,
                                     X_test=X_test,
                                     Y_test=Y_test)

    acc = accuracy(Y_test.cpu().detach().numpy(), y_pred.squeeze(-1).cpu().detach().numpy())
    mae = mean_absolute_error(y_pred.squeeze(-1).cpu().detach().numpy(), Y_test.cpu().detach().numpy())
    rmse = sqrt(mean_squared_error(y_pred.squeeze(-1).cpu().detach().numpy(), Y_test.cpu().detach().numpy()))

    # Computing RUL Error
    Threshold = 0.7 * Rated_Capacity
    idx_true = (Y_test<Threshold).nonzero().squeeze()
    RUL_true = idx_true[0][0]

    if(torch.Tensor(Y_test[int(RUL_true + 1)])>Threshold):
        RUL_true = idx_true[1][0]

    idx_pred = (y_pred<Threshold).nonzero().squeeze()  # search idx less than threshold
    RUL_pred = idx_pred[0][0]                        # first entry is pred RUL
    if(torch.Tensor(Y_test[int(RUL_pred + 1)])>Threshold):
        RUL_pred = idx_pred[1][0]
    RUL_error = RUL_true - RUL_pred
    rul_preds_list[batt] = RUL_pred
    # if positive value, earlier than true RUL; 
    # negative value, later than true RUL
    print(f"RNN (CALCE)\nTest on {batt}")
    print(f"Test Loss: {test_loss:.4f} | MAE: {mae:.4f} | MSE: {rmse:.4f} | Accuracy: {acc:.2f}")
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
    plt.plot(y, Y_test, 'k:', label='Ground Truth')
    plt.plot(y, y_pred, 'r-', label='Predicted')
    plt.xlabel('Discharge cycles')
    plt.ylabel('Capacity (Ah)')
    plt.title(f'One-step Ahead for {batt}')
    plt.legend()
    plt.grid()

    OSP_RNN_CALCE[batt] = {
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
plt.savefig('One-step_RNN_CALCE.pdf')

# Convert to PD DataFrame
OSP_results = pd.DataFrame([OSP_RNN_CALCE['CS2_35'],
                            OSP_RNN_CALCE['CS2_36'],
                            OSP_RNN_CALCE['CS2_37'],
                            OSP_RNN_CALCE['CS2_38']
                            ])
# Saving results
OSP_results.to_pickle("OSP_RNN_CALCE.pkl")
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
