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

def memory_stats(snapshot, key_type='lineno', print_mem=False):
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

# Import modules
import numpy as np
import random
import math
import os
import sys
import copy
import shutil
import zipfile
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from math import sqrt
from pickle import load
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Compute for accuracy
def accuracy(y_test: torch.Tensor, y_pred: torch.Tensor):
    error = torch.abs(y_pred-y_test)/y_test
    acc = torch.ones_like(error) - error
    acc = torch.sum(acc)/len(y_pred)
    return float(acc)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers)
    
    def forward(self, x):
        flat = x.view(x.shape[0], x.shape[1], self.input_size)
        out, h = self.lstm(flat)
        return out, h

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1, num_layers = 1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h):
        out, h = self.lstm(x.unsqueeze(0), h)
        y = self.linear(out.squeeze(0))
        return y, h

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_size = 1, output_size = 1):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size = input_size,
                               hidden_size = hidden_size)

        self.decoder = Decoder(input_size = input_size,
                               hidden_size = hidden_size,
                               output_size = output_size)


    def train_model(self, train_battery, train_list, epochs, target_len, method = 'recursive',
                    tfr = 0.5, lr = 0.01, dynamic_tf = False):
        losses = {}
        for name in train_list:
            losses[name] = np.full(epochs+1, np.nan)
        optimizer = optim.Adam(self.parameters(), lr = lr)
        criterion = nn.HuberLoss()
        for e in range(epochs+1):
            for name in train_list:
                train = to_tensor(train_battery[name]['X'])
                target = to_tensor(train_battery[name]['Y'])
                train = train.to(device)
                target = target.to(device)

                predicted = torch.zeros(target_len, train.shape[1], train.shape[2])
                predicted = predicted.to(device)
                optimizer.zero_grad()
                _, enc_h = self.encoder(train)
                dec_in = train[-1, :, :]
                dec_h = enc_h

                '''
                Recursive training is passing each decoder's output to the decoder's
                input at another step, which is as follows:
                '''
                if method == 'recursive':
                    for t in range(target_len):
                        dec_out, dec_h = self.decoder(dec_in, dec_h)
                        predicted[t] = dec_out
                        dec_in = dec_out

                '''
                Teacher forcing method applies recursive training with probability
                or supplies target values to the encoder with probability tfr:
                '''
                if method == 'teacher_forcing':
                    # use teacher forcing
                    if random.random() < tfr:
                        for t in range(target_len):
                            dec_out, dec_h = self.decoder(dec_in, dec_h)
                            predicted[t] = dec_out
                            dec_in = target[t, :, :]
                    # predict recursively
                    else:
                        for t in range(target_len):
                            dec_out, dec_h = self.decoder(dec_in, dec_h)
                            predicted[t] = dec_out
                            dec_in = dec_out

                '''
                Mixed teacher forcing method mixes the decoder output with 
                target values within the same training epoch:
                '''
                if method == 'mixed_teacher_forcing':
                    # predict using mixed teacher forcing
                    for t in range(target_len):
                        dec_out, dec_h = self.decoder(dec_in, dec_h)
                        predicted[t] = dec_out
                        # predict with teacher forcing
                        if random.random() < tfr:
                            dec_in = target[t, :, :]
                        #predict recursively
                        else:
                            dec_in = dec_out
            
                loss = criterion(predicted, target)
                loss.backward()
                optimizer.step()
                losses[name][e] = loss.item()
                if e % 100 == 0:
                    print(f'Battery: {name} |Epoch {e}/{epochs}: {round(loss.item(), 4)}')
            
                # In some cases, it is helpful to decrease the teacher forcing ration during training
                # dynamic teacher forcing
                if dynamic_tf and tfr > 0:
                    tfr = tfr - 0.02
        return losses
    
    def predict(self, x, target_len):
        y = torch.zeros(target_len, x.shape[1], x.shape[2])
        _, enc_h = self.encoder(x)
        dec_in = x[-1, :, :]
        dec_h = enc_h
        for t in range(target_len):
            dec_out, dec_h = self.decoder(dec_in, dec_h)
            y[t] = dec_out
            dec_in = dec_out
        return y

# transform numpy to tensor
def to_tensor(data):
    return torch.tensor(data).unsqueeze(2).transpose(0,1).float()

def inference_window(x_input, train_period):
    rem = len(x_input) % train_period
    index = abs(rem - len(x_input))
    num_samples = index // train_period
    x = []
    init_x = -index
    for i in range(num_samples):
        '''
        x.append(x_input[-index-a: a])
        a = a + 100
        index =
        '''
        fin_x = init_x + train_period
        if fin_x == 0:
            x.append(x_input[init_x:])
        else:
            x.append(x_input[init_x:fin_x])
        init_x = fin_x
                      
    return x
    
# End memory snapshot for setup
memsnap_end_setup = tracemalloc.take_snapshot()
memuse_end_setup = memory_stats(memsnap_end_setup, print_mem=False)
# End capture memory stats before inference
tracemalloc.stop()
    

Rated_Capacity = 1.1
start_time = timer()
tracemalloc.start()
max_length = 1500
# Setting up device-agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rul_preds_list = {}

# test tensor path
#test_path = "tensors/CS2_35/CS2_35_50.pt"
test_path = "tensors/CS2_35/CS2_35_60.pt"
#test_path = "tensors/CS2_35/CS2_35_70.pt"

# scaled input tensor path
#input_path = "tensors/CS2_35/scaledinput_CS2_35_50.pt"
input_path = "tensors/CS2_35/scaledinput_CS2_35_60.pt"
#input_path = "tensors/CS2_35/scaledinput_CS2_35_70.pt"

# target tensor path
#target_path = "tensors/CS2_35/target_CS2_35_50.pt"
target_path = "tensors/CS2_35/target_CS2_35_60.pt"
#target_path = "tensors/CS2_35/target_CS2_35_70.pt"

# model paths
model_save_path = "models/LSTMCALCE_128_35.pth"
#model_save_path = "models/LSTMCALCE_32_35.pth"

scaler = scaler = load(open('scaler.pkl', 'rb'))

hidden_size = 128
#hidden_size = 32

# train_period
train_period = 80
# prediction length
prediction_period = 80

x_in = torch.load(test_path)
x_input = torch.load(input_path)
x_input = x_input.tolist()
x_target = torch.load(target_path)
x_target = x_target.tolist()

loadedmodel = EncoderDecoder(hidden_size = hidden_size)
loadedmodel.load_state_dict(torch.load(f=model_save_path,
                                        map_location=torch.device('cpu')))

loadedmodel.eval()
output = []
while len(x_input) <= max_length:
    predicted = loadedmodel.predict(x_in, prediction_period)
    pred_seq = predicted[:, -1, :].view(-1).tolist()   
    for i in range(len(pred_seq)):
        x_input.append([pred_seq[i]])
        output.append([pred_seq[i]])
    X = inference_window(x_input, train_period)
    X = np.array(X)
    X = np.squeeze(X)
    x_in = to_tensor(X)

output = scaler.inverse_transform(np.array(output).reshape(-1,1))
output = output.tolist()
acc = accuracy(torch.Tensor(x_target), torch.Tensor(output[:len(x_target)])) * 100
mae = mean_absolute_error(output[:len(x_target)], x_target)
rmse = sqrt(mean_squared_error(output[:len(x_target)], x_target))

# Computing RUL Error
Threshold = 0.7 * Rated_Capacity
idx_true = (torch.Tensor(x_target)<Threshold).nonzero().squeeze()
RUL_true = idx_true[0][0]

if(torch.Tensor(x_target[int(RUL_true + 1)])>Threshold):
    RUL_true = idx_true[0][1]

idx_pred = (torch.Tensor(output[:len(x_target)])<Threshold).nonzero().squeeze()  # search idx less than threshold
RUL_pred = idx_pred[0][0]                        # first entry is pred RUL
RUL_error = RUL_true - RUL_pred
# if positive value, earlier than true RUL; 
# negative value, later than true RUL   
RUL_relative_error = -RUL_error/RUL_true


print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | Accuracy: {acc:.2f}")
print(f"True RUL: {int(RUL_true)} | Prediction RUL: {int(RUL_pred)} | RUL Error: {int(RUL_error)}")
end_time = timer()
snapshot = tracemalloc.take_snapshot()
total_time = print_inference_time(start_time, end_time, device)
memuse_total = memory_stats(snapshot) + memuse_end_setup
print(f"Allocated memory {memuse_total}")
    

#input tensor path
input_path = "tensors/CS2_35/input_CS2_35_50.pt"
x_input = torch.load(input_path)
x_input = x_input.tolist()
#input_path = "tensors/CS2_35/scaledinput_CS2_35_60.pt"
#input_path = "tensors/CS2_35/scaledinput_CS2_35_70.pt"
#input_path = "tensors/CS2_35/scaledinput_CS2_35_80.pt"
import matplotlib as mpl
battery_name = 'CS2_35'
x = range(0, len(x_input) + len(x_target))

plt.figure(figsize=(40,7))
plt.suptitle('LSTM (CALCE) Multi-Step Ahead',size=15,weight='bold')

plt.subplot(1,4,1)
plt.title(f'Prediction for Battery {battery_name}')
plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.grid(True)
plt.plot(x[:-len(x_target)],
         x_input,
         "k-",
         label='Input data')
plt.plot(x[-len(x_target):],
         x_target,
         "k:",
         label='Ground Truth')
plt.plot(x[-len(x_target):],
         output[:len(x_target)],
         "b-",
         label='Predicted')
plt.legend()


results = {
      "model_name": battery_name,
      "mem_usage": memuse_total,
      "exec_time": total_time,
      "acc": acc,
      "mae": mae,
      "rmse": rmse,
      "rul_error": int(RUL_error),
      "RUL_relative_error": float(RUL_relative_error)
}

rul_preds_list[battery_name] = RUL_pred

# create a binary pickle file 
import pickle

f = open(f"{battery_name}.pkl","wb")
pickle.dump(results, f)
f.close()

############################################################################
#############################  CS2_36 ######################################
############################################################################
Rated_Capacity = 1.1
start_time = timer()
tracemalloc.start()
max_length = 1500
# Setting up device-agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# test tensor path
#test_path = "tensors/CS2_36/CS2_36_50.pt"
test_path = "tensors/CS2_36/CS2_36_60.pt"
#test_path = "tensors/CS2_36/CS2_36_70.pt"

# scaled input tensor path
#input_path = "tensors/CS2_36/scaledinput_CS2_36_50.pt"
input_path = "tensors/CS2_36/scaledinput_CS2_36_60.pt"
#input_path = "tensors/CS2_36/scaledinput_CS2_36_70.pt"

# target tensor path
#target_path = "tensors/CS2_36/target_CS2_36_50.pt"
target_path = "tensors/CS2_36/target_CS2_36_60.pt"
#target_path = "tensors/CS2_36/target_CS2_36_70.pt"

# model paths
model_save_path = "models/LSTMCALCE_128_36.pth"
#model_save_path = "models/LSTMCALCE_32_36.pth"

scaler = scaler = load(open('scaler.pkl', 'rb'))

hidden_size = 128
#hidden_size = 32

#train_period
train_period = 80
#prediction_period
prediction_period = 80

x_in = torch.load(test_path)
x_input = torch.load(input_path)
x_input = x_input.tolist()
x_target = torch.load(target_path)
x_target = x_target.tolist()


loadedmodel = EncoderDecoder(hidden_size = hidden_size)
loadedmodel.load_state_dict(torch.load(f=model_save_path,
                                        map_location=torch.device('cpu')))

loadedmodel.eval()
output = []
while len(x_input) <= max_length:
    predicted = loadedmodel.predict(x_in, prediction_period)
    pred_seq = predicted[:, -1, :].view(-1).tolist()   
    for i in range(len(pred_seq)):
        x_input.append([pred_seq[i]])
        output.append([pred_seq[i]])
    X = inference_window(x_input, train_period)
    X = np.array(X)
    X = np.squeeze(X)
    x_in = to_tensor(X)

output = scaler.inverse_transform(np.array(output).reshape(-1,1))
output = output.tolist()
acc = accuracy(torch.Tensor(x_target), torch.Tensor(output[:len(x_target)])) * 100
mae = mean_absolute_error(output[:len(x_target)], x_target)
rmse = sqrt(mean_squared_error(output[:len(x_target)], x_target))

# Computing RUL Error
Threshold = 0.7 * Rated_Capacity
idx_true = (torch.Tensor(x_target)<Threshold).nonzero().squeeze()
RUL_true = idx_true[0][0]

if(torch.Tensor(x_target[int(RUL_true + 1)])>Threshold):
    RUL_true = idx_true[1][0]

idx_pred = (torch.Tensor(output[:len(x_target)])<Threshold).nonzero().squeeze()  # search idx less than threshold
RUL_pred = idx_pred[0][0]                        # first entry is pred RUL
RUL_error = RUL_true - RUL_pred
# if positive value, earlier than true RUL; 
# negative value, later than true RUL    
RUL_relative_error = -RUL_error/RUL_true

print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | Accuracy: {acc:.2f}")
print(f"True RUL: {int(RUL_true)} | Prediction RUL: {int(RUL_pred)} | RUL Error: {int(RUL_error)}")
end_time = timer()
snapshot = tracemalloc.take_snapshot()
total_time = print_inference_time(start_time, end_time, device)
memuse_total = memory_stats(snapshot) + memuse_end_setup
print(f"Allocated memory {memuse_total}")

#input tensor path
input_path = "tensors/CS2_36/input_CS2_36_50.pt"
#input_path = "tensors/CS2_36/input_CS2_36_60.pt"
#input_path = "tensors/CS2_36/input_CS2_36_70.pt"
x_input = torch.load(input_path)
x_input = x_input.tolist()

import matplotlib as mpl
battery_name = 'CS2_36'
x = range(0, len(x_input) + len(x_target))

plt.subplot(1,4,2)
plt.title(f'Prediction for Battery {battery_name}')
plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.grid(True)
plt.plot(x[:-len(x_target)],
         x_input,
         "k-",
         label='Input data')
plt.plot(x[-len(x_target):],
         x_target,
         "k:",
         label='Ground Truth')
plt.plot(x[-len(x_target):],
         output[:len(x_target)],
         "b-",
         label='Predicted')
plt.legend()

results = {
      "model_name": battery_name,
      "mem_usage": memuse_total,
      "exec_time": total_time,
      "acc": acc,
      "mae": mae,
      "rmse": rmse,
      "rul_error": int(RUL_error),
      "RUL_relative_error": float(RUL_relative_error)
}

rul_preds_list[battery_name] = RUL_pred

# create a binary pickle file 
import pickle

f = open(f"{battery_name}.pkl","wb")
pickle.dump(results, f)
f.close()

############################################################################
#############################  CS2_37 ######################################
############################################################################


Rated_Capacity = 1.1
start_time = timer()
tracemalloc.start()
max_length = 1500
# Setting up device-agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# test tensor path
#test_path = "tensors/CS2_37/CS2_37_50.pt"
test_path = "tensors/CS2_37/CS2_37_60.pt"
#test_path = "tensors/CS2_37/CS2_37_70.pt"

# scaled input tensor path
#input_path = "tensors/CS2_37/scaledinput_CS2_37_50.pt"
input_path = "tensors/CS2_37/scaledinput_CS2_37_60.pt"
#input_path = "tensors/CS2_37/scaledinput_CS2_37_70.pt"

# target tensor path
#target_path = "tensors/CS2_37/target_CS2_37_50.pt"
target_path = "tensors/CS2_37/target_CS2_37_60.pt"
#target_path = "tensors/CS2_37/target_CS2_37_70.pt"

# model paths
model_save_path = "models/LSTMCALCE_128_37.pth"
#model_save_path = "models/LSTMCALCE_32_37.pth"

scaler = scaler = load(open('scaler.pkl', 'rb'))

hidden_size = 128
#hidden_size = 32

#train_period
train_period = 80
#prediction_period
prediction_period = 80

x_in = torch.load(test_path)
x_input = torch.load(input_path)
x_input = x_input.tolist()
x_target = torch.load(target_path)
x_target = x_target.tolist()

loadedmodel = EncoderDecoder(hidden_size = hidden_size)
loadedmodel.load_state_dict(torch.load(f=model_save_path,
                                        map_location=torch.device('cpu')))

loadedmodel.eval()
output = []
while len(x_input) <= max_length:
    predicted = loadedmodel.predict(x_in, prediction_period)
    pred_seq = predicted[:, -1, :].view(-1).tolist()   
    for i in range(len(pred_seq)):
        x_input.append([pred_seq[i]])
        output.append([pred_seq[i]])
    X = inference_window(x_input, train_period)
    X = np.array(X)
    X = np.squeeze(X)
    x_in = to_tensor(X)

output = scaler.inverse_transform(np.array(output).reshape(-1,1))
output = output.tolist()
acc = accuracy(torch.Tensor(x_target), torch.Tensor(output[:len(x_target)])) * 100
mae = mean_absolute_error(output[:len(x_target)], x_target)
rmse = sqrt(mean_squared_error(output[:len(x_target)], x_target))

# Computing RUL Error
Threshold = 0.7 * Rated_Capacity
idx_true = (torch.Tensor(x_target)<Threshold).nonzero().squeeze()
RUL_true = idx_true[0][0]

if(torch.Tensor(x_target[int(RUL_true + 1)])>Threshold):
    RUL_true = idx_true[1][0]

idx_pred = (torch.Tensor(output[:len(x_target)])<Threshold).nonzero().squeeze()  # search idx less than threshold
RUL_pred = idx_pred[0][0]                        # first entry is pred RUL
RUL_error = RUL_true - RUL_pred
# if positive value, earlier than true RUL; 
# negative value, later than true RUL    
RUL_relative_error = -RUL_error/RUL_true

print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | Accuracy: {acc:.2f}")
print(f"True RUL: {int(RUL_true)} | Prediction RUL: {int(RUL_pred)} | RUL Error: {int(RUL_error)}")
end_time = timer()
snapshot = tracemalloc.take_snapshot()
total_time = print_inference_time(start_time, end_time, device)
memuse_total = memory_stats(snapshot) + memuse_end_setup
print(f"Allocated memory {memuse_total}")

#input tensor path
input_path = "tensors/CS2_37/input_CS2_37_50.pt"
#input_path = "tensors/CS2_37/input_CS2_37_60.pt"
#input_path = "tensors/CS2_37/input_CS2_37_70.pt"
x_input = torch.load(input_path)
x_input = x_input.tolist()

import matplotlib as mpl
battery_name = 'CS2_37'
x = range(0, len(x_input) + len(x_target))

plt.subplot(1,4,3)
plt.title(f'Prediction for Battery {battery_name}')
plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.grid(True)
plt.plot(x[:-len(x_target)],
         x_input,
         "k-",
         label='Input data')
plt.plot(x[-len(x_target):],
         x_target,
         "k:",
         label='Ground Truth')
plt.plot(x[-len(x_target):],
         output[:len(x_target)],
         "b-",
         label='Predicted')
plt.legend()

results = {
      "model_name": battery_name,
      "mem_usage": memuse_total,
      "exec_time": total_time,
      "acc": acc,
      "mae": mae,
      "rmse": rmse,
      "rul_error": int(RUL_error),
      "RUL_relative_error": float(RUL_relative_error)
}
rul_preds_list[battery_name] = RUL_pred

# create a binary pickle file 
import pickle

f = open(f"{battery_name}.pkl","wb")
pickle.dump(results, f)
f.close()

############################################################################
#############################  CS2_38 ######################################
############################################################################

Rated_Capacity = 1.1
start_time = timer()
tracemalloc.start()
max_length = 1500
# Setting up device-agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# test tensor path
#test_path = "tensors/CS2_38/CS2_38_50.pt"
test_path = "tensors/CS2_38/CS2_38_60.pt"
#test_path = "tensors/CS2_38/CS2_38_70.pt"

# scaled input tensor path
#input_path = "tensors/CS2_38/scaledinput_CS2_38_50.pt"
input_path = "tensors/CS2_38/scaledinput_CS2_38_60.pt"
#input_path = "tensors/CS2_38/scaledinput_CS2_38_70.pt"

# target tensor path
#target_path = "tensors/CS2_38/target_CS2_38_50.pt"
target_path = "tensors/CS2_38/target_CS2_38_60.pt"
#target_path = "tensors/CS2_38/target_CS2_38_70.pt"

# model paths
model_save_path = "models/LSTMCALCE_128_38.pth"
#model_save_path = "models/LSTMCALCE_32_38.pth"

scaler = scaler = load(open('scaler.pkl', 'rb'))

hidden_size = 128
#hidden_size = 32

#train_period
train_period = 80
#prediction_period
prediction_period = 80

x_in = torch.load(test_path)
x_input = torch.load(input_path)
x_input = x_input.tolist()
x_target = torch.load(target_path)
x_target = x_target.tolist()

loadedmodel = EncoderDecoder(hidden_size = hidden_size)
loadedmodel.load_state_dict(torch.load(f=model_save_path,
                                        map_location=torch.device('cpu')))

loadedmodel.eval()
output = []
while len(x_input) <= max_length:
    predicted = loadedmodel.predict(x_in, prediction_period)
    pred_seq = predicted[:, -1, :].view(-1).tolist()   
    for i in range(len(pred_seq)):
        x_input.append([pred_seq[i]])
        output.append([pred_seq[i]])
    X = inference_window(x_input, train_period)
    X = np.array(X)
    X = np.squeeze(X)
    x_in = to_tensor(X)

output = scaler.inverse_transform(np.array(output).reshape(-1,1))
output = output.tolist()
acc = accuracy(torch.Tensor(x_target), torch.Tensor(output[:len(x_target)])) * 100
mae = mean_absolute_error(output[:len(x_target)], x_target)
rmse = sqrt(mean_squared_error(output[:len(x_target)], x_target))

# Computing RUL Error
Threshold = 0.7 * Rated_Capacity
idx_true = (torch.Tensor(x_target)<Threshold).nonzero().squeeze()
RUL_true = idx_true[0][0]

if(torch.Tensor(x_target[int(RUL_true + 1)])>Threshold):
    RUL_true = idx_true[1][0]

idx_pred = (torch.Tensor(output[:len(x_target)])<Threshold).nonzero().squeeze()  # search idx less than threshold
RUL_pred = idx_pred[0][0]                        # first entry is pred RUL
RUL_error = RUL_true - RUL_pred
RUL_relative_error = -RUL_error/RUL_true
# if positive value, earlier than true RUL; 
# negative value, later than true RUL    
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | Accuracy: {acc:.2f}")
print(f"True RUL: {int(RUL_true)} | Prediction RUL: {int(RUL_pred)} | RUL Error: {int(RUL_error)}")
end_time = timer()
snapshot = tracemalloc.take_snapshot()
total_time = print_inference_time(start_time, end_time, device)
memuse_total = memory_stats(snapshot) + memuse_end_setup
print(f"Allocated memory {memuse_total}")

#input tensor path
input_path = "tensors/CS2_38/input_CS2_38_50.pt"
#input_path = "tensors/CS2_38/input_CS2_38_60.pt"
#input_path = "tensors/CS2_38/input_CS2_38_70.pt"
x_input = torch.load(input_path)
x_input = x_input.tolist()

import matplotlib as mpl
battery_name = 'CS2_38'
x = range(0, len(x_input) + len(x_target))
font = {'size'   : 15}
mpl.rc('font', **font)

plt.subplot(1,4,4)
plt.title(f'Prediction for Battery {battery_name}')
plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.grid(True)
plt.plot(x[:-len(x_target)],
         x_input,
         "k-",
         label='Input data')
plt.plot(x[-len(x_target):],
         x_target,
         "k:",
         label='Ground Truth')
plt.plot(x[-len(x_target):],
         output[:len(x_target)],
         "b-",
         label='Predicted')
plt.legend()


results = {
      "model_name": battery_name,
      "mem_usage": memuse_total,
      "exec_time": total_time,
      "acc": acc,
      "mae": mae,
      "rmse": rmse,
      "rul_error": int(RUL_error),
      "RUL_relative_error": float(RUL_relative_error)
}
rul_preds_list[battery_name] = RUL_pred

# create a binary pickle file 
import pickle

f = open(f"{battery_name}.pkl","wb")
pickle.dump(results, f)
f.close()

plt.savefig('Multi-step_LSTM_CALCE.pdf')

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
    for leave_out in ['CS2_35','CS2_36','CS2_37','CS2_38']:
        # Display message
        lcd.clear()
        lcd.message(f'Cycles left for\n{leave_out}: {rul_preds_list[leave_out]}')
        sleep(5)
