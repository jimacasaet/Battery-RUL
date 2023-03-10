# -*- coding: utf-8 -*-
"""inputprofileprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DoVUEfdFtiu4YYi7oTpFrWTMdehCfiER
"""

# Import modules needed for data preprocessing
import numpy as np
import random
import math
import os
import sys
import copy 
import scipy.io
import logging

from math import sqrt
from datetime import datetime


class DataProcessing():
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    # function to get 31-Dim Input from Battery to be used for training/testing
    def MultiChannelIP(self, path, battery_names):
        Battery_names = battery_names
        path = path

        self.logger.info("Loading data...")
        Batt_Val = {}
        Charge_Data = {}
        for name in Battery_names:
            print('Loading Dataset ' + name + '.mat...')
            file_path = path + name + '.mat'
            data = self.loadMat_to_Dict(file_path)
            Batt_Val[name] = self.getDischargeCapacity(data)
            Charge_Data[name] = self.getChargeData(data)

        self.logger.info("Cleaning and Normalizing Charge Profile")
        Charge_Profile = self.NormalizeData(Charge_Data=Charge_Data,
                                       Battery_names=Battery_names)

        self.logger.info("Sampling Charging Profiles...")
        Voltage_Measured = self.ChargeDataSampling(Charge_Data=Charge_Profile,
                                                   Battery_names=Battery_names,
                                                   param_type='Voltage_measured',
                                                   sample_size=10)
        Current_Measured = self.ChargeDataSampling(Charge_Data=Charge_Profile,
                                                   Battery_names=Battery_names,
                                                   param_type='Current_measured',
                                                   sample_size=10)
        Temperature_Measured = self.ChargeDataSampling(Charge_Data=Charge_Profile,
                                                       Battery_names=Battery_names,
                                                       param_type='Temperature_measured',
                                                       sample_size=10)

        """
        self.logger.info("Normalizing Sampled Charging Profiles...")
        Temperature_Measured = self.NormalizeData(ChargingProfile=Temperature_Measured,
                                                  Battery_names=Battery_names)
        Current_Measured = self.NormalizeData(ChargingProfile=Current_Measured,
                                              Battery_names=Battery_names)
        Voltage_Measured = self.NormalizeData(ChargingProfile=Voltage_Measured,
                                              Battery_names=Battery_names)
        """

        self.logger.info("Normalizing Capacity Degradation Data...")
        Capacity = self.NormalizeCapacityData(Batt_Val, Battery_names)

        self.logger.info("Producing 31-Dimension Input...")
        Input_Data = self.concatCPandCap(Voltage_Measured=Voltage_Measured,
                                         Current_Measured=Current_Measured,
                                         Temperature_Measured=Temperature_Measured,
                                         Capacity=Capacity,
                                         Battery_names=Battery_names)
        return Input_Data, Capacity, Batt_Val

    # function to load the .mat data and transform to dict
    def loadMat_to_Dict(self, matfile):
        raw_data = scipy.io.loadmat(matfile)

        # get filename
        filename = matfile.split("/")[-1].split(".")[0]

        col = raw_data[filename]
        # getting 'data'
        col = col[0][0][0][0]
        col_size = col.shape[0]

        data = []
        for i in range(col_size):
            labels = list(col[i][3][0].dtype.fields.keys())
            dict1, dict2 = {}, {}
            if str(col[i][0][0]) != 'impedance':
                for j in range(len(labels)):
                    arr_val = col[i][3][0][0][j][0]
                    val = [arr_val[m] for m in range(len(arr_val))]
                    dict2[labels[j]] = val

            dict1['type'], dict1['temp'], dict1['time'], dict1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(self.datestring_to_time(col[i][2][0])), dict2
            data.append(dict1)

        return data
    
    # convert time_string to datatime
    def datestring_to_time(self, dates):
        year, month, day, hour, minute, second = int(dates[0]), int(dates[1]), int(dates[2]), int(dates[3]), int(dates[4]), int(dates[5])
        return datetime(year=year,
                        month=month,
                        day=day,
                        hour=hour,
                        minute=minute,
                        second=second)

    # function to get capacity data
    def getDischargeCapacity(self, battery):
        cycle, capacity = [], []
        i = 1
        for bat in battery:
            if bat['type'] == 'discharge':
                capacity.append(bat['data']['Capacity'][0])
                cycle.append(i)
                i += 1
        return [cycle, capacity]

    # function to get the charge data of the battery
    def getChargeData(self, battery, type='charge'):
        data = []
        for bat in battery:
            if bat['type'] == type:
                data.append(bat['data'])
        return data

    def ChargeDataSampling(self, Charge_Data, Battery_names, param_type, sample_size):
        sampled_dict = {}
        for name in Battery_names:
            df = copy.deepcopy(Charge_Data[name])
            # remove first and last charging cycle due to few data points
            #df.pop(0) # first list element
            #df.pop(-1) # second list element

            samples = []
            for i in range(len(df)):
                raw_list = list(df[i][param_type])
                sampled_list = self.SystematicSampling(raw_list, sample_size)
                samples.append(sampled_list)
        
            sampled_dict[name] = samples
        return sampled_dict

    # function to get charging profile samples using systemating sampling
    def SystematicSampling(self, a_list, sample_size):
        population = len(a_list)
        step = math.ceil(population/sample_size)
        selected_index = np.arange(1, population, step)
        samples = []
        for i in range(len(selected_index)):
            selected_sample = a_list[selected_index[i]]
            samples.append(selected_sample)
        return samples

    # Normalization function of charging profile parameters
    def NormalizeData(self, Charge_Data, Battery_names):
        Charge_Profile = {}
        parameters = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
        #min_max_scaler = preprocessing.MinMaxScaler()
        # call function to clean charge profile [remove first and last elements due to insufficient data points]
        df = self.CleanCharge_Profile(Charge_Data, Battery_names, parameters)
    
        # call function to find max, min values across all datasets
        for params in parameters:
            oa_max, oa_min = self.FindMinMax(df, Battery_names, params)
            print(f"Max[{params}]: {oa_max} | Min[{params}]: {oa_min}")
            for name in Battery_names:
                for i in range(len(df[name])):
                    for j in range(len(df[name][i][params])):
                        df[name][i][params][j] = (df[name][i][params][j] - oa_min) / (oa_max - oa_min)
        print("Normalization Done!")
        return df

    def FindMinMax(self, df, Battery_names, params):
        final_max = -(sys.maxsize)
        final_min = sys.maxsize
        check_max = -(sys.maxsize)
        check_min = sys.maxsize
        #index_max = 0
        #index_min = 0
        for name in Battery_names:
            for i in range(len(df[name])):
                set_max, set_min = np.nanmax(df[name][i][params]), np.nanmin(df[name][i][params])
                if set_max > check_max:
                    check_max = set_max
                    #index_max = i
                if set_min < check_min:
                    check_min = set_min
                    #index_min = i
                #print(f" Battery[{name}] idx: {i} | Max: {set_max} | Min: {set_min}")
        
            if check_max > final_max:
                final_max = check_max
            
            if check_min < final_min:
                final_min = check_min
        
            #print(f"Dataset: {name} | Max[{index_max}]: {check_max} | Min[{index_min}]: {check_min}")
        return final_max, final_min

    def CleanCharge_Profile(self, Charge_Data, Battery_names, parameters):
        df = copy.deepcopy(Charge_Data)
        for name in Battery_names:
            df[name].pop(32)  # drop index 32 measurement exhibit unusual behavior
            df[name].pop(-1) # drop last index; few datapoints
            print(f"Length of {name}: {len(df[name])}")
        
            for params in parameters:
                for i in range(len(df[name])):
                    # drop nan values
                    df[name][i][params] = [val for val in df[name][i][params] if not(math.isnan(val)) == True]
        print("Done Cleaning Data!")
        return df
    """
    def NormalizeData(self, Charge_Data, Battery_names):
        Charge_Profile = {}
        parameters = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
        for name in Battery_names:
            df = copy.deepcopy(Charge_Data[name])
            # remove the first and last charging cycles due to insufficient data points
            df.pop(0)   # pops out first element
            df.pop(-1)  # pops out last element
            # get length 
            len_df = len(df)
            for i in range(len_df):
                for params in parameters:
                    cp_val = df[i][params]
                    cp_max, cp_min = max(cp_val), min(cp_val)
                    for j in range(len(df[i][params])):
                        df[i][params][j] = (df[i][params][j] - cp_min) / (cp_max - cp_min)
            Charge_Profile[name] = df
        return Charge_Profile
    
    
    # Normalization function for charging profile parameters
    def NormalizeData(self, ChargingProfile, Battery_names):
        df = ChargingProfile
        for name in Battery_names:
            # get max, and min value
            cp_max, cp_min = self.FindMinMax(measured_params=df[name])
            for i in range(len(df[name])):
                for j in range(10):
                    df[name][i][j] = (df[name][i][j] - cp_min) / (cp_max - cp_min)
        return df

    # function to find min,max to be used for normalization
    def FindMinMax(self, measured_params):
        final_max = 0 # arbitrary values
        final_min = 100 # arbitrary values
        for i in range(len(measured_params)):
            max_val = max(measured_params[i])
            min_val = min(measured_params[i])

            if max_val > final_max:
                final_max = max_val
        
            if min_val < final_min:
                final_min = min_val
    
        return final_max, final_min

    # Normalization function for Capacity Degradation values
    def NormalizeCapacityData(self, Batt_Val, Battery_names):
        capacity = {}
        cap_maxmin = {}
        for name in Battery_names:
            rated_cap = 2.0
            df = Batt_Val[name][1]
            c_peaks = {}
            cap_max, cap_min = max(df), min(df)
            for i in range(len(df)):
                if df[i] > rated_cap:
                    df[i] = rated_cap
                df[i] = (df[i] - cap_min) / (rated_cap - cap_min)
        
            capacity[name] = df
            c_peaks['capacity_max'], c_peaks['capacity_min'] = rated_cap, cap_min
            cap_maxmin[name] = c_peaks
    
        return capacity, cap_maxmin
    """
    # Normalization function for Capacity Degradation Values
    def NormalizeCapacityData(self, Batt_Val, Battery_names):
        capacity = {}
        cap_max = -(sys.maxsize)
        cap_min = sys.maxsize
        for name in Battery_names:
            step_max, step_min = np.nanmax(Batt_Val[name][1]), np.nanmin(Batt_Val[name][1])
            print(f"Battery[{name}] | Max Cap: {step_max} | Min Cap: {step_min}")
            if step_max > cap_max:
                cap_max = step_max
            if step_min < cap_min:
                cap_min = step_min
        print(f"Capacity Max: {cap_max} | Min: {cap_min}")
            
        for name in Battery_names:
            df = copy.deepcopy(Batt_Val[name][1])
            c_peaks = {}
            for i in range(len(df)):
                #if df[i] > rated_cap:
                #    df[i] = rated_cap
                df[i] = (df[i] - cap_min) / (cap_max - cap_min)
        
            capacity[name] = df
        print("Capacity Data Normalization Complete!")    
        return capacity

    """
    # Normalization function for Capacity Degradation values
    def NormalizeCapacityData(self, Batt_Val, Battery_names):
        capacity = {}
        cap_maxmin = {}
        for name in Battery_names:
            #rated_cap = 2.2
            df = copy.deepcopy(Batt_Val[name][1])
            c_peaks = {}
            cap_max, cap_min = max(df), min(df)
            for i in range(len(df)):
                #if df[i] > rated_cap:
                #    df[i] = rated_cap
                df[i] = (df[i] - cap_min) / (cap_max - cap_min)
        
            capacity[name] = df
            c_peaks['capacity_max'], c_peaks['capacity_min'] = cap_max, cap_min
            cap_maxmin[name] = c_peaks
    
        return capacity, cap_maxmin
    """    
    # Function to produce Inputs for training/testing
    def concatCPandCap(self, Voltage_Measured, Current_Measured, Temperature_Measured, Capacity, Battery_names):
        Input31Dim = {}
        for name in Battery_names:
            V = Voltage_Measured[name]
            V.pop(-1)
            C = Current_Measured[name]
            C.pop(-1)
            T = Temperature_Measured[name]
            T.pop(-1)
            Cap = Capacity[name]
            Cap.pop(-1)
        
            inputs = []
            for i in range(len(V)):
                onediminput = [*V[i], *C[i], *T[i], Cap[i]]
                inputs.append(onediminput)

            Input31Dim[name] = inputs
        return Input31Dim

    def Denormalize(output, cap_maxmin, name):
        denormal_capacity = []
        for i in range(len(output)):
            cap = cap_maxmin[name]['capacity_min'] + output[i]*(cap_maxmin[name]['capacity_max'] - cap_maxmin[name]['capacity_min'])
            denormal_capacity.append(cap)
        return denormal_capacity


