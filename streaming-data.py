import subprocess
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame, read_csv
from numpy import log
#from statistics import mean

from datetime import datetime
import time
 


from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger

log_dir = "2-layer-32-32-LeakyReLU-update"

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
#logger = TensorBoard(log_dir='log/{}'.format(log_dir), write_graph=True, histogram_freq=1, batch_size=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
csv_logger = CSVLogger('training-2-layer-32-32-LeakyReLU-update.csv')

previous_df = read_csv('memoryUsed9-final.csv')
previous_df.columns = ['timestamp', 'value']
previous_df.set_index('timestamp', inplace=True)
previous_df.dropna(inplace=True)
print(previous_df.head())
print(previous_df.shape)
previous_values = previous_df.values

last_input_value = previous_values[-2]
prev_value = previous_values[-1]

def difference_2(dataset, interval=1):
    print(dataset.ndim)
    print(dataset.shape)
    diff = np.ndarray(shape=(dataset.shape[0]-interval, dataset.shape[1]), dtype=float)
    for i in range(interval, len(dataset)):
        for j in range(dataset.shape[1]):
            diff[i-1][j] = dataset[i][j] - dataset[i - interval][j]
    return diff

diff_values = difference_2(previous_values)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(diff_values)

min_val = scaler.data_min_
max_val = scaler.data_max_


def mean(nums):
    return sum(nums, 0.0) / len(nums)

def streaming():
    counter = 0
    total_instance = 0
    total_mem = list()
    mem_max = 0
    while(counter != 5):
        print("in streaming loop with counter and time in minutes", counter, datetime.now().minute%5)
        instance_stat = subprocess.check_output("cf curl v2/apps/3bbffff8-6834-4383-8140-cd223d9927ee/stats | jq 'with_entries(.value = .value.stats)'", shell=True)
        instance_stat_json = json.loads(instance_stat)
        mem_quota = list()
        instances = list()
        mem_usages = list()
        #mem = 0 
        for instance in instance_stat_json:
            instances.append(int(instance))
            mem_usages.append(instance_stat_json[instance]['usage']['mem'])
            mem_quota.append(instance_stat_json[instance]['mem_quota'])
            #mem = mem + int(instance)*instance_stat_json[instance]['usage']['mem']
        total_mem.append(sum(mem_usages))
        total_instance = len(instances)
        mem_max = max(mem_quota)
        # Wait for 60 seconds
        time.sleep(60)
        counter = counter+1
        if datetime.now().minute % 5 == 0:
            break
    return mean(total_mem), total_instance

def difference(data, prev_val):
    return data-prev_value

def scale_dataset(data, min_val, max_val):
    return (data - min_val)/(max_val - min_val)

def inverse_scale(data, min_val, max_val):
    return data*(max_val - min_val) + min_val

def inverse_diff(data, last_val):
    return data + last_val


def predict(data):
    print("in prediction function with data", data)
    diff_data = difference(data)
    scaled_data = scale_dataset(diff_data)
    train_X = scaled_data.reshape(scaled_data.shape[0], 1, 1)
    yhat= model.predict(train_X, batch_size=1)
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[2])
    yhat_inv = inverse_scale(yhat)
    yhat_final = inverse_diff(yhat_inv, last_value)
    return yhat_final

def scale_decision(memory, instance):
    print("scaling decision function with memory and instance", memory, instance)

def update(data):
    print("update model with next data", data)


#model = load_model('mem0_5-stack-2-32-LeakyReLU.h5')

flag=1
# date_time = datetime.now()
# year, month, day = date_time.year, date_time.month, date_time.day
# hour, minute = date_time.hour, date_time.minute
# rem = minute % 5
# sleep 5-rem 
print("streaming starting from : ", datetime.now())
c=0
while(flag):
    print("in infinite loop with iteration: ", c)
    total_mem, instance_count =  streaming()
    predict(total_mem)
    scale_decision(total_mem, instance_count)
    update(total_mem)
    c = c+1


        
        
        






# history = model.fit(train_X, train_y, epochs=100, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[logger, reduce_lr, csv_logger])


# mem = subprocess.check_output("cf curl v2/apps/3bbffff8-6834-4383-8140-cd223d9927ee/stats | jq 'with_entries(.value = .value.stats.usage.mem)'", shell=True)
# json_mem = json.loads(mem)
# print(json_mem)
# instances = list()
# mem_usages = list()
# for instance in json_mem:
#     print(instance, json_mem[instance])
#     instances.append(int(instance))
#     mem_usages.append(json_mem[instance])

# print(instances)
# print(mem_usages)

# total_instance = len(instances)
# total_mem = sum(mem_usages)
# mem_per_instance = total_mem/total_instance

# print(total_instance)
# print(total_mem)
# print(mem_per_instance)

# instance_stat = subprocess.check_output("cf curl v2/apps/3bbffff8-6834-4383-8140-cd223d9927ee/stats | jq 'with_entries(.value = .value.stats)'", shell=True)
# instance_stat_json = json.loads(instance_stat)
# mem_quota = list()
# instances = list()
# mem_usages = list()
# for instance in instance_stat_json:
#     #print("instance", instance) 
#     #print("content", json_mem[instance])
#     #print("memory quota", instance_stat_json[instance]['mem_quota'])
#     #print("memory usage of the instance", instance_stat_json[instance]['usage']['mem'])
#     instances.append(int(instance))
#     mem_usages.append(instance_stat_json[instance]['usage']['mem'])
#     mem_quota.append(instance_stat_json[instance]['mem_quota'])

# print("instances list", instances)
# print("memory quota list", mem_quota)
# print("memory usage of the instances", mem_usages)

# total_instance = len(instances)
# total_mem = sum(mem_usages)
# mem_per_instance = total_mem/total_instance

# print("toatal number of instances of the app", total_instance)
# print("total memory consumption by all the app instances", total_mem)
# print("average memory consumption per app instance", mem_per_instance)







