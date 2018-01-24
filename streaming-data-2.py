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

log_dir = "streaming-update"

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
#logger = TensorBoard(log_dir='log/{}'.format(log_dir), write_graph=True, histogram_freq=1, batch_size=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
csv_logger = CSVLogger('training-2-layer-32-32-LeakyReLU-streaming-update.csv')


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
    diff_data = difference(data, last_val)
    scaled_data = scale_dataset(diff_data, min_val, max_val)
    train_X = scaled_data.reshape(scaled_data.shape[0], 1, 1)
    yhat= model.predict(train_X, batch_size=1)
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[2])
    yhat_inv = inverse_scale(yhat[0][0])
    yhat_final = inverse_diff(yhat_inv, last_val)
    return yhat_final

def scale_decision(memory, instance):
    print("scaling decision function with memory and instance", memory, instance)

def update(train_X, train_y):
    print("update model with next data", train_X, train_y)
    train_X = train_X.reshape(train_X.shape[0], 1, 1)
    train_y = train_y.reshape(train_X.shape[0], 1, 1)
    history = model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2, shuffle=False, callbacks=[early_stopping, reduce_lr, csv_logger])

    plt.plot(history.history['mean_squared_error'], label='mse')
    plt.plot(history.history['mean_absolute_error'], label='mae')
    plt.plot(history.history['mean_absolute_percentage_error'], label='mape')
    plt.legend(loc='upper right')
    #plt.savefig('accuracy-mem0_5-stack-2-32-LeakyReLU-mem6.png')
    plt.show()

def train():
    initial_data = np.ndarray(shape=(3,1), dtype=float)
    for i in range(3):
        initial_data[i][0], count = streaming()
    last_val = initial_data[2][0]
    data_diff = np.ndarray(shape=(2,1), dtype=float)
    data_diff[0][0] = initial_data[1][0]-initial_data[0][0]
    data_diff[1][0] = initial_data[2][0]-initial_data[1][0]

    min_val = min(data_diff)
    max_val = max(data_diff)

    data_scaled = np.ndarray(shape=(2,1), dtype=float)
    data_scaled = (data_diff - min_val)/(max_val - min_val)

    train_X = data_scaled[0][0]
    train_y = data_scaled[1][0]

    print("initial three data points", initial_data)
    print("data difference of initial data points", data_diff)
    print("scaled data of initial data points", data_scaled)
    print("first training input", train_x)
    print("first training output", train_y)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape(train_X.shape[0], 1, 1)
    train_y = train_y.reshape(train_y.shape[0], 1, train_y.shape[1])
    print(train_X.shape, train_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, train_X.shape[1], train_X.shape[2]), return_sequences=True, name='layer-1'))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=.001))
    # model.add(Activation('softmax'))
    model.add(LSTM(32, return_sequences=True, name='layer-2'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=.001))
    # model.add(Activation('softmax'))
    # model.add(Dropout(0.1))
    model.add(Dense(1, name='output-layer'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2, shuffle=False, callbacks=[early_stopping, reduce_lr, csv_logger])
    print(model.summary())
    plt.plot(history.history['mean_squared_error'], label='mse')
    plt.plot(history.history['mean_absolute_error'], label='mae')
    plt.plot(history.history['mean_absolute_percentage_error'], label='mape')
    plt.legend(loc='upper right')
    #plt.savefig('accuracy-memory-model.png')
    plt.show()
    return model, min_val, max_val, last_val

print("streaming starting from : ", datetime.now())

model, min_val, max_val, last_val = train()

print("minimum value, maximum value and last value", min_val, max_val, last_val)

total_mem, instance_count = streaming()
result = predict(total_mem)
total_mem_2, instance_count = streaming()

print("expected and predicted value", total_mem_2, result)

# c=0

# while(flag):
#     print("in infinite loop with iteration: ", c)
#     total_mem, instance_count =  streaming()
#     predict(total_mem)
#     scale_decision(total_mem, instance_count)
#     update(total_mem)
#     c = c+1

        
        
        












