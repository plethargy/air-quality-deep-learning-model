
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import *
from keras.optimizers import Adam
import numpy as np
import math
import statistics

from keras import callbacks
from keras.metrics import mean_absolute_percentage_error

import csv

import matplotlib.pyplot as plt

from keras.layers import Dropout
from keras import regularizers

train=pd.read_csv("./data/train_clean.csv")
test=pd.read_csv("./data/test_clean.csv")
sample_sub=pd.read_csv("./data/sample_sub.csv")

scaler = preprocessing.StandardScaler()

def scale_values(values):
    scale = scaler.fit(values)
    return scale.transform(values)


def convert_list_to_str(l):
    return ",".join(str(e) for e in l)

def convert_to_image(row):
    image = []
    image.append((row["temp"]))
    image.append((row["precip"]))
    image.append((row["rel_humidity"]))
    image.append((row["wind_dir"]))
    image.append((row["wind_spd"]))
    image.append((row["atmos_press"]))

    return np.array(scale_values(image))

def get_mean(arr):
    return round(statistics.mean(arr), 5)

def get_std_dev(arr):
    return round(statistics.stdev(arr), 5)

def replace_nan(x):
    if x=="nan":
        return np.nan
    else :
        return float(x)
features=["temp","precip","rel_humidity","wind_dir","wind_spd","atmos_press"]
for feature in features : 
    train[feature]=train[feature].apply(lambda x: [ replace_nan(X) for X in x.split(",")])
    test[feature]=test[feature].apply(lambda x: [ replace_nan(X)  for X in x.split(",")])  

train_images = []
train_output = []
val_images = []
val_output = []

rowcount = 0
for index, row in train.iterrows():
    image = convert_to_image(row)
    if rowcount % 4 == 0:
        val_images.append(image)
        val_output.append(row["target"])
    else:
       train_images.append(image)
       train_output.append(row["target"])
    rowcount += 1

for x in range(len(train_images)):
    train_images[x] = train_images[x].reshape(121,6,1)

for x in range(len(val_images)):
    val_images[x] = val_images[x].reshape(121,6,1)

train_images = np.array(train_images)
val_images = np.array(val_images)


model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(121, 6,1)))
model.add(Flatten())
model.add(BatchNormalization(axis=-1))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation='linear'))
opt = Adam(lr=1e-3, decay=1e-3 / 200)

model.compile(loss="mse", optimizer=opt, metrics=['mse', 'mae'])


print("training model...")
generalRMSE = []
trainRMSE = []
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=30)
for x in range(10):

    histo = model.fit(x=train_images, y=train_output, validation_data=(val_images, val_output), epochs=200, batch_size=50, verbose=2, callbacks=[early_stopping])
    
    hist = pd.DataFrame(histo.history)
    
    generalRMSE.append(math.sqrt(hist["val_loss"].values[len(hist) - 1]))
    trainRMSE.append(math.sqrt(hist["loss"].values[len(hist) - 1]))

f = open("cnn_results.txt", "w")
f.write("Average General RMSE: " + str(get_mean(generalRMSE)))
f.write("Average Train RMSE: " + str(get_mean(trainRMSE)))
f.write("General StdDev: " + str(get_std_dev(generalRMSE)))
f.write("Train StdDev: " + str(get_std_dev(trainRMSE)))
f.close()

