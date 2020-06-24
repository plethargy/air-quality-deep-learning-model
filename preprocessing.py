
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math

import csv

import matplotlib.pyplot as plt

from keras.layers import Dropout
from keras import regularizers

train=pd.read_csv("./data/Train.csv")
test=pd.read_csv("./data/Test.csv")
sample_sub=pd.read_csv("./data/sample_sub.csv")



def convert_list_to_str(l):
    return ",".join(str(e) for e in l)


def replace_nan(x):
    if x=="nan":
        return np.nan
    else :
        return float(x)
features=["temp","precip","rel_humidity","wind_dir","wind_spd","atmos_press"]
for feature in features : 
    train[feature]=train[feature].apply(lambda x: [ replace_nan(X) for X in x.split(",")])
    test[feature]=test[feature].apply(lambda x: [ replace_nan(X)  for X in x.split(",")])  



print(len(train["temp"][1]))
print(train["temp"][1])


 

def clear_nan(x):
    count = 0
    total = 0
    for n in x:
        if not math.isnan(n):
            count += 1 
            total += n
    
    avg = total/count
    for n in range(len(x)):
        if (math.isnan(x[n])):
            x[n] = avg
    return [e for e in x]

data=pd.concat([train,test],sort=False).reset_index(drop=True)

for col_name in features:
    train[col_name]=train[col_name].apply(clear_nan)
    test[col_name]=test[col_name].apply(clear_nan)


for feature in features : 
    train[feature]=train[feature].apply(convert_list_to_str)
    test[feature]=test[feature].apply(convert_list_to_str) 



train.to_csv("train_clean.csv", index=False)
test.to_csv("test_clean.csv", index=False)



