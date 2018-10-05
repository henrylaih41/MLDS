import math
import plot1 as p
import numpy as np
import keras as k
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
epoch = 200
batch = 512
model1_name = str(epoch) + "_model1_1200"
model2_name = str(epoch) + "_model2_1200_6"
model3_name = str(epoch) + "_model3_1200_12"
model4_name = str(epoch) + "_model4_1200_26"

def target_function(x,a=4,n=1000):
    y = []
    for points in x:
        sum = 0
        for i in range(1,n):
            sum += math.sin(math.pi * (i ** a) * points)/(math.pi * (i ** a))
        y.append(sum)
    return y

def train_model1(x,y):
   model = k.Sequential()
   model.add(Dense(input_dim=1,units = 400,activation = 'relu'))
   model.add(Dense(units = 1, activation = 'linear'))
   model.compile(loss='mse',optimizer='adam',metrics=['mape'])
   model.summary()
   history = model.fit(x,y,batch_size = batch,epochs = epoch)
   return model,history.history['loss']

def train_model2(x,y):
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 20,activation = 'relu'))
    for _ in range(6):
        model.add(Dense(units = 13, activation = 'relu'))
    model.add(Dense(units = 1,activation = 'linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(x,y,batch_size = batch,epochs = epoch)
    return model,history.history['loss']

def train_model3(x,y):
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 18,activation = 'relu'))
    for _ in range(12):
        model.add(Dense(units = 9, activation = 'relu'))
    model.add(Dense(units = 1,activation = 'linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(x,y,batch_size = batch,epochs = epoch)
    return model,history.history['loss']

def train_model4(x,y):
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 15,activation = 'relu'))
    for _ in range(26):
        model.add(Dense(units = 6, activation = 'relu'))
    model.add(Dense(units = 1,activation = 'linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(x,y,batch_size = batch,epochs = epoch)
    return model,history.history['loss']

def output_func_points(func1,x,path):
    y = func1(x)
    y = np.array(y)
    np.save(path,y)

x = [i/10000 for i in range(10000)]
x = np.array(x)
y = np.load("train_data1.npy")

#model1,model1_history = train_model1(x,y)
#model1.save(model1_name)
#np.save("_his_" + model1_name,model1_history)
#model2,model2_history = train_model2(x,y)
#model2.save(model2_name)
#np.save("_his_" + model2_name,model2_history)
#model3,model3_history = train_model3(x,y)
#model3.save(model3_name)
#np.save("_his_" + model3_name,model3_history)
model4,model4_history = train_model4(x,y)
model4.save(model4_name)
np.save("_his_" + model4_name,model4_history)

p.plot_functions(x,y,model1_name,model2_name,model3_name,model4_name)
p.plot_loss("_his_" + model1_name + ".npy","_his_" + model2_name + ".npy"
           ,"_his_" + model3_name + ".npy","_his_" + model4_name + ".npy")