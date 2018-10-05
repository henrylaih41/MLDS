from math import sin
import plot2 as p
import numpy as np
import keras as k
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
epoch = 10000
batch = 4096
model1_name = str(epoch) + "_model1_600"
model2_name = str(epoch) + "_model2_600_4"
model3_name = str(epoch) + "_model3_600_7"
model4_name = str(epoch) + "_model4_600_11"

def target_function(x):
    y = []
    for points in x:
        value = sin(2*sin(2*sin(2*sin(points))))
        y.append(value)
    return y

def train_model1(x,y):
   model = k.Sequential()
   model.add(Dense(input_dim=1,units = 200,activation = 'relu'))
   model.add(Dense(units = 1, activation = 'linear'))
   model.compile(loss='mse',optimizer='adam',metrics=['mape'])
   model.summary()
   history = model.fit(x,y,batch_size = batch,epochs = epoch)
   return model,history.history['loss']

def train_model2(x,y):
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 20,activation = 'relu'))
    for _ in range(4):
        model.add(Dense(units = 10, activation = 'relu'))
    model.add(Dense(units = 1,activation = 'linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(x,y,batch_size = batch,epochs = epoch)
    return model,history.history['loss']

def train_model3(x,y):
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 20,activation = 'relu'))
    for _ in range(7):
        model.add(Dense(units = 8, activation = 'relu'))
    model.add(Dense(units = 1,activation = 'linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(x,y,batch_size = batch,epochs = epoch)
    return model,history.history['loss']

def train_model4(x,y):
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 20,activation = 'relu'))
    for _ in range(11):
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

x = np.linspace(-0.2, 6.5, 65000)
y = np.load("train_data2.npy")

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