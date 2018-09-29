import math
import plot as p
import numpy as np
import keras as k
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
epoch = 500
batch = 4096
model1_name = str(epoch) + "_model1"

### inputs a x array, outputs the corresponding y array
def target_function(x):
    y = []
    for points in x:
        y.append(math.sin(points))
    return y

### inputs the training data, returns the trained model and loss history.
def train_model1(x,y):
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 3,activation = 'relu'))
    for _ in range(2):
        model.add(Dense(units = 5,activation = 'relu'))
    model.add(Dense(units = 1, activation = 'linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(x,y,batch_size = batch,epochs = epoch)
    return model,history.history['loss']

### inputs the func and x array, saves the y array to path. 
def output_func_points(func1,x,path):
    y = func1(x)
    y = np.array(y)
    np.save(path,y)

### 50000 points between [0,2pi]
x = np.linspace(0,2*math.pi,50000)
### outputs the y points and save it (only need to run it once)
#output_func_points(target_function,x,"train_data1")
y = np.load("train_data1.npy")

### training
model1,model1_history = train_model1(x,y)
model1.save(model1_name)
np.save("_his_" + model1_name,model1_history)

### plotting
p.plot_functions(x,y,model1_name)
p.plot_loss("_his_" + model1_name + ".npy")