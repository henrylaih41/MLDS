import math
import plot2 as p
import numpy as np
import keras as k
from keras.layers.core import Dense, Activation
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
epoch = 10
batch = 4096
model1_name = str(epoch) + "_model1"



def get_gradient_norm_func(model):
    grads = K.gradients(model.total_loss, model.trainable_weights)
    summed_squares = [K.sum(K.square(g)) for g in grads]
    norm = K.sqrt(sum(summed_squares))
    inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
    func = K.function(inputs, [norm])
    return func

def main():
    x = np.linspace(0,2*math.pi,50000)
    x = np.array(x)
    y = np.load("train_data1.npy")
    y = np.array(y)
    model = k.Sequential()
    model.add(Dense(input_dim=1,units = 3,activation = 'relu'))
    for _ in range(2):
        model.add(Dense(units = 5,activation = 'relu'))
    model.add(Dense(units = 1, activation = 'linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    gradient_norm = []
    loss = []
    for _ in range(epoch):
        get_gradient = get_gradient_norm_func(model)

        history = model.fit(x,y,batch_size = batch,epochs = 1)
        print(_)

        gradient_norm.append(get_gradient([x.reshape(50000,1), y.reshape(50000,1), np.ones(len(y)).reshape(50000,)]))
        loss.append(history.history['loss'])

    
    np.save("gradient_norm",gradient_norm)
    np.save("_his_" + model1_name,loss)
    p.plot_gradient_norm("gradient_norm.npy")
    p.plot_loss("_his_" + model1_name + ".npy")
    
        


if  __name__ == '__main__':
    main()
    


