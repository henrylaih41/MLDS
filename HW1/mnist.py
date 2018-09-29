import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

import argparse
import os, sys

def load_data():
    (x_train, y_train),(_,_)=mnist.load_data()
    number=10000
    x_train=x_train[0:number]
    y_train=y_train[0:number]
    x_train=x_train.reshape(number,28*28)
    x_train=x_train.astype('float32')

    y_train = np_utils.to_categorical(y_train,10)
    x_train=x_train/255
    return (x_train,y_train)

def DNNtrain(hidden_layer,dim_size,x_train,y_train):
    model = Sequential()
    model.add(Dense(input_dim=28*28,units=dim_size,activation='relu'))
    for _ in range(hidden_layer):
        model.add(Dense(units=dim_size,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history=model.fit(x_train,y_train,epochs=70,batch_size=128)
    return history

def CNNtrain(hidden_layer,dim_size,x_train,y_train,maxpooling):
    model = Sequential()
    model.add(Conv2D(filters=dim_size,kernel_size=(3,3),input_shape=(28,28,1)))
    if maxpooling:
        model.add(MaxPooling2D(pool_size=(2,2)))
    for _ in range(hidden_layer-1):
        model.add(Conv2D(filters=dim_size,kernel_size=(3,3)))
        if maxpooling:
            model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history=model.fit(x_train,y_train,epochs=70,batch_size=256)
    #history=1
    model.summary()
    return history

def main(opts):
    (x_train,y_train)=load_data()
    if opts.DNN :
        history1= DNNtrain(4,128,x_train,y_train)
        history2= DNNtrain(0,200,x_train,y_train)
        plt.plot(history1.history['loss'])
        plt.plot(history2.history['loss'])
        plt.legend(['1', '2'], loc='upper left')
        plt.title('model loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()
    elif opts.CNN :
        x_train=np.reshape(x_train,(x_train.shape[0],28,28,1))
        if opts.booling:
            history1=CNNtrain(1,8,x_train,y_train,True)
            history2=CNNtrain(2,27,x_train,y_train,True)
            history3=CNNtrain(1,64,x_train,y_train,True)
            history4=CNNtrain(3,77,x_train,y_train,True)
        else:
            history1=CNNtrain(1,8,x_train,y_train,False)
            history2=CNNtrain(2,11,x_train,y_train,False)
            history3=CNNtrain(1,64,x_train,y_train,False)
            history4=CNNtrain(3,71,x_train,y_train,False)
        plt.figure()
        plt.plot(history1.history['acc'])
        plt.plot(history2.history['acc'])
        plt.plot(history3.history['acc'])
        plt.plot(history4.history['acc'])
        plt.legend(['cnn_hidden1_dim8', 'cnn_hidden2_dim27','cnn_hidden1_dim64','cnn_hidden3_dim71'], loc='lower right')
        plt.title('train acc')
        plt.ylabel('acc')
        plt.xlabel('epochs')
        plt.figure()
        plt.plot(history1.history['loss'])
        plt.plot(history2.history['loss'])
        plt.plot(history3.history['loss'])
        plt.plot(history4.history['loss'])
        plt.legend(['cnn_hidden1_dim8', 'cnn_hidden2_dim27','cnn_hidden1_dim64','cnn_hidden3_dim71'], loc='upper right')
        plt.title('train loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.show()
    else :
        print("err input") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--DNN', action='store_true', default=False,dest='DNN', help='Input --DNN to run DNN')
    group.add_argument('--CNN', action='store_true',default=False,dest='CNN', help='Input --CNN to run CNN')
    parser.add_argument('--booling', type=bool, default=True, dest='booling', help='Wanting booling or not')
    opts = parser.parse_args()
    main(opts)

    