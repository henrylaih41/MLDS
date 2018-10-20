import tensorflow as tf
import numpy as np
import math
from keras.utils import np_utils
import matplotlib.pyplot as plt
import sys
import math

##load_data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_data=x_train.shape[0]
x_train=x_train.reshape(-1,784)
x_test=x_test.reshape(-1,784)
y_train=np_utils.to_categorical(y_train, 10)
y_test=np_utils.to_categorical(y_test, 10)

X = tf.placeholder(shape=[None,784], dtype=tf.float32, name="X")
Y = tf.placeholder(shape=[None,10], dtype=tf.float32, name="Y")

##first model
W1_1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))  
B1_1 = tf.Variable(tf.truncated_normal([300], stddev=0.1))
W1_2 = tf.Variable(tf.truncated_normal([300, 200], stddev=0.1))  
B1_2 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
W1_3 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))  
B1_3 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
W1_4 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))  
B1_4 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
W1_5 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))  
B1_5 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
W1_6 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))  
B1_6 = tf.Variable(tf.truncated_normal([100], stddev=0.1))
W1_7 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.1))  
B1_7 = tf.Variable(tf.truncated_normal([50], stddev=0.1))
W1_8 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))  
B1_8 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

total_par=[W1_1,B1_1,W1_2,B1_2,W1_3,B1_3,W1_4,B1_4,W1_5,B1_5,W1_6,B1_6,W1_7,B1_7,W1_8,B1_8]


O1_1 = tf.nn.relu(tf.matmul(X, W1_1) + B1_1)
O1_2 = tf.nn.relu(tf.matmul(O1_1, W1_2) + B1_2)
O1_3 = tf.nn.relu(tf.matmul(O1_2, W1_3) + B1_3)
O1_4 = tf.nn.relu(tf.matmul(O1_3, W1_4) + B1_4)
O1_5 = tf.nn.relu(tf.matmul(O1_4, W1_5) + B1_5)
O1_6 = tf.nn.relu(tf.matmul(O1_5, W1_6) + B1_6)
O1_7 = tf.nn.relu(tf.matmul(O1_6, W1_7) + B1_7)
pred_1 = tf.nn.softmax(tf.matmul(O1_7, W1_8) + B1_8)
loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_1, labels = Y)) 
correct_prediction = tf.equal(tf.argmax(pred_1, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
grad = tf.gradients(loss_1,X)
gradd = tf.convert_to_tensor(grad)
sensitivity = tf.reduce_sum(tf.multiply(gradd,gradd))
if len(sys.argv)==1:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        par1=np.load("60000.npy")
        batch=60000
        train_loss_arr = []
        train_acc_arr = []
        val_loss_arr = []
        val_acc_arr = []
        for i,weights in enumerate(par1):
            assign=total_par[i].assign(weights)
            sess.run(assign)
        
        x_haha=np.array(x_test)
        x_tt=np.array(x_train)
        y_haha=np.array(y_test)
        y_tt=np.array(y_train)
        totalx_train=np.concatenate((x_haha,x_tt),axis=0)
        totaly_train=np.concatenate((y_haha,y_tt))
        summ = []
        for i,_ in enumerate(totalx_train) :
            print(i)
            seperate=sess.run(sensitivity,{X: [totalx_train[i]], Y: [totaly_train[i]]})
            summ.append(seperate)
        train_acc, train_loss = sess.run([acc,loss_1],{X: x_train, Y: y_train})
        val_acc, val_loss = sess.run([acc,loss_1],{X: x_test, Y: y_test})
        a=math.sqrt(sum(summ)/len(summ))


        #np.save("./total",[[batch,a,train_acc,val_acc,train_loss,val_loss]])
        print(a)
        totall=np.load("total.npy")
        temp=np.array([[batch,a,train_acc,val_acc,train_loss,val_loss]])
        b=np.append(totall,temp,axis=0)
        print(b)
        np.save("./total",b)

    
if len(sys.argv)>1:
    b=np.load("total.npy")
    if sys.argv[1]=="acc":
        axis=b[:,0]
        plt.plot(axis,b[:,3],'b:')
        plt.plot(axis,b[:,2],'b')
        plt.plot(axis,b[:,1],'r')
        plt.xscale('log')
        plt.legend(['test_acc', 'train_acc','sensitivity'], loc='upper right')
        plt.show()
    elif sys.argv[1]=="loss" :
        axis=b[:,0]
        plt.plot(axis,b[:,4],'b:')
        plt.plot(axis,b[:,5],'b')
        plt.plot(axis,b[:,1],'r')
        plt.xscale('log')
        plt.legend(['test_loss', 'train_loss','sensitivity'], loc='upper right')
        plt.show()