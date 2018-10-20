import tensorflow as tf
import numpy as np
import math
from keras.utils import np_utils
import sys
# Objective: fitting a function (sinx) #
# Configuration #


batch_size = int(sys.argv[1])
model = "../models/tf_mnist2_" + str(batch_size) + ".ckpt"
epochs = 400
learning_rate = 0.0001
load = False
# Loading data #
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
num_data = x_train.shape[0]
batch_num = math.ceil(num_data/batch_size)
x_train, x_test = x_train/255.0, x_test/255.0
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
# Setup Grpah #
# Input and Output.
X = tf.placeholder(shape=[None,784], dtype=tf.float32, name="X")
Y = tf.placeholder(shape=[None,10], dtype=tf.float32, name="Y")


# Weight and bias.
W1_1 = tf.Variable(tf.truncated_normal([784, 80], stddev=0.1),name = "W1")  
B1_1 = tf.Variable(tf.truncated_normal([80], stddev=0.1),name = "B1")
W1_2 = tf.Variable(tf.truncated_normal([80, 80], stddev=0.1),name = "W2")  
B1_2 = tf.Variable(tf.truncated_normal([80], stddev=0.1),name = "B2")
W1_3 = tf.Variable(tf.truncated_normal([80, 100], stddev=0.1),name = "W3")  
B1_3 = tf.Variable(tf.truncated_normal([100], stddev=0.1),name = "B3")
W1_4 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1),name = "W4")  
B1_4 = tf.Variable(tf.truncated_normal([60], stddev=0.1),name = "B4")
W1_5 = tf.Variable(tf.truncated_normal([60,10], stddev=0.1),name = "W5")  
B1_5 = tf.Variable(tf.truncated_normal([10], stddev=0.1),name = "B5")

O1_1 = tf.nn.relu(tf.matmul(X, W1_1) + B1_1)
O1_2 = tf.nn.relu(tf.matmul(O1_1, W1_2) + B1_2)
O1_3 = tf.nn.relu(tf.matmul(O1_2, W1_3) + B1_3)
O1_4 = tf.nn.relu(tf.matmul(O1_3, W1_4) + B1_4)
pred_1 = tf.nn.softmax(tf.matmul(O1_4, W1_5) + B1_5)
loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_1, labels = Y)) 
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
correct_prediction = tf.equal(tf.argmax(pred_1, 1), tf.argmax(Y, 1))
acc_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
# Training #
with tf.Session() as sess:
    if(not load):
        sess.run(tf.global_variables_initializer())
    else:
        print(saver.restore(sess,model))
        
    for epoch in range(epochs):
        loss_sum, acc_sum = (0,0)
        for num in range(batch_num):
            batch = num * batch_size
            batch_x, batch_y =  x_train[batch : min(num_data,batch + batch_size)],\
                                y_train[batch : min(num_data,batch + batch_size)]
            batch_x = batch_x.reshape(-1,784)
            batch_y = batch_y.reshape(-1,10)
            _, loss, acc = sess.run([train_step,loss_1,acc_num],\
                                  {X: batch_x, Y: batch_y})
            loss_sum += loss
            acc_sum += acc
        print("Epoch: ", epoch,"Loss: ", loss_sum/batch_num, "Acc: ",acc_sum/float(num_data))
    print(saver.save(sess,model))

print("done")
