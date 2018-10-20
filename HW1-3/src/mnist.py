import tensorflow as tf
import numpy as np
import math
from keras.utils import np_utils
import sys

batch_size = 1024
epochs = 200
learning_rate = 0.0001



##load_data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_data=x_train.shape[0]
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
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
correct_prediction = tf.equal(tf.argmax(pred_1, 1), tf.argmax(Y, 1))
acc_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))






batch_num = math.ceil(num_data/batch_size)
print(num_data)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        loss_sum = 0
        acc_sum = 0
        for num in range(batch_num):
            batch = num * batch_size
            batch_x, batch_y =  x_train[batch : min(num_data,batch + batch_size)], y_train[batch : min(num_data,batch + batch_size)]
            batch_x = batch_x.reshape(-1,784)
            batch_y = batch_y.reshape(-1,10)
            _, loss, acc = sess.run([train_step,loss_1,acc_num],\
                                  {X: batch_x, Y: batch_y})
            loss_sum += loss
            acc_sum+= acc
        print("Epoch: ", epoch,"Loss: ", loss_sum, "Acc: ",acc_sum/float(num_data))
    if len(sys.argv)>1:
        par1=sess.run(total_par)
        np.save("./"+sys.argv[1],par1)



 

##set_model
#assign_par=total_par.ass
