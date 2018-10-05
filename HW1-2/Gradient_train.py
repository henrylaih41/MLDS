import tensorflow as tf
from tensorflow.nn import relu
import numpy as np
import math
# Objective: fitting a function (sinx) #
# Configuration #
model = "./models/tf_m0.ckpt"
batch_size = 50000
epochs = 3000
learning_rate = 0.005
num_data = 50000
batch_num = math.ceil(num_data/batch_size)


# Loading data #
# 50000 points between [0,2pi]
train_x = np.linspace(0,2*math.pi,50000)
train_y = np.load("train_data1.npy")


# Input and Output.
X = tf.placeholder(shape=[None,1], dtype=tf.float32, name="X")
Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name="Y")

# Weight and bias.
W1 = tf.Variable(tf.truncated_normal([1, 3], stddev=0.1))  
B1 = tf.Variable(tf.truncated_normal([3], stddev=0.1))  
W2 = tf.Variable(tf.truncated_normal([3, 5], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([5], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([5, 5], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([5], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([5, 1], stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([1], stddev=0.1))

# Making a prediction and comparing it to the true output
XX = tf.reshape(X,[-1,1])
O1 = relu(tf.matmul(XX, W1) + B1)
O2 = relu(tf.matmul(O1, W2) + B2)
O3 = relu(tf.matmul(O2, W3) + B3)
pred = tf.matmul(O3, W4) + B4
YY = tf.reshape(Y,[-1,1])
loss_mse = tf.losses.mean_squared_error(YY,pred)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_mse)

# Training #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        loss_sum = 0
        for num in range(batch_num):
            batch = num * batch_size
            batch_x, batch_y =  train_x[batch : min(num_data,batch + batch_size)], \
                                train_y[batch : min(num_data,batch + batch_size)]
            batch_x = batch_x.reshape(-1,1)
            batch_y = batch_y.reshape(-1,1)
            _, loss = sess.run([train_step,loss_mse],\
                                  {X: batch_x, Y: batch_y})
            loss_sum = loss_sum + loss
        print("Epoch: ", epoch, "Loss: ", loss_sum/batch_num)
    
    # Saving #
    saver = tf.train.Saver()
    print(saver.save(sess,model))

print("done")

    
  
