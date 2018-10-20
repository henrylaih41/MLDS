import tensorflow as tf
import numpy as np
import math
from keras.utils import np_utils
# Objective: fitting a function (sinx) #
# Configuration #
batch_size = 2048
model = "../models/tf_CIFAR1_" + str(batch_size) + ".ckpt"
epochs = 200
learning_rate = 0.005
load = False
# Loading data #
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
num_data = x_train.shape[0]
batch_num = math.ceil(num_data/batch_size)
x_train, x_test = x_train/255.0, x_test/255.0
# Setup Grpah #
# Input and Output.
X = tf.placeholder(shape=[None,32,32,3], dtype=tf.float32, name="X")
Y = tf.placeholder(shape=[None,10], dtype=tf.float32, name="Y")


# Convolution layers.
conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 4], mean=0, stddev=0.1))
conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 4, 8], mean=0, stddev=0.1))
conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 8, 16], mean=0, stddev=0.1))
conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 32], mean=0, stddev=0.1))

### First Layer
conv1 = tf.nn.conv2d(X, conv1_filter, strides=[1,1,1,1], padding='SAME')
### Activation
conv1 = tf.nn.relu(conv1)
### Maxpooling
conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
### Batch_normalization
conv1_bn = tf.layers.batch_normalization(conv1_pool)
### Second Layer
conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
conv2 = tf.nn.relu(conv2)
conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
conv2_bn = tf.layers.batch_normalization(conv2_pool)
### Third Layer
conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
conv3 = tf.nn.relu(conv3)
conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
conv3_bn = tf.layers.batch_normalization(conv3_pool)
### Fourth Layer 
conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
conv4 = tf.nn.relu(conv4)
conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
conv4_bn = tf.layers.batch_normalization(conv4_pool)
### Flatten
flat_para = tf.contrib.layers.flatten(conv4_bn)  
### DNN
L1 = tf.contrib.layers.fully_connected(inputs=flat_para, num_outputs=64, activation_fn=tf.nn.relu)
L1 = tf.layers.batch_normalization(L1)
L2 = tf.contrib.layers.fully_connected(inputs=L1, num_outputs=128, activation_fn=tf.nn.relu)
L2 = tf.layers.batch_normalization(L2)
L3 = tf.contrib.layers.fully_connected(inputs=L2, num_outputs=64, activation_fn=tf.nn.relu)
L3 = tf.layers.batch_normalization(L3)  
Out = tf.contrib.layers.fully_connected(inputs=L3, num_outputs=10, activation_fn=None)
pred = tf.nn.softmax(Out)
### Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred , labels=Y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')
saver = tf.train.Saver()
print("Total: ",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
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
            batch_x = batch_x.reshape(-1,32,32,3)
            batch_y = batch_y.reshape(-1,10)
            print(batch_y.shape)
            _, batch_loss, batch_acc = sess.run([train_step,loss,acc],\
                                  {X: batch_x, Y: batch_y})
            loss_sum += batch_loss
            acc_sum += batch_acc
            print(batch_acc)
        print("Epoch: ", epoch,"Loss: ", loss_sum/batch_num, "Acc: ",acc_sum/batch_num)
    print(saver.save(sess,model))

print("done")
