import tensorflow as tf
from tensorflow.nn import relu
import numpy as np
import math
"""
def tsne_manifold(Weights):
    trans = TSNE(n_components=2).fit_transform(Weights)
    return trans
"""
def add_noise(origin):
    l = []
    for w in origin:
        n_w = w + np.random.normal(scale = 0.0001,size = w.shape)
        l.append(n_w)
    return np.array(l)
     
def to_flat_array(tensors):
    _buf = np.array([ sess.run(p).reshape(-1) for p in tensors])
    flat_list = [item for sublist in _buf for item in sublist]
    flat_array = np.array(flat_list).reshape(-1)
    return flat_array

def get_noise_loss(origin,noise,sess,m_x,m_y):
    temp = origin
    temp_op = sess.run(temp)
    for idx, weights in enumerate(noise):
        assign_op = origin[idx].assign(weights)
        sess.run(assign_op)
    noise_loss = sess.run(loss_mse,{X: m_x,Y: m_y})
    for idx, weights in enumerate(temp_op):
        assign_op2 = origin[idx].assign(weights)
        sess.run(assign_op2)
    return noise_loss

def get_noise_list(origin,num = 10):
    l = []
    for _ in range(num):
        n = add_noise(origin)
        l.append(n)
    return np.array(l)
        


# Objective: fitting a function (sinx) #
# Configuration #
model = "./models/tf_m0.ckpt"

# Loading data #
# 50000 points between [0,2pi]
train_x = np.linspace(0,2*math.pi,50000)
train_y = np.load("train_data1.npy")
m_x = train_x.reshape(-1,1)
m_y = train_y.reshape(-1,1)
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

saver = tf.train.Saver()
# Training #
with tf.Session() as sess:
    for epoch in range(6):
        loss_list = []
        saver.restore(sess,"./models/tf_model" + str(epoch) + ".ckpt")
        origin_loss = sess.run(loss_mse,{X: m_x,Y: m_y})
        loss_list.append(origin_loss)
        _paras = [W1,W2,W3,W4,B1,B2,B3,B4]
        flat_weights = [to_flat_array(_paras)]
        noise_list = get_noise_list(_paras,num=50)
        for noise in noise_list:
            noise_loss = get_noise_loss(_paras,noise,sess,m_x,m_y)
            loss_list.append(noise_loss)
            flat_noise = to_flat_array(noise)
            flat_weights.append(flat_noise)

        flat_weights = np.array(flat_weights)    
        loss_list = np.array(loss_list)
        np.save("./data/cordin50_" + str(epoch),flat_weights)
        np.save("./data/loss50_" + str(epoch),loss_list)
        print("Epoch: ",epoch)