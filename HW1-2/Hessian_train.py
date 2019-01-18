import tensorflow as tf
from tensorflow.nn import relu
import numpy as np
import math
import plot as p
from sklearn.manifold import TSNE
        
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

# Hessian
grad=tf.gradients(loss_mse,[W1,B1,W2,B2,W3,B3,W4,B4])
staki=[tf.reshape(k,[-1]) for k in grad]
stakin=tf.concat(staki,-1)
staking=tf.unstack(stakin)
second=[tf.gradients(k,[W1,B1,W2,B2,W3,B3,W4,B4]) for k in staking]
secondlinear=[tf.reshape(k,[-1]) for j in second for k in j]
secondllinear=[tf.reshape(k,[-1]) for k in secondlinear]
stakingg=tf.concat(secondllinear,-1)
hession=tf.reshape(stakingg,[62,62])
rand_var_1 = np.random.normal(scale = 0.1,size=[62,62])
hess= hession + rand_var_1
invers_hession=tf.linalg.inv(hess)
updating=tf.matmul([stakin],0.1 * invers_hession)
updatings=tf.squeeze(updating)

w1,b1,w2,b2,w3,b3,w4,b4=tf.split(updatings,[3,3,15,5,25,5,5,1])
ww1=tf.reshape(w1,[1,3])
bb1=tf.reshape(b1,[3])
ww2=tf.reshape(w2,[3,5])
bb2=tf.reshape(b2,[5])
ww3=tf.reshape(w3,[5,5])
bb3=tf.reshape(b3,[5])
ww4=tf.reshape(w4,[5,1])
bb4=tf.reshape(b4,[1])

matrix=[ww1,bb1,ww2,bb2,ww3,bb3,ww4,bb4]
weights=[W1,B1,W2,B2,W3,B3,W4,B4]
update=[v.assign_add(-k) for v,k in zip(weights,matrix)]   ###更新參數

saver = tf.train.Saver()
# Training #
loss_list = []
with tf.Session() as sess:
    saver.restore(sess,model)
    for epoch in range(5):
        _, loss = sess.run([update,loss_mse],\
                                 {X: m_x, Y: m_y})
        print("Epoch: ", epoch, "Loss: ", loss)
        print(saver.save(sess,model,global_step=epoch))