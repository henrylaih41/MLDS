import tensorflow as tf
from tensorflow.nn import relu
import numpy as np
import math
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import time as t
from keras.utils import np_utils
import plot as p 
def time_it(func):
    def inner_func(*args,**kwargs):
        t1 = t.time()
        result = func(*args,**kwargs)
        t2 = t.time()
        print(func.__name__ + " Done! Total time: %.3f sec " % (t2-t1))
        return result
    return inner_func

def add_noise(origin,noise_size):
    l = []
    for w,size in zip(origin,noise_size):
        n_w = w + np.random.normal(size=w.shape,scale=size/5)
        #n_w = w + np.random.random(w.shape)/(size*300)
        l.append(n_w)
    return np.array(l)

@time_it
def get_noise_list(origin_value,num = 10):
    l = [origin_value]
    noise_size = []
    ### calculate weight mean
    for w in origin_value:
        abs_w = np.absolute(w)
        noise_size.append(abs_w.mean())
    ### Adding noise
    for _ in range(num):
        n = add_noise(origin_value,noise_size)
        l.append(n)
    return np.array(l)
        
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    num_data = x_train.shape[0]
    #batch_num = math.ceil(num_data/batch_size)
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape(-1,784)
    y_train = y_train.reshape(-1,10)
    x_test = x_test.reshape(-1,784)
    y_test = y_test.reshape(-1,10)
    return x_train,y_train,x_test,y_test

def generate_noise_data(num,model):
    with tf.Session() as sess:
        # Load Model #
        loader = tf.train.import_meta_graph(model + ".meta")
        loader.restore(sess,model)
        _paras_name = ["W1","W2","W3","W4","W5","B1","B2","B3","B4","B5"]
        _paras_tensor = [tf.get_default_graph().get_tensor_by_name(name + ":0") for name in _paras_name]
        _paras_value = [sess.run(tensor) for tensor in _paras_tensor]
        noise_list = get_noise_list(_paras_value,num)
        return noise_list

@time_it
def get_noise_max_loss(noise_list,train_x,train_y,test_x,test_y,test=False):
    # Loads data #
    train_x,train_y,test_x,test_y = load_data()
    loss_list = []
    # Setup Graph #
    # Input
    X = tf.placeholder(shape=[None,784], dtype=tf.float32, name="X")
    Y = tf.placeholder(shape=[None,10], dtype=tf.float32, name="Y")

    # Weight and bias.
    W1 = tf.placeholder(shape=[784, 80], dtype=tf.float32, name = "W1")  
    B1 = tf.placeholder(shape=[80],    dtype=tf.float32, name = "B1")  
    W2 = tf.placeholder(shape=[80, 80], dtype=tf.float32, name = "W2")
    B2 = tf.placeholder(shape=[80],    dtype=tf.float32, name = "B2")
    W3 = tf.placeholder(shape=[80, 100], dtype=tf.float32, name = "W3")
    B3 = tf.placeholder(shape=[100],    dtype=tf.float32, name = "B3")
    W4 = tf.placeholder(shape=[100, 60], dtype=tf.float32, name = "W4")
    B4 = tf.placeholder(shape=[60],    dtype=tf.float32, name = "B4")
    W5 = tf.placeholder(shape=[60, 10], dtype=tf.float32, name = "W5")
    B5 = tf.placeholder(shape=[10],    dtype=tf.float32, name = "B5")

    # Define Operations to calculate loss_mse
    O1_1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    O1_2 = tf.nn.relu(tf.matmul(O1_1, W2) + B2)
    O1_3 = tf.nn.relu(tf.matmul(O1_2, W3) + B3)
    O1_4 = tf.nn.relu(tf.matmul(O1_3, W4) + B4)
    pred_1 = tf.nn.softmax(tf.matmul(O1_4, W5) + B5)
    loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_1, labels = Y)) 
    #train_step = tf.train.AdamOptimizer(learn
    # ing_rate).minimize(loss_1)
    correct_prediction = tf.equal(tf.argmax(pred_1, 1), tf.argmax(Y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    if(test):
        target_x, target_y = test_x, test_y
    else:
        target_x, target_y = train_x, train_y
    with tf.Session() as sess:
        min_acc = 100
        max_loss = 0
        first = True
        for weights in noise_list:
            loss, acc = sess.run([loss_1,acc_num/target_x.shape[0]],{
                X:target_x,  Y:target_y,  
                W1:weights[0], B1:weights[5],  
                W2:weights[1], B2:weights[6],
                W3:weights[2], B3:weights[7],
                W4:weights[3], B4:weights[8],
                W5:weights[4], B5:weights[9]   
            })
            if (first):
                origin_acc = acc
                origin_loss = loss
                first = False
            if (loss > max_loss): max_loss = loss
            if (acc < min_acc): min_acc = acc
            print(loss,acc)
        print(origin_loss,max_loss,origin_acc,min_acc)
    return origin_loss, max_loss, origin_acc, min_acc

def tsne_and_plot(n_list,loss_list):
    flat = []
    for weights in n_list:
        _buf = np.array([])
        for w in weights:
            _buf = np.append(_buf,np.hstack(w))
        flat.append(_buf)
    flat = np.array(flat)
    print(flat.shape)
    n_dim = TSNE(n_components=2).fit_transform(flat)
    print(n_dim.shape)
    threeD_error_surface(n_dim,loss_list)

def threeD_error_surface(cordin,loss):
    x = np.array(cordin[:,0])
    y = np.array(cordin[:,1])
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x, y, loss, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()   

train_x,train_y,test_x,test_y = load_data()
sharpness_list = []
loss_list = []
acc_list = []
t_loss_list = []
t_acc_list = []
acc_diff = []
for i in range(5,16):
    model_name = "../models/tf_mnist2_" + str(2 ** i) + ".ckpt"
    n_list = generate_noise_data(num=200,model=model_name)
    origin_loss, max_loss, origin_acc, min_acc = get_noise_max_loss(n_list,train_x,train_y,test_x,test_y)
    sharpness = (max_loss - origin_loss)/(1 + origin_loss)
    loss_list.append(origin_loss)
    sharpness_list.append(sharpness)
    acc_list.append(origin_acc)
    print("Test")
    t_loss, tm_loss, t_acc, tm_acc = get_noise_max_loss(n_list,train_x,train_y,test_x,test_y,test=True)
    t_loss_list.append(t_loss)
    t_acc_list.append(t_acc)
for acc,t_acc in zip(acc_list,t_acc_list):
    acc_diff.append(((acc - t_acc)/acc * 100))
p.plot_acc_diff_sharpness(acc_diff,sharpness_list)
p.plot_loss_sharpness(loss_list,t_loss_list,sharpness_list)
p.plot_acc_sharpness(acc_list,t_acc_list,sharpness_list)
exit()