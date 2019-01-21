import numpy as np
import tensorflow as tf

def conv2d(input_image, output_dim, f_h=3, f_w=3, 
           s_h=2, s_w=2, stddev=0.02, name="conv",batch_norm=False):
    ### filter: [height, width, in_channels, output_channels]
    conv_filter = tf.get_variable(name + '_w', [f_h, f_w, input_image.shape[-1], output_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=stddev))

    conv = tf.nn.conv2d(input_image, conv_filter, strides=[1, s_h, s_w, 1], padding='SAME')

    biases = tf.get_variable(name + '_b', [output_dim], 
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
   
    conv = tf.nn.bias_add(conv, biases)
    if(batch_norm):
        conv = tf.layers.batch_normalization(conv,training=True)
    conv = tf.nn.relu(conv)

    return conv

def DNN(inputs, shape, idx, activate=True):
    W = tf.get_variable("W" + str(idx),shape,tf.float32,
                        tf.truncated_normal_initializer(stddev=0.02))
    B = tf.get_variable("B" + str(idx),shape[-1],tf.float32,
                        tf.truncated_normal_initializer(stddev=0.02))
    output = tf.matmul(inputs,W) + B
    if activate:
        output = tf.nn.relu(output)
        
    return output

def get_real_reward(r_list, gamma=0.9):
    ### reference to https://github.com/mlitb/pong-cnn/blob/master/pong.py
    discounted_reward = np.zeros((len(r_list), 1))
    future_reward = 0
    for i in range(len(r_list) - 1, -1, -1):
        if r_list[i] != 0: # reset future reward after each score
            future_reward = 0
        discounted_reward[i][0] = r_list[i] + gamma * future_reward
        future_reward = discounted_reward[i][0]
    #discounted_reward -= np.mean(discounted_reward)
    #discounted_reward /= np.std(discounted_reward)
    #for idx,i in enumerate(discounted_reward):
        #if i < 0:
            #discounted_reward[idx] = 0
    return discounted_reward

def get_weights(V_s,rt):
    W = []
    for i in range(0,len(V_s) - 1):
        W.append(V_s[i+1] - V_s[i])    
    W.append(rt)
    return np.vstack(W)
#V_s = [1,2,3,4,5,6,2,1]
#rt = 1
#print(get_weights(V_s,rt))
r = [0,0,0,1,0,0,0,-1,0,0,1,-1,1,0,0,-1]
print(r)
print(get_real_reward(r))
