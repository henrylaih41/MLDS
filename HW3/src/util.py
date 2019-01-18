import numpy as np
import tensorflow as tf
def genNoise(bs,nos_dim=100, mode='uniform',ball=True):
        if (mode == 'uniform'):
            if(ball):
                x = np.random.uniform(-1, 1, [bs, nos_dim]).astype(np.float32)
                x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
                
                return (x / 4)
            else:
                return np.random.uniform(-1, 1, [bs, nos_dim]).astype(np.float32)
        elif(mode == 'normal'):
            if(ball):
                x = np.random.normal(0,0.7,[bs, nos_dim]).astype(np.float32)
                return (x / np.linalg.norm(x, axis=1)[:, np.newaxis])
            else:
                return np.random.normal(0,0.7,[bs, nos_dim]).astype(np.float32)
        else:
            raise RuntimeError('mode %s is not defined' % (mode))

def deconv2d(input_block, output_shape,f_h=5, f_w=5, 
             s_h=2, s_w=2, stddev=0.02,name="deconv",relu=True,batch_norm=False):
    ### filter: [height, width, output_channels, in_channels]
    deconv_filter = tf.get_variable(name + '_w', [f_h, f_w, output_shape[-1], input_block.shape[-1]],
                                    initializer = tf.truncated_normal_initializer(stddev=stddev))
   
    deconv = tf.nn.conv2d_transpose(input_block,  deconv_filter, 
                                    output_shape, [1, s_h, s_w, 1]) # (input,filter,output_shape,strides)

    biases = tf.get_variable(name + '_b', [output_shape[-1]], 
                             initializer= tf.truncated_normal_initializer(stddev=stddev))

    deconv = tf.nn.bias_add(deconv, biases)
    if(batch_norm):
        deconv = tf.layers.batch_normalization(deconv,training=True)
    if(relu):
        deconv = tf.nn.relu(deconv)
    
    return deconv
    
def conv2d(input_image, output_dim, f_h=5, f_w=5, 
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
    conv = tf.nn.leaky_relu(conv)

    return conv

def DNN(inputs, shape, idx, activate=True):
    W = tf.get_variable("W" + str(idx),shape,tf.float32,
                        tf.truncated_normal_initializer(stddev=0.02))
    B = tf.get_variable("B" + str(idx),shape[-1],tf.float32,
                        tf.truncated_normal_initializer(stddev=0.02))
    output = tf.matmul(inputs,W) + B
    if activate:
        output = tf.nn.leaky_relu(output)
        
    return output
        
def random_tags(batch_size):
    tags = []
    for _ in range(batch_size):
        hair_idx = np.random.random_integers(0,11)
        hairs = [0.0]*12
        hairs[hair_idx] = 1.0
        eye_idx = np.random.random_integers(0,9)
        eyes= [0.0]*10
        eyes[eye_idx] = 1.0
        tag = np.concatenate([hairs,eyes])
        tags.append(tag)
    tags = np.array(tags)
    return tags
