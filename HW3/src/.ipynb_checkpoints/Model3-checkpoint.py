import tensorflow as tf
import numpy as np
import util as ut
from dataProcess import Dataset
import skimage
import argparse
### tanh, leaky Relu, Adam, batch_norm, gaussian noise, 7.
class DCGAN():
    def __init__(self, noise_dim=100, batch_size=128, lr=0.0002, beta1=0.5):
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.d_loss_fake = None
        self.d_loss_real = None
        self.image_size = 96
        self.image_depth = 3
        self.label_dir = "../data/filtered_faces2/"
        self.name = "DCGAN_3"
        self.save_path = "../models3/"
        self.d_update_num = 1
        self.g_update_num = 3
    def build_model(self,z,real_image):
        def build_generator():
            ### First block shape
            dim = 6 # 96/16
            depth = 1024
            ### DNN layer: transform inputs (bs,noise_dim) to higher dims (bs,4,4,1024)
            with tf.variable_scope("generator") as scope:
                W1 = tf.get_variable("W1",[self.noise_dim,dim*dim*depth],tf.float32,
                                    tf.truncated_normal_initializer(stddev=0.02))
                B1 = tf.get_variable("B1",[dim*dim*depth],tf.float32,
                                    tf.truncated_normal_initializer(stddev=0.02))
                DNN_output = tf.matmul(z,W1) + B1
                
                DNN_output = tf.nn.relu(DNN_output)
                DNN_output = tf.reshape(DNN_output,[-1,dim,dim,depth]) ###(bs,dim,dim,depth)
                print(DNN_output.shape)
                DNN_output = tf.layers.batch_normalization(DNN_output,training=True)
                
                ### Deconv layers
                bs, d_w, d_h, depth = DNN_output.get_shape().as_list()
                deconv1 = ut.deconv2d(DNN_output,[bs,d_w*2,d_h*2,int(depth/2)],name="deconv1")
                print(deconv1.shape)
                deconv1 = tf.layers.batch_normalization(deconv1,training=True)
                bs, d_w, d_h, depth = deconv1.get_shape().as_list()
                deconv2 = ut.deconv2d(deconv1,[bs,d_w*2,d_h*2,int(depth/2)],name="deconv2")
                print(deconv2.shape)
                deconv2 = tf.layers.batch_normalization(deconv2,training=True)
                bs, d_w, d_h, depth = deconv2.get_shape().as_list()
                deconv3 = ut.deconv2d(deconv2,[bs,d_w*2,d_h*2,int(depth/2)],name="deconv3")
                print(deconv3.shape)
                deconv3 = tf.layers.batch_normalization(deconv3,training=True)
                bs, d_w, d_h, depth = deconv3.get_shape().as_list()
                gen_image = ut.deconv2d(deconv3,[bs,d_w*2,d_h*2,self.image_depth],
                                        f_h=7,f_w=7,name="deconv4",relu=False)
                gen_image = tf.nn.tanh(gen_image)
                print(gen_image.shape)
            return gen_image
        
        def build_discriminator(image,reuse=False):
            conv_dim = 64
            with tf.variable_scope("discriminator") as scope:
                if (reuse):
                    scope.reuse_variables()
                
                f1 = ut.conv2d(image,conv_dim,name="conv1")
                print(f1.shape)
                f2 = ut.conv2d(f1,f1.shape[-1]*2,name="conv2")
                print(f2.shape)
                f2 = tf.layers.batch_normalization(f2,training=True)
                f3 = ut.conv2d(f2,f2.shape[-1]*2,name="conv3")
                print(f3.shape)
                f3 = tf.layers.batch_normalization(f3,training=True)
                f4 = ut.conv2d(f3,f3.shape[-1]*2,name="conv4")
                f4 = tf.layers.batch_normalization(f4,training=True)
                print(f4.shape)

                ### DNN layer
                flatten = tf.reshape(f4,[self.batch_size,-1])
                #flatten = tf.layers.batch_normalization(flatten,training=True)
                print(flatten.shape)
                W2 = tf.get_variable("W2",[flatten.shape[-1],1],tf.float32,
                                     tf.truncated_normal_initializer(stddev=0.02))
                B2 = tf.get_variable("B2",[1],tf.float32,
                                     tf.truncated_normal_initializer(stddev=0.02))
                score = tf.matmul(flatten,W2) + B2
                print(score.shape)
            
            return score

        self.gen_image = build_generator()
        fake_score = build_discriminator(self.gen_image)  # (bs,1)
        real_score = build_discriminator(real_image,reuse=True) # (bs,1) 
        ### Calculate loss
        self.d_loss_fake = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, 
                                                                   labels=tf.zeros_like(fake_score)))
        self.d_loss_real = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, 
                                                                   labels=tf.ones_like(fake_score)))
        self.g_loss      = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, 
                                                                   labels=tf.ones_like(fake_score)))
        d_loss = self.d_loss_fake + self.d_loss_real

        ### Get variables of d and g
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
        print(d_vars)
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
        print(g_vars)

        ### Optimizer of d and g
        self.d_opt = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(d_loss, var_list=d_vars)
        self.g_opt = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
        
    def train(self,epoch=100,load=False):
        noise      = tf.placeholder(tf.float32,[self.batch_size,self.noise_dim])
        real_image = tf.placeholder(tf.float32,[self.batch_size,self.image_size,
                                                self.image_size,self.image_depth])
        self.build_model(noise,real_image)
        datasetTrain = Dataset(self.label_dir,self.batch_size)
        # (bs,self.image_size,self.image_size,self.image_depth)
        
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=3)

            if not load:
                sess.run(tf.global_variables_initializer())
                print("New Model: ", self.name)
            else:
                latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
                saver.restore(sess, latest_checkpoint)
                print("Saver Loaded: " + latest_checkpoint)

            PATH = self.save_path + self.name + ".ckpt"
            
            for epo in range(0, epoch):
                if (epo % 1 == 0):
                    batch_z   = ut.genNoise(self.batch_size,mode="normal")
                    gen_image = sess.run(self.gen_image,feed_dict ={noise:batch_z})
                    idx = np.random.randint(0,self.batch_size-2)
                    gen_image = np.array(gen_image[ idx : idx+2])
                    for i in range(2):
                        skimage.io.imsave("../genImage/" + self.name + "_" + str(epo)
                                                + "_" + str(i) + ".jpg", gen_image[i])
                if (epo != 0):
                    ckpt_path = saver.save(sess, PATH, global_step=epo)
                    print("Model saved: ",ckpt_path)
                
                datasetTrain.shuffle_perm()
                num_steps = int(datasetTrain.batch_max_size/self.batch_size)
                epo_loss  = [0,0,0]
                ### Training
                for i in range(0, num_steps):
                    images, _ = datasetTrain.next_batch()
                    images    = np.array(images).astype(np.float32)
                    batch_z   = ut.genNoise(self.batch_size,mode="normal")
                    total_dl_fake = 0 
                    total_dl_real = 0
                    total_gl      = 0

                    for j in range(self.d_update_num):
                        #print("Start training")
                        _, dl_fake, dl_real = sess.run([self.d_opt,self.d_loss_fake,self.d_loss_real],
                                              feed_dict = {noise      : batch_z,
                                                           real_image : images
                                              })
                        total_dl_fake += dl_fake/self.d_update_num
                        total_dl_real += dl_real/self.d_update_num
                        #print("Done Dis")

                    for j in range(self.g_update_num):
                        _, gl = sess.run([self.g_opt,self.g_loss],
                                          feed_dict = {noise      : batch_z,
                                                       real_image : images
                                          })
                        total_gl += gl/self.g_update_num
                        #print("Done Gen")

                    print("steps: %i gloss: %f dloss_fake: %f dloss_real %f" % 
                                  (i, total_gl, total_dl_fake, total_dl_real))

                    for i, loss in enumerate([total_gl, total_dl_fake, total_dl_real]):
                        epo_loss[i] += loss
                epo_loss = np.array(epo_loss)
                print("\n[FINISHED] Epoch " + str(epo) + \
                        ", Training Loss (per epoch): ", epo_loss/num_steps)
            print('\n\nTraining finished!')
        print("Success")

    def test(self):
        pass

model = DCGAN()
model.train(load=True)
exit()
    
    