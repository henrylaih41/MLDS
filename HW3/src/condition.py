import tensorflow as tf
import numpy as np
import util as ut
from dataProcessv3 import TrainDataset
import skimage
import argparse
import cv2

class DCGAN():
    def __init__(self, args, noise_dim=100, batch_size=256, lr=0.00005, beta1=0.5):
        self.d_rate = 1
        self.noise_dim = noise_dim
        self.word_dim = 22
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.d_loss_fake = None
        self.d_loss_real = None
        self.image_size = args.image_size
        self.image_depth = 3
        self.label_dir = "../data/tag_faces/"
        self.label_csv = "../data/extra_data/tags.csv"
        self.save_dict = "./"
        self.name = args.name
        self.save_path = args.save_path
        self.d_update_num = args.d_update_num
        self.g_update_num = args.g_update_num
        self.ball = args.ball
        self.noise_mode = args.noise_mode
        self.cv2 = args.cv2
        self.image_save_path = args.image_save_path
        self.genImage_num = args.genImage_num
        self.nor = args.nor
    def build_model(self,z,real_image,real_tags,random_tags):
        def build_generator():
            ### First block shape
            dim = int(self.image_size/16) # 96/16
            depth = 1024
            ### DNN layer: transform inputs (bs,noise_dim) to higher dims (bs,4,4,1024)
            with tf.variable_scope("generator") as scope:
                W1 = tf.get_variable("W1",[self.noise_dim+self.word_dim,dim*dim*depth],tf.float32,tf.truncated_normal_initializer(stddev=0.02))
                B1 = tf.get_variable("B1",[dim*dim*depth],tf.float32,
                                    tf.truncated_normal_initializer(stddev=0.02))
                DNN_output = tf.matmul(z,W1) + B1
                DNN_output = tf.nn.relu(DNN_output)
                DNN_output = tf.reshape(DNN_output,[-1,dim,dim,depth]) ###(bs,dim,dim,depth)
                print(DNN_output.shape)

                ### Deconv layers
                bs, d_w, d_h, depth = DNN_output.get_shape().as_list()
                deconv1 = ut.deconv2d(DNN_output,[bs,d_w*2,d_h*2,int(depth/2)],name="deconv1")
                print(deconv1.shape)
                bs, d_w, d_h, depth = deconv1.get_shape().as_list()
                deconv2 = ut.deconv2d(deconv1,[bs,d_w*2,d_h*2,int(depth/2)],name="deconv2")
                print(deconv2.shape)
                bs, d_w, d_h, depth = deconv2.get_shape().as_list()
                deconv3 = ut.deconv2d(deconv2,[bs,d_w*2,d_h*2,int(depth/2)],name="deconv3")
                print(deconv3.shape)
                bs, d_w, d_h, depth = deconv3.get_shape().as_list()
                gen_image = ut.deconv2d(deconv3,[bs,d_w*2,d_h*2,self.image_depth],
                                        f_h=3,f_w=3,name="deconv4",relu=False)
                gen_image = tf.nn.tanh(gen_image)
                print(gen_image.shape)
            return gen_image
        
        def build_discriminator(image,cond,reuse=False):
            conv_dim = 64
            with tf.variable_scope("discriminator") as scope:
                if (reuse):
                    scope.reuse_variables()
                
                f1 = ut.conv2d(image,conv_dim,f_h=7,f_w=7,name="conv1")
                print(f1.shape)
                f2 = ut.conv2d(f1,f1.shape[-1]*2,name="conv2")
                print(f2.shape)
                f3 = ut.conv2d(f2,f2.shape[-1]*2,name="conv3")
                print(f3.shape)
                f4 = ut.conv2d(f3,f3.shape[-1]*2,name="conv4")
                print(f4.shape)
                
                features1D = tf.reshape(f4,[self.batch_size,-1])
                print(features1D.shape)
           
                ### condition matching
                concat_features = tf.concat([features1D,cond], axis=1)
                C1 = ut.DNN(concat_features, [concat_features.shape[-1],1000], 4)
                score = ut.DNN(C1, [1000,1], 6, False)
                
                return score
            
        self.gen_image = build_generator()
        fake_score = build_discriminator(self.gen_image,random_tags)  # (bs,1)
        real_score = build_discriminator(real_image,real_tags,reuse=True) # (bs,1) 
        rft_score  = build_discriminator(real_image,random_tags,reuse=True) 
        ### Calculate loss
        self.d_loss_fake = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, 
                                                                   labels=tf.zeros_like(fake_score)))
        self.d_loss_real = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, 
                                                                   labels=tf.ones_like(real_score)))
        self.d_loss_rft  = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(logits=rft_score, 
                                                                   labels=tf.zeros_like(rft_score)))
        self.g_loss      = tf.reduce_mean(
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, 
                                                                   labels=tf.ones_like(fake_score)))
        d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_rft
        ### Get variables of d and g
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
        print(d_vars)
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
        print(g_vars)

        ### Optimizer of d and g
        self.d_opt = tf.train.AdamOptimizer(self.lr*self.d_rate, beta1=self.beta1) \
                      .minimize(d_loss, var_list=d_vars)
        self.g_opt = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
        
    def train(self,epoch=100,load=False):
        noise       = tf.placeholder(tf.float32,[self.batch_size,self.noise_dim+self.word_dim])
        real_image  = tf.placeholder(tf.float32,[self.batch_size,self.image_size,
                                                 self.image_size,self.image_depth])
        real_tags   = tf.placeholder(tf.float32,[self.batch_size,self.word_dim])
        random_tags = tf.placeholder(tf.float32,[self.batch_size,self.word_dim])
        self.build_model(noise,real_image,real_tags,random_tags)
        
        if(self.image_size == 64):
            datasetTrain = TrainDataset(self.label_csv, self.label_dir, self.save_dict, self.batch_size, self.nor)
        elif(self.image_size == 96):
            datasetTrain = TrainDataset(self.label_dir,self.batch_size,cv2=self.cv2,mode="origin"
                                        ,normalized=self.nor)
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=3)

            if not load:
                sess.run(tf.global_variables_initializer())
                print("New Model: ", self.name)
            else:
                latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
                saver.restore(sess, latest_checkpoint)
                print("Saver Loaded: " + latest_checkpoint)

            PATH = self.save_path + self.name + "_" + ".ckpt"
            if(self.nor):
                mean, std = datasetTrain.reset_normalized_par()
            for epo in range(0, epoch):
                if (epo % 1 == 0):
                    batch_z   = ut.genNoise(self.batch_size,mode=self.noise_mode,
                                            ball=self.ball)
                    word_test = ut.random_tags(self.batch_size)
                    batch_z = np.concatenate((batch_z, word_test), axis=1)
                    gen_image = sess.run(self.gen_image,feed_dict ={noise:batch_z})
                    idx = np.random.randint(0,self.batch_size-self.genImage_num + 1)
                    if(self.nor):
                        gen_image = std*np.array(gen_image[ idx : idx+self.genImage_num]) + mean
                    else:
                        gen_image = np.array(gen_image[ idx : idx+self.genImage_num])*255
                    print(gen_image)
                    for i in range(self.genImage_num):
                        print(word_test[i+idx])
                        if(self.cv2):
                            cv2.imwrite( self.image_save_path + self.name + "_" + 
                                         str(epo) + "_" + str(i) + ".jpg", gen_image[i])
                        else:
                            skimage.io.imsave("../genImage/DCGAN2_80/" + self.name + "_" + str(epo)
                                                    + "_" + str(i) + ".jpg", gen_image[i])
                if (epo != 0):
                    ckpt_path = saver.save(sess, PATH, global_step=epo)
                    print("Model saved: ",ckpt_path)
                
                datasetTrain.shuffle_perm()
                num_steps = int(datasetTrain.batch_max_size/self.batch_size)
                epo_loss  = [0,0,0,0]
                ### Training
                for i in range(0, num_steps):
                    images, words, _ = datasetTrain.next_batch()
                    images    = np.array(images).astype(np.float32)
                    batch_z   = ut.genNoise(self.batch_size,mode=self.noise_mode,
                                            ball=self.ball)
                    noise_tag = ut.random_tags(self.batch_size)
                    batch_z = np.concatenate((batch_z, noise_tag), axis=1)
                    total_dl_fake = 0 
                    total_dl_real = 0
                    total_dl_rft  = 0
                    total_gl      = 0

                    for j in range(self.d_update_num):
                        #print("Start training")
                        _, dl_fake, dl_real, dl_rft = sess.run([self.d_opt, self.d_loss_fake, self.d_loss_real, self.d_loss_rft],
                                              feed_dict = {noise      : batch_z,
                                                           real_image : images,
                                                           real_tags  : words,
                                                           random_tags: noise_tag
                                              })
                        total_dl_fake += dl_fake/self.d_update_num
                        total_dl_real += dl_real/self.d_update_num
                        total_dl_rft  += dl_rft/self.d_update_num
                        #print("Done Dis")

                    for j in range(self.g_update_num):
                        _, gl = sess.run([self.g_opt,self.g_loss],
                                          feed_dict = {noise      : batch_z,
                                                       real_image : images
                                          })
                        total_gl += gl/self.g_update_num
                        #print("Done Gen")

                    print("steps: %i gloss: %f dloss_fake: %f dloss_real %f dloss_rft %f" % 
                                  (i, total_gl, total_dl_fake, total_dl_real, total_dl_rft))

                    for i, loss in enumerate([total_gl, total_dl_fake, total_dl_real, total_dl_rft]):
                        epo_loss[i] += loss
                epo_loss = np.array(epo_loss)
                print("\n[FINISHED] Epoch " + str(epo) + \
                        ", Training Loss (per epoch): ", epo_loss/num_steps)
            print('\n\nTraining finished!')
        print("Success")

    def test(self):
        pass
    
parser = argparse.ArgumentParser()
parser.add_argument('--ball', type=int, default=1)
parser.add_argument('--noise_mode', type=str, default="uniform")
parser.add_argument('--name', type=str, default="DCGAN_conditional")
parser.add_argument('--save_path', type=str, default="../Henry_model2/")
parser.add_argument('--image_save_path', type=str, default="../Henry_model2/")
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--d_update_num', type=int, default=1)
parser.add_argument('--g_update_num', type=int, default=1)
parser.add_argument('--genImage_num', type=int, default=3)
parser.add_argument('--cv2',type=int, default=1)
parser.add_argument('--epoch',type=int, default=80)
parser.add_argument('--load',type=int, default=0)
parser.add_argument('--nor',type=int, default=0)
args = parser.parse_args()
model = DCGAN(args)
model.train(epoch=args.epoch,load=args.load)
exit()
    
    