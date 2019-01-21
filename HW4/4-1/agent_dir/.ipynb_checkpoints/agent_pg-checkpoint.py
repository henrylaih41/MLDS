from agent_dir.agent import Agent
import scipy
import numpy as np
from agent_dir.util import conv2d, DNN, get_real_reward
import tensorflow as tf
import time
def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, [80,80])
    resized = np.expand_dims(resized.astype(np.float32),axis=2)
    return np.expand_dims(resized.astype(np.float32),axis=0)

def prepro_v2(I):
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = np.expand_dims(I.astype(np.float32),axis=2)
    return np.expand_dims(I.astype(np.float32),axis=0)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        self.epoch = args.epoch
        self.lr = args.lr
        self.N = args.N
        self.load = args.load
        self.name = args.name
        self.gl_a = args.gl_a
        self.argmax = args.argmax
        self.save_path = args.save_path
        super(Agent_PG,self).__init__(env)
        self.s_p = np.zeros([1,80,80,1])
        self.build_model()
        
        if args.test_pg or args.load:
            #you can load your model here
            loader = tf.train.Saver()
            self.sess = tf.Session()
            loader.restore(self.sess, self.save_path + "pong_vanilla_v2max.ckpt-1000")
            print('loading trained model')
        print(self.env.get_action_space())
        print(self.env.get_random_action())
        print(self.env.get_observation_space())
        #self.env.reset()
        #for _ in range(2100):
            #_, r, done, info = self.env.step(0)
            #print(r,done,info)
            #if(done):
                #self.env.reset()
        
        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def build_model(self):
        ### Action Making
        self.diff = tf.placeholder(tf.float32,[None,6400])
        
        def build_CNN(frame,reuse=False):
            output_dim = 2 
            with tf.variable_scope("CNN") as scope:
                if (reuse):
                    scope.reuse_variables()
                C1 = conv2d(frame,output_dim,3,3,2,2,name="conv1")
                print(C1.shape)
                flatten = tf.reshape(C1,[-1,3200])
                print(flatten.shape)
                return flatten 
        ###diff_feature = build_CNN(diff)

        #frames = tf.concat([feature1,feature2],1)
        D1 = DNN(self.diff,[6400,100],0)
        D2 = DNN(D1,[100,2],1)
        self.action_prob = D2
        self.action = tf.nn.softmax(self.action_prob)
        
        ### Calculating loss  
        self.actions = tf.placeholder(tf.int32,[None])
        self.weights = tf.placeholder(tf.float32,[None,1])
        w_loss = tf.losses.log_loss(labels=tf.one_hot(self.actions,2),
                                  predictions=self.action,
                                  weights=self.weights)
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.action_prob,labels=tf.one_hot(self.actions,2))
        #w_loss = tf.multiply(loss,self.weights)
        self.opt_without_gl = tf.train.AdamOptimizer(self.lr).minimize(w_loss)
                
    
    def train(self):
        """
        Implement your training algorithm here
        """
        running_average = []
        his_r = []
        c = 0
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=3)
            if not self.load:
                sess.run(tf.global_variables_initializer())
                print("New Model: ", self.name)
            else:
                latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
                saver.restore(sess, latest_checkpoint)
                print("Saver Loaded: " + latest_checkpoint)
            PATH = self.save_path + self.name
            pre_s = np.zeros([1,80,80,1])
            for epo in range(self.epoch):
                if (epo % 500 == 0):
                    ckpt_path = saver.save(sess, PATH, global_step=epo)
                    print("Model saved: ",ckpt_path)
                count = 0
                self.env.reset()
                step = 0
                tup_list = []
                while(count < self.N):
                    
                    s, r, _, _ = self.env.step(step)
                    s = prepro_v2(s)
                    state = s - pre_s
                    state = state.ravel()
                    action_prob = sess.run(self.action, 
                                           feed_dict = {
                                               self.diff : state.reshape(1,-1)
                                           })
                    if (self.argmax):
                        step = np.argmax(action_prob) + 2
                    else:
                        step = np.random.choice(2, 1, p=action_prob[0])[0] + 2
                    pre_s = s
                    if (r!=0.0):
                        count = count + 1
                    tup = (state,r,step-2)
                    tup_list.append(tup)
###             
                s_l, r_l, a_l = zip(*tup_list)
                r_sum = np.sum(np.array(r_l))
                his_r.append(r_sum)
                r_l = get_real_reward(r_l)
                s_l = np.vstack(s_l)
                print(s_l.shape)
                a_l = np.array(a_l)
                
                _ = sess.run([self.opt_without_gl,], 
                             feed_dict = {
                                 self.diff : s_l,
                                 self.actions : a_l,
                                 self.weights : r_l
                             })
                
                print(action_prob,step)
                print("epo reward",epo,r_sum)
                if(len(running_average) <= 50):
                    running_average.append(r_sum)
                else:
                    if(c == 50):
                        c = 0
                    running_average[c] = r_sum
                    c = c + 1
                print(np.mean(running_average))
            his_r = np.array(his_r)
            np.save(self.save_path + "0.0005_hd1000" + ".npy",his_r)
                
                    
                
        
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        s = prepro_v2(observation)
        diff = s - self.s_p
        diff = diff.ravel()
        self.s_p = s
        action_prob = self.sess.run(self.action, 
                                feed_dict = {
                                    self.diff : diff.reshape(1,-1)
                                }) 
        step = np.argmax(action_prob) + 2
        return step
    
    