from agent_dir.agent import Agent
import scipy
import numpy as np
from agent_dir.util import conv2d, DNN, get_real_reward
import tensorflow as tf
import time
import argparse
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
        self.build_model()
        
        if args.test_pg or args.load:
            #you can load your model here
            print('loading trained model')
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
        self.w_loss = tf.losses.log_loss(labels=tf.one_hot(self.actions,2),
                                  predictions=self.action)
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.action_prob,labels=tf.one_hot(self.actions,2))
        #w_loss = tf.multiply(loss,self.weights)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.w_loss)
                
    
    def pretrain(self):
        """
        Implement your training algorithm here
        """
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=1)
            if not self.load:
                sess.run(tf.global_variables_initializer())
                print("New Model: ", self.name)
            else:
                latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
                saver.restore(sess, latest_checkpoint)
                print("Saver Loaded: " + latest_checkpoint)
            PATH = self.save_path + self.name + "max.ckpt"
            labels = np.load("./labels.npy")
            diff = []
            action = []
            for tag in labels:
                diff.append(prepro_v2(tag[1]-tag[0]).ravel())
                action.append(tag[2])
            print(action)
            for epo in range(self.epoch):
                if (epo % 1 == 0):
                    ckpt_path = saver.save(sess, PATH, global_step=epo)
                    print("Model saved: ",ckpt_path)
                
                _, loss = sess.run([self.opt,self.w_loss], 
                             feed_dict = {
                                 self.diff : diff,
                                 self.actions : action
                             })
                
                print("epo loss",epo,loss)
        
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
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()
    
def parse():
    parser = argparse.ArgumentParser(description="MLDS 2018 HW4")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    env = None
    agent = Agent_PG(env, args)
    agent.pretrain()
