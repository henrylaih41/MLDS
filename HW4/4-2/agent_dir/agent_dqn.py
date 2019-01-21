from agent_dir.agent import Agent
import tensorflow as tf
from collections import namedtuple
import numpy as np
import random
import sys
from scipy.misc import imsave

Transition = namedtuple('Transition',('current_state', 'action', 'next_state', 'reward', 'done'))
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        #self.dueling = args.dueling_dqn
        self.gamma = args.gamma
        self.n_actions = env.action_space.n
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.memory = self.ReplayMemory(args.replay_memory_size)
        self.num_episodes = args.num_episodes
        self.epsilon = 1.0
        self.replace_num = args.replace_num
        self.update_num = args.update_num
        self.epsilon_decay_constant = args.epsilon_decay_constant
        self.epsilon_end = args.epsilon_end
        self.step = 0
        self.saver_steps = args.saver_steps
        self.output_logs = args.output_logs 
        self.dueling = args.dueling_dqn


        self.input_active_frame = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_active_frame')
        self.input_fixed_frame = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_fixed_frame')
        self.input_y = tf.placeholder(tf.float32, [None])
        self.input_action = tf.placeholder(tf.float32, [None, self.n_actions])

        self.q_active = self.build_network(self.input_active_frame, "active_q")
        self.q_fixed = self.build_network(self.input_fixed_frame, "fixed_q")

        self.build_optimizer()
        self.avtive_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='active_q')
        self.fixed_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fixed_q')

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.fixed_params, self.avtive_params)] 

        self.sess = tf.Session()

        self.saver = tf.train.Saver(max_to_keep = 3)
        self.ckpts_path = args.save_dir + "dqn.ckpt"
        if args.test_dqn:
            print("load_testing")
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            print(ckpt)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.step = self.sess.run(self.global_step)
        else:
            if args.load_saver :
                latest_checkpoint = tf.train.latest_checkpoint(args.save_dir)
                self.saver.restore(self.sess, latest_checkpoint)
                print("Saver Loaded: " + latest_checkpoint)
                self.step = self.sess.run(self.global_step)
                print('load step: ', self.step)
            else :
                self.sess.run(tf.global_variables_initializer())
                print("New Model~~")

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        current_loss = 0
        rewards = 0
        #episode_len = 0.0
        file_loss = open(self.output_logs, "a")
        file_loss.write("episode,step,epsilon,reward,loss\n")
        for episode in range(self.num_episodes):
            prev_state = self.env.reset()
            #self.init_game_setting()
            loss = 0
            episode_reward = 0
            while True:
                action = self.make_action(prev_state, test=False)
                #self.env.env.render()
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.storeTransition(prev_state, action, next_state, reward, done)
                #episode_len += 1
                if len(self.memory) > self.batch_size:
                    if self.step%self.update_num==0:
                        loss += self.update_network()
                if self.step % self.saver_steps==0 and episode>0:
                    self.saver.save(self.sess, self.ckpts_path, global_step = self.step)
                    print("save_model")
                if done:
                    break
                prev_state = next_state
            rewards+= episode_reward
            current_loss+= loss
            if episode%50==0:
                file_loss.write(str(episode) + "," + str(self.step) + "," + "{:.4f}".format(self.epsilon) + "," + "{:.2f}".format(rewards/50.0) + "," + "{:.4f}".format(current_loss/50.0) + "\n")
                file_loss.flush()
                print("episode:",episode)
                print("reword:",rewards/50.0)
                print("loss:", current_loss/50.0)
                print("step:", self.step)
                rewards = 0
                current_loss = 0
            


        
    def update_network(self):
        transitions = self.memory.sample(self.batch_size)
        minibatch = Transition(*zip(*transitions))

        #state_batch = [np.array(s).astype(np.float32) / 255.0 for s in list(minibatch.current_state)]
        state_batch = np.array(minibatch.current_state)
        state_batch = state_batch.astype(np.float32)/255.0
        #next_state_batch = [np.array(s_).astype(np.float32) / 255.0 for s_ in list(minibatch.next_state)]
        next_state_batch = np.array(minibatch.next_state)
        next_state_batch = next_state_batch.astype(np.float32)/255.0
        action_batch = []
        for act in list(minibatch.action):
            one_hot_action = np.zeros(self.n_actions)
            one_hot_action[act] = 1
            action_batch.append(one_hot_action)
        reward_batch = list(minibatch.reward)
        reward_batch = [np.array(data).astype(np.float32) for data in reward_batch]
        done_batch = list(minibatch.done)
        
        y_batch = []
        q_batch = self.sess.run(self.q_fixed, feed_dict={self.input_fixed_frame: next_state_batch})
        for i in range(self.batch_size):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y = reward_batch[i] + self.gamma * np.max(q_batch[i])
                y_batch.append(y)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict = { self.input_active_frame: state_batch, self.input_y: y_batch, self.input_action: action_batch})
        if self.step % self.replace_num ==0:
            self.sess.run(self.replace_target_op)
            print("Target replaced")
        return loss


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = observation.reshape((1, 84, 84, 4))
        q_value = self.sess.run(self.q_active, feed_dict={self.input_active_frame: state})[0]
        if test:
            if random.random() <= 0.025:
                return random.randrange(self.n_actions)
            return np.argmax(q_value)

        if random.random() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else : 
            action = np.argmax(q_value)

        if self.epsilon > self.epsilon_end and self.step > 30000:
            self.epsilon -= self.epsilon_decay_constant

        return action

    class ReplayMemory(object):
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, *args):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    def storeTransition(self, s, action, s_, reward, done):
        """
        Store transition in this step
        Input:
            s: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            s_: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            action: int (0, 1, 2, 3)
                the predicted action from trained model
            reward: float64 (0, +1, -1)
                the reward from selected action
        Return:
            None
        """
        self.step = self.sess.run(self.add_global)
        
        #np.set_printoptions(threshold=np.nan)
        assert np.amin(s) >= 0.0
        assert np.amax(s) <= 1.0
        
        s  = (s * 255).round().astype(np.uint8)
        s_ = (s_ * 255).round().astype(np.uint8)
        #print(sys.getsizeof(image)) # 28352, uint8
        #print(sys.getsizeof(s)) # 113024, float32
        
        self.memory.push(s, int(action), s_, int(reward), done)

    def build_network(self, input_frame, var_name):
        with tf.variable_scope(var_name):
            print ("input shape: ",input_frame.shape)
            conv1 = self.conv2d(input_frame, 32, 8, 8, s_h=4, s_w=4, name="conv1")
            print("conv1 shape: ", conv1.shape)
            conv2 = self.conv2d(conv1, 64, 4, 4, name="conv2")
            print("conv2 shape: ", conv2.shape)
            conv3 = self.conv2d(conv2, 64, 3,3, s_h=1, s_w=1, name="conv3")
            print("conv3 shape: ", conv3.shape)

            flatten = tf.reshape(conv3,[-1,3136])
            print("flattend shape: " , flatten.shape)

            W1 = tf.get_variable("W1",[3136,512],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            B1 = tf.get_variable("B1",[512],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            flatten_1 = tf.matmul(flatten,W1) + B1
            print("DNN_1: ",flatten_1.shape)
            if self.dueling:
                vW2 = tf.get_variable("vW2",[512,1],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                vB2 = tf.get_variable("vB2",[1],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                v_value = tf.matmul(flatten_1, vW2) + vB2
                print("v_value:",v_value)
                action_W2 = tf.get_variable("action_vW2",[512,self.n_actions],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                action_B2 =  tf.get_variable("action_B2",[self.n_actions],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                action_value = tf.matmul(flatten_1, action_W2) + action_B2
                print("action_value", action_value)
                print(tf.reduce_mean(action_value, axis = 1, keep_dims=True))
                print(tf.reduce_mean(action_value, axis = 1,))
                q_value = v_value + (action_value-tf.reduce_mean(action_value, axis = 1, keep_dims=True))
            else:
                W2 = tf.get_variable("W2",[512,self.n_actions],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                B2 = tf.get_variable("B2",[self.n_actions],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                q_value = tf.matmul(flatten_1,W2) + B2
                print("DNN_2: ",q_value.shape)


        return q_value

    def build_optimizer(self):
        self.q_value = tf.reduce_sum(tf.multiply(self.q_active, self.input_action), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.input_y - self.q_value))
        self.train_op = self.build_rmsprop_optimizer(self.lr, 0.99, 1e-6, 1)


    def conv2d(self, input_image, output_dim, f_h=5, f_w=5, 
           s_h=2, s_w=2, stddev=0.02, name="conv"):
        ### filter: [height, width, in_channels, output_channels]
        conv_filter = tf.get_variable(name + '_w', [f_h, f_w, input_image.shape[-1], output_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_image, conv_filter, strides=[1, s_h, s_w, 1], padding='VALID')

        biases = tf.get_variable(name + '_b', [output_dim], 
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
    
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.leaky_relu(conv)

        return conv

    # https://github.com/Jabberwockyll/deep_rl_ale
    def build_rmsprop_optimizer(self, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip):

        with tf.name_scope('rmsprop'):
            optimizer = None
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=0.0, epsilon=rmsprop_constant)

            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads = [gv[0] for gv in grads_and_vars]
            params = [gv[1] for gv in grads_and_vars]
            # print(grads)
            if gradient_clip > 0:
                grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

            grads = [grad for grad in grads if grad != None]

            return optimizer.apply_gradients(zip(grads, params))

