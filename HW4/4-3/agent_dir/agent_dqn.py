from agent_dir.agent import Agent
import tensorflow as tf
from collections import namedtuple
import numpy as np
import random
import sys

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)

        self.batch_size = args.batch_size
        #self.dueling = args.dueling_dqn
        self.gamma = args.gamma
        self.n_actions = env.action_space.n
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.num_episodes = args.num_episodes
        self.epsilon = 1
        self.replace_num = args.replace_num
        self.update_num = args.update_num
        self.epsilon_decay_constant = args.epsilon_decay_constant
        self.epsilon_end = args.epsilon_end
        self.step = 0
        self.saver_steps = args.saver_steps
        self.output_logs = args.output_logs 
        self.num_training = 0
        self.training_steps = args.training_steps

        self.input_fixed_frame = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_fixed_frame')
        self.input_active_frame = tf.placeholder(tf.float32, [None, 84, 84, 4], name='input_active_frame')
        self.intput_action = tf.placeholder(shape=[None, self.n_actions], dtype=tf.float32,name='intput_action')
        self.input_advantage = tf.placeholder(shape=[None,1], dtype=tf.float32,name='input_advantage')
        self.input_y = tf.placeholder(tf.float32, [None,1],name='input_y')
        self.place_epsilon = tf.placeholder(tf.float32,name='input_y')
        

        self.active_pi, self.active_value = self.build_actor_critic_net(self.input_active_frame, "active_network")
        self.fixed_pi, self.fixed_value = self.build_actor_critic_net(self.input_fixed_frame,"fixed_network")
        self.avtive_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='active_network')
        self.fixed_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fixed_network')
        self.assign_network = [tf.assign(t, e) for t, e in zip(self.fixed_params, self.avtive_params)]
        self.if_epsilon = args.if_epsilon
        self.one = tf.constant(1,dtype=tf.float32)
        self.four = tf.constant(self.n_actions,dtype=tf.float32)

        print("input_y",self.input_y)
        print("activ_value",self.active_value)
        self.adv = self.input_y - self.active_value
        self.critic_loss = tf.reduce_mean(tf.square(self.adv))

        
        #ratio = self.active_pi.prob(self.intput_action) / self.fixed_pi.prob(self.intput_action)
        if self.if_epsilon:
            pi_prob = tf.boolean_mask(self.active_pi,self.intput_action)
            pi_prob = pi_prob*(self.one - self.epsilon) + self.epsilon/self.four
            print("pi_prob", pi_prob)
            old_pi_prob = tf.boolean_mask(self.fixed_pi,self.intput_action)
            old_pi_prob = old_pi_prob*(self.one - self.epsilon) + self.epsilon/self.four
            ratio = pi_prob/old_pi_prob
        else :
            pi_prob = tf.boolean_mask(self.active_pi,self.intput_action)
            print("pi_prob", pi_prob)
            old_pi_prob = tf.boolean_mask(self.fixed_pi,self.intput_action)
            ratio = pi_prob/old_pi_prob
        print("raion:",ratio)
        pg_loss1 = self.input_advantage * tf.clip_by_value(ratio, 0.8, 1.2)
        pg_loss2 = self.input_advantage * ratio
        self.actor_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))

        self.train_op_actor = tf.train.AdamOptimizer(args.actor_learning_rate).minimize(self.actor_loss, var_list=self.avtive_params)
        self.train_op_critic = tf.train.AdamOptimizer(args.critic_learning_rate).minimize(self.critic_loss, var_list=self.avtive_params)
        
    
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
        current_loss_a = 0
        current_loss_c = 0
        rewards = 0
    
        file_loss = open(self.output_logs, "a")
        file_loss.write("episode,step,epsilon,reward,loss_actor,loss_critic\n")
        self.done = 0
        for episode in range(self.num_episodes):
            prev_state = self.env.reset()
            self.done = 0
            loss_a = 0
            loss_c = 0
            episode_reward = 0
            update_count = 0
            while self.done==0:
                states_buf = []
                actions_buf = []
                rewards_buf = []
                for _ in range(self.batch_size):
                    self.step = self.sess.run(self.add_global)
                    action = self.make_action(prev_state, test = False)
                    self.env.env.render()
                    next_state, reward, self.done, _ = self.env.step(action)
                    episode_reward += reward
                    states_buf.append(prev_state)
                    actions_buf.append(action)
                    rewards_buf.append(reward)
                    prev_state = next_state
                    if self.done:
                        break
                last_value = self.sess.run(self.active_value, feed_dict = {self.input_active_frame: next_state.reshape((1,84,84,4))})[0]
                if states_buf:
                    update_count+=1
                    discounted_r = []
                    v_s = last_value
                    print(v_s)
                    for r in rewards_buf[::-1]:
                        v_s = r + v_s*self.gamma
                        discounted_r.append(v_s)
                    discounted_r.reverse()
                    action_batch = []
                    for act in list(actions_buf):
                        one_hot_action = np.zeros(self.n_actions)
                        one_hot_action[act] = 1
                        action_batch.append(one_hot_action)
                    bs, ba, br = np.array(states_buf), np.vstack(action_batch), np.array(discounted_r)
                    #print(br.shape,br)
                    loss_a, loss_c = self.learn(bs,ba,br)
                    current_loss_a += loss_a
                    current_loss_c += loss_c
            if update_count>0 :
                current_loss_a /= update_count
                current_loss_c /= update_count
                update_count = 0
            rewards+= episode_reward
            loss_a += current_loss_a
            loss_c += current_loss_c
            if episode%25==0 and episode>0:
                file_loss.write(str(episode) + "," + str(self.step) + "," + "{:.4f}".format(self.epsilon) + "," + "{:.2f}".format(rewards/25.0) + "," + "{:.4f}".format(loss_a) + "," +"{:.4f}".format(loss_c) + "\n")
                file_loss.flush()
                print("episode:",episode)
                print("step:", self.step)
                print("reword:",rewards/25.0)
                print("loss_a:", current_loss_a)
                print("loss_c:", current_loss_c)
                rewards = 0
                loss_a = 0
                loss_c = 0
                
            
#self.learn
    def learn(self, state, action, reward):
        self.sess.run(self.assign_network)
        self.num_training += 1
        print(reward.dtype)
        adv = self.sess.run(self.adv, feed_dict={self.input_active_frame: state,self.input_y: reward})

        if self.if_epsilon:
            [self.sess.run([self.train_op_actor, self.train_op_critic], feed_dict={self.input_active_frame: state, self.input_fixed_frame: state, self.intput_action: action,self.input_advantage: adv,self.input_y: reward, self.place_epsilon: self.epsilon}) for _ in range(self.training_steps)]
        else:
            [self.sess.run([self.train_op_actor, self.train_op_critic], feed_dict={self.input_active_frame: state, self.input_fixed_frame: state, self.intput_action: action,self.input_advantage: adv,self.input_y: reward}) for _ in range(self.training_steps)]
        
        #[self.sess.run(self.train_op_critic, feed_dict={self.input_active_frame: state, self.input_y: reward})]
        loss_a, loss_c = self.sess.run([self.actor_loss, self.critic_loss], feed_dict={self.input_active_frame: state, self.input_fixed_frame: state, self.intput_action: action,self.input_advantage: adv ,self.input_y: reward})
        if self.num_training%500==0 :
            self.saver.save(self.sess, self.ckpts_path, global_step = self.step)
            print("save_model, update:", self.num_training*10)

        return loss_a, loss_c
        


                    
                        



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
        state = observation.reshape((1,84,84,4))

        if test:
            print("cool")
        else:
            if self.if_epsilon:
                if random.random() <= self.epsilon:
                    action = random.randrange(self.n_actions)
                else :
                    action_prob = self.sess.run(self.active_pi.sample(1),feed_dict={self.input_active_frame: state})
                    action = np.random.choice(self.n_actions, 1, p = np.squeeze(action_prob))
                    action = np.squeeze(action)
            else:
                action_prob = self.sess.run(self.active_pi,feed_dict={self.input_active_frame: state})
                action = np.random.choice(self.n_actions, 1, p = np.squeeze(action_prob))
                action = np.squeeze(action)
        if self.epsilon > self.epsilon_end and self.step > 30000:
            self.epsilon -= self.epsilon_decay_constant
        return action

    
    

    def build_actor_critic_net(self, input_frame, var_name):
        with tf.variable_scope(var_name):
            conv1 = self.conv2d(input_frame, 32, 8, 8, s_h=4, s_w=4, name="conv1")
            conv2 = self.conv2d(conv1, 64, 4, 4, name="conv2")
            conv3 = self.conv2d(conv2, 64, 3,3, s_h=1, s_w=1, name="conv3")
            flatten = tf.reshape(conv3,[-1,3136])
            W1 = tf.get_variable("W1",[3136,512],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            B1 = tf.get_variable("B1",[512],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            flatten_1 = tf.matmul(flatten,W1) + B1
            print("DNN_1: ",flatten_1.shape)

            actor_w = tf.get_variable("actor_w",[512,self.n_actions],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            actor_b = tf.get_variable("actor_b",[self.n_actions],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            action_pre = tf.matmul(flatten_1,actor_w) + actor_b
            action_prob = tf.nn.softmax(action_pre)

            critique_w = tf.get_variable("critique_w",[512,1],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            critique_b = tf.get_variable("critique_b",[1],tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            value = tf.matmul(flatten_1,critique_w) + critique_b
            print("value",value)
            return action_prob, value
            


            


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

