from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import tensorflow as tf
import numpy as np
import os
import sys
import json
import pandas as pd
import argparse
import random
import pickle
import json
from preprocessDataTrain import DatasetTrain
from preprocessDataTest import DatasetTest
from util import linear_decay, dec_print_train, dec_print_val, dec_print_test
from subprocess import call

FLAGS = None

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

n_inputs        = 4096
n_hidden        = 512
val_batch_size  = 100 #100
n_frames        = 80
max_caption_len = 50
dropout_prob    = 0.8
n_attention     = n_hidden

special_tokens  = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
phases = {'train': 0, 'val': 1, 'test': 2}

class S2VT:
    def __init__(self, vocab_num = 0, 
                       with_attention = True, 
                       lr = 1e-4):

        self.vocab_num = vocab_num
        self.with_attention = with_attention
        self.learning_rate = lr
        self.saver = None
    def set_saver(self,saver):
        self.saver = saver
    def build_model(self, feat, captions=None, cap_len=None, sampling=None, phase=0):

        weights = {
            'W_feat': tf.Variable( tf.random_uniform([n_inputs, n_hidden], -0.1, 0.1), name='W_feat'), 
            'W_dec': tf.Variable(tf.random_uniform([n_hidden, self.vocab_num], -0.1, 0.1), name='W_dec')
        }
        biases = {
            'b_feat':  tf.Variable( tf.zeros([n_hidden]), name='b_feat'),
            'b_dec': tf.Variable(tf.zeros([self.vocab_num]), name='b_dec')
        }   
        embeddings = {
         'emb': tf.Variable(tf.random_uniform([self.vocab_num, n_hidden], -0.1, 0.1), name='emb')
        }

        if self.with_attention:   ###with attention
            weights['w_enc_out'] =  tf.Variable(tf.random_uniform([n_hidden, n_hidden]), 
                dtype=tf.float32, name='w_enc_out')
            weights['w_dec_state'] =  tf.Variable(tf.random_uniform([n_hidden, n_hidden]), 
                dtype=tf.float32, name='w_dec_state')
            weights['v'] = tf.Variable(tf.random_uniform([n_hidden, 1]), 
                dtype=tf.float32, name='v')

        batch_size = tf.shape(feat)[0]

        if phase != phases['test']: 
            cap_mask = tf.sequence_mask(cap_len, max_caption_len, dtype=tf.float32)
     
        feat = tf.reshape(feat, [-1, n_inputs])           ###encoding feat to n_hidden length
        image_emb = tf.matmul(feat, weights['W_feat']) + biases['b_feat']
        image_emb = tf.reshape(image_emb, [-1, n_frames, n_hidden])
        image_emb = tf.transpose(image_emb, perm=[1, 0, 2])
        
        with tf.variable_scope('LSTM1'):
            lstm_red = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
            if phase == phases['train']:
                lstm_red = tf.contrib.rnn.DropoutWrapper(lstm_red, output_keep_prob=dropout_prob)    
        with tf.variable_scope('LSTM2'):
            lstm_gre = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
            if phase == phases['train']:
                lstm_gre = tf.contrib.rnn.DropoutWrapper(lstm_gre, output_keep_prob=dropout_prob)    

        state_red = lstm_red.zero_state(batch_size, dtype=tf.float32)
        state_gre = lstm_gre.zero_state(batch_size, dtype=tf.float32)

        if self.with_attention:   
            padding = tf.zeros([batch_size, n_hidden + n_attention])
        else:
            padding = tf.zeros([batch_size, n_hidden])

        h_src = []        
        for i in range(0, n_frames):
            with tf.variable_scope("LSTM1", reuse=(i!=0)):
                output_red, state_red = lstm_red(image_emb[i,:,:], state_red)
                h_src.append(output_red)
            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output_gre, state_gre = lstm_gre(tf.concat([padding, output_red], axis=1), state_gre)
                 # even though padding is augmented, output_gre/state_gre's shape not change

        h_src = tf.stack(h_src, axis = 0)
        bos = tf.ones([batch_size, n_hidden])
        padding_in = tf.zeros([batch_size, n_hidden])
        logits = []
        max_prob_index = None

        if self.with_attention:
            def bahdanau_attention(time, prev_output=None):
                
                if time == 0:
                    H_t = h_src[-1,:, :] # encoder last output as first target input, H_t
                else:
                    H_t = prev_output

                H_t = tf.matmul(H_t, weights['w_dec_state'])
                H_s = tf.identity(h_src) # copy
                    
                H_s = tf.reshape(H_s, (-1, n_hidden))
                score = tf.matmul(H_s, weights['w_enc_out'])
                score = tf.reshape(score, (-1, batch_size, n_hidden))
                score = tf.add(score, tf.expand_dims(H_t, 0))
                
                score = tf.reshape(score, (-1, n_hidden))
                score = tf.matmul(tf.tanh(score), weights['v'])
                score = tf.reshape(score, (n_frames, batch_size, 1))
                score = tf.nn.softmax(score, dim=-1, name='alpha')

                H_s = tf.reshape(H_s, (-1, batch_size, n_hidden))
                C_i = tf.reduce_sum(tf.multiply(H_s, score), axis=0)
                return C_i

        cross_ent_list = []
        for i in range(0, max_caption_len):
            with tf.variable_scope("LSTM1", reuse=(i!=0)):
                output_red, state_red = lstm_red(padding_in, state_red)

            if i == 0:
                with tf.variable_scope("LSTM2", reuse=(i!=0)):
                    con = tf.concat([bos, output_red], axis=1)
                    if self.with_attention:
                        C_i = bahdanau_attention(i)
                        con = tf.concat([con, C_i], axis=1)

                    output_gre, state_gre = lstm_gre(con, state_gre)
            else:
                if phase == phases['train']:
                    if sampling[i] == True:
                        feed_in = captions[:, i - 1]
                    else:
                        feed_in = tf.argmax(logit_words, 1)
                else:
                    feed_in = tf.argmax(logit_words, 1)
                with tf.device("/cpu:0"):
                    embed_result = tf.nn.embedding_lookup(embeddings['emb'], feed_in)
                with tf.variable_scope("LSTM2"):
                    con = tf.concat([embed_result, output_red], axis=1)
                    if self.with_attention:
                        C_i = bahdanau_attention(i, state_gre[1]) 
                        con = tf.concat([con, C_i], axis=1)
                    output_gre, state_gre = lstm_gre(con, state_gre)

            logit_words = tf.matmul(output_gre, weights['W_dec']) + biases['b_dec']
            logits.append(logit_words)

            if phase != phases['test']:
                labels = captions[:, i]
                one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value = 1, off_value = None, axis = 1) 
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=one_hot_labels)
                cross_entropy = cross_entropy * cap_mask[:, i]
                cross_ent_list.append(cross_entropy)
        
        loss = 0.0
        if phase != phases['test']:
            cross_entropy_tensor = tf.stack(cross_ent_list, 1)
            loss = tf.reduce_sum(cross_entropy_tensor, axis=1)
            loss = tf.divide(loss, tf.cast(cap_len, tf.float32))
            loss = tf.reduce_mean(loss, axis=0)

        logits = tf.stack(logits, axis = 0)
        logits = tf.reshape(logits, (max_caption_len, batch_size, self.vocab_num))
        logits = tf.transpose(logits, [1, 0, 2])
        
        return logits, loss

    def inference(self, logits):
        dec_pred = tf.argmax(logits, 2)
        return dec_pred

    def optimize(self, loss_op):
        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, params))

        return train_op


def train(_):
    datasetTrain = DatasetTrain(FLAGS.data_dir, FLAGS.batch_size)
    datasetTrain.build_train_data_obj_list()
    vocab_num = datasetTrain.dump_tokenizer()

    print("vocab_num: " + str(vocab_num))

    feat = tf.placeholder(tf.float32, [None, n_frames, n_inputs], name='video_features')
    captions = tf.placeholder(tf.int32, [None, max_caption_len], name='captions')
    sampling = tf.placeholder(tf.bool, [max_caption_len], name='sampling')
    cap_len = tf.placeholder(tf.int32, [None], name='cap_len')
    model = S2VT(vocab_num=vocab_num, with_attention=FLAGS.with_attention, 
                    lr=FLAGS.learning_rate)
    logits, loss_op = model.build_model(feat, captions, cap_len, sampling, phases['train'])
    dec_pred = model.inference(logits)
    train_op = model.optimize(loss_op)

    with tf.Session() as train_sess:

        ###saver
        model.set_saver(tf.train.Saver(max_to_keep = 5))
        if not FLAGS.load_saver:
            train_sess.run(tf.global_variables_initializer())
            print("No saver was loaded")
        else:
            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
            model.saver.restore(train_sess, latest_checkpoint)
            print("Saver Loaded: " + latest_checkpoint)
        ckpts_path = FLAGS.save_dir + "save.ckpt"
        
        for epo in range(0, FLAGS.num_epoches):
            datasetTrain.shuffle_perm()
            num_steps = int( datasetTrain.batch_max_size / FLAGS.batch_size )
            epo_loss = 0
            print(num_steps,datasetTrain.batch_max_size)
            for i in range(0, num_steps):
                data_batch, label_batch, caption_lens_batch, id_batch = datasetTrain.next_batch()
                samp = datasetTrain.schedule_sampling(epo, caption_lens_batch)
                _, loss, p = train_sess.run([train_op, loss_op, dec_pred], 
                                    feed_dict={feat: data_batch,
                                            captions: label_batch,
                                            cap_len: caption_lens_batch,
                                            sampling: samp})

                epo_loss += loss
                print("num_step: ", i," loss: ",loss)
            print("\n[FINISHED] Epoch " + str(epo) + \
                    ", (Training Loss (per epoch): " + "{:.4f}".format(epo_loss))
            if epo % FLAGS.num_saver_epoches == 0:
                ckpt_path = model.saver.save(train_sess, ckpts_path, global_step=1)
                print("\nSaver saved: " + ckpt_path)

def test(_):
    datasetTest = DatasetTest(FLAGS.data_dir, FLAGS.test_dir, 100)
    datasetTest.build_test_data_obj_list()
    vocab_num = datasetTest.load_tokenizer()

    feat = tf.placeholder(tf.float32, [None, n_frames, n_inputs], name='video_features')
    model = S2VT(vocab_num=vocab_num, with_attention=FLAGS.with_attention)
    logits, _, = model.build_model(feat, phase=phases['test'])
    dec_pred = model.inference(logits)

    model.set_saver(tf.train.Saver(max_to_keep=5))

    saver_path = FLAGS.save_dir
    print('saver path: ' + saver_path)
    latest_checkpoint = tf.train.latest_checkpoint(saver_path)

    num_steps = int( datasetTest.batch_max_size / 100)

    with tf.Session() as train_sess:
        output = []
        model.saver.restore(train_sess, latest_checkpoint)
        print("Saver Loaded: " + latest_checkpoint)
        for i in range(0, num_steps):
            data_batch, id_batch = datasetTest.next_batch()
            p = train_sess.run(dec_pred, feed_dict={feat: data_batch})
            seq = dec_print_test(p, datasetTest.idx_to_word, 100, id_batch)
            this_answer = []
            for j in range(0, 100):
                this_answer.append((seq[j],id_batch[j]))
            print("Inference: " + str((i+1) * 100) + "/" + \
                    str(datasetTest.batch_max_size) + ", done..." )

        with open(FLAGS.output_filename, 'w') as fout:
            for ans in this_answer:
               fout.write(ans[1] + ',' + ans[0] + '\n')
        print('\n\nTesting finished.')
        print('\n Save file: ' + FLAGS.output_filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-e', '--num_epoches', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-t', '--test_mode', type=int, default=0)
    parser.add_argument('-d', '--num_display_steps', type=int, default=15)
    parser.add_argument('-ns', '--num_saver_epoches', type=int, default=5)
    parser.add_argument('-s', '--save_dir', type=str, default='save/')    
    parser.add_argument('-l', '--log_dir', type=str, default='logs/')
    parser.add_argument('-o', '--output_filename', type=str, default='output.txt')
    parser.add_argument('-lo', '--load_saver', type=int, default=0)
    parser.add_argument('-at', '--with_attention', type=int, default=1)
    parser.add_argument('--data_dir', type=str, 
        default=('./MLDS_hw2_data')
    )
    parser.add_argument('--test_dir', type=str, 
        default=('./MLDS_hw2_data/testing_data')
    )
    opts = parser.parse_args()
    FLAGS, unparsed = parser.parse_known_args()
    if(FLAGS.test_mode):
        tf.app.run(main=test, argv=[sys.argv[0]] + unparsed)
    else:	
        tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)


