import tensorflow as tf
import pickle
import numpy as np
from my_data import Dataset
from my_data import test_Dataset
class AttentionModel():
    def __init__(self,lr,n_hidden,batch_size,vocab_num,model_name,path,test=False):
        self.lr = lr
        self.loss = None
        self.logits = None
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.wv_dim = 300  ### dim of word vector is 300
        self.i_maxlen = 30   ### max length of input and output sentence is 30 words
        self.o_maxlen = 29
        self.vocab_num = vocab_num
        self.grad_rate = 5.0
        self.save_path = path
        self.name = model_name
        self.test_flag = test
    
    def inference(self, pred):    
        index = tf.argmax(pred, 2)
        return index
    
    def load_index2vec(self):
        with open("./data/index2vec.pkl", "rb") as r:
            index2vec_list = pickle.load(r)
            emb = np.array(index2vec_list,dtype=np.float32)
        return emb

    def onehot_to_index(self, logits):
        indexs = tf.argmax(logits, 2)
        return indexs

    def build_model(self,input_s,output_s,label_len,mask):
        
        ### for label to word vector look up
        index2vec = tf.constant(self.load_index2vec()) # shape (self.vocab_num,self.wv_dim)

        w = {
            ### a layer that transform dim of word vector (300) to dim of n_hidden
            'wv_to_hidden': tf.Variable(tf.truncated_normal([self.wv_dim, self.n_hidden], -0.5, 0.5), name='wv_to_hidden'),
            'attention1' : tf.Variable(tf.truncated_normal([2*self.n_hidden,self.n_hidden],-0.5,0.5)),
            'attention2' : tf.Variable(tf.truncated_normal([self.n_hidden,self.n_hidden],-0.5,0.5)),
            'hidden_to_output' : tf.Variable(tf.truncated_normal([self.n_hidden,self.vocab_num],-0.5,0.5))
        }

        ### bias
        b = {
            'wv_to_hidden' : tf.Variable(tf.truncated_normal([self.n_hidden], -0.5, 0.5), name='Bwv_to_hidden'),
            'attention1' : tf.Variable(tf.truncated_normal([self.n_hidden],-0.5,0.5)),
            'attention2' : tf.Variable(tf.truncated_normal([self.n_hidden],-0.5,0.5)),
            'hidden_to_output' : tf.Variable(tf.truncated_normal([self.vocab_num],-0.5,0.5))
        }
        
        with tf.variable_scope('LSTM'):
            ### Two stage of LSTM, one for encoding and one for decoding. state is a tuple ([n_hidden],[n_hidden])
            encoder = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True) 
            decoder = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        #lstm_gre = tf.contrib.rnn.DropoutWrapper(lstm_gre, output_keep_prob=dropout_prob)

        ### Initial states: h0,z0 
        e_state = encoder.zero_state(self.batch_size, dtype=tf.float32) # (batch_size,(n_hidden,n_hidden))
        d_state = decoder.zero_state(self.batch_size, dtype=tf.float32) # state is in form of tuple

        ### Always with attention, so the input would be the prediction(dim=n_hidden) +
        ### attention(dim=n_hideen), dim = 2*n_hidden
        #padding = tf.zeros([batch_size, n_hidden + n_hidden])
        ### Embedding input to n_hidden dim
        input_s = tf.reshape(input_s,[-1,self.wv_dim])
        emb_input = tf.matmul(input_s,w['wv_to_hidden']) + b['wv_to_hidden']
        emb_input = tf.reshape(emb_input,[self.batch_size,self.i_maxlen,self.n_hidden])
        h_database = []        
        for i in range(0, self.i_maxlen):
            with tf.variable_scope("LSTM", reuse=(i!=0)):
                ### Encoding stage, eats one word vector at a time.
                e_output, e_state = encoder(emb_input[:,i,:], e_state)
                h_database.append(e_output) # store the outputs to perform attention
        h_database = tf.stack(h_database,axis=1) # (batch_size,i_maxlen,n_hidden)
        ### Attention, Z shape = (batch_size,n_hidden)
        ### runs Z into a same DNN then multiply it with h_database to get alpha
        ### finally use alpha to get C
        def attention(Z):
            state_tensor = tf.concat(Z ,axis=1) # (batch_size,n_hidden*2)
            op_1 = tf.matmul(state_tensor,w['attention1']) + b['attention1'] # (bs,n_hidden)
            op_2 = tf.matmul(op_1,w['attention2']) + b['attention2'] # (bs,n_hidden)
            alpha = tf.matmul(h_database,tf.expand_dims(op_2,2)) # op_2 to (bs,n_hidden,1), alpha (bs,i_maxlen,1)
            softmax_alpha = tf.nn.softmax(alpha) # (bs,i_maxlen,1)
            softmax_alpha = tf.transpose(softmax_alpha,[0,2,1]) # (bs,1,i_maxlen)
            C_i = tf.matmul(softmax_alpha,h_database) # (bs,1,n_hidden)
            C_i = tf.squeeze(C_i) ##(bs,n_hidden)
            return C_i

        ### Decoding stage, output shape = (batch_size,n_hidden)
        ### for i = 0, when there is no ouput yet.
        pad_in = tf.zeros([self.batch_size, self.n_hidden])
        outputs = []
        total_cross_entropy = []
        for i in range(0, self.o_maxlen):
            C_i = attention(d_state)
            if i == 0:
                #with tf.variable_scope("LSTM", reuse=(i!=0)):
                d_output, d_state = decoder(tf.concat([C_i, pad_in], axis=1), e_state) # d_output shape (bs,n_hidden)
            else:
                if (not self.test_flag):
                    feed_in_label = output_s[:,i-1]
                    label_wv = tf.nn.embedding_lookup(index2vec, feed_in_label) # (bs,word_vec_dim)
                    emb_label = tf.matmul(label_wv,w['wv_to_hidden']) + b['wv_to_hidden'] # (bs,n_hidden)
                    d_output, d_state = decoder(tf.concat([C_i, emb_label], axis=1), d_state)
                else:
                    d_output, d_state = decoder(tf.concat([C_i, d_output], axis=1), d_state)
            output = tf.matmul(d_output,w['hidden_to_output']) + b['hidden_to_output'] # output shape (bs,vocab_num)
            outputs.append(output) 
            ### Calculating cross_entropy 
            labels = output_s[:,i] # output_s shape (bs,o_maxlen)
            one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value = 1, off_value = None, axis = 1) # one_hot_labels (bs,vocal_num)
            cross_entropy_per_word = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=one_hot_labels) # (bs,)
            cross_entropy_per_word = cross_entropy_per_word * mask[:,i] 
            total_cross_entropy.append(cross_entropy_per_word) # (o_maxlen,bs,)
        
        ### Calculating total loss
        cross_entropy_tensor = tf.stack(total_cross_entropy,axis=1) # cross_entropy_tensr (bs,o_maxlen)
        loss = tf.reduce_sum(cross_entropy_tensor, axis=1) # loss (bs,)
        loss = tf.divide(loss,tf.cast(label_len,tf.float32)) # label_len is (bs,)
        loss = tf.reduce_mean(loss,axis=0) # loss is a float32

        ### Optimizer
        paras = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.lr)
        grad, var = zip(*opt.compute_gradients(loss))
        grad, _ = tf.clip_by_global_norm(grad, self.grad_rate)
        train_op = opt.apply_gradients(zip(grad, paras))

        ### pred, outputs (maxlen,bs,vocab_num)
        pred = tf.stack(outputs,axis = 1) # pred (bs,o_maxlen,vocab_num)

        return pred, loss, train_op

    def train(self,epoch,load=False):
        datasetTrain = Dataset(self.batch_size)
        datasetTrain.word2vec()
        input_s = tf.placeholder(tf.float32,[self.batch_size,self.i_maxlen,self.wv_dim],name='questions')
        output_s = tf.placeholder(tf.int32,[self.batch_size,self.o_maxlen])
        label_len = tf.placeholder(tf.int32,[self.batch_size])
        mask = tf.sequence_mask(label_len, self.o_maxlen, dtype=tf.float32)
        pred, loss, train_op = self.build_model(input_s,output_s,label_len,mask)

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=2)
            if not load:
                sess.run(tf.global_variables_initializer())
                print("New Model: ", self.name)
            else:
                latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
                saver.restore(sess, latest_checkpoint)
                print("Saver Loaded: " + latest_checkpoint)
            PATH = self.save_path + self.name + ".ckpt"
            
            for epo in range(0, epoch):
                if (epo != 0):
                    ckpt_path = saver.save(sess, PATH, global_step=epo)
                    print("Model saved: ",ckpt_path)
                datasetTrain.shuffle_perm()
                num_steps = int(datasetTrain.batch_max_size/self.batch_size)
                epo_loss = 0
                for i in range(0, num_steps):
                    data_batch, label_batch, caption_lens_batch = datasetTrain.next_batch()
                    _, l = sess.run([train_op, loss], 
                                        feed_dict={input_s: data_batch,
                                                output_s: label_batch,
                                                label_len: caption_lens_batch,
                                            })

                    epo_loss += l
                    print("num_step: ", i," loss: ",l)
                print("\n[FINISHED] Epoch " + str(epo) + \
                        ", (Training Loss (per epoch): " + "{:.4f}".format(epo_loss))
            print('\n\nTraining finished!')
            
    def test(self):
        datasetTrain = Dataset(self.batch_size)
        datasetTrain.word2vec()
        batch_size = 10
        datasetTest = test_Dataset(batch_size)
        vocab_num = 38103
        input_s = tf.placeholder(tf.float32,[self.batch_size,self.i_maxlen,self.wv_dim],name='questions')
        output_s = tf.placeholder(tf.int32,[self.batch_size,self.o_maxlen])
        label_len = tf.placeholder(tf.int32,[self.batch_size])
        mask = tf.sequence_mask(label_len, self.o_maxlen, dtype=tf.float32)
        
        pred, _, _ = self.build_model(input_s,output_s,label_len,mask)
        pred_index = self.inference(pred)
        
        saver = tf.train.Saver(max_to_keep=6)
        latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
        
        num_steps = int( datasetTest.batch_max_size / self.batch_size)
        txt = open("outputs.txt", 'w')
        with tf.Session() as sess:
            saver.restore(sess, latest_checkpoint)
            print("Saver Loaded: " + latest_checkpoint)
            for i in range(0, num_steps):
                _, label_batch, caption_lens_batch = datasetTrain.next_batch()
                data_batch = datasetTest.next_batch()
                p = sess.run(pred_index, feed_dict={input_s: data_batch,
                                                    output_s: label_batch,
                                                    label_len: caption_lens_batch})
                seq = datasetTest.index_to_sentence(p)
                for j in range(0, self.batch_size):
                    for w in range(len(seq[j])):
                        txt.write(seq[j][w] + " ")
                    txt.write("\n")
                print("Inference: " + str((i+1) * self.batch_size) + "/" + \
                        str(datasetTest.batch_max_size) + ", done..." )
            txt.close()
            print('\n\nTesting finished.')
            print('\n Save file: ' + "output.txt")

