import pandas as pd
import numpy as np
import pickle
import sys

embedding_length = 30
max_caption_len = 29

class DataObject:
    def __init__(self, input_sentence, output_index, sentence_len = None):
        self.input_sentence = input_sentence
        self.output_index = output_index
        self.sentence_len = sentence_len # no EOS, e.g. ['I', 'love', 'you']

class Dataset:    #for train
    def __init__(self,batch_size):
        self.perm = None # permutation numpy array
        self.batch_size = batch_size
        self.data_obj_list = []
        self.batch_max_size = 0
        self.batch_index = 0
        
    def word2vec(self):
        quest_data_path = './data/quest_data.txt'
        word2vec_path   = './data/word2vec_v2.pkl'
        with open(quest_data_path, 'rb') as f:     #load the quest_data
            quest_data = pickle.load(f)

        with open(word2vec_path, 'rb') as w:       #load the word2vec.pkl
            w2v = pickle.load(w)
        
        input_mat = []

        for sentence in range (len(quest_data)):   #through 900000 sentences
            input_mat.append([])
            for words in range (embedding_length): #max length of sentence = 27, so choose 30
                input_mat[sentence].append([])
                if words == 0:                     #BOS = np.full(0.9), EOS=np.full(-0.9), PAD = np.zeros
                    input_mat[sentence][words] = w2v['BOS']
                elif words == len(quest_data[sentence]) + 1:
                    input_mat[sentence][words] = w2v['EOS']
                elif words > len(quest_data[sentence]) + 1:
                    input_mat[sentence][words] = w2v['PAD']
                else:
                    input_mat[sentence][words] = w2v[quest_data[sentence][words - 1]]
        
        length = np.load("./data/mask.npy")
        onehot = np.load("./data/output_id.npy")
        for i in range(len(length)):
            obj = DataObject(input_mat[i],onehot[i],length[i])
            self.data_obj_list.append(obj)
        self.batch_max_size = len(length)
        self.data_obj_list = np.array(self.data_obj_list)
        self.perm = np.arange( self.batch_max_size, dtype=np.int )

    def shuffle_perm(self):
        np.random.shuffle( self.perm )
    
    def next_batch(self): 
        current_index = self.batch_index
        max_size = self.batch_max_size

        if current_index + self.batch_size <= max_size:
            dat_list = self.data_obj_list[self.perm[current_index:(current_index + self.batch_size)]]
            self.batch_index += self.batch_size
        else:
            right = self.batch_size - (max_size - current_index)
            dat_list = np.append(self.data_obj_list[self.perm[current_index:max_size]], 
                    self.data_obj_list[self.perm[0: right]])
            self.batch_index = right

        input_batch = []
        cap_batch = []
        cap_len = []
        for d in dat_list:
            input_batch.append(d.input_sentence)
            cap_batch.append(d.output_index)
            cap_len.append(d.sentence_len)
        return np.array(input_batch),np.array(cap_batch),np.array(cap_len)
    
    def schedule_sampling(self, sampling_prob, cap_len_batch):

        sampling = np.ones(max_caption_len, dtype = bool)
        for l in range(max_caption_len):
            if np.random.uniform(0,1,1) < sampling_prob:
                sampling[l] = True
            else:
                sampling[l] = False
         
        sampling[0] = True
        return sampling
    def index_to_sentence(self, pred):
        all_sentence = []
        for i in range(0, self.batch_size):
            word = []
            for j in range(0, max_caption_len):
                word.append(self.onehot[pred[i][j]])
            all_sentence.append(word)
        return all_sentence

class test_Dataset:    #for test
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.batch_index = 0
        self.data = np.load("./data/test_data2.npy")
        self.batch_max_size = self.data.shape[0]
        with open('./data/index2word.pkl', 'rb') as f:
            self.onehot = pickle.load(f)
    def next_batch(self):
        current_index = self.batch_index
        max_size = self.batch_max_size
        if current_index + self.batch_size <= max_size:
            self.batch_index += self.batch_size
        else:
            print("batch_error")
            sys.exit()
        return self.data[current_index:current_index + self.batch_size]
    def index_to_sentence(self, pred):
        all_sentence = []
        for i in range(0, self.batch_size):
            word = []
            for j in range(0, max_caption_len):
                if pred[i][j]==2:
                    break
                else:
                    if j>0:
                        if pred[i][j] != pred[i][j-1]:
                            word.append(self.onehot[pred[i][j]])
                            
                        else:
                            pass
                        
                    else:
                        word.append(self.onehot[pred[i][j]])
            print(len(word))
            all_sentence.append(word)
        return all_sentence
    def logits_to_sentence_less_EOS(self, logits): ###numpy(batch_size,max_length,vocab_length)
        index = numpy.argmax(logits[:,:,2],axis = 1)
        
        


    





