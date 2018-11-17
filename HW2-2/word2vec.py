import pandas as pd
import numpy as np
import pickle

embedding_length = 30

def get_original_len(list1):         #eat quest_data
    orig_len = []
    for i in range(len(list1)):
        orig_len.append(len(list1[i]))
    return orig_len

def word2vec(quest_data_path, word2vec_path):
    with open(quest_data_path, 'rb') as f:     #load the quest_data
        quest_data = pickle.load(f)

    with open(word2vec_path, 'rb') as w:       #load the word2vec.pkl
        w2v = pickle.load(w)
    print(type(w2v))
    print(len(w2v))
    print(w2v['EOS'])    
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
    ### Get question origin len
    origin_len = get_original_len(quest_data)
    
    return input_mat, origin_len

def index2vec():
    with open("word2vec_v2.pkl", 'rb') as w:       #load the word2vec.pkl
        w2v = pickle.load(w)
    with open("index2word.pkl", "rb") as w:
        i2w = pickle.load(w)
    print(len(i2w))
    word_vec = []
    for i in range(38103):
        if i2w[i] != 'UNK':
            vec = w2v[i2w[i]]
        else:
            vec = np.zeros(300)
        word_vec.append(vec)
    save_path = open('index2vec.pkl', 'wb')
    pickle.dump(word_vec,save_path)

#word2vec("quest_data","word2vec_v2.pkl")
indexs = np.random.randint(0,38103,size=100)
index2vec()