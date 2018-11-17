import pandas as pd
import numpy as np
import pickle
embedding_length = 30


def word2vec(length):
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
    cool = []
    for i in range(length):
        cool.append(input_mat[i])
    input_mat = np.array(cool)
    print(input_mat.shape)
    np.save("./test_data",input_mat)
    print(input_mat[1])
word2vec(2000)