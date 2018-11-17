import word2vec as w
import numpy as np
quest_data_path = 'quest_data'
word2vec_path   = 'word2vec_v2.pkl'

class DataProcesser:
    def __init__(self):
        self.input = None
        self.input_len = None
        self.output = None
        self.output_len = None

    def readData(self):
        self.input, self.input_len = w.word2vec(quest_data_path, word2vec_path)
    
    def nextBatch(self):
        pass

D = DataProcesser()
D.readData()
n = D.input[0:50]
n = np.array(n)
print(n.shape)
d = D.input_len
d = np.array(d)
print(d.shape)  