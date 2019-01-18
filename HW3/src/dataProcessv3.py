from os import listdir
import numpy as np
import pandas as pd
import cv2
import sys

class DataObject:
    def __init__(self, one_hot_label, image_name):
        self.one_hot_label = one_hot_label
        self.image_name = image_name

class TrainDataset:  #label_cvs need the label csv_file, image_dir need the directory of images, 
                     #save_dict is the name of .npy you want to save dictionary 
    def __init__(self, label_csv, image_dir, save_dict_dir, batch_size, normalized=False):
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.data = []
        self.batch_max_size = 0
        self.batch_index = 0
        self.perm = []
        self.normalized = normalized
        self.lib_len = 0
        self.process_label(label_csv, save_dict_dir)
        if normalized:
            self.reset_normalized_par()

    def process_label(self, label_csv, dict_path):
        df = pd.read_csv(label_csv, header = None) 
        library = []
        library.extend(df[1].values)
        temp_word = []
        for i in library:
            temp = []
            w = i.split()
            temp.append(w[0])
            temp.append(w[2])
            temp_word.append(temp)
        temp_word = np.array(temp_word)
        hair_library = set(temp_word[:,0])
        hair_library = [i for i in hair_library]
        hair_library.sort()
        print(hair_library)

        eyes_library = set(temp_word[:,1])
        eyes_library = [i for i in eyes_library]
        eyes_library.sort()
        print(eyes_library)
        np.save(dict_path+"_hair_library", hair_library)
        np.save(dict_path+"_eyes_library", eyes_library)
        
        hair_length = len(hair_library)
        eyes_length = len(eyes_library)
        total_dir = listdir(self.image_dir)
        self.batch_max_size = len(total_dir)
        self.perm = np.arange( self.batch_max_size, dtype=np.int )# permutation numpy array
        for i in total_dir:
            try:
                index = int(i.replace(".jpg",""))
            except:
                print("fil error")
                print(i)
                sys.exit()
            zero = np.zeros(hair_length+eyes_length)
            k = np.searchsorted(hair_library, temp_word[index][0])
            w = np.searchsorted(eyes_library,temp_word[index][1]) + hair_length
            zero[k] = 1
            zero[w] = 1
            buffer = DataObject(zero,i)
            self.data.append(buffer)
        self.data = np.array(self.data)
        print(self.data[0].one_hot_label)


    def reset_normalized_par(self):
        data = [self.get_image(self.image_dir+name,True)for name in listdir(self.image_dir)]
        self.mean = np.mean(data)
        self.std = np.std(data)
        print("mean: ",self.mean)
        print("std ",self.std)
        return self.mean, self.std

    def shuffle_perm(self):
        np.random.shuffle( self.perm )

    def get_image(self,image_path,reset = True):
        #記得在生產圖片的地方換成  cv.imwrite(filename, img)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))
            
        if not reset :
            if(not self.normalized):
                img = img / 255.0
            else:
                img = (img - self.mean) / self.std
        return img

    def next_batch(self):
        current_index = self.batch_index
        max_size = self.batch_max_size
        if current_index + self.batch_size <= max_size:
            data_list = self.data[self.perm[current_index:(current_index + self.batch_size)]]
            self.batch_index += self.batch_size
        else:
            right = self.batch_size - (max_size - current_index)
            data_list = np.append(self.data[self.perm[current_index:max_size]], self.data[self.perm[0: right]])
            self.batch_index = right
        batch_word = []
        data_name = []
        for i in data_list:
            batch_word.append(i.one_hot_label)
            data_name.append(i.image_name)
        batch_image = [self.get_image(self.image_dir+name,False) for name in data_name]
        return batch_image, batch_word, data_name


class Testdataset():
    def __init__(self, label_csv, save_dict, batch_size, normalized=False):
        self.batch_size = batch_size
        self.batch_index = 0      
        df = pd.read_csv(label_csv, header = None)
        hair_library = np.load(save_dict+"_hair_library.npy")
        eyes_library = np.load(save_dict+"_eyes_library.npy")
        self.label = []
        library = []
        library.extend(df[1].values)


        hair_length = len(hair_library)
        eyes_length = len(eyes_library)
        temp_word = []
        for i in library:
            temp = []
            w = i.split()
            temp.append(w[0])
            temp.append(w[2])
            temp_word.append(temp)
        for i in temp_word:
            zero = np.zeros(hair_length+eyes_length)
            k = np.searchsorted(hair_library, i[0])
            w = np.searchsorted(eyes_library,i[1]) + hair_length
            zero[k] = 1
            zero[w] = 1
            self.label.append(zero)

        self.label = np.array(self.label)
        self.batch_max_size = len(self.label)


    def next_batch(self):
        if self.batch_index >= self.batch_max_size:
            print("Call to more")
        current_index = self.batch_index
        batch_word = self.label[current_index:min((current_index + self.batch_size),self.batch_max_size)]
        self.batch_index += self.batch_size
        return batch_word
    
    
'''
###用法
save_dict = "./"
label_csv =  "../../Henry/HW3/data/extra_data/tags.csv"
file_dir = "../../Henry/HW3/data/tag_faces/"
batch_size = 5000

datasetTrain = TrainDataset(label_csv,file_dir,save_dict,batch_size)    
#datasetTrain.shuffle_perm()
a, b, c =datasetTrain.next_batch()
print(a[0])
print(b[0])
print(c[0])
a[0] = a[0]*255
cv2.imwrite("./cool.jpg", a[0])

test = Testdataset(label_csv,save_dict,batch_size)
a = test.next_batch()
'''
batch_size = 25
save_dict = "./"
label_csv =  "./tags.csv"
test = Testdataset(label_csv,save_dict,batch_size)
a = test.next_batch()
print(a)
