from os import listdir
import numpy as np
import skimage
import cv2 as cv

class Dataset:
    def __init__(self, file_dir, batch_size, mode="compress", normalized=True, cv2=True):
        self.batch_size = batch_size
        self.file_dir = file_dir
        self.data = np.array(listdir(file_dir))
        self.batch_max_size = len(self.data)
        self.batch_index = 0
        self.perm = np.arange( self.batch_max_size, dtype=np.int )# permutation numpy array
        self.mode = mode
        self.normalized = normalized
        self.cv2 = cv2
        if self.mode == "origin":  #這邊是測出來的(用cv2)，如果不放心可以先跑reset_normalized_par會幫你重新取值，大概需要10秒鐘
            self.mean = 165.0294144611144
            self.std = 67.43082924601117
        elif mode == "crop":
            self.mean = 174.66443061094887
            self.std = 66.75514889309804
        elif mode == "compress" :
            self.mean = 164.97142601310438
            self.std = 66.07747407441585
        
    def reset_normalized_par(self):
        data = [self.get_image(True,self.file_dir+name)for name in listdir(self.file_dir)]
        self.mean = np.mean(data)
        self.std = np.std(data)
        print("mean: ",self.mean)
        print("std ",self.std)
        return self.mean, self.std

    def shuffle_perm(self):
        np.random.shuffle( self.perm )

    def get_image(self,reset,image_path, input_height=64, input_width=64):
        #print(image_path)
        
        #記得在生產圖片的地方換成  cv.imwrite(filename, img)
        if self.mode == "origin": 
            if(self.cv2):
                img = cv.imread(image_path)
            else:
                img = skimage.io.imread(image_path)
            
        elif(self.mode == "crop"):
            if(self.cv2):
                img = cv.imread(image_path)
            else:
                img = skimage.io.imread(image_path)
            h, w = img.shape[:2]
            j = int(round((h - input_height)/2.))
            i = int(round((w - input_width)/2.))
            img = img[j:j+input_height, i:i+input_width]
        
        elif(self.mode== "compress"):
            if(self.cv2):
                img = cv.imread(image_path)
                img = cv.resize(img, (64, 64))
            else:
                img = skimage.transform.resize(img, (64, 64,3))

        else:
            raise RuntimeError("mode %s is not defined" % self.mode)
            
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
            data_name = self.data[self.perm[current_index:(current_index + self.batch_size)]]
            self.batch_index += self.batch_size
        else:
            right = self.batch_size - (max_size - current_index)
            data_name = np.append(self.data[self.perm[current_index:max_size]], self.data[self.perm[0: right]])
            self.batch_index = right

        batch_image = [self.get_image(False,self.file_dir+name) for name in data_name]
        return batch_image, data_name
    
    
"""
def genNoise(bs,nos_dim=100, mode='uniform'):
        if (mode == 'uniform'):
            x = np.random.uniform(-1, 1, [bs, nos_dim]).astype(np.float32)
            return (x / np.linalg.norm(x, axis=1)[:, np.newaxis])
        elif(mode == 'normal'):
            x = np.random.normal(0,0.7,[bs, nos_dim]).astype(np.float32)
            return (x / np.linalg.norm(x, axis=1)[:, np.newaxis])
        else:
            raise RuntimeError('mode %s is not defined' % (mode))

"""         

###用法
"""file_dir = "./faces/"
batch_size = 10

datasetTrain = Dataset(file_dir,batch_size)
datasetTrain.shuffle_perm()
a, b =datasetTrain.next_batch()
print(a[0])
print(b[0])
"""
