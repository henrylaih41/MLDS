from sklearn.decomposition import PCA
import numpy as np
import math
import keras
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
color = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0.5, 0.5, 0.5)]
def do_pca(weight, dimension, count, color, layer):
    pca = PCA(n_components = dimension)
    pca.fit(weight)
    result = pca.transform(weight)
    eigen_value = pca.explained_variance_
    np.save('pca/pca_' + layer + '_' + str(count), result)
    #np.save('eigen_value/eigen_value' + str(count), eigen_value)
    return

def take_first_layer_weight(filepath):
    model = load_model(filepath)
    weight = model.get_weights()[0]
    weight = np.reshape(weight, (1, -1))
    return weight

def take_whole_model_weight(filepath):
    model = load_model(filepath)
    weight = model.get_weights()[0]
    for i in range(2, 21, 2):
        temp = model.get_weights()[i]
        weight = np.append(weight, temp)
    weight = np.reshape(weight, (1, -1))
    return weight

def shit():
    for i in range(8):
        whole_model_pca = np.load('./PCA/PCA_whole_' + str(i + 1) + '.npy')
        for j in range(100):
            loss = np.load('./history/all_loss_' + str(i + 1) + '.npy')
            plt.scatter(whole_model_pca[j][0], whole_model_pca[j][1], c = color[i], alpha = 1 - loss[j])
    print('model' + str(i + 1) + ' taken')
    plt.show()

def loss():
    for i in range(8):
        name = './history_mnist/_his_model' + str(i + 1)
        his = np.load(name + '_3.npy')
        loss = np.array(his[2])
        for j in range(6, 31, 3):
            new_his = np.load(name + '_' + str(j) + '.npy')
            loss = np.append(loss, new_his[2])
            print(str(j) + ' taken')
        np.save('history_mnist/all_loss_' + str(i + 1) + '.npy', loss)

for count in range(8):
    model_path = './model_mnist/model' + str(count + 1) + '_'
    whole_model = take_first_layer_weight(model_path + '3.h5')
    print('model' + str(count + 1) + '_1203 taken')

    for i in range(6, 301, 3):
        weight = take_first_layer_weight(model_path + str(i) + '.h5')
        whole_model = np.concatenate((whole_model, weight), axis = 0)
        print('model' + str(count + 1) + '_' + str(i) + ' taken')
    np.save('weight_mnist/first_layer_of_model' + str(count + 1), whole_model)
    do_pca(whole_model, 2, count + 1, color[count], 'first')

for i in range(8):
    pca_mnist_first = np.load('PCA_mnist/PCA_first_' + str(i + 1) + '.npy')
    plt.scatter(pca_mnist_first[:,0], pca_mnist_first[:,1], color = color[i])
plt.show()

for i in range(8):
    pca_mnist_first = np.load('PCA_mnist/PCA_first_' + str(i + 1) + '.npy')
    loss = np.load('history/all_loss_' + str(i + 1) + '.npy')
    plt.scatter(pca_mnist_first[:,0], pca_mnist_first[:,1], c = loss, cmap='Spectral')
