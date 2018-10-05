import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import plot as plt
f_w = []
loss = []
for epoch in range(6):
    w = np.load("./data/cordin50_" + str(epoch) + ".npy")
    f_w.append(w)
    l = np.load("./data/loss50_" + str(epoch) + ".npy")
    loss.append(l)
f_w = np.array(f_w)
loss = np.array(loss)
fl = np.array([item for sublist in f_w for item in sublist])
ll = np.array([item for sublist in loss for item in sublist])
np.save("loss_300",ll)
n_dim = TSNE(n_components=2).fit_transform(fl)
np.save("N18_6_50",n_dim)
plt.threeD_error_surface()