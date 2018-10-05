from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
def plot_functions(x,y,func1):
    model1 = k.models.load_model(func1)
    predict1 = model1.predict(x)
    plt.plot(x, predict1, label = "model")
    plt.plot(x, y,label = "target")
    plt.legend()
    plt.show()

def plot_loss(model1):
    history1 = np.load(model1)
    plt.plot(history1)
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()

def plot_dot(x,y,y1):
    #plt.plot(x, y1, label = "model")
    plt.plot(x, y, label = "target")
    plt.legend()
    plt.show()


def threeD_error_surface():
    cordin = np.load("N18_6_50.npy")
    loss = np.load("loss_300.npy")
    x = np.array(cordin[:,0])
    y = np.array(cordin[:,1])
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x, y, loss, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()