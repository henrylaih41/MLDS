import numpy as np
import matplotlib.pyplot as plt
import keras as k
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
