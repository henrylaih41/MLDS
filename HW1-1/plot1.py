import numpy as np
import matplotlib.pyplot as plt
import keras as k
def plot_with_finction(function1, function2, start_point, end_point, point_num):
    #set x
    x = np.linspace(start_point, end_point, point_num)  #place point_num points uniformly in the region
    y1 = function1(x)                                   #for the function
    y2 = function2(x) 

    #plot
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.ylim(0, 0.4)                                 #set the range of y, depending on your function
    plt.show()

def plot_functions(x,y,func1,func2,func3,func4):
    model1 = k.models.load_model(func1)
    model2 = k.models.load_model(func2)
    model3 = k.models.load_model(func3)
    model4 = k.models.load_model(func4)
    predict1 = model1.predict(x)
    predict2 = model2.predict(x)
    predict3 = model3.predict(x)
    predict4 = model4.predict(x)
    plt.plot(x, predict4, label = "24 layers")
    plt.plot(x, predict3, label = "12 layers")
    plt.plot(x, predict2, label = "6 layers")
    plt.plot(x, predict1, label = '2 layers')
    plt.plot(x, y,label = "target")
    plt.legend()
    plt.ylim(0, 0.36)
    plt.show()

def plot_loss(model1,model2,model3,model4):
    history1 = np.load(model1)
    history2 = np.load(model2)
    history3 = np.load(model3)
    history4 = np.load(model4)
    plt.plot(history4,label = "26 layers")
    plt.plot(history1,label = "2 layers")
    plt.plot(history3,label = '12 layers')
    plt.plot(history2,label = "6 layers")
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.show()
