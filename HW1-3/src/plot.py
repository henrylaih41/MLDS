import matplotlib.pyplot as plt

def plot_loss_sharpness(loss,t_loss,sharpness):
    x = [2 ** i for i in range(5,16)]
    print(x)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("batch_size")
    ax1.set_ylabel('loss', color=color)
    ax1.plot(x, loss, color=color,label="train")
    ax1.plot(x, t_loss, color=color,label="test",linestyle=":")
    plt.xscale("log")

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('sharpness', color=color)  
    ax2.plot(x, sharpness, color=color,label="sharpness")
    fig.tight_layout()  
    plt.legend()
    plt.show()

def plot_acc_sharpness(acc,t_acc,sharpness):
    x = [2 ** i for i in range(5,16)]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("batch_size")
    ax1.set_ylabel('accuracy', color=color)
    ax1.plot(x, acc, color=color,label="train")
    ax1.plot(x, t_acc, color=color,label="test",linestyle=":")
    plt.xscale("log")
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('sharpness', color=color)  
    ax2.plot(x, sharpness, color=color,label="sharpness")
    fig.tight_layout()  
    plt.legend()
    plt.show()


def plot_acc_diff_sharpness(acc_diff,sharpness):
    x = [2 ** i for i in range(5,16)]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("batch_size")
    ax1.set_ylabel('accuracy difference (%)', color=color)
    ax1.plot(x, acc_diff, color=color)
    plt.xscale("log")
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('sharpness', color=color)  
    ax2.plot(x, sharpness, color=color,label="sharpness")
    fig.tight_layout()  
    plt.legend()
    plt.show()