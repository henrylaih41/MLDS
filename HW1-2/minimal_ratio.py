import tensorflow as tf
import numpy as np
import math
### Objective: fitting a function (sinx) ###

def add_noise(origin):
    l = []
    for w in origin:
        #n_w = w + np.random.normal(scale = 0.000001,size = w.shape)
        n_w = w + tf.multiply(tf.cast(np.random.normal(scale = 0.00001,size = w.shape),dtype=tf.float32),w)
        l.append(n_w)
    return np.array(l)
     
def to_flat_array(tensors):
    _buf = np.array([ sess.run(p).reshape(-1) for p in tensors])
    flat_list = [item for sublist in _buf for item in sublist]
    flat_array = np.array(flat_list).reshape(-1)
    return flat_array

def get_noise_loss(origin,noise,sess,m_x,m_y):
    temporary=origin
    tempor=sess.run(temporary)
    for idx, weights in enumerate(noise):
        assign_op = origin[idx].assign(weights)
        sess.run(assign_op)
    #print(sess.run(origin[1]))
    noise_loss = sess.run(loss_mse,{X: m_x,Y: m_y})
    for idx, weights in enumerate(tempor):
        assign_opp = origin[idx].assign(weights)
        sess.run(assign_opp)
    #print(noise_loss)
    return noise_loss
def get_noise_list(origin,num = 10):
    l = []
    for _ in range(num):
        n = add_noise(origin)
        l.append(n)
    return np.array(l)

### Configuration ###
logs_path = 'sin_x'
batch_size = 50000
epochs = 10000
learning_rate = 0.01
num_data = 50000
batch_num = math.ceil(num_data/batch_size)


### Loading data ###
### 50000 points between [0,2pi]
train_x = np.linspace(0,2*math.pi,50000)
train_y = np.load("train_data1.npy")


# Input and Output.
X = tf.placeholder(shape=[None,1], dtype=tf.float32, name="X")
Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name="Y")

# Weight and bias.
# Computing hessians is currently only supported for one-dimensional tensors,
# so I did not bothered reshaping some 2D arrays for the parameters.
W1 = tf.Variable(tf.truncated_normal([1, 3], stddev=0.1))  
B1 = tf.Variable(tf.truncated_normal([3], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([3, 5], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([5], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([5, 5], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([5], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([5, 1], stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([1], stddev=0.1))
# Making a prediction and comparing it to the true output
XX = tf.reshape(X,[-1,1])
O1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
O2 = tf.nn.relu(tf.matmul(O1, W2) + B2)
O3 = tf.nn.relu(tf.matmul(O2, W3) + B3)
pred = tf.matmul(O3, W4) + B4
YY = tf.reshape(Y,[-1,1])
loss_mse = tf.losses.mean_squared_error(YY,pred)

###算二階反矩陣更新參數
grad=tf.gradients(loss_mse,[W1,B1,W2,B2,W3,B3,W4,B4])
staki=[tf.reshape(k,[-1]) for k in grad]
stakin=tf.concat(staki,-1)
staking=tf.unstack(stakin)
second=[tf.gradients(k,[W1,B1,W2,B2,W3,B3,W4,B4]) for k in staking]
secondlinear=[tf.reshape(k,[-1]) for j in second for k in j]
secondllinear=[tf.reshape(k,[-1]) for k in secondlinear]
stakingg=tf.concat(secondllinear,-1)
hession=tf.reshape(stakingg,[62,62])
rand_var_1 = np.random.normal(scale = 0.1,size=[62,62])
hess= hession + rand_var_1
invers_hession=tf.linalg.inv(hess)
updating=tf.matmul([stakin],0.05 * invers_hession)
updatings=tf.squeeze(updating)

w1,b1,w2,b2,w3,b3,w4,b4=tf.split(updatings,[3,3,15,5,25,5,5,1])
ww1=tf.reshape(w1,[1,3])
bb1=tf.reshape(b1,[3])
ww2=tf.reshape(w2,[3,5])
bb2=tf.reshape(b2,[5])
ww3=tf.reshape(w3,[5,5])
bb3=tf.reshape(b3,[5])
ww4=tf.reshape(w4,[5,1])
bb4=tf.reshape(b4,[1])

matrix=[ww1,bb1,ww2,bb2,ww3,bb3,ww4,bb4]
weights=[W1,B1,W2,B2,W3,B3,W4,B4]
update=[v.assign_add(-k) for v,k in zip(weights,matrix)]   ###更新參數






#secondflat=[tf.reshape(k,[-1]) for k in second]
#line=tf.concat(secondflat,-1)
"""gradw=tf.gradients(loss_mse,[W1,W2,W3,W4])
secondw= [tf.gradients(g, [W1,W2,W3,W4]) for g in gradw]
staking=[tf.reshape(k,[-1]) for k in secondw]"""


train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_mse)
tf.summary.scalar("loss", loss_mse) #
summary_op = tf.summary.merge_all() #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph()) 
    for epoch in range(epochs):
        loss_sum = 0
        for num in range(batch_num):
            batch = num * batch_size
            batch_x, batch_y =  train_x[batch : min(num_data,batch + batch_size)], \
                                train_y[batch : min(num_data,batch + batch_size)]
            batch_x = batch_x.reshape(-1,1)
            batch_y = batch_y.reshape(-1,1)
            _, loss = sess.run([train_step,loss_mse],\
                                  {X: batch_x, Y: batch_y})
            loss_sum = loss_sum + loss
            #writer.add_summary(summary,\
                               #batch_num * epoch + num)
        #if epoch % 2 == 0:
        print("Epoch: ", epoch, "Loss: ", loss_sum/batch_num)
    train_x = train_x.reshape(-1,1)
    train_y = train_y.reshape(-1,1)
#疊 second optimization
    for epoch in range(50):
        _, loss = sess.run([update,loss_mse],\
                                  {X: train_x, Y: train_y})
        print("Epoch: ", epoch, "Loss: ", loss)

    origin_loss = sess.run(loss_mse,{X: train_x,Y: train_y})
    _paras = [W1,W2,W3,W4,B1,B2,B3,B4]
    noise_list = get_noise_list(_paras,num=50)
    success=0
    print("original loss:",origin_loss)
    for i,noise in enumerate(noise_list):
        arg=sess.run(_paras)
        noise_loss = get_noise_loss(_paras,noise,sess,train_x,train_y)
        if noise_loss>=origin_loss :
            success+=1
        print("epoch",i+1," : ",success/float(i+1))
        print("loss", noise_loss)
    print("success",success/50.0)

    #print("regr" ,sess.run(ww1, {X: train_x, Y: train_y}))
    loss,hesss = sess.run([loss_mse,hession],\
                                  {X: train_x, Y: train_y})
    print(loss)
    


   # print(second)
    
print("done")