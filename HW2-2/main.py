import model as m
import tensorflow as tf
import sys

if(sys.argv[1] =='--train'):
    model = m.AttentionModel(0.0005,300,1024,38103,"final_500",path="models3/")
    model.train(500,load=True)
elif(sys.argv[1] == '--test'):
    model = m.AttentionModel(0.0005,300,10,38103,"final_500",path="models3/")
    model.test()