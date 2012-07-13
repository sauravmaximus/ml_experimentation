#!/usr/bin/python 

##fix all imports 
import numpy
from numpy import *
import pybrain
from pybrain.structure import FeedForwardNetwork 
from pybrain.structure import LinearLayer , SigmoidLayer
from pybrain.structure import FullConnection
import time 
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError 
from pybrain.tools.shortcuts import buildNetwork 
from pybrain.supervised.trainers import BackpropTrainer 
import pylab

if __name__ == "__main__":
 ##now we must load in the dataset 
 ## our dataset consists of four attributes for each data point and it has to be classified as left , balanced or right
 ## we will here classify using an ANN with 4 input nodes , 5 hidden nodes and 3 output nodes ( class encoded )
 ###thus we first load the dataset 
 
 inputs = numpy.genfromtxt("balance-scale.data" , delimiter="," , usecols=(0,1,2,3,4) , dtype=[('mystring','S1'),('myf1','float'),('myf2','float'),('myf3','float'),('myf4','float')])
 
 input= numpy.asarray(inputs)
 ##note that input is a numpy.ndarray so it has to be indexed as input[i][j] not input[i,j] unlike standard numpy arrays 
 ##replace L by 0 , B by 1 , R by 2 
 
 num_features =input.size
 input_features = zeros([num_features,4])
 target_features = zeros([num_features,1])
 for i in range(0,num_features):
   for j in range(0,4):
      input_features[i][j] = input[i][j+1]
   if input[i][0] == "L":
      target_features[i] = 0 
   if input[i][0] == "B":
      target_features[i] = 1 
   if input[i][0] == "R":
      target_features[i] = 2 

 print "Dataset loaded into workspace ...."
 time.sleep(3)

 data = ClassificationDataSet(4,1,nb_classes=3)
 for val in range(0,num_features):
    inp = input_features[val,:]
    targ = target_features[val]
    data.addSample(inp,[targ])

 print "Dataset created successfully" 
 
 ##split into training and testing data 
 tstdata , trndata =  data.splitWithProportion(0.30)
 trndata._convertToOneOfMany()
 tstdata._convertToOneOfMany()
 print "Training data inp dimension :" ,trndata.indim
 print "\n Training data outp dimension :" , trndata.outdim

 ##now create the neural network 
 print "Creating neural network .."
 time.sleep(2)

 fnn = FeedForwardNetwork() 
 inLayer = LinearLayer(trndata.indim) 
 hiddenLayer = SigmoidLayer(5)
 outLayer = SigmoidLayer(trndata.outdim)
 w_in = FullConnection(inLayer, hiddenLayer)
 w_out = FullConnection(hiddenLayer , outLayer) 
 
 fnn.addInputModule(inLayer)
 fnn.addModule(hiddenLayer)
 fnn.addOutputModule(outLayer)

 fnn.addConnection(w_in)
 fnn.addConnection(w_out) 

 fnn.sortModules()
 
 ##now create a trainer to train the dataset 
 
 trainer = BackpropTrainer(fnn, dataset = trndata , verbose = True)
 plot_train=[]
 avg_trnresult=0
 for i in range(500):
     trainer.trainEpochs(1)
     trnresult = percentError(trainer.testOnClassData() , trndata['class'])
     avg_trnresult = avg_trnresult + trnresult 
     if i%10==0:
        plot_train.append(avg_trnresult/10)
        avg_trnresult = 0    
    
     print "Epoch: %4d" %trainer.totalepochs,\
           "train error : %5.2f%%" % trnresult , 
     
  
 #now check on the test data 
 tstresult = percentError(trainer.testOnClassData(dataset=tstdata),tstdata['class'])
 print "test error : %5.2f%%" %tstresult

 plot_trainvals = numpy.asarray(plot_train[1:])

 ##printing some diagnostics 
 ##like the training error collected during training 

  
 pylab.plot(plot_trainvals)
 pylab.show()

  
    
 

