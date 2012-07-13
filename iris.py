#!/usr/bin/python 
#read the dataset 
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
  

##read dataset 
  inputs = numpy.loadtxt("iris.data", delimiter=",",usecols=(0,1,2,3))
  targets =  numpy.zeros([150,1])

  for i in range(0,50):
    targets[i] = 0 

  for i in range(50,100):
    targets[i] = 1 
 
  for i in range(100,150):
    targets[i] = 2 


  print "Inputs:\n" 
  print inputs 


  print "Targets:\n"
  print targets


  ##now create the network
  n =  FeedForwardNetwork()
  inLayer = LinearLayer(4) 
  hiddenLayer = SigmoidLayer(6)
  outLayer = SigmoidLayer(1) 

  #now add the layers to the network 
  n.addInputModule(inLayer) 
  n.addModule(hiddenLayer)
  n.addOutputModule(outLayer)
  
  in_to_hidden = FullConnection(inLayer , hiddenLayer)
  hidden_to_out = FullConnection(hiddenLayer , outLayer) 

  n.addConnection(in_to_hidden)
  n.addConnection(hidden_to_out)

  n.sortModules()

  ##the network has been designed 
  ##now proceed to train and test 

  print "Loading the dataset...."
  time.sleep(5) 
  
  data = ClassificationDataSet(4,1,nb_classes=3)
  for vals in range(0,150):
    input = inputs[vals,:]
    klass = targets[vals]  
    data.addSample(input,[klass])
     
  print "Dataset prepared successfully" 
  
  #split into training and testing sets 
  tstdata , trndata = data.splitWithProportion(0.25) 

  #encode classes such that there is one output neuron per class , original targets are separately stored in an integer field named class
 
  trndata._convertToOneOfMany()
  tstdata._convertToOneOfMany()

  ##brief look at the dataset 
  print "Number of training patterns :" , len(trndata)
  print "Input and Output dimensions :" , trndata.indim , trndata.outdim 
  time.sleep(3)
  print "First sample (input , target ,class):"
  print trndata['input'][0] , trndata['target'][0] , trndata['class'][0]
  print "Succesful" 

  ##now start the training  
  fnn = buildNetwork(trndata.indim , 6, trndata.outdim , outclass = SigmoidLayer)
  trainer = BackpropTrainer(fnn , dataset = trndata , verbose = True )
  
  plot_train =[]
  avg_trnresult = 0
  for i in range(1000):
     trainer.trainEpochs(1)
     trnresult = percentError(trainer.testOnClassData() , trndata['class'])
     avg_trnresult = avg_trnresult + trnresult 
     if i%10 ==0:
        plot_train.append(avg_trnresult/10)
        avg_trnresult = 0    
    
     print "Epoch: %4d" %trainer.totalepochs,\
           "train error : %5.2f%%" % trnresult , 
     

         
  #now check on the test data 
  tstresult = percentError(trainer.testOnClassData(dataset=tstdata),tstdata['class'])
  print "test error : %5.2f%%" %tstresult

  plot_trainvals = numpy.asarray(plot_train)

  ##printing some diagnostics 
  ##like the training error collected during training 

  
  pylab.plot(plot_trainvals)
  pylab.show()

  


   
