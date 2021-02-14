# -*- coding: utf-8 -*-
"""tensorflow lower layers.ipynb
created By Manar Al-Kali
can be used for reasech purposes 
Original file is located at
    https://github.com/radex86/Building-CNN-in-Tf-2.0-with-Eager-Execution.git
"""

# importing required modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist,cifar10
import matplotlib.pyplot as plt

#def one hot encoding function
def y2ind(y,K=10):
  N = y.shape[0]
  ind=np.zeros([N,K])
  for i in range(N):
    ind[i, y[i]] =1
  return ind 

# getting the dataset and preprocessing data 
def get_data():
  (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
  xtrain, xtest = xtrain/255 , xtest/255
  #plt.imshow(xtrain[0])

  # reshape for CNN2D
  xtrain = xtrain.reshape(-1,28,28,1)
  xtest = xtest.reshape(-1,28,28,1)
  #xtrain= np.expand_dims(xtrain, axis=1)

  # one_hot encoding
  ytrain_ind= y2ind(ytrain)
  ytest_ind= y2ind(ytest)

  xtrain, xval = xtrain[:50000], xtrain[50000:]
  ytrain, yval = ytrain_ind[:50000], ytrain_ind[50000:]

  return xtrain, xval,xtest, ytrain, yval, ytest_ind

# Building the Model
# Required functions for The CNN

def weight_baias_init(n_classes):
  # defining the weight and baises function (mnist uses 2 CNN Layers, 1 FNN layers  )
  weights = {
      'wc1': tf.Variable(tf.random.normal([5,5,1,32])),
      'wc2': tf.Variable(tf.random.normal([5,5,32,64])),
      'wd1': tf.Variable(tf.random.normal([7*7*64, 1024])),
      'out': tf.Variable(tf.random.normal([1024, n_classes]))
  }
  baises= {
      'bc1': tf.Variable(tf.random.normal([32])),
      'bc2': tf.Variable(tf.random.normal([64])),
      'bd1': tf.Variable(tf.random.normal([1024])),
      'out': tf.Variable(tf.random.normal([n_classes]))
  }
  return weights, baises

# define the convPoollayer
# strides works exactly like max pooling so we have to choose only one 
def convPool(X,W,b, strides=1, k=2):
  t = tf.nn.conv2d(X,W, strides=[1,strides, strides, 1], padding='SAME')
  t= tf.nn.bias_add(t, b)
  t= tf.nn.max_pool(
      t, 
      ksize=[1,k,k,1],
      strides=[1,k,k,1],
      padding='SAME'
  )
  return t

# creating the conv_net Model
# layer 1 CNN
def conv_model(X, W,b, dropout=0.75):
    Conv1 = convPool(X,W['wc1'],b['bc1'])

    # layer 2 CNN
    Conv2 = convPool(Conv1,W['wc2'],b['bc2'])

    #FDD
    #flatten
    fc1 = tf.reshape(Conv2, [-1, W['wd1'].get_shape().as_list()[0]])
    #1st layer
    fc1 = tf.add(tf.matmul(fc1, W['wd1']), b['bd1'])
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(fc1, W['out']), b['out'])
    #checking the outputshape
    #print(out)
    return out

# Defining the hyperprameters (depend on the dataset)
xtrain, xval,xtest, ytrain, yval,ytest = get_data()
learning_rate = 0.001 
epochs = 10
batch_sz = 128
n_classes = 10  # MNIST total classes (0-9 digits)
#dropout = 0.75  # dropout (probability to keep units)
n_batchs = xtrain.shape[0] // batch_sz
loss=[]
val_loss=[]
weights, baises = weight_baias_init(n_classes)[0] , weight_baias_init(n_classes)[1] 

for epoch in range(epochs):
  for j in range(n_batchs):
    xbatch =xtrain[j*batch_sz:(j*batch_sz+batch_sz),:,:,:]  
    ybatch = ytrain[j*batch_sz:(j*batch_sz+batch_sz),:] 

    #convert the data into tensors
    inputx = tf.convert_to_tensor(xbatch, tf.float32)
    labels = tf.convert_to_tensor(ybatch,  tf.float32)
    # Model

    if len(xbatch) == batch_sz:
       
      # logits for predictions
      logits = tf.nn.softmax(conv_model(inputx, weights, baises, dropout))
    
      # training loss
      cost = lambda : tf.reduce_mean( \
             tf.nn.softmax_cross_entropy_with_logits(logits= conv_model(inputx, weights, baises, dropout), labels=labels))
            
      opt= tf.optimizers.Adam(learning_rate=learning_rate)
      opt.minimize(cost, var_list=[weights, baises])
      
      correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(ybatch, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
      #testing the accuracy on the validation set
      if j%10 == 0:
        for k in range(len(xtest)// batch_sz):
              xtch =xval[k*batch_sz:(k*batch_sz+batch_sz),:,:,:]  
              ytch = yval[k*batch_sz:(k*batch_sz+batch_sz),:]
              #convert the data into tensors
              inputtx = tf.convert_to_tensor(xtch, tf.float32)
              labelts = tf.convert_to_tensor(ytch,  tf.float32)
             
              logitts = conv_model(inputtx, weights, baises, dropout)
              val_cost = lambda: tf.reduce_mean(\
                 tf.nn.softmax_cross_entropy_with_logits(logits=conv_model(inputtx, weights, baises, dropout), labels=labelts))    
              val_correct_pred = tf.equal(tf.argmax(logitts, 1), tf.argmax(ybatch, 1))
              val_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        if j%20 == 0: 
           loss.append(cost())
           val_loss.append(val_cost()) 
        
        print(f'Epoch {epoch+1}, Batch {j} -Loss: {cost()} Accuracy: {val_accuracy} - Val_loss: {val_cost()} Val_Accuracy: {val_accuracy}.')

prediction = tf.equal(tf.argmax(conv_model(xtest, weights, baises, dropout), 1), tf.argmax(ytest, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print('Testing Accuracy: {}'.format(acc))

plt.plot(loss, label='loss', color='blue')
plt.plot(val_loss, label='val loss', color='green')
plt.legend()
