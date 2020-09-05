#!/usr/bin/env python
# coding: utf-8

# # TRAFFIC-SIGNS-CLASSIFICATIONS

# Dataset taken from kaggle (https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed)😊
# 15-16 july 2020 🚀

# tried to make it in a more simpilar manner
# around 43 different traffic signs 
# let me find out that traffic sign for you...😜

# In[ ]:





# made using tensorflow 👀

# In[1]:


import matplotlib.pyplot as mlt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import random
import pickle


# In[2]:


with open("C:/Users/kumar/Documents/traffic-signs-data/train.pickle", mode='rb') as training_data:
  train = pickle.load(training_data)
with open("C:/Users/kumar/Documents/traffic-signs-data/valid.pickle", mode='rb') as validation_data:
  valid = pickle.load(validation_data)
with open("C:/Users/kumar/Documents/traffic-signs-data/test.pickle", mode='rb') as testing_data:
  test = pickle.load(testing_data)


# In[3]:


x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']


# In[4]:


x_valid.shape


# In[5]:


y_train.shape


# In[6]:


i = np.random.randint(1, len(x_train))
mlt.imshow(x_train[i])
y_train[i]


# In[ ]:





# In[7]:


w_grid = 10
l_grid = 10

fig, axes = mlt.subplots(l_grid, w_grid, figsize=(15,15))
axes = axes.ravel()
n_training = len(x_train)

for i in np.arange(0, w_grid*l_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(x_train[index])
    axes[i].set_title(y_train[index], fontsize = 15)
    axes[i].axis('off')
    
mlt.subplots_adjust(hspace = 0.4)


# In[8]:


from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train)


# In[9]:


x_train_gray = np.sum(x_train/3, axis=3, keepdims = True)
x_valid_gray = np.sum(x_valid/3, axis=3, keepdims = True)
x_test_gray = np.sum(x_test/3, axis=3, keepdims = True)


# In[10]:


x_train_gray


# In[11]:


x_train_gray.shape


# In[12]:


#normal form
x_train_gray_normal = (x_train_gray - 128)/128
x_valid_gray_normal = (x_train_gray - 128)/128
x_test_gray_normal = (x_train_gray - 128)/128


# In[13]:


x_train_gray_normal


# In[14]:



x_valid_gray_normal.shape


# In[15]:


i = random.randint(1, len(x_train_gray_normal))
mlt.imshow(x_train_gray_normal[i].squeeze(), cmap='gray')
mlt.figure()


# In[16]:


from tensorflow.keras import datasets, layers, models

CNN = models.Sequential()
CNN.add(layers.Conv2D(6, (5, 5), activation = 'relu', input_shape = (32, 32, 1)))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Dropout(0.2))

CNN.add(layers.Conv2D(16, (5, 5), activation = 'relu'))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Flatten())

CNN.add(layers.Dense(120, activation = 'relu'))
CNN.add(layers.Dense(84, activation = 'relu'))
CNN.add(layers.Dense(43 , activation = 'softmax'))

CNN.summary()


# In[17]:


CNN.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[18]:


history = CNN.fit(x_train_gray_normal,
                 y_train,
                 batch_size = 500,
                 epochs = 5,
                 verbose = 1,
                 validation_data = (x_valid_gray_normal, y_valid))


# Acurracy of 80% and can be increased with more epochs.😁

# In[ ]:





# In[ ]:




