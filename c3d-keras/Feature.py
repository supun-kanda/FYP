#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import json
import h5py
import sys
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import c3d_model
#from diagnose import diagnose


# In[2]:

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras import backend as K
import tensorflow as tf
with K.tf.device('/gpu:1'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=4, allow_soft_placement=True,device_count = {'CPU' : 1, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)
import keras.backend as K
from keras.layers import Dense
from keras.models import model_from_json


# In[3]:


#features = h5py.File('/home/jayathu/DAMBA/feature.c3d.hdf5', 'r')
features = h5py.File('/mnt/data/c3d_features/feature.c3d.hdf5', 'r')


# In[4]:


#Get video files and limit them to 20
number_of_vids = 100
with open('/home/supunK/GIT/c3d-keras/files') as f:
    content = f.readlines()
videos = [x.strip().split('.')[0] for x in content[:number_of_vids]] 


# In[5]:


diagnose_plots = False
model_dir = '/home/supunK/GIT/c3d-keras/models'

model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')


# In[6]:


#Customizing model to make last layer 500D
model = model_from_json(open(model_json_filename, 'r').read())
model_img_filename = os.path.join(model_dir, 'c3d_model.png')
if not os.path.exists(model_img_filename):
    from keras.utils import plot_model
    plot_model(model, to_file=model_img_filename)

model.load_weights(model_weight_filename)
model.layers.pop()
model.get_layer('conv1').trainable=False
model.get_layer('conv2').trainable=False
model.get_layer('conv3a').trainable=False
model.get_layer('conv3b').trainable=False
model.get_layer('conv4a').trainable=False
model.get_layer('conv4b').trainable=False
model.get_layer('conv5a').trainable=False
model.get_layer('conv5b').trainable=False
model.get_layer('fc6').trainable=False
model.get_layer('fc7').trainable=False
model.add(Dense(2048, activation='relu', name='pre_custom', trainable=True))
model.add(Dense(500, activation='relu', name='custom', trainable=True))

model.summary()
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
exit()

# In[14]:


#Create Dataset including labels
mean_cube = np.load('models/train01_16_128_171_mean.npy')
mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
stride = 8
resolution = 16

#videos=['/home/jayathu/DAMBA/DataSet/v__VPf75tGIHQ.mp4','grass.mp4']

jump=5
for s in range(0,number_of_vids,jump):
    print("Processing %d Set"%s)
    stack = []
    c3d_feat=[]
    for video in videos[s:s+jump]:
        vid_array=features[video]['c3d_features'].value
        for frame in vid_array:
            c3d_feat.append(frame)

        cap = cv2.VideoCapture("/home/supunK/GIT/c3d-keras/Vids/%s.mp4"%(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            vid.append(cv2.resize(img, (171, 128)))
        vid = np.array(vid, dtype=np.float32)
        end = vid.shape[0]
        #print('Started processing Vid: %s, FPS: %f'%(video,fps))
        for start_frame in range(0,end,stride):
            #print('frame no: %d/%d'%(start_frame,end))
            X = vid[start_frame:(start_frame + resolution), :, :, :]
            if(X.shape[0]<resolution):
                continue
            #X -= mean_cube
            X = X[:, 8:120, 30:142, :] # (l, h, w, c)
            stack.append(X)

        ls=len(stack)
        lc=len(c3d_feat)
        if(ls>lc):
            stack=stack[:lc]
        elif(lc>ls):
            c3d_feat=c3d_feat[:ls]

    labels = np.array(c3d_feat)
    arr = np.array(stack)
    model.fit(x=arr, y=labels, batch_size=50, epochs=20, verbose=1)


# In[ ]:


#model.fit(x=arr, y=labels, batch_size=50, epochs=10, verbose=1)


# In[8]:


model.save("model.json")
model.save_weights("model_weights.h5")
