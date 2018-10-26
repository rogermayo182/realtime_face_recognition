# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import facenet
import os
from os.path import join 
import sys
import pickle
from sklearn.svm import SVC
from sklearn import metrics

data_dir = './model_train_data/'
image_size = 160
facenet_model_dir = './facenet_model/20180613-102209'

with tf.Graph().as_default():
      
    with tf.Session() as sess:
            
        np.random.seed(seed = 42)

        dataset = facenet.get_dataset(data_dir)
        train_set, test_set = facenet.split_dataset(dataset, 0.1, 2, 'SPLIT_IMAGES')

        dataset = train_set
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        class_names = [ cls.name.replace('_', ' ') for cls in dataset]
        print(class_names)

        # 加载模型,模型位于models目录下
        print('Loading feature extraction model')
        facenet.load_model(facenet_model_dir)

        # 获取输入和输出 tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        embedding_size = embeddings.get_shape()[1]
        image_size = images_placeholder.get_shape()[1]
        
        nrof_images = len(paths)
        emb_array = np.zeros((nrof_images, embedding_size))
        
        images = facenet.load_data(paths, do_random_crop=False, do_random_flip=False,
                                           image_size=image_size, do_prewhiten=True)
        
        feed_dict = {images_placeholder:images, phase_train_placeholder:False }

        # fix the size error
        emb_array[0:nrof_images,:] = sess.run(embeddings, feed_dict=feed_dict)

        classifier_filename_exp = os.path.expanduser('./facenet_model/feature_models.pkl')

        # Train classifier
        #model = KNeighborsClassifier() # accuracy: 77.70%
        #model = SVC(kernel='linear', probability=True)
        #model = SVC(kernel='poly',degree=2,gamma=1,coef0=0,probability=True) # accuracy: 77.03%
        model = SVC(kernel='poly',degree=10,gamma=1,coef0=0,probability=True) #accuracy: 87.16%
        
        model.fit(emb_array, labels)
        
        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)


        # 验证
        paths, labels = facenet.get_image_paths_and_labels(test_set)

        nrof_images = len(paths)
        emb_array = np.zeros((nrof_images, embedding_size))

        images = facenet.load_data(paths, do_random_crop=False, do_random_flip=False,
                                   image_size=image_size, do_prewhiten=True)

        feed_dict = {images_placeholder: images, phase_train_placeholder: False}

        # fix the size error
        emb_array[0:nrof_images, :] = sess.run(embeddings, feed_dict=feed_dict)

        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
        predict = model.predict(emb_array)
        accuracy = metrics.accuracy_score(labels, predict)
        print ('accuracy: %.2f%%' % (100 * accuracy))
