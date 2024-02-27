#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sagemaker tensorflow opencv-python-headless')


# In[2]:


import tensorflow as tf
import sagemaker
from sagemaker.tensorflow import TensorFlow
import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time

print(f"TensorFlow version: {tf.__version__}")
print(f"SageMaker SDK version: {sagemaker.__version__}")


# In[3]:


bucket_name = 'distributedmachinelearning'
prefix = 'distributedMLtrainingModel'
output_path = f's3://{bucket_name}/{prefix}/output'
print(output_path)


# In[4]:


def build_and_compile_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[6]:


hyperparameters = {
    "learning_rate": "0.001",  # Corresponds to 'eta' in gradient boosting
    "dropout_rate": "0.5",     # Dropout rate to prevent overfitting, not directly related but serves a regularization purpose similar to 'gamma'
    "batch_size": "32",        # The number of samples processed before the model is updated
    "epochs": "5",            # The number of complete passes through the training dataset
    "conv_layers": "3",        # Number of convolutional layers, similar in concept to 'max_depth' as it affects model complexity
    "filters": "64",           # Number of filters in the first Conv layer, can increase with depth
    "kernel_size": "3",        # The size of the convolutional filters
    "pool_size": "2",          # The size of the pooling window
    "dense_neurons": "128",    # The number of neurons in the dense layer after convolutional layers
    "activation": "relu",      # Activation function for the convolutional layers
    "final_activation": "softmax", # Final activation function, for binary classification it could be 'sigmoid'
    "optimizer": "adam"        # Optimization algorithm
}


# In[7]:


role = sagemaker.get_execution_role()
estimator = TensorFlow(entry_point='test_script.py', 
                       hyperparameters=hyperparameters,
                       role=role,
                       instance_count=2,
                       instance_type='ml.m5.2xlarge',
                       framework_version='2.3.1',
                       py_version='py37',
                       output_path=output_path,
                       use_spot_instances=True,
                       max_run=300,
                       max_wait=600,
                       distribution={'parameter_server': {'enabled': True}})


# In[8]:


estimator.fit('s3://distributedmachinelearning/FaceAll')

