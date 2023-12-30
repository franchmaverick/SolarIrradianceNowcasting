!pip install tensorflow==2.3.0 #Preparing the machine learning environment: (1) installing depndencies and framework
 
import tensorflow as tf #nickname
 
tf.__version__#need to check because the version tungod kay naay specific code and protocol kada mag change ang version.
 
!nvidia-smi  #to check what kind of GPU are we using.
 
#Preparing the machine learning environment:
 
# (2) import the libraries as shown below
 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
 
#from keras.applications.resnet50 import Resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import imageZ
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt
 
# re-size all the images to this
IMAGE_SIZE = [224, 224]
 
train_path = '/content/drive/MyDrive/Personal Files/DENG Energy Systems/Disertation 1/Labelled Data March /train'
valid_path = '/content/drive/MyDrive/Personal Files/DENG Energy Systems/Disertation 1/Labelled Data March /valid'
 
# Import the Resnet50 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights
 
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
 
#from tensorflow.python.keras.applications.resnet import ResNet50
# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False
 
  # useful for getting number of output classes
folders = glob('/content/drive/MyDrive/Personal Files/DENG Energy Systems/Disertation 1/Labelled Data March /train/*')
 
folders
 
# our layers - you can add more if you want
x = Flatten()(resnet.output)
 
prediction = Dense(len(folders), activation='softmax')(x)
 
# create a model object
model = Model(inputs=resnet.input, outputs=prediction)
 
# view the structure of the model
model.summary()
 
# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
 
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
 
test_datagen = ImageDataGenerator(rescale = 1./255)
 
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Personal Files/DENG Energy Systems/Disertation 1/Labelled Data March /train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
 
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/Personal Files/DENG Energy Systems/Disertation 1/Labelled Data March /test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
 
# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=121,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
 
import matplotlib.pyplot as plt
 
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
 
# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
