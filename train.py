
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class DataGenerator(Sequence):
    def __init__(self,
                 input_directory_1,
                 output_directory,
                 batch_size=16,
                 shuffle=True):
        'Initialization'
        self.input_directory_1 = input_directory_1
        self.output_directory = output_directory
        self.list_ids_1 = [f for f in os.listdir(input_directory_1) if os.path.isfile(os.path.join(input_directory_1,f))]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids_1) / self.batch_size))
    
    def __getitem__(self,index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_ids_temp = [self.list_ids_1[k] for k in indexes]

        # Generate data
        
        X,Y = self._data_generation(list_ids_temp,self.input_directory_1,self.output_directory)

        return X,Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids_1))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _data_generation(self,list_ids_temp,input_directory_1,output_directory):
        'Generates data containing batch_size samples'
        
        input_batch_1 = []
        input_batch_2 = []
        input_batch_3 = []
        output_batch = []

        for i,ID in enumerate(list_ids_temp):
            
            input_imgs_1,input_imgs_2,input_imgs_3 = read_image(input_directory_1,ID)
            output_img = read_image_HDR(output_directory,ID)
            
            
            input_batch_1.append(input_imgs_1)
            input_batch_2.append(input_imgs_2)
            input_batch_3.append(input_imgs_3)
            output_batch.append(output_img)

        batch_X = [input_batch_1,input_batch_2,input_batch_3]
        batch_Y = np.array(output_batch)
        return batch_X,batch_Y


def read(filename):
  img = plt.imread(filename)
  img = img/255
  img = cv2.resize(img,(512,512)) 
  return img

def read_image(path,file_name):
    p = path[:-3]
    s = path.split('/')
    if  'valid' in s:
      path_2 = p+'r2/'
      path_3 = p+'r3/'
    else:
      path_2 = p+'r2/'
      path_3 = p+'r3/'
    num = file_name.split('_')[1]

    img_1 = read(path+file_name)
    img_2 = read(path_2+'img_{}_2.png'.format(num))
    img_3 = read(path_3+'img_{}_3.png'.format(num))

    
    return (img_1,img_2,img_3)

def read_image_HDR(path,file_name):

  img = cv2.imread(path+file_name)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  img = img/255
  img = cv2.resize(img,(512,512))
  return img

def feature(str_,network):
  subnet = keras.models.Sequential()
  for l in network.layers:
    subnet.add(l)
    if l.name == str_:
      break
  return subnet

def create_model(input_shape):
  #__________input images__________
  input_1 = keras.layers.Input(input_shape)
  input_2 = keras.layers.Input(input_shape)
  input_3 = keras.layers.Input(input_shape)
  
  #__________original network__________
  network = keras.models.Sequential([])
  network.add(keras.layers.Conv2D(32, (7, 7), activation=tf.nn.relu, padding='SAME',input_shape=input_shape))
  network.add(keras.layers.Conv2D(32, (7, 7), activation=tf.nn.relu, padding='SAME', name='a'))
  network.add(keras.layers.MaxPool2D((2, 2)))
  network.add(keras.layers.BatchNormalization())
  network.add(keras.layers.Dropout(0.3))

  network.add(keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu, padding='SAME'))
  network.add(keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu, padding='SAME', name='b'))
  network.add(keras.layers.MaxPool2D((2, 2)))
  network.add(keras.layers.BatchNormalization())
  network.add(keras.layers.Dropout(0.3))

  network.add(keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='SAME'))
  network.add(keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='SAME', name='c'))
  network.add(keras.layers.MaxPool2D((2, 2)))
  network.add(keras.layers.BatchNormalization())
  network.add(keras.layers.Dropout(0.3))

  network.add(keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu, padding='SAME'))
  network.add(keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu, padding='SAME', name='d'))
  network.add(keras.layers.MaxPool2D((2, 2)))
  network.add(keras.layers.BatchNormalization())
  network.add(keras.layers.Dropout(0.3))

  network.add(keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME'))
  network.add(keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME', name='e'))
  network.add(keras.layers.MaxPool2D((2, 2)))
  network.add(keras.layers.BatchNormalization())
  network.add(keras.layers.Dropout(0.3))
  
  network.add(keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME'))
  network.add(keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME', name='f'))
  network.add(keras.layers.MaxPool2D((2, 2)))
  network.add(keras.layers.BatchNormalization())
  network.add(keras.layers.Dropout(0.3))

  network.add(keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME'))
  network.add(keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME', name='g'))
  network.add(keras.layers.MaxPool2D((2, 2)))
  network.add(keras.layers.BatchNormalization())
  network.add(keras.layers.Dropout(0.3))

  # network.summary()
  #__________feeding images__________
  encoded_1 = network(input_1)
  encoded_2 = network(input_2)
  encoded_3 = network(input_3)
  
  #__________getting features__________
  subnet_a = feature('a',network)
  subnet_b = feature('b',network)
  subnet_c = feature('c',network)
  subnet_d = feature('d',network)
  subnet_e = feature('e',network)
  subnet_f = feature('f',network)
  subnet_g = feature('g',network)
  
  a1 = subnet_a(input_1)
  a2 = subnet_a(input_2)
  a3 = subnet_a(input_3)
  
  b1 = subnet_b(input_1)
  b2 = subnet_b(input_2)
  b3 = subnet_b(input_3)

  c1 = subnet_c(input_1)
  c2 = subnet_c(input_2)
  c3 = subnet_c(input_3)

  d1 = subnet_d(input_1)
  d2 = subnet_d(input_2)
  d3 = subnet_d(input_3)

  e1 = subnet_e(input_1)
  e2 = subnet_e(input_2)
  e3 = subnet_e(input_3)

  f1 = subnet_f(input_1)
  f2 = subnet_f(input_2)
  f3 = subnet_f(input_3)

  g1 = subnet_g(input_1)
  g2 = subnet_g(input_2)
  g3 = subnet_g(input_3)

  #__________Concatenation__________
  x = keras.layers.Concatenate()([encoded_1,encoded_2,encoded_3])
  a = keras.layers.Concatenate()([a1,a2,a3])
  b = keras.layers.Concatenate()([b1,b2,b3])
  c = keras.layers.Concatenate()([c1,c2,c3])
  d = keras.layers.Concatenate()([d1,d2,d3])
  e = keras.layers.Concatenate()([e1,e2,e3])
  f = keras.layers.Concatenate()([f1,f2,f3])
  g = keras.layers.Concatenate()([g1,g2,g3])
  

  x = keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME')(x)
  x = keras.layers.UpSampling2D((2,2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.3)(x)

  x = keras.layers.Concatenate()([x,g])
  x = keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME')(x)
  x = keras.layers.UpSampling2D((2,2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.3)(x)
  
  x = keras.layers.Concatenate()([x,f])
  x = keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, padding='SAME')(x)
  x = keras.layers.UpSampling2D((2,2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.3)(x)
  
  x = keras.layers.Concatenate()([x,e])
  x = keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu, padding='SAME')(x)
  x = keras.layers.UpSampling2D((2,2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.3)(x)
  
  x = keras.layers.Concatenate()([x,d])
  x = keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='SAME')(x)
  x = keras.layers.UpSampling2D((2,2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.3)(x)
  
  x = keras.layers.Concatenate()([x,c])
  x = keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu, padding='SAME')(x)
  x = keras.layers.UpSampling2D((2,2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.3)(x)
  
  x = keras.layers.Concatenate()([x,b])
  x = keras.layers.Conv2D(32, (7, 7), activation=tf.nn.relu, padding='SAME')(x)
  x = keras.layers.UpSampling2D((2,2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(0.3)(x)
  
  x = keras.layers.Concatenate()([x,a])
  decoded = keras.layers.Conv2D(3, (3, 3), activation=tf.nn.relu, padding='SAME')(x)
  
  siamese = keras.Model(inputs=[input_1,input_2,input_3],outputs=decoded)
  return siamese

#_______________________training___________________

LDR_Path_1 = os.path.join('/Train/LDR/r1/')
HDR_Path = os.path.join('/Train/HDR/')

Valid_LDR_1 = os.path.join('/valid/LDR/r1/')
Valid_HDR = os.path.join('/valid/HDR/')

file_names_LDR = [f for f in os.listdir(LDR_Path_1) if os.path.isfile(os.path.join(LDR_Path_1,f))]
valid_file_names_LDR = [f for f in os.listdir(Valid_LDR_1) if os.path.isfile(os.path.join(Valid_LDR_1,f))]

bsize = 2

train_generator = DataGenerator(input_directory_1 = LDR_Path_1, output_directory = HDR_Path, batch_size = bsize)
valid_generator = DataGenerator(input_directory_1 = Valid_LDR_1, output_directory = Valid_HDR, batch_size = bsize)

model=create_model((512,512,3))

save = ModelCheckpoint('weights.hdf5', save_weights_only=True)


model.compile(optimizer='adam', loss=keras.losses.mean_absolute_error)

history = model.fit_generator(train_generator, steps_per_epoch = len(file_names_LDR)//train_bsize, epochs=2, verbose=1, callbacks=[save]
                              , validation_data = valid_generator, 
                              validation_steps = len(valid_file_names_LDR)//valid_bsize)
