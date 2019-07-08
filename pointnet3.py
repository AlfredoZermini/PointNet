import os, time
import glob
import h5py
import numpy as np
import wget
import math
from keras import backend as K
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Convolution1D, LeakyReLU, MaxPooling1D, InputLayer, BatchNormalization, Dropout, Activation, concatenate, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

DATA_DIR = 'data'
S3DIS = 'indoor3d_sem_seg_hdf5_data'
NUM_CLASSES = 13

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 20.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def testing(x_test, y_test):

   pointnet_class = load_model(os.path.join(os.getcwd(),'pointnet') )
   pointnet_class.summary()
   output = pointnet_class.predict(x_test, batch_size = 32, verbose=1)

   _, acc = pointnet_class.evaluate(x_test, y_test,batch_size=32)
   #print('Test score:', score)
   print('Test accuracy:', acc)

def pointnet(x_train, y_train):

   [n_samples, n_points, n_features] = x_train.shape

   input1 = Input(shape=(n_points,n_features) )

   MLP1 = ( Convolution1D(64, kernel_size=(1), strides=(1), padding='valid' ))(input1)
   MLP1 = ( BatchNormalization() )(MLP1)
   MLP1 = ( LeakyReLU())(MLP1)
   MLP1 = ( Convolution1D(64, kernel_size=(1), strides=(1), padding='valid' ))(MLP1)
   MLP1 = ( BatchNormalization() )(MLP1)
   MLP1 = ( LeakyReLU())(MLP1)
   MLP1 = ( Convolution1D(128, kernel_size=(1), strides=(1), padding='valid' ) )(MLP1)
   MLP1 = ( BatchNormalization() )(MLP1)
   MLP1 = ( LeakyReLU())(MLP1)
   MLP1 = ( Convolution1D(1024, kernel_size=(1), strides=(1), padding='valid') )(MLP1)
   MLP1 = ( BatchNormalization() )(MLP1)
   out1 = ( LeakyReLU())(MLP1) #(None,4096,1024)
   maxp = ( MaxPooling1D(pool_size=(n_points)))(out1)

   MLP2 = Sequential()
   MLP2.add( Convolution1D(256, kernel_size=(1), strides=(1), padding='valid', input_shape=(1,1024) ) )
   MLP2.add( BatchNormalization() )
   MLP2.add( LeakyReLU())
   MLP2.add( Convolution1D(128, kernel_size=(1), strides=(1), padding='valid', kernel_initializer='ones') )
   MLP2.add( BatchNormalization() )
   MLP2.add( LeakyReLU())
   out2 = MLP2(maxp) #(None,1,128)

   out2_rep = Lambda(lambda x: K.repeat_elements(x, 4096, axis=1))(out2)
   conc = concatenate([out1, out2_rep], axis=2)

   MLP3 = Sequential()
   MLP3.add( Convolution1D(512, kernel_size=(1), strides=(1), padding='valid', input_shape=(4096,1152) ) )
   MLP3.add( BatchNormalization() )
   MLP3.add( LeakyReLU())
   MLP3.add( Convolution1D(256, kernel_size=(1), strides=(1), padding='valid') )
   MLP3.add( Dropout(0.3) )
   out3 = MLP3(conc)

   classifier = Sequential()
   classifier.add( Dense(NUM_CLASSES,activation='softmax', input_shape=(4096,256) ) )
   out4 = classifier(out3)

   pointnet_class = Model(input1, out4)
   pointnet_class.summary()

   # compile
   adam = Adam(lr=lrate, beta_1=0.9, decay=0.0)
   pointnet_class.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])

   # save after every epoch
   checkpoint = ModelCheckpoint( os.path.join(os.getcwd(),'pointnet') , monitor='val_loss', verbose=1, save_best_only=False, mode='min')

   lrate = LearningRateScheduler(step_decay)
   pointnet_class.fit(x_train, y_train, epochs=40, batch_size=32, callbacks=[lrate,checkpoint])

   print "\n%s - Finished training" % (time.ctime())

   return pointnet_class


### LOAD
save_train_data_path = os.path.join(os.getcwd(),'training_data')

with h5py.File(save_train_data_path, 'r') as hf:
    x_train = hf.get('x_train')
    y_train = hf.get('y_train')

    x_train = np.array(x_train)
    y_train = np.array(y_train)

#TRAINING
pointnet(x_train, y_train)

#TESTING
save_test_data_path = os.path.join(os.getcwd(),'testing_data')
with h5py.File(save_test_data_path, 'r') as hf:
    x_test = hf.get('x_test')
    y_test = hf.get('y_test')

    x_test = np.array(x_test)
    y_test = np.array(y_test)

testing(x_test,y_test)




