

from __future__ import print_function
import keras # this is needed
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
#import tensorflow

number_classes = 5
image_rows,image_columns= 48, 48
batch_size = 32

# The absolute path of your training/validation data
train_data_path = '/Users/ganesh/Ganesh/fer2013/train'
validation_data_path = '/Users/ganesh/Ganesh/fer2013/validation'

# train data generator
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    shear_range=0.3,
                    zoom_range=0.3,
                    width_shift_range=0.4,
                    height_shift_range=0.4,
                    horizontal_flip=True,
                    fill_mode='nearest')

# validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)

# Train generator
train_generator = train_datagen.flow_from_directory(
                    train_data_path,
                    color_mode='grayscale',
                    target_size=(image_rows,image_columns),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=True)

#validation generator
validation_generator = validation_datagen.flow_from_directory(
                          validation_data_path,
                          color_mode='grayscale',
                          target_size=(image_rows,image_columns),
                          batch_size=batch_size,
                          class_mode='categorical',
                          shuffle=True)

# define our model you can name it whatever you want
model = Sequential()

# layer 1
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(image_rows,image_columns,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(image_rows,image_columns,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# layer 2
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# layer 3
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# layer 4
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# layer 5
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# layer 6
model.add(Dense(number_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

# Display the total neurons in our model named model
print(model.summary())

# checkpoint 
checkpoint = ModelCheckpoint('weights.h5',
                              monitor='val_loss',
                              mode='min',
                              save_best_only=True,
                              verbose=1)
# early stopping
earlystop = EarlyStopping(monitor='val_loss', 
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

# reduce linear regression on plateau
reduce_lr= ReduceLROnPlateau(monitor='val_loss',
                             factor=0.2,
                             patience=3,
                             verbose=1,
                             min_delta=0.0001)

# callbacks
callbacks = [earlystop,checkpoint,reduce_lr]

# compile our model and optimize using Adam 
model.compile(loss='categorical_crossentropy',
               optimizer=Adam(lr=0.001),
               metrics=['accuracy'])

# number of samples respectively 
nb_train_samples=24176
nb_validation_samples=3006

#epoch=25 is not the primary value. this value was the last time i trained the model with.
epochs=25

# for back propagation
history=model.fit_generator(
              train_generator,
              steps_per_epoch=nb_train_samples//batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=validation_generator,
              validation_steps=nb_validation_samples//batch_size)
