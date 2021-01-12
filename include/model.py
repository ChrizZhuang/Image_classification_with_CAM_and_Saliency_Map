# import required packages
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPooling2D,GlobalAveragePooling2D

# Build the classifier
class classifier():
    """
    Construct the classification model 
    """
    def __init__(self, input_shape=(300,300,3),kernel_size=(3,3),act_func='relu',padding='same', num_classes=2):

        model = Sequential()
        model.add(Conv2D(16,input_shape=input_shape,kernel_size=kernel_size,activation=act_func, padding=padding))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(32,kernel_size=kernel_size,activation=act_func, padding=padding))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,kernel_size=kernel_size,activation=act_func, padding=padding))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128,kernel_size=kernel_size,activation=act_func, padding=padding))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(num_classes,activation='softmax'))s

