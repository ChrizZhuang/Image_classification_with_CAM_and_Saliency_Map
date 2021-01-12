# import required packages
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from tensorflow.keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,GlobalAveragePooling2D
import cv2 # import OpenCV2
import "../include/model.py" # import the classifier

# Load the data and create the train set
train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)

def augmentimages(image, label):
    """ Define a function that takes in animage and label. """

    # cast to float
    image = tf.cast(image, tf.float32)
    # normalize the pixel values
    image = (image/255)
    # resize to 300 x 300
    image = tf.image.resize(image,(300,300))

    return image, label

augmented_training_data = train_data.map(augmentimages)
train_batches = augmented_training_data.shuffle(1024).batch(32)

# Build the classifier
model = Sequential()
model.add(Conv2D(16,input_shape=(300,300,3),kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(2,activation='softmax'))

def do_salience(image, model, label, prefix):
    '''
    Generates the saliency map of a given image.

    Args:
        image (file) -- picture that the model will classify
        model (keras Model) -- your cats and dogs classifier
        label (int) -- ground truth label of the image
        prefix (string) -- prefix to add to the filename of the saliency map
    '''

    # Read the image and convert channel order from BGR to RGB
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    #print(type(img))

    # Resize the image to 300 x 300 and normalize pixel values to the range [0, 1]
    img = cv2.resize(img, (300, 300)) / 255.0


    # Add an additional dimension (for the batch), and save this in a new variable
    image_batch = np.expand_dims(img, axis=0)

    # Declare the number of classes
    num_classes = 2

    # Define the expected output array by one-hot encoding the label
    # The length of the array is equal to the number of classes
    expected_output = tf.one_hot([label] * image_batch.shape[0], num_classes)

    # Witin the GradientTape block:
    with tf.GradientTape() as tape:
        # Cast the image as a tf.float32
        inputs = tf.cast(image_batch, tf.float32)
        # Use the tape to watch the float32 image
        tape.watch(inputs)
        # Get the model's prediction by passing in the float32 image
        predictions = model(inputs)
        # print(predictions.shape)
        # Compute an appropriate loss
        loss = tf.keras.losses.categorical_crossentropy(
            expected_output, predictions
        )

    # get the gradients of the loss with respect to the model's input image
    gradients = tape.gradient(loss, inputs)

    # generate the grayscale tensor
    grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)

    # normalize the pixel values to be in the range [0, 255].
    # the max value in the grayscale tensor will be pushed to 255.
    # the min value will be pushed to 0.
    # Use the formula: 255 * (x - min) / (max - min)
    # Use tf.reduce_max, tf.reduce_min
    # Cast the tensor as a tf.uint8
    normalized_tensor = tf.cast(
        255
        * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
        / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
        tf.uint8,
    )
        
    # Remove dimensions that are size 1
    normalized_tensor = tf.squeeze(normalized_tensor)
    
    # plot the normalized tensor
    # Set the figure size to 8 by 8
    # do not display the axis
    # use the 'gray' colormap
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(normalized_tensor, cmap='gray')
    plt.show()

    # superimpose the saliency map with the original image, then display it.
    gradient_color = cv2.applyColorMap(normalized_tensor.numpy(), cv2.COLORMAP_HOT)
    gradient_color = gradient_color / 255.0
    super_imposed = cv2.addWeighted(img, 0.5, gradient_color, 0.5, 0.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(super_imposed)
    plt.axis('off')
    plt.show()


    # save the normalized tensor image to a file. this is already provided for you.
    salient_image_name = prefix + image
    normalized_tensor = tf.expand_dims(normalized_tensor, -1)
    normalized_tensor = tf.io.encode_jpeg(normalized_tensor, quality=100, format='grayscale')
    writer = tf.io.write_file(salient_image_name, normalized_tensor)

# load different weights
#model.load_weights('../pretrained-weight/0_epochs.h5')
#model.load_weights('../pretrained-weight/15_epochs.h5')
model.load_weights('../pretrained-weight/95_epochs.h5')

# generate the saliency maps for the 5 test images
do_salience('cat1.jpg', model, 0, 'epoch0_salient')
do_salience('cat2.jpg', model, 0, 'epoch0_salient')
do_salience('catanddog.jpg', model, 0, 'epoch0_salient')
do_salience('dog1.jpg', model, 1, 'epoch0_salient')
do_salience('dog2.jpg', model, 1, 'epoch0_salient')

# compile the model
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=tf.keras.optimizers.RMSprop(lr=0.001))

# load pre-trained weights
model.load_weights('../pretrained-weight/15_epochs.h5')

# train the model for just 3 epochs
model.fit(train_batches, epochs=10)

do_salience('cat1.jpg', model, 0, "epoch95_salient")
do_salience('cat2.jpg', model, 0, "epoch95_salient")
do_salience('catanddog.jpg', model, 0, "epoch95_salient")
do_salience('dog1.jpg', model, 1, "epoch95_salient")
do_salience('dog2.jpg', model, 1, "epoch95_salient")

# wrap up the files
from zipfile import ZipFile

filenames = ['cat1.jpg', 'cat2.jpg', 'catanddog.jpg', 'dog1.jpg', 'dog2.jpg']

# writing files to a zipfile 
with ZipFile('images.zip','w') as zip:
  for file in filenames:
    zip.write('salient' + file)

print("images.zip generated!")