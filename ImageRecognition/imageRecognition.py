#-- Importing Keras libraries and packages, Using TensorFlow as Backend
from keras.models import Sequential #used to initialize the Neural Network, 
from keras.layers import Convolution2D #used to do make the convolutional layers
from keras.layers import MaxPooling2D # this will add our pooling layers
from keras.layers import Flatten # converts pool matrix to a vector
from keras.layers import Dense # use to add fully connected layers


# all images must have the same format and size our images will be converted to 3D arrays
#intialize CNN
classifier = Sequential()

#---------- convolution, using multiple feature detectors to make feature maps -------------------
classifier.add(Convolution2D(32, 3, 3, input_shape =(64 ,64, 3 ), activation = 'relu')) 
# 32 feature maps of 3 by 3 size, input_shape 3 = 3D array of size 64 by 64, using this activation function will stop negative values
#convolution is good to make sure we arent just comparing one pixel. 
#--------- pooling step---------------------- this reduces the size by half because of the stride =2
#apply max pooling to convolutional layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#------------------------ Flattening ---------------------
classifier.add(Flatten())
#---------------------- Make an ANN -------------
classifier.add(Dense(output_dim = 128, activation ='relu'))
#we have 128 nodes in the hidden layer

#below is the output layer
classifier.add(Dense(output_dim = 1, activation ='sigmoid')) #binary outcome so use sigmoid function
#compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer chooses the stochastic gradient descent


#---------- fitting the CNN to the images-----------
#1) image augentation, this prevents our image from overfitting. we need to recognize pixels not independent variables
#     this will increase the amount of our images by alot, these are random transformation, this will reduce overfitting. 10,000 images is too little

from keras.preprocessing.image import ImageDataGenerator # this is to let us manipulate the images

train_datagen = ImageDataGenerator(rescale = 1./255, # all of our pixel values are from 0 to 1
                                   shear_range = 0.2, # random geometric transformation
                                   zoom_range = 0.2, # random zoom
                                   horizontal_flip = True)
        # batches are made to train by flipping some images and zooming and disorting, creates more test data
test_datagen = ImageDataGenerator(rescale = 1./255)

#creating training set 
training_set = train_datagen.flow_from_directory('dataset/training_set',  # this is the directory where the imaages are
                                                 target_size = (64, 64), # the expected size of our images
                                                 batch_size = 32,        # 32 images before retraining CNN 
                                                 class_mode = 'binary') #based on dependent variables

test_set = test_datagen.flow_from_directory('dataset/test_set', 
                                            target_size = (64, 64), # this will resize the image for us
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32, # images per training set
                         epochs = 25, # 50 is better but have to wait very long
                         validation_data = test_set,
                         validation_steps = 2000/32) 