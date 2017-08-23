#Convolutional Neural Network

#In this example, rather than solve a business problem, the class had us solve a classification
#of photos of cats and dogs

#The training and test data has already been put together in a folder structure
#that Keras can read

#Part 1 - Building the CNN
# Import Keras library and other packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1- Convolution
#Common practice is to use 32 filters/feature detectors, and 3x3 dimension
classifier.add(Convolution2D(filters=32,kernel_size=(3,3), input_shape = (64, 64, 3), activation = 'relu'))

#Step 2- Pooling - usually do a 2x2 matrix for pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Add a second Convolutional Layer and apply Pooling (this is after making a first run)
#NOTE: do not need the input_shape parameter because Keras knows that you have a prior layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Step 3- Flattening
classifier.add(Flatten())

#Step 4- Full Connection (we don't use the average inputs/outputs to determine size of hidden
#layer rule of thumb here, have to keep enough features to process the image)
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the images
#We'll need to preprocess the images to augment them, which will prevent overfitting
#One of the situations that usually leads to overfitting is when we don't have enough data
#in training.  For images, we need a LOT of images to make sure we don't overfit, but we don't
#have enough.  Augmentation will create many batches of the images, and on each batch
#it will perform transformations on random sets of the images.  Sort of an oversampling method.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #must match the input shape given above^^
        batch_size=32,
        class_mode='binary') #have two classes, cats and dogs, so this is fine

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32, #according to new documentation, this should typically be set
                                 #to the number of unique samples (8000) divided by the batch size(32)
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/32) #same explanation as above