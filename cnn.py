from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=1)
test_gen = ImageDataGenerator(rescale=1 / 255)
training_set = train_gen.flow_from_directory('/Users/rld1996/Documents/CNN_Data/training_set',
                                             target_size=(64, 64), batch_size=32, class_mode='binary')
testing_set = test_gen.flow_from_directory('/Users/rld1996/Documents/CNN_Data/test_set',
                                           target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=testing_set,
                         validation_steps=2000)
from keras.models import load_model
from keras.models import save_model

save_model('cd.h5')
