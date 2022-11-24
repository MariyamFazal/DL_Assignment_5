# Importing Models

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D

from keras.utils import to_categorical


import warnings
warnings.filterwarnings('ignore')

from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalize
X_train = X_train/255
X_test = X_test/255
# One-Hot-Encoding
Y_train_en = to_categorical(Y_train,10)
Y_test_en = to_categorical(Y_test,10)

# Model
model = Sequential()
model.add(Conv2D(32,(4,4),input_shape = (32,32,3),activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(32,(4,4),input_shape = (32,32,3),activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(128, activation ="relu"))
model.add(Dense(10, activation ="softmax"))
model.compile(loss ="categorical_crossentropy", optimizer ="adam", metrics =["accuracy"])

model.summary()
History = model.fit(X_train, Y_train_en, epochs = 20, verbose=1,validation_data=(X_test,Y_test_en))
