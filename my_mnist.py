import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs

import cv2
import matplotlib.pyplot as plt

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 2

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


model.save("my1stmodel.h5")

"""
## Predict 
"""

model= keras.models.load_model("my1stmodel.h5")

mnist_model = keras.models.load_model('./my1stmodel.h5')  
mnist_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## in-house generated test images
f_name = './3.PNG'

img = cv2.imread(f_name, cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img, (28,28)) # absolute size

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('original')
plt.subplot(1,2,2)
plt.imshow(img2)
plt.title('resized abs')

X_madeVal = []
X_madeVal.append(img2)
X_madeVal_np = np.array(X_madeVal)
X_madeVal_f = X_madeVal_np.reshape(X_madeVal_np.shape[0], 28, 28, 1).astype('float32') / 255

mnist_prediction = mnist_model.predict(X_madeVal_f)
y_predict = mnist_classes[mnist_prediction.argmax()]
print('The figure is '+ str(y_predict))
print("Accuracy : " + str(round(100*mnist_prediction.max(),2)) + " %")
