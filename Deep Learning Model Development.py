
# Simple Deep Learning Model - MNIST Classification
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
import keras
from keras import layers


# Step 1: Load dataset (MNIST handwritten digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Step 2: Preprocess data (normalize to 0-1 range)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 3: Build the model (Simple Neural Network)
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # Flatten 28x28 images into 784 vector
    layers.Dense(128, activation='relu'),   # Hidden layer
    layers.Dense(10, activation='softmax')  # Output layer (10 classes)
])

# Step 4: Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Step 6: Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Step 7 (Optional): Save model for deployment
model.save("mnist_model.h5")
