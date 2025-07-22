import tensorflow as tf
from tensorflow import keras
from keras import layers 
from keras import models 

import numpy as np
import matplotlib.pyplot as plt

print("\n----- 1. Loading and Initial Data Preparation -----")

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"Original x_train shape: {x_train.shape}")   # (60000, 28, 28)
print(f"Original y_train shape: {y_train.shape}")   # (60000,)
print(f"Original x_test shape: {x_test.shape}")     # (10000, 28, 28)
print(f"Original y_test shape: {y_test.shape}")     # (10000,)

# Define class names for better understanding
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("-"*100)

print("\n----- 2. Data Preprocessing for CNNs -----")

# Instead of flattening as with the MLP, we need to add a "channels" dimension
    # Normalisation: Same as before (0-255 to 0.0-1.0)
    # Reshaping: For a grayscale image, Conv2D layers expect input shape to be (height, width, channels). 
        # Since Fashion MNIST images are grayscale, they have 1 channel. So, a 28x28 image needs to become 28x28x1.

# Convert integer pixel values to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalise pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"\nx_train shape after dtype conversion and normalisation: {x_train.shape}") # (60000, 28, 28)
print(f"x_test shape after dtype conversion and normalisation: {x_test.shape}")     # (10000, 28, 28)

# Reshape the data to add the channels dimension
# For grayscale, channels=1
# For color (RGB), channels=3
# Input shape for Conv2D should be (height, width, channels)
x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(f"x_train_cnn shape after reshaping for CNN: {x_train_cnn.shape}")            # (60000, 28, 28, 1)
print(f"x_test_cnn shape after reshaping for CNN: {x_test_cnn.shape}")              # (10000, 28, 28, 1)

# y_train and y_test are already integer labels (0-9), which is suitable for SparseCategoricalCrossentropy

print("-"*100)

print("\n----- 3. Model Definition: Building the CNN Architecture -----")

# Define the Keras Sequential model for CNN with Regularisation
model_cnn_regularised = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)), # Remove activation, as it is explicity used after normalisation
    layers.BatchNormalization(),            # Normalise after Conv, before Activation (or combined activation)
    layers.Activation('relu'),              # Explicit ReLU activation, i.e. introduce non-linearity
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second Convolutional Block
    layers.Conv2D(filters=64, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten the output
    layers.Flatten(),

    # Dense layers for classification with Dropout
    layers.Dense(128),
    layers.BatchNormalization(),            # Optional: Can add BN to dense layers too
    layers.Activation('relu'),
    layers.Dropout(0.5),                    # Dropout after activation (or after Dense with activation), e.g., 50% dropout

    layers.Dense(10, activation='softmax')  # Output layer
])

model_cnn_regularised.summary()

"""
BatchNormalization layer adds 4 parameters for every feature (or channel)
1. gamma (γ): A learnable scaling factor.
2. beta (β): A learnable offset (shift).
3. moving_mean (μ): A non-learnable running average of the batch means.
4. moving_variance (σ²): A non-learnable running average of the batch variances.

- gamma and beta are trainable parameters because the network learns their optimal values during backpropagation
- moving_mean and moving_variance are non-trainable parameters because they are statistics computed from the data (running averages) and are used for inference (when the model is evaluated or making predictions) to ensure consistent normalisation.
"""

# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │ # input_shape=(28, 28, 1); kernel_size=(3,3), filters=32; Total parameters = (kernel_height * kernel_width * input_channels + 1 (bias)) * number_of_filters = (3 * 3 * 1 + 1) * 32 = (9 + 1) * 32 = 10 * 32 = 320
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization                  │ (None, 26, 26, 32)          │             128 │ # Parameters = 4 * 32 = 128 (4 is the number of parameters BatchNormalization adds as described above)
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation (Activation)              │ (None, 26, 26, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │ # Total parameters = (kernel_height * kernel_width * input_channels + 1) * number_of_filters = (3 * 3 * 32 + 1) * 64 = (288 + 1) * 64 = 289 * 64 = 18496
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization_1                │ (None, 11, 11, 64)          │             256 │ # Parameters = 4 * 64 = 256
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation_1 (Activation)            │ (None, 11, 11, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 1600)                │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 128)                 │         204,928 │ # Total parameters = (input_features * neurons) + neurons = (1600 * 128) + 128 = 204800 + 128 = 204928
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ batch_normalization_2                │ (None, 128)                 │             512 │ # Parameters = 4 * 128 = 512
# │ (BatchNormalization)                 │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ activation_2 (Activation)            │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │ # Total parameters = (input_features * neurons) + neurons = (128 * 10) + 10 = 1280 + 10 = 1290
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 225,930 (882.54 KB)
#  Trainable params: 225,482 (880.79 KB)
#  Non-trainable params: 448 (1.75 KB)      # see note above for why there are non-trainable parameters

print("-"*100)

print("\n----- 4. Model Compilation -----")

# Same as the MLP
# Compile the CNN model
model_cnn_regularised.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

print("-"*100)

print("\n----- 5. Model Training -----")

# Using the CNN-reshaped data (x_train_cnn, x_test_cnn) for training and evaluation

# Train the CNN model
print("\nStarting CNN training...")
history_cnn = model_cnn_regularised.fit(
    x_train_cnn, y_train,
    epochs=10, 
    batch_size=32,
    validation_split=0.1
    )

# Starting CNN training...
# Epoch 1/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 13s 8ms/step - accuracy: 0.7799 - loss: 0.6321 - val_accuracy: 0.8858 - val_loss: 0.3026
# Epoch 2/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 18s 10ms/step - accuracy: 0.8771 - loss: 0.3449 - val_accuracy: 0.8963 - val_loss: 0.2739
# Epoch 3/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 19s 11ms/step - accuracy: 0.8901 - loss: 0.3010 - val_accuracy: 0.8787 - val_loss: 0.3220
# Epoch 4/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 19s 11ms/step - accuracy: 0.9038 - loss: 0.2684 - val_accuracy: 0.8778 - val_loss: 0.3175
# Epoch 5/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 20s 12ms/step - accuracy: 0.9057 - loss: 0.2550 - val_accuracy: 0.9023 - val_loss: 0.2618
# Epoch 6/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 20s 12ms/step - accuracy: 0.9154 - loss: 0.2337 - val_accuracy: 0.9048 - val_loss: 0.2624
# Epoch 7/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 20s 12ms/step - accuracy: 0.9218 - loss: 0.2178 - val_accuracy: 0.9148 - val_loss: 0.2319
# Epoch 8/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 19s 11ms/step - accuracy: 0.9234 - loss: 0.2054 - val_accuracy: 0.8970 - val_loss: 0.2794
# Epoch 9/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 20s 12ms/step - accuracy: 0.9288 - loss: 0.1928 - val_accuracy: 0.9172 - val_loss: 0.2284  # Minimal loss in validation set
# Epoch 10/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 21s 12ms/step - accuracy: 0.9299 - loss: 0.1882 - val_accuracy: 0.9090 - val_loss: 0.2477

print("-"*100)

print("\n----- 6. Model Evaluation -----")

# Evaluate the CNN model on the test data
test_loss_cnn, test_accuracy_cnn = model_cnn_regularised.evaluate(x_test_cnn, y_test, verbose=1)

print(f"\nCNN Test Loss: {test_loss_cnn:.4f}")          # 0.2695
print(f"CNN Test Accuracy: {test_accuracy_cnn:.4f}")    # 0.9066


print("-"*100)

print("\n----- 7. Making Predictions -----")

# Make predictions on a few test samples using the CNN model
predictions_cnn = model_cnn_regularised.predict(x_test_cnn[:5])
predicted_classes_cnn = np.argmax(predictions_cnn, axis=1)

print(f"\nCNN Predictions for the first 5 test samples (classes): {predicted_classes_cnn}") # [9 2 1 1 6]
print(f"Actual classes for the first 5 samples: {y_test[:5]}")                              # [9 2 1 1 6]

print("\nCNN Comparison of predicted vs actual:")
for i in range(5):
    print(f"Sample {i+1}: Predicted: {class_names[predicted_classes_cnn[i]]} (Index: {predicted_classes_cnn[i]}), Actual: {class_names[y_test[i]]} (Index: {y_test[i]})")

# CNN Comparison of predicted vs actual:
# Sample 1: Predicted: Ankle boot (Index: 9), Actual: Ankle boot (Index: 9)
# Sample 2: Predicted: Pullover (Index: 2), Actual: Pullover (Index: 2)
# Sample 3: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 4: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 5: Predicted: Shirt (Index: 6), Actual: Shirt (Index: 6)

"""
Changing dropout rates:

Dropout Rates   | Test Accuracy     | Test Loss     | Epoch with minimal loss in validation set
----------------|-------------------|---------------|-------------------------------------------
0.2             | 0.9102            | 0.2941        | 7
0.3             | 0.9098            | 0.2677        | 7
0.4             | 0.9073            | 0.2840        | 9
0.5             | 0.9066            | 0.2695        | 9
Previous CNN    | 0.9101            | 0.2874        | 7

"""
