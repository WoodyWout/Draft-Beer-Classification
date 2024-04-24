#!/usr/bin/env python3
from keras import Sequential, regularizers
from keras.optimizers import Adam # SGD
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, PReLU #, Input, Layer, RandomFlip, RandomRotation
from keras.initializers import Constant

"""Model to classify draft beers

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 16

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 100

# @tf.function
def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different beers.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    # Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """

    # Model Architecture
    model = Sequential()

    # Convolutional Layers 32
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=input_layer.shape[1:]))
    model.add(PReLU(alpha_initializer=Constant(value=0.1)))# added the `PReLU()` layer right after each `Conv2D` layer. This is necessary because `PReLU` has trainable parameters and must be added as a separate layer rather than as an activation function parameter within `Conv2D`.
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(PReLU(alpha_initializer=Constant(value=0.1)))# added the `PReLU()` layer right after each `Conv2D` layer. This is necessary because `PReLU` has trainable parameters and must be added as a separate layer rather than as an activation function parameter within `Conv2D`.
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3))
    model.add(MaxPooling2D((2, 2)))  # ✅ Good practice ✅ : After each convolution, add a Max-Pooling layer
    model.add(Dropout(0.3))

    # Convolutional Layers 64
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(PReLU(alpha_initializer=Constant(value=0.1)))# added the `PReLU()` layer right after each `Conv2D` layer. This is necessary because `PReLU` has trainable parameters and must be added as a separate layer rather than as an activation function parameter within `Conv2D`.
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(PReLU(alpha_initializer=Constant(value=0.1)))# added the `PReLU()` layer right after each `Conv2D` layer. This is necessary because `PReLU` has trainable parameters and must be added as a separate layer rather than as an activation function parameter within `Conv2D`.
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # Convolutional Layers 128
    model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(PReLU(alpha_initializer=Constant(value=0.1)))# added the `PReLU()` layer right after each `Conv2D` layer. This is necessary because `PReLU` has trainable parameters and must be added as a separate layer rather than as an activation function parameter within `Conv2D`.
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3))

    model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(PReLU(alpha_initializer=Constant(value=0.1)))# added the `PReLU()` layer right after each `Conv2D` layer. This is necessary because `PReLU` has trainable parameters and must be added as a separate layer rather than as an activation function parameter within `Conv2D`.

    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Flatten the feature maps
    model.add(Flatten())

    # Dense Layers
    # Regularization
    reg_l2 = regularizers.L2(0.001) # weight regularization or weight decay : during training, an additional term is added to the loss function that penalizes large weights in the dense layer.

    model.add(Dense(128, kernel_regularizer=reg_l2))
    # model.add(Dense(128, kernel_initializer='he_uniform')) #❗️
    model.add(PReLU(alpha_initializer=Constant(value=0.1)))# added the `PReLU()` layer right after each `Conv2D` layer. This is necessary because `PReLU` has trainable parameters and must be added as a separate layer rather than as an activation function parameter within `Conv2D`.
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(5, activation='softmax')) # 'softmax' : multiclass classification problems

    print("✅ Model initialized")

    initial_learning_rate = 0.005

    adam_op = Adam(
                    learning_rate=initial_learning_rate, #❗️
                    beta_1=0.9, # Beta1 : controls the exponential decay rate for the first moment estimates (mean) in Adam. It should be a value between 0 and 1. A common default value is 0.9, which means that the first moment estimate decays quickly, allowing the optimizer to adapt to rapid changes in gradients.
                    beta_2=0.999, # Beta2 : controls the exponential decay rate for the second moment estimates (variance) in Adam. Like beta 1, it should be a value between 0 and 1. A typical default value is 0.999, which means that the second moment estimate decays more slowly, providing smoother updates to the model weights.
                    #  clipvalue=0.5 # gradients will be clipped to the range [-0.5, 0.5]
                    )

    model.compile(
                    loss='sparse_categorical_crossentropy', # multiclasses
                    optimizer=adam_op,
                    metrics=['accuracy'])

    print("✅ Model compiled")

    return model
