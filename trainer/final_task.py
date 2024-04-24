#!/usr/bin/env python3

"""Train and export the model

This file trains the model upon all data with the arguments it got via
the gcloud command."""

import os
import argparse
import logging

from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf

import trainer.data as data
import trainer.model as model

from keras.callbacks import ReduceLROnPlateau
# from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback


def prepare_prediction_image(image_str_tensor):
    """Prepare an image tensor for prediction.
    Takes a string tensor containing a binary jpeg image and returns
    a tensor object of the image with dtype float32.

    Parameters:
        image_str_tensor: a tensor containing a binary jpeg image as a string
    Returns:
        image: A tensor representing an image.
    """
    image_str_tensor = tf.cast(image_str_tensor, tf.string)
    image = tf.image.decode_jpeg(image_str_tensor, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = data.preprocess_image(image) # I would have done a condition to check if the images need to be normalize

    # image = needs_normalization(image)

    return image


def prepare_prediction_image_batch(image_str_tensor):
    """Prepare a batch of images for prediction."""
    # tf.map_fn function in TensorFlow is used to apply a specified function (fn) to each element in the input tensor and return a new tensor
    # containing the results. It's particularly useful for applying a transformation function to each element of a batch of data or any higher-dimensional tensor,
    # where you want the transformation to be applied element-wise.
    return tf.map_fn(prepare_prediction_image,
                     image_str_tensor,
                     dtype=tf.float32) # dtype argument will be removed in a future version

    ###### WARNING:tensorflow:From /home/quentin/.pyenv/versions/portfol/lib/python3.10/site-packages/tensorflow/python/util/deprecation.py:629: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
    # Instructions for updating: Use fn_output_signature instead

def export_model(ml_model, export_dir, model_dir='exported_model'):
    """Prepare model for strings representing image data and export it.

    Before the model is exported the initial layers of the model need to be
    adapted to handle prediction of images contained in JSON files.

    Parameters:
        ml_model: A compiled model
        export_dir: A string specifying the
        model_dir: A string specifying the name of the directory to
            which the model is written.
    """
    prediction_input = tf.keras.Input(dtype=tf.string,
                                      name='bytes',
                                      shape=())

    prediction_output = tf.keras.layers.Lambda(prepare_prediction_image_batch)(prediction_input)

    prediction_output = ml_model(prediction_output)
    prediction_output = tf.keras.layers.Lambda(lambda x: x,
                                               name='PROBABILITIES')(prediction_output)
    prediction_class = tf.keras.layers.Lambda(lambda x: tf.argmax(x, 1),
                                              name='CLASSES')(prediction_output)

    ml_model = tf.keras.models.Model(prediction_input,
                                     outputs=[prediction_class, prediction_output])

    model_path = Path(export_dir) / model_dir
    if model_path.exists():
        timestamp = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
        model_path = Path(str(model_path) + timestamp)

    tf.saved_model.save(ml_model, str(model_path))

def train_and_export_model(params):
    """The function gets the training data from the training folder and
    the eval folder.
    Your solution in the model.py file is trained with this training data.
    The evaluation in this method is not important since all data was already
    used to train.

    Parameters:
        params: Parameters for training and exporting the model
    """
    (train_data, train_labels) = data.create_data_with_labels("data/train/")
    (eval_data, eval_labels) = data.create_data_with_labels("data/eval/")

    train_data = np.append(train_data, eval_data, axis=0)
    train_labels = np.append(train_labels, eval_labels, axis=0)

    img_shape = train_data.shape[1:] # (160, 160, 3)
    input_layer = tf.keras.Input(shape=img_shape, name='input_image') # <KerasTensor: shape=(None, 160, 160, 3) dtype=float32 (created by layer 'input_image')>

    # ------------------------------- #
    #         Data Augmentation       #
    # ------------------------------- #

    train_generator = data.create_train_generator(train_data, train_labels, batch_size=20)

    # ------------------------------- #
    #         Early Stopping          #
    # ------------------------------- #

    # es = EarlyStopping(
    #                     monitor='val_loss',  # Metric to monitor
    #                     patience=10,         # How many epochs to wait after min has been hit
    #                     verbose=1,
    #                     restore_best_weights=True  # Restores model weights from the epoch with the minimum monitored metric.
    #                   )

    # ------------------------------- #
    #     Learning Rate Scheduler     #
    # ------------------------------- #

    # Step Plateau

    # reduce_lr = ReduceLROnPlateau(
    #                           monitor='val_loss',
    #                           factor=0.2,   # New learning rate = factor * current learning rate
    #                           patience=5, # The factor .2 and patience=5 indicate that the learning rate will be reduced if no improvement in val_loss is seen for 5 epochs
    #                           min_lr=0.0005,  # Lower bound on the learning rate
    #                           verbose=1
    #                       )

    # Step decay

    # ---------------------------------------- #
    #  Integrate Class Weights into Training   #
    # ---------------------------------------- #


    ml_model = model.solution(input_layer)
    print("✅ Model is now fitting")

    ml_model.fit(
                train_generator,
                # train_data,
                 # train_labels,
                batch_size=model.get_batch_size(),
                epochs=model.get_epochs(),
                # callbacks = [es,reduce_lr],
                # callbacks = [reduce_lr]
                )

    export_model(ml_model, export_dir=params.job_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default='output',
        help='directory to store checkpoints'
    )

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level // 10)

    train_and_export_model(args)
