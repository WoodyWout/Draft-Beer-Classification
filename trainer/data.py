#!/usr/bin/env python3

"""Load image files and labels

This file contains the method that creates data and labels from a directory.
"""
import os
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2


def create_data_with_labels(dataset_dir):
    """Gets numpy data and label array from images that are in the folders
    that are in the folder which was given as a parameter. The folders
    that are in that folder are identified by the beers they represent and
    the folder name starts with the label.

    Parameters:
        dataset_dir: A string specifying the directory of a dataset
    Returns:
        data: A numpy array containing the images
        labels: A numpy array containing labels corresponding to the images
    """
    image_paths_per_label = collect_paths_to_files(dataset_dir)

    images = []
    labels = []
    for label, image_paths in image_paths_per_label.items():
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)

    data = np.array([preprocess_image(image.astype(np.float32))
                     for image in images])
    labels = np.array(labels)

    return data, labels


def collect_paths_to_files(dataset_dir):
    """Returns a dict with labels for each subdirectory of the given directory
    as keys and lists of the subdirectory's contents as values.

    Parameters:
        dataset_dir: A string containing the path to a directory containing
            subdirectories to different classes.
    Returns:
        image_paths_per_label: A dict with labels as keys and lists of file
        paths as values.
    """
    dataset_dir = Path(dataset_dir)
    beer_dirs = [f for f in sorted(os.listdir(dataset_dir)) if not f.startswith('.')]
    image_paths_per_label = {
        label: [
            dataset_dir / beer_dir / '{0}'.format(f)
            for f in os.listdir(dataset_dir / beer_dir) if not f.startswith('.')
        ]
        for label, beer_dir in enumerate(beer_dirs)
    }
    return image_paths_per_label


def preprocess_image(image):
    """Returns a preprocessed image.

    Parameters:
        image: A RGB image with pixel values in range [0, 255].
    Returns
        image: The preprocessed image.
    """
    image = image / 255.

    return image

# ------------------------------- #
#         Data Augmentation       #
# ------------------------------- #

def create_train_generator(train_data, train_labels, batch_size=32):
    """
    Creates a training data generator with data augmentation.

    Parameters:
        train_data: The training data (images).
        train_labels: The training labels.
        batch_size: Size of the batches of data (default: 32).

    Returns:
        train_generator: An augmented data generator for training.
    """
    # Define the data augmentation
    train_datagen = ImageDataGenerator(
                                    width_shift_range=0.1, # Range for random horizontal shifts
                                    height_shift_range=0.1, # Range for random vertical shifts
                                    horizontal_flip=True, # Randomly flip inputs horizontally
                                    )

    # Create the generator
    train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)

    return train_generator

# ------------------------------- #
#     Normalized image or not     #
# ------------------------------- #


# def needs_normalization(images):
#     """Checks if the images need to be normalized.

#     Parameters:
#     images: A numpy array of RGB images with pixel values in range [0, 255].

#     Returns:
#     bool: True if the images need to be normalized, False otherwise.
#     """
#     max_pixel_value = np.max(images)
#     min_pixel_value = np.min(images)
#     if max_pixel_value > 1.0 or min_pixel_value < 0.0:
#         return True

#     else:
#         return False

# ---------------------------------------- #
#  Integrate Class Weights into Training   #
# ---------------------------------------- #

# def calculate_class_distribution(dataset_dir):

#     """
#     Collects paths to files, calculates class distribution, and computes class weights.

#     Parameters:
#     - dataset_dir: The directory path where the dataset is stored.

#     Returns:
#     - A dictionary with class weights like :
#      {0: 0.9259259259259259,
#       1: 1.098901098901099,
#       2: 1.2048192771084338,
#       3: 0.8771929824561403,
#       4: 0.9615384615384616}
#     """
#     # Calculate total counts per class from the file paths
#     image_paths_per_label = collect_paths_to_files(dataset_dir)
#     total_counts = {key: len(value) for key, value in image_paths_per_label.items()}

#     # Assuming 'train_labels' are available and match the classes in 'total_counts'
#     classes = np.array(list(total_counts.keys()))
#     train_labels = np.concatenate([[key] * value for key, value in total_counts.items()])
#     class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
#     class_weight_dict = dict(zip(classes, class_weights))

#     return class_weight_dict


# def should_use_class_weights(class_weight_dict, threshold=2.0):
#     """
#     Determine if class weights should be applied based on a threshold.

#     Args:
#     - class_weight_dict: Dictionary with class IDs as keys and their weights as values.
#     - threshold: A float value to determine the imbalance ratio that necessitates class weights.

#     Returns:
#     - A boolean indicating whether class weights should be applied or not.
#     """
#     # Calculate the ratio of the maximum weight to the minimum weight
#     max_weight = max(class_weight_dict.values())
#     min_weight = min(class_weight_dict.values())
#     ratio = max_weight / min_weight

#     # Apply class weights if the ratio exceeds the threshold
#     return ratio > threshold
