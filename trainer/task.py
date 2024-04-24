#!/usr/bin/env python3

"""Train and evaluate the model

This file trains the model upon the training data and evaluates it with
the eval data.
It uses the arguments it got via the gcloud command."""

import os
import argparse
import logging

import tensorflow as tf

import trainer.data as data
import trainer.model as model


def train_model(params):
    """The function gets the training data from the training folder,
    the evaluation data from the eval folder and trains your solutioninput
    from the model.py file with it.

    Parameters:
        params: parameters for training the model
    """
    (train_data, train_labels) = data.create_data_with_labels("data/train/")
    (eval_data, eval_labels) = data.create_data_with_labels("data/eval/")

    img_shape = train_data.shape[1:] # (160, 160, 3)
    input_layer = tf.keras.Input(shape=img_shape, name='input_image') # <KerasTensor: shape=(None, 160, 160, 3) dtype=float32 (created by layer 'input_image')>

    # Model using best_params from Bayesian Optimizer

    ml_model = model.solution(input_layer)


    if ml_model is None:
        print("No model found. You need to implement one in model.py")
    else:
        print("✅ Model is now fitting")
        ml_model.fit(
                     train_data,
                     train_labels,
                     batch_size=int(model.get_batch_size()),
                     epochs=model.get_epochs(),
                    #  validation_split = 0.3, # Few considerations : Randomness, Data Splitting, Data Leakage
                    #  callbacks = [es],
                    #  callbacks = [lr_exp_scheduler],
                     )

        ml_model.evaluate(eval_data, eval_labels, verbose=1)
    print(f'losses is = {round(ml_model.evaluate(eval_data, eval_labels, verbose=1)[0],3)} , accuracy is = {round(ml_model.evaluate(eval_data, eval_labels, verbose=1)[1],3)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_logger.level // 10)

    train_model(args)
