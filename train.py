#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:10:29 2020

@author: asabater
"""

import random
from scipy.special import comb
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
from tensorflow.keras.metrics import Precision, Recall
from metrics_logger import MetricsLogger
import os

import json

from data_generator import triplet_data_generator_deterministic, triplet_data_generator, get_scaler_filename, get_num_feats
from train_callbacks import get_lr_metric  # eval_one_shot_callback, eval_one_shot_therapies_callback, 
import train_utils
from shutil import copyfile

from models.TCN_classifier import TCN_clf
# tf.config.experimental_run_functions_eagerly(True)

from dataset_scripts.ntu120_utils.triplet_ntu_callback import eval_ntu_one_shot_triplets_callback
from dataset_scripts.therapies.triplet_therapies_callback import eval_therapies_triplet_callback

from remove_suboptimal_weights import remove_path_weights


SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def set_all_weights_to_value(model, value=0.01):
    for layer in model.layers:
        # Recursively access sub-layers (important for Sequential inside custom models)
        if isinstance(layer, tf.keras.Model):
            set_all_weights_to_value(layer, value)
        else:
            weights = layer.get_weights()
            if weights:
                new_weights = [np.full_like(w, value) for w in weights]
                layer.set_weights(new_weights)


class BatchLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename='batch_losses.json'):
        super().__init__()
        self.filename = filename
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            self.batch_losses.append(loss)

    def on_train_end(self, logs=None):
        with open(self.filename, 'w') as f:
            json.dump(self.batch_losses, f)


def main(model_params):
    train_verbose = 1

    model_params.update({
                'path_model': train_utils.create_model_folder(model_params['path_results'], model_params['model_name']),
                'num_jcd_feats': int(comb(model_params['joints_num'],2)), 
                'num_feats': int(comb(model_params['joints_num'],2)) + model_params['joints_dim']*model_params['joints_num'],
            })
    model_params['num_feats'] = get_num_feats(**model_params)
    json.dump(model_params, open(model_params['path_model']+'model_params.json', 'w'))
    print(model_params)
    
    with open(model_params['train_annotations'], 'r') as f: num_train_files = len(f.read().splitlines())
    if model_params['val_annotations']  == '': num_val_files = 0
    else:
        with open(model_params['val_annotations'], 'r') as f: num_val_files = len(f.read().splitlines())
    print(num_train_files, num_val_files)
    
    if model_params['scale_data']:
        scaler_filename = get_scaler_filename(**model_params)
        copyfile(scaler_filename, model_params['path_model'] + '/scaler.pckl')    
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    metrics_save_dir = os.path.join(parent_dir, 'Conversion comparison')
    os.makedirs(metrics_save_dir, exist_ok=True)

    
    # Set model parameters
    print(' * Setting model parameters')
    model = TCN_clf(**model_params)
    
    # Build model
    print(' * Building model')
    model.build((None, None, model_params['num_feats']))    
    set_all_weights_to_value(model, value=0.01)
    # print([w.numpy() for w in model.weights])  # Keras

    
    # Initialise dummy input and test model outputs
    print(' * Initialising dummy input and checking model outputs')
    print(' * model_params[batch_size]:', model_params['batch_size'])
    dummy_input = np.random.rand(
        model_params['batch_size'],
        max(abs(model_params['max_seq_len']), 123),
        model_params['num_feats']
    )

    print(' * dummy_input shape:', dummy_input.shape)

    # Forward pass via direct call
    model_output_direct = model(dummy_input)
    print(' * model_output_direct shapes:', [output.shape for output in model_output_direct])

    # Forward pass via predict()
    model_output_predict = model.predict(dummy_input)
    print(' * model_output_predict shapes:', [output.shape for output in model_output_predict])

    # Save each output (assumes model returns a list of tensors)
    for i, output in enumerate(model_output_predict):
        np.save(f'keras_output_{i}.npy', output)

    # If you want a text file viewable line-by-line (works best for 2D arrays):
    for i, output in enumerate(model_output_predict):
        reshaped = output.reshape(output.shape[0], -1)  # Flatten inner dims if needed
        np.savetxt(f'keras_output_{i}.txt', reshaped)

    # Get only the embedding output
    dummy_embedding = model.get_embedding(dummy_input)
    print(' * dummy_embedding shape:', dummy_embedding.shape)
    
    # Set optimizer
    print(' * Setting optimizer')
    optimizer = Adam(model_params['init_lr'], clipnorm=1.)
    losses, metrics, loss_weights, sample_weights_mode = {}, {}, {}, {}
    losses['output_1'] = tf.keras.losses.CategoricalCrossentropy()
    loss_weights['output_1'] = 0.4
    # loss_weights = None
    # loss_weights = [ 1.0 ]
    #metrics = [ 'accuracy', get_lr_metric(optimizer) ]
    metrics = [ 'accuracy', Precision(name='precision'), Recall(name='recall'), get_lr_metric(optimizer) ]


    # Show losses, metrics and loss_weights
    print(' * losses:', losses)
    print(' * loss_weights:', loss_weights)
    if sample_weights_mode == {}: sample_weights_mode = None
    print(' * sample_weights_mode:', sample_weights_mode)

    # Show model summary
    print(' * Model summary')
    model.summary(100)

    monitor = model_params.get('monitor', 'val_loss')
    print(' * Monitor:', monitor)
    model_chkpt_path = 'ep{epoch:03d}-loss{loss:.5f}-' + monitor + '{' + monitor + ':.5f}.ckpt'
    callbacks = [ 
                    TensorBoard(log_dir = model_params['path_model'], profile_batch=0),
                    ModelCheckpoint(model_params['path_model'] + 'weights/' + model_chkpt_path,
                                             monitor=monitor, save_weights_only=True, 
                                             save_best_only=True, save_freq='epoch'),
                    ReduceLROnPlateau(monitor=monitor, min_delta=0.001, factor=0.1, patience=3, verbose=1, min_lr=1e-7),
                    EarlyStopping(monitor=monitor, min_delta=0.001, patience=6, verbose=1),
                ]

    val_steps = num_val_files // model_params['batch_size']


    file_writer = tf.summary.create_file_writer(model_params['path_model'] + "/metrics")
    file_writer.set_as_default()

    #if model_params['eval_ntu']: 
    #    callbacks = [LambdaCallback(_supports_tf_logs = True, 
    #                                on_epoch_end=eval_ntu_one_shot_triplets_callback(model, model_params.copy(), file_writer))] + callbacks
    #if model_params['eval_therapies']: 
    #    callbacks = [LambdaCallback(_supports_tf_logs = True, 
    #                                on_epoch_end=eval_therapies_triplet_callback(model, model_params.copy(), file_writer, 'full'))] + callbacks
    #    callbacks = [LambdaCallback(_supports_tf_logs = True, 
    #                                on_epoch_end=eval_therapies_triplet_callback(model, model_params.copy(), file_writer, 'sample'))] + callbacks
    #print(callbacks)
    
    
    print(' * metrics:', metrics)
    print(' * sample_weights_mode:', sample_weights_mode)
    
    
    model.compile(optimizer=optimizer,
                  loss = losses,
                  metrics = metrics,
                  loss_weights = loss_weights,
                  sample_weight_mode=sample_weights_mode
                  )


    # Save model
    model.save(model_params['path_model'] + 'model')
    
    print('\n\n')
    print(' * Model saved to:', model_params['path_model'] + 'model')
    
    # TODO: change everything back to normal and not deterministic
    # train_gen = triplet_data_generator(pose_annotations_file=model_params['train_annotations'], 
    #                         validation=False, 
    #                         in_memory_generator=model_params['in_memory_generator_train'],
    #                         **model_params)
    
    # Deterministic Batches
    train_gen = triplet_data_generator_deterministic(pose_annotations_file=model_params['train_annotations'], 
                                validation=False, 
                                in_memory_generator=model_params['in_memory_generator_train'],
                                  **model_params)

    if model_params['val_annotations'] == '': val_gen = None
    else:
        print(' * Creating validation data generator')
        val_gen = triplet_data_generator_deterministic(pose_annotations_file=model_params['val_annotations'], 
                        validation=True, 
                        in_memory_generator=model_params['in_memory_generator_val'],
                           **model_params)

    if val_gen is not None:
        validation_steps = None if num_val_files == 0 else num_val_files//model_params['batch_size'],
        metrics_logger = MetricsLogger(validation_steps, pose_annotations_file=model_params['val_annotations'], 
                                       metrics_save_dir=metrics_save_dir, in_memory_generator=model_params['in_memory_generator_val'], **model_params)
        callbacks.append(metrics_logger)

    # Print the labels for the first 2 batches
    # for i in range(2):
    #     X_train, Y_train, sample_weights_train = next(train_gen)
    #     print(f"\nBatch {i+1} Input Shape:", X_train.shape)
    #     print(f"Batch {i+1} Labels Shape:", Y_train.shape)
    #     # If y_raw is not None, print its shape and contents
    #     X_val, Y_val, sample_weights_val = next(val_gen) if val_gen is not None else (None, None, None)
    #     if X_val is not None:
    #         print(f"\nValidation Batch {i+1} Input Shape:", X_val.shape)
    #         print(f"Validation Batch {i+1} Labels Shape:", Y_val.shape)
    #         print(f"\nSample Weights Validation {i+1} Labels:", sample_weights_val)

    # batch_loss_logger = BatchLossLogger(filename='batch_losses.json')
    
    # batch = next(iter(train_gen))  # or your generator
    # print(type(batch), len(batch))
    # TODO: Check why batch len is 4 instead of 3
    
    
    model.fit(
            train_gen,
            #validation_data = val_gen,
            steps_per_epoch = num_train_files//model_params['batch_size'],
            #validation_steps = None if num_val_files == 0 else num_val_files//model_params['batch_size'],
            epochs = 10,
            # epochs = 50, 
            # epochs = 300, 
            # steps_per_epoch = 10,         # num_val_files//model_params['batch_size'],
            # validation_steps = 10,
            verbose = train_verbose,
            callbacks = callbacks,
        )
    
    # Extract weights and biases from softmax output layer (Dense layer named 'out_clf')
    softmax_weights, softmax_biases = model.clf_out.get_weights()

    # Save them as numpy files for easy loading later
    np.save('softmax_weights.npy', softmax_weights)
    np.save('softmax_biases.npy', softmax_biases)

    model.summary(100)
    
    # Assuming train_gen is your training data generator
    last_batch_input, _ = next(iter(train_gen))

    np.savez('tf_results.npz',
        weights=model.get_weights(),
        outputs=model.predict(last_batch_input))

    # Save sample batch for comparison
    np.save('tf_sample_batch.npy', last_batch_input)

    del train_gen; del val_gen
    #del callbacks

    # Remove suboptimal weights
    remove_path_weights(model_params['path_model'], model_params['monitor'], model_params['min_monitor'])


if __name__ == "__main__":

    model_params = {
        "path_results": "./pretrained_models/",

        # # NTU-120 Data sets to optimize the therapy data
        "train_annotations": "./ntu_annotations/one_shot_aux_set_train_full8.txt",
        "val_annotations": "./ntu_annotations/one_shot_aux_set_val_full8.txt",
        # "eval_therapies": False,       ### Therapy data needed for its evaluation
        # "eval_therapies_triplets_dataset": "./therapies_annotations/triplets/triplets_dataset.pckl",
        # "eval_therapies_triplets_bgnd_dataset": "./therapies_annotations/triplets/triplets_ther_pat_bgnd_dataset.pckl",
        # "eval_therapies_video_skels": "./therapies_annotations/video_skels.pckl",
        "h_flip": True,
        # "skip_frames": [2, 3],

        # NTU-120 Data sets to optimize the NTU one-shot benchmark
        # "train_annotations": "./ntu_annotations/one_shot_aux_set.txt",
        # "val_annotations": "",
        # "eval_therapies": False,
        # "h_flip": False,
        # "monitor": "ntu_one_shot_acc_euc",
        # "min_monitor": False,
        # "skip_frames": [2],

        "in_memory_generator_train": False,
        "in_memory_generator_val": False,
        #"in_memory_callback": True,

        # "eval_ntu": True,
        # "eval_ntu_one_shot_eval_anchors_file": "./ntu_annotations/one_shot_eval_anchors.txt",
        # "eval_ntu_one_shot_eval_set_file": "./ntu_annotations/one_shot_eval_set.txt",

        "joints_num": 25,
        "joints_dim": 3,
        "init_lr": 0.0001,
        "max_seq_len": -32,

        # Set True to use a fitted data scaler. The one from the pre-trained models can also be used
        "scale_data": False,       
        "lstm_recurrent_dropout": 0.0,
        "lstm_dropout": 0.2,
        "num_layers": 2,
        "num_neurons": 256,
        "batch_size": 64,
        "masking": True,
        "center_skels": True,
        "scale_by_torso": True,
        "temporal_scale": [0.8, 1.2],
        "classification": True, "triplet": False, "decoder": False, "reverse_decoder": False,
        "num_classes": 120,
        "clf_neurons": 0,

        "model_name": "train_TCN",
        "conv_params": [256, 4, 2, True, "causal", [4]],
        "is_tcn": False,
        "use_jcd_features": True,
        "use_speeds": False,
        "use_coords_raw": False,
        "use_coords": True,
        "use_jcd_diff": False,
        "use_bone_angles": True,
        "use_bone_angles_cent": False,
        "average_wrong_skels": True,
        "average_wrong_skels_method": 'mean',   
        }
    
    main(model_params)
