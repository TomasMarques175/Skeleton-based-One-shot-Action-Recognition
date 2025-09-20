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
from tqdm import tqdm
import os
import pickle
from demo_speed import ther_batch_iterator

import json

from data_generator import triplet_data_generator, get_scaler_filename, get_num_feats, therapy_data_generator
from train_callbacks import get_lr_metric  # eval_one_shot_callback, eval_one_shot_therapies_callback, 
import train_utils
from shutil import copyfile

from models.TCN_classifier import TCN_clf
# tf.config.experimental_run_functions_eagerly(True)

from dataset_scripts.ntu120_utils.triplet_ntu_callback import eval_ntu_one_shot_triplets_callback
from dataset_scripts.therapies.triplet_therapies_callback import eval_therapies_triplet_callback

from remove_suboptimal_weights import remove_path_weights
from evaluation_metrics import get_therapies_metrics, get_video_distances
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from sklearn.utils import resample
import pandas as pd

# Format: MM_DD_HH_MM (month_day_hour_minute)
timestamp = datetime.now().strftime('%m_%d_%H_%M_%S')

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import prediction_utils
import argparse

def get_weights_filename(path_model, loss_name, verbose=False, num_file=None):
    weights = sorted([ w for w in os.listdir(path_model + '/weights') if 'index' in w ])
    if verbose: print(weights)
    if loss_name is not None:
        weights = [ w for w in weights  if loss_name in w ][0]
    else:
        if num_file is not None:
            weights = weights[num_file]
        # weights = weights[0]
        elif 'mon' in weights[0]: 		# and False
            if verbose:  print('weights by monitor')
            weights = max(weights, key=lambda w: [ float(s[3:]) for s in w.replace('.ckpt.index', '').split('-') if s.startswith('mon') ][0])
        elif 'val_loss' not in weights[0]:
            if verbose: print('weights by last')
            weights = weights[-1]
        else:
            if verbose: print('weights by val_loss')
            losses = [ float(w.split('-')[2][8:15]) for w in weights ]
            weights = weights[losses.index(min(losses))]
    weights = weights[:-6]
    return path_model + '/weights/' + weights


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

def balance_classes(df, target_col='action', method='oversample', random_state=42):
    classes = df[target_col].unique()
    grouped = [df[df[target_col] == cls] for cls in classes]

    if method == 'oversample':
        max_size = max(len(g) for g in grouped)
        balanced = [
            resample(g, replace=True, n_samples=max_size, random_state=random_state)
            if len(g) < max_size else g
            for g in grouped
        ]
    elif method == 'undersample':
        min_size = min(len(g) for g in grouped)
        balanced = [
            resample(g, replace=False, n_samples=min_size, random_state=random_state)
            if len(g) > min_size else g
            for g in grouped
        ]
    else:
        raise ValueError("method must be 'oversample' or 'undersample'")

    return pd.concat(balanced).sample(frac=1, random_state=random_state).reset_index(drop=True)


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
    metrics_save_dir = os.path.join(parent_dir, f'Conversion comparison {timestamp}')
    os.makedirs(metrics_save_dir, exist_ok=True)

    # Set model parameters
    print(' * Setting model parameters')
    model = TCN_clf(**model_params)
    
    # Build model
    print(' * Building model')
    model.build((None, None, model_params['num_feats']))    
    # set_all_weights_to_value(model, value=0.01)
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
    #for i, output in enumerate(model_output_predict):
    #    np.save(f'keras_output_{i}.npy', output)

    # If you want a text file viewable line-by-line (works best for 2D arrays):
    #for i, output in enumerate(model_output_predict):
    #    reshaped = output.reshape(output.shape[0], -1)  # Flatten inner dims if needed
    #    np.savetxt(f'keras_output_{i}.txt', reshaped)

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
                    # ReduceLROnPlateau(monitor=monitor, min_delta=0.001, factor=0.1, patience=3, verbose=1, min_lr=1e-5),
                    EarlyStopping(monitor=monitor, min_delta=0.001, patience=6, verbose=1),
                    # EarlyStopping(monitor=monitor, min_delta=0.001, patience=10, verbose=1, restore_best_weights=True),
                ]

    file_writer = tf.summary.create_file_writer(model_params['path_model'] + "/metrics")
    file_writer.set_as_default()

    #if model_params['eval_ntu']: 
    #    callbacks = [LambdaCallback(_supports_tf_logs = True, 
    #                                on_epoch_end=eval_ntu_one_shot_triplets_callback(model, model_params.copy(), file_writer))] + callbacks
    # if model_params['eval_therapies']: 
    #     callbacks = [LambdaCallback(_supports_tf_logs = True, 
    #                                 on_epoch_end=eval_therapies_triplet_callback(model, model_params.copy(), file_writer, 'full'))] + callbacks
    #     callbacks = [LambdaCallback(_supports_tf_logs = True, 
    #                                 on_epoch_end=eval_therapies_triplet_callback(model, model_params.copy(), file_writer, 'sample'))] + callbacks
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
    
    train_gen = triplet_data_generator(pose_annotations_file=model_params['train_annotations'], 
                            validation=False, 
                            in_memory_generator=model_params['in_memory_generator_train'],
                            **model_params)
    
    # train_gen = triplet_data_generator(pose_annotations_file=model_params['train_annotations'], 
    #                             validation=False, 
    #                             in_memory_generator=model_params['in_memory_generator_train'],
    #                             **model_params)

    if model_params['val_annotations'] == '': val_gen = None
    else:
        print(' * Creating validation data generator')
        val_gen = triplet_data_generator(pose_annotations_file=model_params['val_annotations'], 
                        validation=True, 
                        in_memory_generator=model_params['in_memory_generator_val'],
                        **model_params)
    
    
    ## Get therapy data
    #print(' * Getting therapy data')
    #raw_data_path = './datasets/therapies_dataset/'
    #
    #actions_data = pickle.load(
    #    open(os.path.join(raw_data_path, 'actions_data_v2.pckl'), 'rb'))
    #actions_data = actions_data[~actions_data.action.isin(['no', 'si'])]
    #actions_data = actions_data.sort_values(
    #    by=['patient', 'session', 'video', 'ex_num'])
    #
    #video_skels = pickle.load(
    #    open(os.path.join(raw_data_path, 'video_skels_v2.pckl'), 'rb'))
    #
    ## Split while keeping action label balance
    #splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

    #for train_idx, val_idx in splitter.split(actions_data, actions_data['action']):
    #    train_data = actions_data.iloc[train_idx].reset_index(drop=True)
    #    val_data = actions_data.iloc[val_idx].reset_index(drop=True)


    #num_train_files = len(train_data)
    #print(f"\nNumber of training files: {num_train_files}")
    ## Check class distributions
    #print("Balanced train class distribution:")
    #print(train_data['action'].value_counts().sort_index())

    #balanced_train_data = balance_classes(train_data, method='oversample')

    #num_train_files = len(balanced_train_data)
    #print(f"\nNumber of training files: {num_train_files}")
    ## Check class distributions
    #print("Balanced train class distribution:")
    #print(balanced_train_data['action'].value_counts().sort_index())

    #num_val_files = len(val_data)
    #print(f"\nNumber of validation files: {num_val_files}")
    #print("\nValidation class distribution:")
    #print(val_data['action'].value_counts().sort_index())

    #exit()
    ## Create data generators
    #train_gen = therapy_data_generator(balanced_train_data, video_skels, pose_annotations_file=None, 
    #                                validation=False,
    #                                in_memory_generator=model_params['in_memory_generator_train'],
    #                                **model_params)

    #val_gen = therapy_data_generator(val_data, video_skels, pose_annotations_file=None, 
    #                                validation=True,
    #                                in_memory_generator=model_params['in_memory_generator_val'],
    #                                **model_params)
    #
    #print("\n========== First 2 batches from train_gen ==========")
    #for i in range(2):
    #    print(f"\n--- Train Batch {i+1} ---")
    #    X_train, Y_train, sample_weights_train = next(train_gen)
    #    print(f"Input shape: {X_train.shape}")
    #    print(f"Labels shape: {Y_train.shape}")
    #    
    ## Therapy data Obtained
    #print(' * Therapy data obtained')


    # if val_gen is not None:
    #     # Assume you already created val_gen and have model_params, etc.
    #     val_steps = len(val_data) // model_params['batch_size']  # or use your own logic to define validation_steps
    #     print(' * Validation steps:', val_steps)
    #     metrics_logger = MetricsLogger(
    #             validation_steps=val_steps,
    #             val_data=val_data,
    #             metrics_save_dir=metrics_save_dir,  # change this path as needed
    #             in_memory_generator=model_params['in_memory_generator_val'],
    #             model_params=model_params,
    #             validation_generator=val_gen 
    #         )
    #     callbacks.append(metrics_logger)

    if val_gen is not None:
        validation_steps = None if num_val_files == 0 else num_val_files//model_params['batch_size']
        print(' * Validation steps:', validation_steps)
        metrics_logger = MetricsLogger(validation_steps=validation_steps, pose_annotations_file=model_params['val_annotations'], 
                                        metrics_save_dir=metrics_save_dir, 
                                        in_memory_generator=model_params['in_memory_generator_val'], 
                                        model_params=model_params,
                                        validation_generator=val_gen)
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

    
    steps_per_epoch = num_train_files//model_params['batch_size']
    print(' * Steps per epoch:', steps_per_epoch)

    print(f' Num train files: {num_train_files}, Num val files: {num_val_files}')
    
    model.fit(
            train_gen,
            validation_data = val_gen,
            steps_per_epoch = num_train_files//model_params['batch_size'],
            validation_steps = None if num_val_files == 0 else num_val_files//model_params['batch_size'],
            epochs = 300,
            # epochs = 50, 
            # epochs = 300, 
            # steps_per_epoch = 10,         # num_val_files//model_params['batch_size'],
            # validation_steps = 10,
            verbose = train_verbose,
            callbacks = callbacks,
        )
    
    model.summary(100)
    
    del train_gen; del val_gen
    #del callbacks

    # Remove suboptimal weights
    # remove_path_weights(model_params['path_model'], model_params['monitor'], model_params['min_monitor'])


if __name__ == "__main__":

    model_params = {
        #"path_results": "./therapist_pretrained_models/",
        "path_results": "./pretrained_models/",
        
        # # NTU-120 Data sets to optimize the therapy data
        "train_annotations": "./ntu_annotations/one_shot_aux_set_train_full8.txt",
        "val_annotations": "./ntu_annotations/one_shot_aux_set_val_full8.txt",
        # "eval_therapies": False,       ### Therapy data needed for its evaluation
        # "eval_therapies_triplets_dataset": "./therapies_annotations/triplets/triplets_dataset.pckl",
        # "eval_therapies_triplets_bgnd_dataset": "./therapies_annotations/triplets/triplets_ther_pat_bgnd_dataset.pckl",
        # "eval_therapies_video_skels": "./therapies_annotations/video_skels.pckl",
        "h_flip": True,
        "skip_frames": [2, 3],

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

        # "batch_size": 3,       
        # "init_lr": 0.0001,
        # "init_lr": 0.001,
        # "lstm_recurrent_dropout": 0.0,
        # "lstm_dropout": 0.2,

        "batch_size": 64,       
        "init_lr": 0.0001,
        # "init_lr": 0.001,
        "lstm_recurrent_dropout": 0.0,
        "lstm_dropout": 0.2,

        "joints_num": 25,
        "joints_dim": 3,
        "max_seq_len": -32,

        # Set True to use a fitted data scaler. The one from the pre-trained models can also be used
        "scale_data": False,       
        "num_layers": 2,
        "num_neurons": 256,
        "masking": True,
        "center_skels": True,
        "scale_by_torso": True,
        "temporal_scale": [0.8, 1.2],
        "classification": True, 
        "triplet": False, 
        "decoder": False, 
        "reverse_decoder": False,
        "num_classes": 14,
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
        
        }
    
    main(model_params)
